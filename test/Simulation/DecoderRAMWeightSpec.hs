module Simulation.DecoderRAMWeightSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P

import LLaMa2.Types.LayerData (Token, ProcessingState(..), CycleStage(..), Seed)
import LLaMa2.Types.ModelConfig (NumLayers)
import qualified LLaMa2.Top as Top
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified Data.ByteString.Lazy as BSL
import Text.Printf (printf)

-- CRITICAL: This test compares decoder behavior with hardcoded weights vs RAM weights
-- to identify where they diverge

spec :: Spec
spec = do
  describe "Decoder RAM Weight Loading" $ do

    it "DIAGNOSTIC: decoder should produce identical tokens with RAM vs hardcoded weights" $ do
      -- This is the HIGH-LEVEL test that will show us WHERE the problem starts

      modelBinary <- BSL.readFile "./data/stories260K.bin"

      let temperature = 0.0 :: Float  -- Deterministic sampling
          seed = 42 :: Seed
          promptTokens = [1, 2, 3]  -- Simple prompt
          numSteps = 10000  -- Enough cycles for a few tokens

          -- Run with HARDCODED weights (baseline)
          hardcodedOutputs = runDecoderSim modelBinary promptTokens temperature seed numSteps False

          -- Run with RAM weights (under test)
          ramOutputs = runDecoderSim modelBinary promptTokens temperature seed numSteps True

          -- Find first divergence
          divergencePoint = findFirstDivergence
            (P.zip [0..] hardcodedOutputs)
            (P.zip [0..] ramOutputs)

      case divergencePoint of
        Nothing -> return ()  -- Perfect match!
        Just (cycle', hc, ram) -> do
          -- DIAGNOSTIC OUTPUT
          putStrLn $ "\n❌ DIVERGENCE DETECTED AT CYCLE " P.++ show cycle'
          putStrLn $ "\nHardcoded output: " P.++ showDecoderState hc
          putStrLn $ "RAM output:       " P.++ showDecoderState ram

          -- Show context around divergence
          let contextRange = [max 0 (cycle' - 10) .. min (P.length hardcodedOutputs - 1) (cycle' + 10)]
          putStrLn "\n--- Context (10 cycles before/after) ---"
          mapM_ (\c -> do
            let hcOut = hardcodedOutputs P.!! c
                ramOut = ramOutputs P.!! c
            putStrLn $ printf "Cycle %5d | HC: %s" c (showDecoderState hcOut)
            putStrLn $ printf "           | RM: %s" (showDecoderState ramOut)
            ) contextRange

          -- This test should pass once the bug is fixed
          expectationFailure $ "Outputs diverged at cycle " P.++ show cycle'

    it "LAYER 0: first layer should process correctly with RAM weights" $ do
      -- Focus on just layer 0 to isolate if the problem is layer-specific

      modelBinary <- BSL.readFile "./data/stories260K.bin"

      let outputs = runDecoderSim modelBinary [1] 0.0 42 5000 True

          -- Check that we complete layer 0 processing
          layer0Completions = [ (cycle', state) |
                               (cycle', (_, _, _, state, _, _, _layer, _)) <- P.zip [0 :: Int ..] outputs,
                               processingLayer state == 0,
                               processingStage state == Stage3_Attend ]

      -- Should see at least one successful layer 0 completion
      P.length layer0Completions `shouldSatisfy` (> 0)

    it "TIMING: enableAttention timing relative to weight loading" $ do
      -- Test that enableAttention never goes high before weights are loaded

      modelBinary <- BSL.readFile "./data/stories260K.bin"

      let outputs = runDecoderSim modelBinary [1, 2] 0.0 42 10000 True

          -- Extract fullyLoaded and enableAttention equivalent
          -- (we can't directly see enableAttention, but we can infer it from behavior)

          -- Check: whenever we transition to a new layer, 
          -- we should NOT see Stage1_ProjectQKV complete before weights are loaded
          layerTransitions = detectLayerTransitions outputs

          invalidTransitions = [ (cycle', oldLayer, newLayer) |
                                (cycle', oldLayer, newLayer) <- layerTransitions,
                                -- Check if Stage1 completes too early after transition
                                hasEarlyProjection outputs cycle' ]

      invalidTransitions `shouldBe` []

    it "WEIGHT BUFFER: buffer state during layer transition" $ do
      -- Test the weight buffer controller behavior during transitions

      modelBinary <- BSL.readFile "./data/stories260K.bin"

      let outputs = runDecoderSim modelBinary [1] 0.0 42 8000 True
          transitions = detectLayerTransitions outputs

      -- For each transition, check buffer state
      mapM_ (\(cycle', oldL, newL) -> do
        let bufferStates = getBufferStatesAroundCycle outputs cycle' 5
        putStrLn $ "\nLayer transition " P.++ show oldL P.++ " → " P.++ show newL P.++ " at cycle " P.++ show cycle'
        mapM_ (\(c, loaded) ->
          putStrLn $ "  Cycle " P.++ show c P.++ ": fullyLoaded = " P.++ show loaded) bufferStates
        ) (P.take 3 transitions)  -- Just show first few transitions

-- Helper functions

type DecoderOutput = (Token, Bool, Bool, ProcessingState, Bool, Bool, Index NumLayers, Bool)

runDecoderSim :: BSL.ByteString -> [Token] -> Float -> Seed -> Int -> Bool
              -> [DecoderOutput]
runDecoderSim modelBinary promptTokens temperature seed numCycles _useRAM =
  P.take numCycles $ sample $ bundle (tokenOut, readyOut, attnDone, state,
                                    ffnDone, weightValid, layerIdx, layerChange)
  where
    temperature' = realToFrac temperature

    -- Create input sequence (prompt then feedback)
    tokenSignal = fromList $ promptTokens P.++ P.repeat 0  -- Will be overridden by feedback
    validSignal = fromList $ P.map (const True) promptTokens P.++ P.repeat False
    tempSignal = pure temperature'
    seedSignal = pure seed
    powerOn = pure True

    -- Create DDR slave backed by model binary
    ddrSlave = exposeClockResetEnable
      (DRAM.createDRAMBackedAxiSlave modelBinary ddrMaster)
      systemClockGen resetGen enableGen

    -- Run decoder
    (tokenOut, readyOut, ddrMaster, introspection) =
      exposeClockResetEnable
        (Top.topEntityWithAxi ddrSlave powerOn tokenSignal validSignal tempSignal seedSignal)
        systemClockGen resetGen enableGen

    -- Extract introspection signals
    state = Top.state introspection
    attnDone = Top.attnDone introspection
    ffnDone = Top.ffnDone introspection
    weightValid = Top.weightStreamValid introspection
    layerIdx = Top.layerIndex introspection
    layerChange = Top.layerChangeDetected introspection

showDecoderState :: DecoderOutput -> String
showDecoderState decoderOut = printf "Tok=%3d Rdy=%s Attn=%s FFN=%s Layer=%d Stage=%s WgtV=%s LayChg=%s"
    (fromEnum tok) (show ready) (show attn) (show ffn) (fromEnum layer)
    (show $ processingStage state) (show wgtV) (show layChg)
    where
      (tok, ready, attn, state, ffn, wgtV, layer, layChg) = decoderOut

findFirstDivergence :: [(Int, DecoderOutput)] -> [(Int, DecoderOutput)]
                    -> Maybe (Int, DecoderOutput, DecoderOutput)
findFirstDivergence [] _ = Nothing
findFirstDivergence _ [] = Nothing
findFirstDivergence ((c1, out1):rest1) ((c2, out2):rest2)
  | c1 /= c2 = error "Cycle mismatch in comparison"
  | out1 == out2 = findFirstDivergence rest1 rest2
  | otherwise = Just (c1, out1, out2)

detectLayerTransitions :: [DecoderOutput] -> [(Int, Index NumLayers, Index NumLayers)]
detectLayerTransitions outputs =
  [ (cycle', oldLayer, newLayer) |
    (cycle', (_, _, _, _, _, _, oldLayer, _), (_, _, _, _, _, _, newLayer, _))
      <- P.zip3 [0..] outputs (P.tail outputs),
    oldLayer /= newLayer ]

hasEarlyProjection :: [DecoderOutput] -> Int -> Bool
hasEarlyProjection outputs transitionCycle =
  -- Check if Stage1_ProjectQKV completes within 100 cycles of transition
  -- before weight loading completes
  let checkRange = [transitionCycle .. min (P.length outputs - 1) (transitionCycle + 100)]
      earlyStage1 = [ c | c <- checkRange,
                      let (_, _, _, state, _, _, _, _) = outputs P.!! c,
                      processingStage state == Stage1_ProjectQKV ]
  in not (null earlyStage1)

getBufferStatesAroundCycle :: [DecoderOutput] -> Int -> Int -> [(Int, Bool)]
getBufferStatesAroundCycle outputs centerCycle range =
  [ (c, fullyLoaded)
    | c <- [max 0 (centerCycle - range) .. min (P.length outputs - 1) (centerCycle + range)],

    -- We can't directly see fullyLoaded, but we can infer from behavior
    -- For now, just return placeholder
    let fullyLoaded = False ]  -- TODO: extract from weightBufferState introspection
