module Simulation.SingleLayerOutputSpec (spec) where

import           Test.Hspec
import           Clash.Prelude
import qualified Prelude as P

import           LLaMa2.Types.LayerData (Temperature, Seed, Token, LayerData (..))
import           LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Decoder.SimplifiedDecoder as Decoder
import Simulation.Parameters (DecoderParameters)
import qualified Simulation.ParamsPlaceholder as PARAM
import Control.Monad (unless, when)
import qualified System.Directory as DIR

-- Create autoregressive feedback loop
-- State: (current token, remaining prompt tokens, using prompt)
type TokenState = (Token, [Token], Bool)

vectorNorm :: Vec a FixedPoint -> Float
vectorNorm v = sqrt $ sum $ P.map (\x -> let f = realToFrac x in f * f) (toList v)

spec :: Spec
spec = do
  describe "Single Layer Output Comparison" $ do
    it "Layer 0 data and output norms should match expected values with autoregressive feeding" $ do
        -- Check file exists
        fileExists <- DIR.doesFileExist "./data/stories260K.bin"
        cwd <- DIR.getCurrentDirectory
        putStrLn $ "Test running from: " P.++ cwd
        unless fileExists $
          expectationFailure "Model file not found at ./data/stories260K.bin"

        let promptTokens = [1, 320, 417]  -- "Hi" encoded as BOS + [320, 417]
            temperature = pure (0.0 :: Temperature)
            seed        = pure (123 :: Seed)
            powerOn     = pure True
            totalCycles = 10_000  -- Need more cycles to process the full prompt

        let
          params :: DecoderParameters
          params = PARAM.decoderConst

          ddrSlave = exposeClockResetEnable
            (DRAM.createDRAMBackedAxiSlave params ddrMaster)
            systemClockGen
            resetGen
            enableGen

          initialTokenState :: TokenState
          initialTokenState = case promptTokens of
            (t:ts) -> (t, ts, True)
            []     -> (1, [], True)

          advanceTokenState :: TokenState -> (Bool, Token) -> (TokenState, (Token, Bool))
          advanceTokenState (current, remaining, usingPrompt) (isReady, sampled) =
            if not isReady
            then ((current, remaining, usingPrompt), (current, usingPrompt))
            else case remaining of
              (next:rest) -> ((next, rest, True), (next, True))
              []          -> ((sampled, [], False), (sampled, False))

          -- inputToken and inputValid depend on the decoder's ready signal (circular definition)
          (inputToken, inputValid) = unbundle $
            exposeClockResetEnable
              (mealy advanceTokenState initialTokenState (bundle (rdy, tok)))
              systemClockGen
              resetGen
              enableGen

          -- Create the feedback loop: masters drive slaves, slaves feed back to decoder
          (tok, rdy, ddrMaster, intro) =
            exposeClockResetEnable
              (Decoder.decoder ddrSlave powerOn params inputToken inputValid temperature seed)
              systemClockGen
              resetGen
              enableGen

        -- -------------------------------------------------------------------
        -- Sample *all* internal signals we need
        -- -------------------------------------------------------------------
        let sampledLayerIdx   = sampleN totalCycles (Decoder.layerIndex intro)
            sampledFFNDone    = sampleN totalCycles (Decoder.ffnDone intro)
            sampledLayerOut   = sampleN totalCycles (Decoder.layerOutput intro)
            sampledLayerData  = sampleN totalCycles (Decoder.layerData intro)
            sampledInputToken = sampleN totalCycles inputToken
            sampledReady      = sampleN totalCycles rdy

        -- -------------------------------------------------------------------
        -- Pair everything with a cycle index
        -- -------------------------------------------------------------------
        let
          cycles = [0..totalCycles-1] :: [Int]
          zipped = zipWith6' (,,,,,)
                              cycles
                              sampledLayerIdx
                              sampledFFNDone
                              (P.zip sampledLayerOut sampledLayerData)
                              sampledInputToken
                              sampledReady

        -- -------------------------------------------------------------------
        -- Keep only cycles where layer 0 has just finished its FFN
        -- -------------------------------------------------------------------
        let layer0Completions =
              [ (i, output, layerData, inTok, rdy')
              | (i, layerIdx, ffnDone, (output, layerData), inTok, rdy') <- zipped
              , layerIdx == 0 && ffnDone ]

        putStrLn $ "\n=== Layer 0 Completions Found: " P.++ show (P.length layer0Completions) P.++ " ==="

        -- Print info about each completion
        P.mapM_ (\(i :: Int, idx) -> do
            let (cycle', output, layerData, inTok, _) = layer0Completions P.!! idx
            let outNorm = vectorNorm output
            let LayerData { attentionOutput, feedForwardOutput } = layerData
            let attnNorm   = vectorNorm attentionOutput   :: Float
            let ffnOutNorm = vectorNorm feedForwardOutput :: Float
            putStrLn $ "\nCompletion #" P.++ show i P.++ ":"
            putStrLn $ "  Cycle: " P.++ show cycle'
            putStrLn $ "  Input Token: " P.++ show inTok
            putStrLn $ "  Output Norm: " P.++ show outNorm
            putStrLn $ "  Attention Norm: " P.++ show attnNorm
            putStrLn $ "  FFN Output Norm: " P.++ show ffnOutNorm
          ) (P.zip [0..] [0 .. P.length layer0Completions - 1])

        case layer0Completions of
          [] -> expectationFailure "Layer 0 never completed"

          _ -> do
            -- Check the FIRST completion (processing token 1 - BOS)
            let (cycle', output, layerData, inTok, _) = P.head layer0Completions
            let outNorm = vectorNorm output
            putStrLn $ "\n=== Checking First Layer 0 Completion (Token " P.++ show inTok P.++ ") ==="
            putStrLn $ "Cycle: " P.++ show cycle'

            let LayerData { attentionOutput, feedForwardOutput } = layerData
            let attnNorm   = vectorNorm attentionOutput   :: Float
                ffnOutNorm = vectorNorm feedForwardOutput :: Float

            putStrLn $ "attentionOutput norm : " P.++ show attnNorm
            putStrLn $ "feedForwardOutput norm : " P.++ show ffnOutNorm
            putStrLn $ "output norm : " P.++ show outNorm

            -- Reference values for token 1 (BOS) obtained from simulation
            abs (attnNorm   - 3.04) `shouldSatisfy` (< 0.1)
            abs (ffnOutNorm - 8.24) `shouldSatisfy` (< 0.1)
            abs (outNorm    - 8.24) `shouldSatisfy` (< 0.1)

            -- If there's a second completion (processing token 320), check it too
            when (P.length layer0Completions >= 2) $ do
              let (cycle2, output2, layerData2, inTok2, _) = layer0Completions P.!! 1
              let outNorm2 = vectorNorm output2
              putStrLn $ "\n=== Checking Second Layer 0 Completion (Token " P.++ show inTok2 P.++ ") ==="
              putStrLn $ "Cycle: " P.++ show cycle2

              let LayerData { attentionOutput = attnOut2, feedForwardOutput = ffnOut2 } = layerData2
              let attnNorm2   = vectorNorm attnOut2   :: Float
                  ffnOutNorm2 = vectorNorm ffnOut2 :: Float

              putStrLn $ "attentionOutput norm : " P.++ show attnNorm2
              putStrLn $ "feedForwardOutput norm : " P.++ show ffnOutNorm2
              putStrLn $ "output norm : " P.++ show outNorm2

              -- You can add assertions here once you know the expected values

zipWith6' :: (a -> b -> c -> d -> e -> f -> g) -> [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [g]
zipWith6' _ []      _      _      _      _      _      = []
zipWith6' _ _      []      _      _      _      _      = []
zipWith6' _ _      _      []      _      _      _      = []
zipWith6' _ _      _      _      []      _      _      = []
zipWith6' _ _      _      _      _      []      _      = []
zipWith6' _ _      _      _      _      _      []      = []
zipWith6' fn (x:xs) (y:ys) (z:zs) (w:ws) (v:vs) (u:us) =
    fn x y z w v u : zipWith6' fn xs ys zs ws vs us
