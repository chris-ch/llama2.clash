module Simulation.SingleLayerOutputSpec (spec) where

import           Test.Hspec
import           Clash.Prelude
import qualified Prelude as P

import           LLaMa2.Types.LayerData (Temperature, Seed, Token, LayerData (..), ProcessingState, CycleStage (..))
import           LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Decoder.Decoder as Decoder
import Simulation.Parameters (DecoderParameters)
import qualified Simulation.ParamsPlaceholder as PARAM
import Control.Monad (unless, when)
import qualified System.Directory as DIR
import Data.Foldable (forM_)

-- Create autoregressive feedback loop
-- State: (current token, remaining prompt tokens, using prompt)
type TokenState = (Token, [Token], Bool)

vectorNorm :: Vec a FixedPoint -> Float
vectorNorm v = sqrt $ sum $ P.map (\x -> let f = realToFrac x in f * f) (toList v)

-- Reference values for each token completion at layer 0
data TokenReference = TokenReference
  { refToken        :: Token
  , refQNorm        :: Float  -- Q[0] head norm
  , refKNorm        :: Float  -- K[0] head norm  
  , refVNorm        :: Float  -- V[0] head norm
  , refAttnNorm     :: Float  -- Attention output norm
  , refFFNOutNorm   :: Float  -- FFN output norm
  , refLayerOutNorm :: Float  -- Final layer output norm
  , tolerance       :: Float  -- Allowed deviation
  } deriving (Show)

-- Expected values obtained from baseline simulation
-- TODO: Replace these with actual values from your baseline run
tokenReferences :: [TokenReference]
tokenReferences =
  [ TokenReference  -- Token 1 (BOS)
      { refToken        = 1
      , refQNorm        = 4.662631
      , refKNorm        = 9.931277
      , refVNorm        = 1.387717
      , refAttnNorm     = 3.038473
      , refFFNOutNorm   = 8.235682
      , refLayerOutNorm = 8.235682
      , tolerance       = 0.0001
      }
  , TokenReference  -- Token 320
      { refToken        = 320
      , refQNorm        = 4.251084
      , refKNorm        = 12.635413
      , refVNorm        = 1.1689112
      , refAttnNorm     = 2.7742534  
      , refFFNOutNorm   = 3.8518157
      , refLayerOutNorm = 3.8518157
      , tolerance       = 0.0001
      }
  , TokenReference  -- Token 417
      { refToken        = 417
      , refQNorm        = 17.01927
      , refKNorm        = 4.388498
      , refVNorm        = 0.8723053
      , refAttnNorm     = 2.7814713
      , refFFNOutNorm   = 2.822887
      , refLayerOutNorm = 2.822887
      , tolerance       = 0.0001
      }
  ]

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
        -- Sample *all* internal signals we need (Phase 2: simplified signals)
        -- -------------------------------------------------------------------
        let sampledLayerIdx   = sampleN totalCycles (Decoder.layerIndex intro)
            sampledFFNDone    = sampleN totalCycles (Decoder.ffnDone intro)
            sampledAttnDone   = sampleN totalCycles (Decoder.attnDone intro)  -- Now includes QKV+Write+Attend
            sampledLayerOut   = sampleN totalCycles (Decoder.layerOutput intro)
            sampledLayerData  = sampleN totalCycles (Decoder.layerData intro)
            sampledInputToken = sampleN totalCycles inputToken
            sampledInputValid = sampleN totalCycles inputValid
            sampledReady      = sampleN totalCycles rdy
            sampledState      = sampleN totalCycles (Decoder.state intro)

        -- -------------------------------------------------------------------
        -- Pair everything with a cycle index (Phase 2: 8 signals instead of 9)
        -- -------------------------------------------------------------------
        let
          cycles = [0..totalCycles-1] :: [Int]
          zipped = zipWith8' (,,,,,,,)
                              cycles
                              sampledLayerIdx
                              sampledAttnDone  -- Single attention completion signal
                              sampledFFNDone
                              (P.zip sampledLayerOut sampledLayerData)
                              sampledInputToken
                              sampledInputValid
                              sampledReady
                              
        putStrLn "\n=== DIAGNOSTIC: Per-Layer Completions ==="
        forM_ [0, 1, 2, 3, 4] $ \layer -> do
          let attnCount = P.length $ filter 
                (\(_, layerIdx, attnDone, _, _, _, _, _) -> layerIdx == layer && attnDone) 
                zipped
              ffnCount = P.length $ filter 
                (\(_, layerIdx, _, ffnDone, _, _, _, _) -> layerIdx == layer && ffnDone) 
                zipped
          
          putStrLn $ "Layer " P.++ show layer P.++ ": attn=" P.++ show attnCount P.++ " ffn=" P.++ show ffnCount

        -- -------------------------------------------------------------------
        -- Extract stage completions for layer 0 (Phase 2: simplified)
        -- -------------------------------------------------------------------
        let layer0AttnCompletions =
              [ (i, output, layerData, inTok)
              | (i, layerIdx, attnDone, _, (output, layerData), inTok, _, _) <- zipped
              , layerIdx == 0 && attnDone ]

        let layer0FFNCompletions =
              [ (i, output, layerData, inTok, rdy')
              | (i, layerIdx, _, ffnDone, (output, layerData), inTok, _, rdy') <- zipped
              , layerIdx == 0 && ffnDone ]

        -- -------------------------------------------------------------------
        -- Diagnostic output (Phase 2: simplified)
        -- -------------------------------------------------------------------
        putStrLn $ "\n=== Layer 0 Stage Completions (Phase 2) ==="
        putStrLn $ "Attention Completions (full): " P.++ show (P.length layer0AttnCompletions)
        putStrLn $ "FFN Completions: " P.++ show (P.length layer0FFNCompletions)

        -- Print detailed info about each FFN completion
        putStrLn "\n=== Detailed FFN Completion Data ==="
        P.mapM_ (\(idx :: Int) -> do
            let (cycle', output, layerData, inTok, _) = layer0FFNCompletions P.!! idx
            let outNorm = vectorNorm output
            let LayerData { attentionOutput, feedForwardOutput, queryVectors, keyVectors, valueVectors } = layerData
            let attnNorm   = vectorNorm attentionOutput   :: Float
            let ffnOutNorm = vectorNorm feedForwardOutput :: Float
            let qNorm = vectorNorm (P.head $ toList queryVectors) :: Float  -- First Q head
            let kNorm = vectorNorm (P.head $ toList keyVectors) :: Float    -- First K head
            let vNorm = vectorNorm (P.head $ toList valueVectors) :: Float  -- First V head
            
            putStrLn $ "\n--- FFN Completion #" P.++ show idx P.++ " (Token " P.++ show inTok P.++ ") ---"
            putStrLn $ "  Cycle: " P.++ show cycle'
            putStrLn $ "  Q[0] Norm: " P.++ show qNorm
            putStrLn $ "  K[0] Norm: " P.++ show kNorm
            putStrLn $ "  V[0] Norm: " P.++ show vNorm
            putStrLn $ "  Attention Output Norm: " P.++ show attnNorm
            putStrLn $ "  FFN Output Norm: " P.++ show ffnOutNorm
            putStrLn $ "  Layer Output Norm: " P.++ show outNorm
          ) [0 .. P.length layer0FFNCompletions - 1]

        -- -------------------------------------------------------------------
        -- Verify stage completion timing (Phase 2: Attn → FFN)
        -- -------------------------------------------------------------------
        putStrLn "\n=== Stage Completion Timing Analysis (Phase 2) ==="
        
        -- For each FFN completion, find the corresponding Attn completion
        P.mapM_ (\(idx :: Int) -> do
            let (ffnCycle, _, _, inTok, _) = layer0FFNCompletions P.!! idx
            
            -- Find matching Attn completion for this token
            let attnMatches = P.filter (\(_, _, _, tok) -> tok == inTok) layer0AttnCompletions
            
            putStrLn $ "\nToken " P.++ show inTok P.++ " stage timing:"
            
            case attnMatches of
              ((attnCycle, _, _, _):_) -> 
                putStrLn $ "  Attention done at cycle: " P.++ show attnCycle
              [] -> 
                putStrLn "  Attention completion not found!"
            
            putStrLn $ "  FFN done at cycle: " P.++ show ffnCycle
            
            -- Verify ordering (Attention < FFN)
            case attnMatches of
              ((attnCycle, _, _, _):_) -> do
                when (attnCycle >= ffnCycle) $
                  putStrLn "  WARNING: Attention completed after or at same time as FFN!"
              _ -> putStrLn "  WARNING: Missing attention completion!"
          ) [0 .. P.length layer0FFNCompletions - 1]

        let readyCount = P.length $ filter id sampledReady
        let attnCount  = P.length $ filter id sampledAttnDone
        let ffnCount   = P.length $ filter id sampledFFNDone

        putStrLn "\n=== Controller Activity Summary ==="
        putStrLn $ "ready pulses: " P.++ show readyCount
        putStrLn $ "attention dones: " P.++ show attnCount
        putStrLn $ "ffn dones: " P.++ show ffnCount

        -- -------------------------------------------------------------------
        -- Assertions against reference values
        -- -------------------------------------------------------------------
        putStrLn "\n=== Verifying Against Reference Values ==="
        
        case layer0FFNCompletions of
          [] -> expectationFailure "Layer 0 never completed"

          _ -> do
            -- Verify we got the expected number of completions
            let numCompletions = P.length layer0FFNCompletions
            let expectedCompletions = P.length promptTokens
            
            putStrLn $ "Expected " P.++ show expectedCompletions P.++ " completions, got " P.++ show numCompletions
            
            -- Check each completion against its reference
            P.mapM_ (\(idx :: Int) -> do
                when (idx < P.length tokenReferences) $ do
                  let ref = tokenReferences P.!! idx
                  let (cycle', output, layerData, inTok, _) = layer0FFNCompletions P.!! idx
                  
                  let LayerData { attentionOutput, feedForwardOutput, queryVectors, keyVectors, valueVectors } = layerData
                  let qNorm = vectorNorm (P.head $ toList queryVectors) :: Float
                  let kNorm = vectorNorm (P.head $ toList keyVectors) :: Float
                  let vNorm = vectorNorm (P.head $ toList valueVectors) :: Float
                  let attnNorm   = vectorNorm attentionOutput   :: Float
                  let ffnOutNorm = vectorNorm feedForwardOutput :: Float
                  let outNorm = vectorNorm output :: Float
                  
                  putStrLn $ "\nVerifying completion #" P.++ show idx P.++ " (Token " P.++ show inTok P.++ "):"
                  
                  -- Check token matches
                  inTok `shouldBe` refToken ref
                  
                  -- Check Q norm
                  let qError = abs (qNorm - refQNorm ref)
                  putStrLn $ "  Q[0] norm: " P.++ show qNorm P.++ " (ref: " P.++ show (refQNorm ref) P.++ ", error: " P.++ show qError P.++ ")"
                  qError `shouldSatisfy` (< tolerance ref)
                  
                  -- Check K norm
                  let kError = abs (kNorm - refKNorm ref)
                  putStrLn $ "  K[0] norm: " P.++ show kNorm P.++ " (ref: " P.++ show (refKNorm ref) P.++ ", error: " P.++ show kError P.++ ")"
                  kError `shouldSatisfy` (< tolerance ref)
                  
                  -- Check V norm
                  let vError = abs (vNorm - refVNorm ref)
                  putStrLn $ "  V[0] norm: " P.++ show vNorm P.++ " (ref: " P.++ show (refVNorm ref) P.++ ", error: " P.++ show vError P.++ ")"
                  vError `shouldSatisfy` (< tolerance ref)
                  
                  -- Check attention output norm
                  let attnError = abs (attnNorm - refAttnNorm ref)
                  putStrLn $ "  Attention norm: " P.++ show attnNorm P.++ " (ref: " P.++ show (refAttnNorm ref) P.++ ", error: " P.++ show attnError P.++ ")"
                  attnError `shouldSatisfy` (< tolerance ref)
                  
                  -- Check FFN output norm
                  let ffnError = abs (ffnOutNorm - refFFNOutNorm ref)
                  putStrLn $ "  FFN output norm: " P.++ show ffnOutNorm P.++ " (ref: " P.++ show (refFFNOutNorm ref) P.++ ", error: " P.++ show ffnError P.++ ")"
                  ffnError `shouldSatisfy` (< tolerance ref)
                  
                  -- Check layer output norm
                  let outError = abs (outNorm - refLayerOutNorm ref)
                  putStrLn $ "  Layer output norm: " P.++ show outNorm P.++ " (ref: " P.++ show (refLayerOutNorm ref) P.++ ", error: " P.++ show outError P.++ ")"
                  outError `shouldSatisfy` (< tolerance ref)
                  
                  putStrLn "  ✓ All checks passed"
              ) [0 .. P.min numCompletions (P.length tokenReferences) - 1]


-- Helper function to zip 8 lists (Phase 2: simplified from 9)
zipWith8' :: (a -> b -> c -> d -> e -> f -> g -> h -> i) 
          -> [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [g] -> [h] -> [i]
zipWith8' _ []     _      _      _      _      _      _      _      = []
zipWith8' _ _      []     _      _      _      _      _      _      = []
zipWith8' _ _      _      []     _      _      _      _      _      = []
zipWith8' _ _      _      _      []     _      _      _      _      = []
zipWith8' _ _      _      _      _      []     _      _      _      = []
zipWith8' _ _      _      _      _      _      []     _      _      = []
zipWith8' _ _      _      _      _      _      _      []     _      = []
zipWith8' _ _      _      _      _      _      _      _      []     = []
zipWith8' fn (x:xs) (y:ys) (z:zs) (w:ws) (v:vs) (u:us) (t:ts) (s:ss) =
    fn x y z w v u t s : zipWith8' fn xs ys zs ws vs us ts ss
