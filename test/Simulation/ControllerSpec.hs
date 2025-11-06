module Simulation.ControllerSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Types.LayerData (CycleStage(..), ProcessingState(..))
import qualified LLaMa2.Decoder.SequenceController as Controller
import LLaMa2.Types.ModelConfig (NumLayers)

spec :: Spec
spec = do
  describe "SequenceController Phase 2" $ do
    
    it "should advance through a single token's stages" $ do
      let totalCycles = 100
          
          -- Simulate: Attention completes at cycle 10, FFN at cycle 20
          attnDone = fromList $ P.replicate 10 False P.++ [True] P.++ P.replicate (totalCycles - 11) False
          ffnDone  = fromList $ P.replicate 20 False P.++ [True] P.++ P.replicate (totalCycles - 21) False
          classifierDone = fromList $ P.replicate 30 False P.++ [True] P.++ P.replicate (totalCycles - 31) False
          
          controller = exposeClockResetEnable
            (Controller.sequenceController attnDone ffnDone classifierDone)
            systemClockGen resetGen enableGen
          
          -- Sample all the important signals
          sampledStage = sampleN totalCycles (Controller.processingState controller)
          sampledLayer = sampleN totalCycles (Controller.currentLayer controller)
          sampledReady = sampleN totalCycles (Controller.readyPulse controller)
          sampledEnableAttn = sampleN totalCycles (Controller.enableAttention controller)
          sampledEnableFFN = sampleN totalCycles (Controller.enableFFN controller)
          sampledEnableCls = sampleN totalCycles (Controller.enableClassifier controller)
      
      -- Print what we see
      putStrLn "\n=== Controller Behavior (1 Token, Layer 0 Only) ==="
      putStrLn $ "Total cycles sampled: " P.++ show totalCycles
      
      -- Show key cycles
      let showCycle i = do
            let stage = processingStage (sampledStage P.!! i)
                layer = sampledLayer P.!! i
                ready = sampledReady P.!! i
                eAttn = sampledEnableAttn P.!! i
                eFFN = sampledEnableFFN P.!! i
                eCls = sampledEnableCls P.!! i
            putStrLn $ "Cycle " P.++ show i P.++ ": "
              P.++ "stage=" P.++ show stage P.++ " "
              P.++ "layer=" P.++ show layer P.++ " "
              P.++ "ready=" P.++ show ready P.++ " "
              P.++ "enables: attn=" P.++ show eAttn 
              P.++ " ffn=" P.++ show eFFN
              P.++ " cls=" P.++ show eCls
      
      -- Show a sample of cycles around the transitions
      putStrLn "\nCycles around Attention done (cycle 10):"
      P.mapM_ showCycle [8, 9, 10, 11, 12]
      
      putStrLn "\nCycles around FFN done (cycle 20):"
      P.mapM_ showCycle [18, 19, 20, 21, 22]
      
      putStrLn "\nCycles around Classifier done (cycle 30):"
      P.mapM_ showCycle [28, 29, 30, 31, 32]
      
      -- Basic assertions
      let stage10 = processingStage (sampledStage P.!! 10)
          stage11 = processingStage (sampledStage P.!! 11)
          stage20 = processingStage (sampledStage P.!! 20)
          stage21 = processingStage (sampledStage P.!! 21)
          stage30 = processingStage (sampledStage P.!! 30)
          stage31 = processingStage (sampledStage P.!! 31)
      
      putStrLn "\n=== Key Transitions ==="
      putStrLn $ "Cycle 10: " P.++ show stage10 P.++ " (attn done arrives)"
      putStrLn $ "Cycle 11: " P.++ show stage11 P.++ " (should advance to FFN)"
      putStrLn $ "Cycle 20: " P.++ show stage20 P.++ " (ffn done arrives)"
      putStrLn $ "Cycle 21: " P.++ show stage21 P.++ " (should advance to Classifier)"
      putStrLn $ "Cycle 30: " P.++ show stage30 P.++ " (classifier done arrives)"
      putStrLn $ "Cycle 31: " P.++ show stage31 P.++ " (should advance to Attention for next token)"
      
      -- Check if ready pulse fires
      let readyPulses = P.filter snd (P.zip [0..] sampledReady)
      putStrLn $ "\nReady pulses at cycles: " P.++ show (P.map fst readyPulses)
      
      -- Expectations
      stage11 `shouldBe` Stage_FeedForward
      stage21 `shouldBe` Stage_Classifier
      stage31 `shouldBe` Stage_Attention  -- Back to start for next token
      
      -- Should have exactly 1 ready pulse at cycle 31
      P.length readyPulses `shouldBe` 1
      
    it "should handle multiple layers" $ do
      let totalCycles = 200
          numLayers = natToNum @NumLayers :: Integer
          
          -- Simulate each layer completing
          -- Layer 0: attn@10, ffn@20
          -- Layer 1: attn@30, ffn@40
          -- ...
          attnDoneCycles = [10, 30, 50, 70, 90, 110]
          ffnDoneCycles = [20, 40, 60, 80, 100, 120]
          clsDoneCycle = 130
          
          makeSignal cycles = fromList $ P.map (\i -> i `P.elem` cycles) [0..totalCycles-1]
          
          attnDone = makeSignal attnDoneCycles
          ffnDone = makeSignal ffnDoneCycles
          classifierDone = makeSignal [clsDoneCycle]
          
          controller = exposeClockResetEnable
            (Controller.sequenceController attnDone ffnDone classifierDone)
            systemClockGen resetGen enableGen
          
          sampledLayer = sampleN totalCycles (Controller.currentLayer controller)
          sampledStage = sampleN totalCycles (Controller.processingState controller)
          sampledReady = sampleN totalCycles (Controller.readyPulse controller)
      
      putStrLn "\n=== Multi-Layer Test ==="
      putStrLn $ "Number of layers: " P.++ show numLayers
      
      -- Check layer progression
      let layerAt c = sampledLayer P.!! c
          stageAt c = processingStage (sampledStage P.!! c)
      
      putStrLn "\nLayer progression:"
      putStrLn $ "Cycle 11 (after layer 0 attn+ffn): layer=" P.++ show (layerAt 11)
      putStrLn $ "Cycle 21 (should be layer 0 FFN→layer 1 Attn): layer=" P.++ show (layerAt 21) 
                  P.++ " stage=" P.++ show (stageAt 21)
      putStrLn $ "Cycle 31 (after layer 1 attn): layer=" P.++ show (layerAt 31)
      putStrLn $ "Cycle 41 (should be layer 1 FFN→layer 2 Attn): layer=" P.++ show (layerAt 41)
                  P.++ " stage=" P.++ show (stageAt 41)
      
      -- Check if we reach the last layer
      let maxLayerReached = P.maximum sampledLayer
      putStrLn $ "\nMax layer reached: " P.++ show maxLayerReached
      putStrLn $ "Expected max layer: " P.++ show (fromInteger (numLayers - 1) :: Index NumLayers)
      
      -- Check if classifier is reached
      let classifierReached = P.any (\s -> processingStage s == Stage_Classifier) sampledStage
      putStrLn $ "Classifier stage reached: " P.++ show classifierReached
      
      -- Check ready pulse
      let readyPulses = P.filter snd (P.zip [0..] sampledReady)
      putStrLn $ "Ready pulse fired: " P.++ show (not $ P.null readyPulses)
      
      -- Assertions
      classifierReached `shouldBe` True
      not (P.null readyPulses) `shouldBe` True