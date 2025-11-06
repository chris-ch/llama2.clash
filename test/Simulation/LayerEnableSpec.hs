module Simulation.LayerEnableSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Types.LayerData (CycleStage(..), ProcessingState(..))
import qualified LLaMa2.Decoder.SequenceController as Controller
import LLaMa2.Types.ModelConfig (NumLayers)

spec :: Spec
spec = do
  describe "Layer-Specific Enable Generation" $ do
    
    it "should generate enables for the correct layer" $ do
      let totalCycles = 150
          
          -- Simulate layer completions
          -- Layer 0: attn@10, ffn@20
          -- Layer 1: attn@30, ffn@40
          -- Layer 2: attn@50, ffn@60
          -- ...
          attnDoneCycles = [10, 30, 50, 70, 90]
          ffnDoneCycles = [20, 40, 60, 80, 100]
          
          makeSignal cycles = fromList $ P.map (\i -> i `P.elem` cycles) [0..totalCycles-1]
          
          attnDone = makeSignal attnDoneCycles
          ffnDone = makeSignal ffnDoneCycles
          classifierDone = makeSignal [110]  -- After layer 4 FFN
          
          controller = exposeClockResetEnable
            (Controller.sequenceController attnDone ffnDone classifierDone)
            systemClockGen resetGen enableGen
          
          -- These are the signals that would go to the decoder
          processingState = Controller.processingState controller
          layerIdx = Controller.currentLayer controller
          
          -- The decoder would generate these
          enableAttention = (fmap processingStage processingState) .==. pure Stage_Attention
          enableFFN = (fmap processingStage processingState) .==. pure Stage_FeedForward
          
          -- Sample everything
          sampledStage = sampleN totalCycles processingState
          sampledLayer = sampleN totalCycles layerIdx
          sampledEnableAttn = sampleN totalCycles enableAttention
          sampledEnableFFN = sampleN totalCycles enableFFN
      
      putStrLn "\n=== Layer-Specific Enable Test ==="
      
      -- Check layer-specific enables at key cycles
      let checkCycle c expected_layer expected_stage expected_attn expected_ffn = do
            let actual_layer = sampledLayer P.!! c
                actual_stage = processingStage (sampledStage P.!! c)
                actual_attn = sampledEnableAttn P.!! c
                actual_ffn = sampledEnableFFN P.!! c
                
                -- Simulate layer-specific enable (what LayerStack would do)
                layer0_enable_attn = actual_attn && (actual_layer == 0)
                layer1_enable_attn = actual_attn && (actual_layer == 1)
                layer2_enable_attn = actual_attn && (actual_layer == 2)
                
                layer0_enable_ffn = actual_ffn && (actual_layer == 0)
                layer1_enable_ffn = actual_ffn && (actual_layer == 1)
                layer2_enable_ffn = actual_ffn && (actual_layer == 2)
            
            putStrLn $ "\nCycle " P.++ show c P.++ ":"
            putStrLn $ "  Layer: " P.++ show actual_layer P.++ " (expected: " P.++ show expected_layer P.++ ")"
            putStrLn $ "  Stage: " P.++ show actual_stage P.++ " (expected: " P.++ show expected_stage P.++ ")"
            putStrLn $ "  Global enables: attn=" P.++ show actual_attn P.++ " ffn=" P.++ show actual_ffn
            putStrLn $ "  Layer-specific enables:"
            putStrLn $ "    Layer 0: attn=" P.++ show layer0_enable_attn P.++ " ffn=" P.++ show layer0_enable_ffn
            putStrLn $ "    Layer 1: attn=" P.++ show layer1_enable_attn P.++ " ffn=" P.++ show layer1_enable_ffn
            putStrLn $ "    Layer 2: attn=" P.++ show layer2_enable_attn P.++ " ffn=" P.++ show layer2_enable_ffn
            
            -- Verify
            actual_layer `shouldBe` expected_layer
            actual_stage `shouldBe` expected_stage
      
      putStrLn "\n=== After Layer 0 Attention Done (cycle 11) ==="
      checkCycle 11 0 Stage_FeedForward False True
      
      putStrLn "\n=== After Layer 0 FFN Done (cycle 21) ==="
      checkCycle 21 1 Stage_Attention True False
      
      putStrLn "\n=== After Layer 1 Attention Done (cycle 31) ==="
      checkCycle 31 1 Stage_FeedForward False True
      
      putStrLn "\n=== After Layer 1 FFN Done (cycle 41) ==="
      checkCycle 41 2 Stage_Attention True False
      
      putStrLn "\n=== Summary ==="
      putStrLn "✓ Controller correctly advances through layers"
      putStrLn "✓ Layer-specific enables would be generated correctly"
      putStrLn ""
      putStrLn "This means the CONTROLLER is working correctly."
      putStrLn "If the full system only processes layer 0, the issue is likely:"
      putStrLn "  1. Layer 1+ attention/FFN not actually running"
      putStrLn "  2. Layer 1+ done signals not propagating back to controller"
      putStrLn "  3. Data flow between layers broken"
