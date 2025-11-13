module Simulation.DRAMSimpleSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified Simulation.ParamsPlaceholder as PARAM
import qualified LLaMa2.Memory.WeightLoader as WEIGHTS (calculateLayerBaseAddress, weightManagementSystem)
import qualified Data.List as DL
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import qualified LLaMa2.Memory.AXI.Master as Master

spec :: Spec
spec = describe "DRAM Memory Simulation" $ do
  
    it "DRAM returns correct data when reading layer 0 Q weights" $ do
        let params = PARAM.decoderConst
            mem = DRAM.buildMemoryFromParams params
            
            -- Where layer 0 starts
            layer0Base = WEIGHTS.calculateLayerBaseAddress 0
            
            -- Skip RmsAtt (65 bytes), first Q byte is at layer0Base + 65
            firstQByteAddr = layer0Base + 65
            
            -- Convert to word index
            wordIdx = fromIntegral (firstQByteAddr `div` 64) :: Int
            byteOffset = fromIntegral (firstQByteAddr `mod` 64) :: Int
            
            -- Read from memory
            word = mem !! wordIdx
            bytes = bitCoerce word :: Vec 64 (BitVector 8)
            actualByte = bitCoerce (bytes !! byteOffset) :: Signed 8
            
        putStrLn "\n=== DRAM Read Verification ==="
        putStrLn "Reading layer 0, first Q byte"
        putStrLn $ "Address: " P.++ show firstQByteAddr
        putStrLn $ "Word index: " P.++ show wordIdx P.++ ", byte offset: " P.++ show byteOffset
        putStrLn $ "Value read: " P.++ show actualByte
        putStrLn "Expected: 11"

        actualByte `shouldBe` 11

    it "Decoder requests correct address for layer 0 Q weights" $ do
        let params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ [True] P.++ P.repeat False
            sinkRdy = pure True
            totalCycles = 100
            
            (ddrMaster, _, _, _) =
                withClockResetEnable systemClockGen resetGen enableGen $
                let ddrSlave = DRAM.createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', weightStream, streamValid, sysState) =
                        WEIGHTS.weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (ddrMaster', weightStream, streamValid, sysState)
            
            sampledArValid = sampleN totalCycles (Master.arvalid ddrMaster)
            sampledArData = sampleN totalCycles (Master.ardata ddrMaster)
            
            -- Find first read request
            firstRequest = DL.find fst $ P.zip sampledArValid sampledArData
            
        case firstRequest of
            Just (_, arData) -> do
                let requestedAddr = AXITypes.araddr arData
                    expectedBase = WEIGHTS.calculateLayerBaseAddress 0
                
                putStrLn "\n=== Master Address Request ==="
                putStrLn $ "First address requested: " P.++ show requestedAddr
                putStrLn $ "Expected layer 0 base: " P.++ show expectedBase
                
                requestedAddr `shouldBe` expectedBase
            Nothing -> expectationFailure "No AXI request issued!"
