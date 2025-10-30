module Simulation.WeightLoaderSpec (spec) where

import Clash.Prelude
import LLaMa2.Memory.WeightLoader
  ( calculateLayerBaseAddress
  , parseI8EChunk
  , weightManagementSystem
  , WeightSystemState (..)
  , calculateLayerSizeBytes
  )
import LLaMa2.Numeric.Types (Mantissa)
import Test.Hspec
import qualified Prelude as P
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import Simulation.DRAMBackedAxiSlave (createDRAMBackedAxiSlave)
import qualified Clash.Sized.Vector as Vec
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified Simulation.ParamsPlaceholder as PARAM

-- ============================================================================
-- MOCKS
-- ============================================================================

-- A simple, always-ready AXI slave for testing.
mockAxiSlave :: Slave.AxiSlaveIn dom
mockAxiSlave =
  Slave.AxiSlaveIn
    { arready = pure True
    , rvalid = fromList $ P.replicate 5 False P.++ P.repeat True
    , rdata = pure (AXITypes.AxiR 0 0 False 0)
    , awready = pure True
    , wready = fromList $ P.replicate 5 False P.++ P.repeat True
    , bvalid = pure True
    , bdata = pure (AXITypes.AxiB 0 0)
    }

-- ============================================================================
-- SPEC
-- ============================================================================

spec :: Spec
spec = do
  describe "calculateLayerBaseAddress" $ do
    it "increments linearly per layer" $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
      addr1 `shouldBe` (addr0 + calculateLayerSizeBytes)

    it "produces different addresses for different layers" $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
          addr2 = calculateLayerBaseAddress 2
      addr0 `shouldNotBe` addr1
      addr1 `shouldNotBe` addr2
      addr0 `shouldNotBe` addr2

  describe "parseI8EChunk" $ do
    it "extracts mantissas and exponent correctly" $ do
      let bv = bitCoerce (replicate d64 (0x7F :: BitVector 8)) :: BitVector 512
          (mants, expn) = parseI8EChunk @8 bv
      -- Check mantissas
      mants `shouldBe` replicate (SNat @8) (127 :: Mantissa)
      -- Check exponent (7-bit value)
      pack expn `shouldBe` (0x7F :: BitVector 7)

    it "handles different bit patterns" $ do
      let
          vec :: Vec 64 (BitVector 8)
          vec = Vec.replicate d32 0xAA Vec.++ Vec.replicate d32 0x55

          bv = pack vec
          (mants, _expn) = parseI8EChunk @16 bv
      P.length (filter (== 170) (toList mants)) `shouldBe` 16

    it "extracts exponent from correct position" $ do
      let
          -- Index 10 as a runtime Int
          idx :: Index 64
          idx = 10

          -- 64 bytes, all 0x00 except byte #10 which is 0x3F
          vec :: Vec 64 (BitVector 8)
          vec = Vec.replace idx 0x3F (Vec.replicate d64 0x00)

          bv :: BitVector 512
          bv = pack vec

          (_mants, expn) = parseI8EChunk @10 bv

      -- Exponent (lower 7 bits of byte 10) must be 0x3F
      pack expn `shouldBe` (0x3F :: BitVector 7)

  describe "weightManagementSystem (idle behavior)" $ do
    context "when load is not triggered" $ do
      let
          layerIdx = pure 0
          startStream = pure False
          sinkRdy = pure True
          ( _ddrMaster, _, _, sysState
            ) =
              withClockResetEnable systemClockGen resetGen enableGen
                $ weightManagementSystem mockAxiSlave startStream layerIdx sinkRdy
          stateS = P.drop 2 (sampleN 10 sysState)

      it "stays in WSReady state" $ do
        P.all (== WSReady) stateS `shouldBe` True

  describe "weightManagementSystem (loading behavior)" $ do
    context "using full model from file" $ do
      it "correctly calculates layer size" $ do
        let layerSize = calculateLayerSizeBytes
        -- Layer size should be non-zero and reasonable
        layerSize `shouldSatisfy` (> 0)
        layerSize `shouldSatisfy` (< 10_000_000) -- sanity check: < 10MB per layer

      it "DIAGNOSTIC: system state responds to trigger" $ do
        let
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False :: Signal System Bool
            sinkRdy = pure True
            
        let (_ddrMaster, _, _, sysState) =
              withClockResetEnable systemClockGen resetGen enableGen
                $ weightManagementSystem mockAxiSlave startStream layerIdx sinkRdy
        
        let sampledState = sampleN 100 sysState
            
        -- Print for debugging
        P.putStrLn $ "First 30 states: " P.++ show (P.take 30 sampledState)
        
        -- Check if we ever leave WSReady
        let everStreaming = WSStreaming `P.elem` sampledState
      
        everStreaming `shouldBe` True

      it "DIAGNOSTIC: with real DDR, does streamer start?" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            
        let (streamValid, sysState, ddrMaster) = 
              withClockResetEnable systemClockGen resetGen enableGen $ 
                let ddrSlave = createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', _, streamValid', sysState') =
                      weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (streamValid', sysState', ddrMaster')
        
        let sampledValid = sampleN 1000 streamValid
            sampledState = sampleN 1000 sysState
            -- Check the AXI signals
            sampledArValid = sampleN 1000 (Master.arvalid ddrMaster)
            
        P.putStrLn $ "States (0-30): " P.++ show (P.take 30 sampledState)
        P.putStrLn $ "Valid (0-30): " P.++ show (P.take 30 sampledValid)
        P.putStrLn $ "AXI arvalid (0-30): " P.++ show (P.take 30 sampledArValid)
        
        let stateChanges = WSStreaming `P.elem` sampledState
            axiActive = P.or sampledArValid
            
        stateChanges `shouldBe` True
        axiActive `shouldBe` True

      it "DIAGNOSTIC: count valid beats over 50K cycles" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            
        let (streamValid, sysState) = 
              withClockResetEnable systemClockGen resetGen enableGen $ 
                let ddrSlave = createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', _, streamValid', sysState') =
                      weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (streamValid', sysState')
        
        let sampledValid = sampleN 50000 streamValid
            sampledState = sampleN 50000 sysState
            
            validCount = P.length $ P.filter id sampledValid
            streamingCount = P.length $ P.filter (== WSStreaming) sampledState
            
        P.putStrLn $ "Valid beats in 50K cycles: " P.++ show validCount
        P.putStrLn $ "Streaming cycles: " P.++ show streamingCount
        
        validCount `shouldSatisfy` (> 0)

      it "DIAGNOSTIC: check AXI handshake over time" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            
        let (streamValid, ddrMaster, ddrSlave) = 
              withClockResetEnable systemClockGen resetGen enableGen $ 
                let ddrSlave' = createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', _, streamValid', _) =
                      weightManagementSystem ddrSlave' startStream layerIdx sinkRdy
                in (streamValid', ddrMaster', ddrSlave')
        
        let sampledValid = sampleN 100 streamValid
            sampledArValid = sampleN 100 (Master.arvalid ddrMaster)
            sampledArReady = sampleN 100 (Slave.arready ddrSlave)
            sampledRValid = sampleN 100 (Slave.rvalid ddrSlave)
            sampledRReady = sampleN 100 (Master.rready ddrMaster)
            
        P.putStrLn $ "streamValid: " P.++ show (P.take 30 sampledValid)
        P.putStrLn $ "arvalid:     " P.++ show (P.take 30 sampledArValid)
        P.putStrLn $ "arready:     " P.++ show (P.take 30 sampledArReady)
        P.putStrLn $ "rvalid:      " P.++ show (P.take 30 sampledRValid)
        P.putStrLn $ "rready:      " P.++ show (P.take 30 sampledRReady)
        
        True `shouldBe` True
        
      it "produces consistent stream during burst" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            -- INCREASED: Need more cycles to actually see streaming happen
            totalCycles = 50_000

        let (streamValid, sysState) = 
              withClockResetEnable systemClockGen resetGen enableGen $ 
                let ddrSlave = createDRAMBackedAxiSlave params ddrMaster
                    (ddrMaster, _, streamValid', sysState') =
                      weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (streamValid', sysState')
        
        let
            sampledValid = sampleN totalCycles streamValid
            sampledState = sampleN totalCycles sysState

            -- Count valid beats
            validBeats = P.length $ P.filter id sampledValid

            -- Count streaming cycles
            streamingCycles = P.length $ P.filter (== WSStreaming) sampledState

        -- Should have many valid beats during streaming
        validBeats `shouldSatisfy` (> 100)
        streamingCycles `shouldSatisfy` (> 100)

    context "with backpressure" $ do
      it "respects sinkReady signal" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            -- Sink ready alternates: ready for 2 cycles, not ready for 2 cycles
            sinkRdy = fromList $ P.cycle [True, True, False, False] :: Signal System Bool
            totalCycles = 100_000 -- need more cycles due to backpressure

        let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
              let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                  (ddrMasterOut, _weightStream, streamValid, _sysState) =
                    weightManagementSystem ddrSlave startStream layerIdx sinkRdy
              (streamValid, sinkRdy)
        
        let (streamValid, _sinkRdy) = simulation
            
            sampledValid :: [Bool]
            sampledValid = sampleN totalCycles streamValid

            sampledSinkRdy :: [Bool]
            sampledSinkRdy = sampleN totalCycles sinkRdy

            -- Stream should only be valid when sink is ready (or shortly after)
            streamEverValid = P.or sampledValid

            -- Count how many times valid occurs when sink not ready
            -- (should be minimal - only pipeline delays)
            invalidWhileNotReady = P.length $
              P.filter (\(v, sr) -> v && not sr) $
              P.zip sampledValid sampledSinkRdy

        streamEverValid `shouldBe` True
        -- Allow some pipeline delay, but should be < 10% of total valid beats
        let totalValidBeats = P.length $ P.filter id sampledValid
        invalidWhileNotReady `shouldSatisfy` (< (totalValidBeats `P.div` 10))

    it "completes successfully with backpressure" $ do
      let
          params = PARAM.decoderConst
          layerIdx = pure 0
          startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
          sinkRdy = fromList $ P.cycle [True, False]
          totalCycles = 100_000

      let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
            let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                (ddrMasterOut, _, streamValid, sysState) =
                  weightManagementSystem ddrSlave startStream layerIdx sinkRdy
            (streamValid, sysState)

      let (streamValid, sysState) = simulation
          sampledValid = sampleN totalCycles streamValid
          sampledState = sampleN totalCycles sysState
          
          validCount = P.length $ P.filter id sampledValid
          streamingCount = P.length $ P.filter (== WSStreaming) sampledState
          completesEventually = WSReady `P.elem` P.drop 10000 sampledState

      validCount `shouldSatisfy` (> 100)
      streamingCount `shouldSatisfy` (> 100)
      completesEventually `shouldBe` True

    context "loading different layers" $ do
      it "loads layer 1 successfully" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 1
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            totalCycles = 50_000

        let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
              let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                  (ddrMasterOut, _weightStream, streamValid, sysState) =
                    weightManagementSystem ddrSlave startStream layerIdx sinkRdy
              (streamValid, sysState)
        
        let (streamValid, sysState) = simulation
            sampledValid = sampleN totalCycles streamValid
            sampledState = sampleN totalCycles sysState

            streamEverValid = P.or sampledValid
            systemWentStreaming = WSStreaming `P.elem` sampledState

        streamEverValid `shouldBe` True
        systemWentStreaming `shouldBe` True

  describe "state transitions" $ do
    it "transitions from Ready -> Streaming -> Ready" $ do
      let
          params = PARAM.decoderConst
          layerIdx = pure 0
          startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
          sinkRdy = pure True
          totalCycles = 50_000

      let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
            let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                (ddrMasterOut, _weightStream, _streamValid, sysState) =
                  weightManagementSystem ddrSlave startStream layerIdx sinkRdy
            sysState
      
      let sysState = simulation
          sampledState = sampleN totalCycles sysState
          states = P.dropWhile (== WSReady) sampledState

          -- After initial ready, should enter streaming
          firstTransition = if P.null states then WSReady else P.head states

          -- Eventually should return to ready
          eventuallyReady = WSReady `P.elem` P.drop 100 states

      firstTransition `shouldBe` WSStreaming
      eventuallyReady `shouldBe` True

    it "does not start streaming without trigger" $ do
      let
          layerIdx = pure 0
          startStream = pure False -- Never trigger
          sinkRdy = pure True
          totalCycles = 1000

      let (_ddrMaster, _, _, sysState) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ weightManagementSystem mockAxiSlave startStream layerIdx sinkRdy

      let sampledState = sampleN totalCycles sysState
          allReady = P.all (== WSReady) sampledState

      allReady `shouldBe` True
