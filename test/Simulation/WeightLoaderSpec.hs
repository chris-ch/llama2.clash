module Simulation.WeightLoaderSpec (spec) where

import Clash.Prelude
import LLaMa2.Memory.WeightLoader (calculateLayerBaseAddress, parseI8EChunk, weightManagementSystem, WeightSystemState (..), calculateLayerSizeBytes)
import LLaMa2.Numeric.Types (Mantissa)
import Test.Hspec
import qualified Prelude as P
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import qualified Data.ByteString.Lazy as BSL
import Simulation.RamBackedAxiSlave (createRamBackedAxiSlave, WriteState (..))

-- ============================================================================
-- MOCKS
-- ============================================================================

-- A simple, always-ready AXI slave for testing.
mockAxiSlave :: Slave.AxiSlaveIn dom
mockAxiSlave =
  Slave.AxiSlaveIn
    { arready = pure True,
      rvalid = fromList $ P.replicate 5 False P.++ P.repeat True,
      rdata = pure (AXITypes.AxiR 0 0 False 0),
      awready = pure True,
      wready = fromList $ P.replicate 5 False P.++ P.repeat True,
      bvalid = pure True,
      bdata = pure (AXITypes.AxiB 0 0)
    }

-- ============================================================================
-- SPEC
-- ============================================================================

spec :: Spec
spec = do
  describe "calculateLayerBaseAddress"
    $ it "increments linearly per layer"
    $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
      addr1 `shouldBe` (addr0 + calculateLayerSizeBytes)

  describe "parseI8EChunk"
    $ it "extracts mantissas and exponent correctly"
    $ do
      let bv = bitCoerce (replicate d64 (0x7F :: BitVector 8)) :: BitVector 512
          (mants, expn) = parseI8EChunk @8 bv
      -- Correct: provide SNat 8 explicitly + value
      mants `shouldBe` replicate (SNat @8) (127 :: Mantissa)
      -- Exponent needs bit manipulation to extract 7-bit value
      pack expn `shouldBe` (0x7F :: BitVector 7)

  describe "weightManagementSystem (bypass)"
    $ context "immediately reports WSReady when bypass=True"
    $ do
      let
          powerOn = pure True
          layerIdx = pure 0
          loadTrig = pure False
          sinkRdy = pure True
          (_ddrMaster, _, _, sysReady, sysState
            ) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ weightManagementSystem mockAxiSlave powerOn layerIdx loadTrig sinkRdy

          readyS = P.drop 2 (sampleN 10 sysReady)
          stateS = P.drop 2 (sampleN 10 sysState)
      it "all ready" $ do
        P.and readyS `shouldBe` True
      it "all states ready" $ do
        P.all (== WSReady) stateS `shouldBe` True

  describe "loading smoke test: DDR gets written" $ do
    context "using full model" $ do
      it "DDR has weights from file" $ do
        modelBinary <- BSL.readFile "data/stories260K.bin"

        let powerOn   = pure True
            layerIdx  = pure 0
            loadTrig  = pure False
            sinkRdy   = pure True
            totalCycles = 400_000   -- generous budget; adjust after profiling
            
        withClockResetEnable systemClockGen resetGen enableGen $ do
          let
              (ddrSlave, _rState, wState)        = createRamBackedAxiSlave modelBinary ddrMasterOut
              (ddrMasterOut, _weightStream, streamValid,
                _sysReady, _sysState
                ) = weightManagementSystem ddrSlave powerOn layerIdx loadTrig sinkRdy

          let
              sampledWrite = sampleN totalCycles wState
              sampledValid = sampleN totalCycles streamValid

              ddrBurstObserved = WBursting `elem` sampledWrite
              streamEverValid  = or sampledValid

          ddrBurstObserved `shouldBe` True
          streamEverValid  `shouldBe` True
