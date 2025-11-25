module LLaMa2.Layer.Attention.WeightLoaderSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import LLaMa2.Layer.Attention.WeightLoader (weightLoader, WeightLoaderOutput(..))
import LLaMa2.Types.ModelConfig
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import qualified Simulation.ParamsPlaceholder as PARAM
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..))

-- Use your predefined test parameters (decoderConst)
-- Make sure PARAM.decoderConst contains testMatrix in the wqHeadQ fields.
testParams :: PARAM.DecoderParameters
testParams = PARAM.decoderConst

-- Row request sequence: run 0..7 once, then idle
rowReqs :: [Index HeadDimension]
rowReqs = [0..7] P.++ P.replicate 40 0

rowReqValid :: [Bool]
rowReqValid = P.replicate 8 True P.++ P.replicate 40 False

spec :: Spec
spec = do
  describe "weightLoader DRAM vs hardcoded equivalence" $ do
    it "returns identical rows for all requested indices" $ do

      let maxCycles = 200

          reqSig      = fromList rowReqs
          reqValidSig = fromList rowReqValid
          readySig    = pure True

          -- Build DRAM contents (Vec 65536 WordData) from test params
          dramContents :: Vec 65536 DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams

          -- Small helper to create an instantiated DRAM slave that takes a master and
          -- returns a Slave.AxiSlaveIn. This mirrors TransformerLayerSpec pattern.
          realDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec (DRAMSlave.DRAMConfig 1 1 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          ----------------------------------------------------------------
          -- 1) DRAM-backed weightLoader (master <-> realDRAM slave)
          -- Expose the loader: give it the realDRAM function as its slave.
          ----------------------------------------------------------------
          (axiDRAM, outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          ----------------------------------------------------------------
          -- 2) Hardcoded path loader: attach a dummy slave (no AXI traffic)
          --    We create a minimal static slave with all channels idle.
          ----------------------------------------------------------------
          dummySlave :: Slave.AxiSlaveIn System
          dummySlave = Slave.AxiSlaveIn
                        { arready = pure False
                        , rvalid  = pure False
                        , rdata   = pure (AxiR 0 0 False 0)
                        , awready = pure False
                        , wready  = pure False
                        , bvalid  = pure False
                        , bdata   = pure (AxiB 0 0)
                        }

          (_axiHC, outHC, _dvHC, _readyHC) =
            exposeClockResetEnable
              (weightLoader dummySlave 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          ----------------------------------------------------------------
          -- 3) Sample outputs
          ----------------------------------------------------------------
          hcRows   = sampleN maxCycles (hcRowOut outHC)
          dramRows = sampleN maxCycles (dramRowOut outDRAM)
          valids   = sampleN maxCycles dvDRAM

          compared =
            [ (n, hcRows P.!! n, dramRows P.!! n)
            | n <- [0..maxCycles-1], valids P.!! n ]

      -- For every cycle where dram reports data valid, the hardcoded row should match it.
      mapM_ (\(_cy, hc, dr) -> hc `shouldBe` dr) compared
