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
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Numeric.Types (Mantissa)
import Control.Monad (zipWithM_)

testParams :: PARAM.DecoderParameters
testParams = PARAM.decoderConst

-- Stimulus pattern: 8 valid requests for rows 0..7, then idle
stimulusLength :: Int
stimulusLength = 350  -- comfortably larger than any maxCycles used

rowReqs :: [Index HeadDimension]
rowReqs = [0..7] P.++ P.replicate (stimulusLength - 8) 0

rowReqValid :: [Bool]
rowReqValid = P.replicate 8 True P.++ P.replicate (stimulusLength - 8) False

spec :: Spec
spec = do
  --------------------------------------------------------------------------
  -- Test 1: hardcoded-only path sanity (no DRAM attached)
  -- NOTE: check hcRowOut directly (hardcoded path). dram-valid is irrelevant here.
  --------------------------------------------------------------------------
  describe "weightLoader hardcoded path" $ do
    it "produces valid outputs for requested rows" $ do
      let maxCycles = 100
          -- NOTE: make signal infinite by appending repeat 0 to avoid finite-list issues
          reqSig      = fromList (rowReqs P.++ P.repeat 0)
          reqValidSig = fromList (rowReqValid P.++ P.repeat False)
          readySig    = pure True

          -- a simple dummy AXI slave (never responds). It's safe for the HC path.
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

          -- Run the weightLoader under the System clock/reset/enable.
          (_, weightsOut, _validHC, _ready) =
            exposeClockResetEnable
              (weightLoader dummySlave 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample the hard-coded row mantissa 0 over time
          hcMant0s :: [Mantissa]
          hcMant0s = sampleN maxCycles ( (!! (0 :: Int)) . rowMantissas <$> hcRowOut weightsOut )

      -- We expect the hardcoded rows (mantissas > 0) to appear at least 8 times
      P.length (P.filter (> 0) hcMant0s) `shouldSatisfy` (>= 8)

  --------------------------------------------------------------------------
  -- Test 2: DRAM-backed path (fixed compilation + wiring)
  -- NOTE: make the request signals infinite so sampleN() cannot exhaust them.
  --------------------------------------------------------------------------
  describe "weightLoader DRAM path" $ do
    it "produces valid outputs from DRAM" $ do
      let maxCycles = 500
          -- append infinite tail to avoid 'finite list' XException
          reqSig      = fromList (rowReqs P.++ P.repeat 0)
          reqValidSig = fromList (rowReqValid P.++ P.repeat False)
          readySig    = pure True

          -- Build the DRAM contents once from params (Vec 65536 WordData)
          dramContents :: Vec 65536 DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams

          -- realDRAM helper (recursive wiring)
          realDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec (DRAMSlave.DRAMConfig 1 1 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Recursive let: weightLoader produces axiMaster which is fed back to the realDRAM helper.
          (axiDRAM, outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample outputs
          valids   = sampleN maxCycles dvDRAM
          totalValids = P.length (P.filter id valids)

      -- We expect at least some valid completions (exact number depends on config/latency)
      totalValids `shouldSatisfy` (> 0)
      totalValids `shouldSatisfy` (<= maxCycles)

  --------------------------------------------------------------------------
  -- Test 3: Equivalence HC vs DRAM, serialized requests (with dynamic offset)
  --------------------------------------------------------------------------
  describe "weightLoader equivalence (robust timing)" $ do
    it "DRAM and hardcoded paths produce matching rows" $ do
      let maxCycles = 2000
          readySig  = pure True

          -- Compute offset with safety margin
          modelDim = natToNum @ModelDimension
          wordsPerRow = ceiling (modelDim / 63 :: Double)
          readLat = 1

          -- Conservative estimate
          estimatedOffset = 2 + readLat + wordsPerRow + 2
          cyclesPerRequest = estimatedOffset + 20

          -- Generate well-separated requests
          requestGroups = [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                          | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqList = P.map fst requestPairs
          validList = P.map snd requestPairs

          reqSig = fromList (reqList P.++ P.repeat 0)
          reqValidSig = fromList (validList P.++ P.repeat False)

          -- DRAM-backed loader
          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                 (DRAMSlave.DRAMConfig readLat 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample ALL signals for debugging
          reqsSampled = sampleN maxCycles reqSig
          reqValidSampled = sampleN maxCycles reqValidSig
          readySampled = sampleN maxCycles readyDRAM
          dramRows = sampleN maxCycles (dramRowOut outDRAM)
          valids = sampleN maxCycles dvDRAM

          -- Track request acceptance: both reqValid AND dramReady must be true
          acceptedRequests = [(n, reqsSampled P.!! n)
                             | n <- [0..maxCycles-1]
                             , reqValidSampled P.!! n
                             , readySampled P.!! n]

          -- Track DRAM outputs
          dramOutputs = [(n, dramRows P.!! n) | n <- [0..maxCycles-1], valids P.!! n]

          -- Get HC reference
          hcWeights = PARAM.wqHeadQ (head (PARAM.headsQ (PARAM.multiHeadAttention (head (PARAM.modelLayers testParams)))))

      -- DIAGNOSTIC OUTPUT
      P.putStrLn "\n=== DIAGNOSTIC OUTPUT ==="
      P.putStrLn $ "Requests issued (reqValid=True): " P.++ show (P.length [n | n <- [0..maxCycles-1], reqValidSampled P.!! n])
      P.putStrLn $ "Requests accepted (reqValid AND dramReady): " P.++ show (P.length acceptedRequests)
      P.putStrLn $ "Accepted at cycles: " P.++ show (P.map fst acceptedRequests)
      P.putStrLn $ "Accepted row indices: " P.++ show (P.map snd acceptedRequests)
      P.putStrLn $ "DRAM outputs produced: " P.++ show (P.length dramOutputs)
      P.putStrLn $ "Output at cycles: " P.++ show (P.map fst dramOutputs)
      P.putStrLn "========================\n"

      -- Verify we issued 8 requests
      P.length [n | n <- [0..maxCycles-1], reqValidSampled P.!! n] `shouldBe` 8

      -- Check how many were actually accepted
      P.length acceptedRequests `shouldBe` 8

      -- Check how many outputs we got
      P.length dramOutputs `shouldBe` 8

      -- If we got this far, compare the outputs
      zipWithM_ (\(_, dramRow) (_, expectedIdx) -> do
        let expectedRow = hcWeights !! expectedIdx
        dramRow `shouldBe` expectedRow
        ) dramOutputs acceptedRequests
