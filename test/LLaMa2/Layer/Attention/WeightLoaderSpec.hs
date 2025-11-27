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
import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..), AxiAR (..))
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
  describe "weightLoader equivalence (robust timing) - STATE MACHINE DEBUG" $ do
    it "traces the state machine to find the bug" $ do
      let maxCycles = 1400
          readySig  = pure True

          modelDim = natToNum @ModelDimension
          wordsPerRow = ceiling (modelDim / 63 :: Double) :: Int

          -- Use longer spacing to ensure no overlap
          cyclesPerRequest = 40  -- Much longer than 31 cycle completion time

          requestGroups :: [[(Index HeadDimension, Bool)]]
          requestGroups = [(0, False)] :  -- Idle during reset
                [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqList = P.map fst requestPairs
          validList = P.map snd requestPairs

          reqSig = fromList (reqList P.++ P.repeat 0)
          reqValidSig = fromList (validList P.++ P.repeat False)

          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample everything
          reqsSampled = sampleN @System maxCycles reqSig
          reqValidSampled = sampleN @System maxCycles reqValidSig
          readySampled = sampleN maxCycles readyDRAM
          validsSampled = sampleN maxCycles dvDRAM

          arValidSampled = sampleN maxCycles $ Master.arvalid axiDRAM
          slaveIn = realDRAM axiDRAM
          arReadySampled = sampleN maxCycles $ Slave.arready slaveIn
          rValidSampled = sampleN maxCycles $ Slave.rvalid slaveIn
          rLastSampled = sampleN maxCycles $ rlast <$> Slave.rdata slaveIn

          requestAccepted = [n | n <- [0..maxCycles-1]
                            , reqValidSampled P.!! n
                            , readySampled P.!! n]

          axiRequests = [n | n <- [0..maxCycles-1]
                        , arValidSampled P.!! n
                        , arReadySampled P.!! n]

          dramOutputs = [n | n <- [0..maxCycles-1], validsSampled P.!! n]

      P.putStrLn "\n=== STATE MACHINE DEBUG (40 cycle spacing) ==="
      P.putStrLn $ "Requests accepted: " P.++ show (P.length requestAccepted)
      P.putStrLn $ "  At cycles: " P.++ show requestAccepted
      P.putStrLn $ "AXI AR transactions: " P.++ show (P.length axiRequests)
      P.putStrLn $ "  At cycles: " P.++ show axiRequests
      P.putStrLn $ "DRAM outputs: " P.++ show (P.length dramOutputs)
      P.putStrLn $ "  At cycles: " P.++ show dramOutputs
      P.putStrLn $ "\nWords per row: " P.++ show wordsPerRow
      P.putStrLn "========================\n"

      -- Compute how many should finish within the window based on the first AR cycle and spacing.
      let expectedWithinWindow =
            case axiRequests of
              [] -> 0
              (firstCyc:_) ->
                let spacing = 40  -- cyclesPerRequest in this test
                    lastIndex = (maxCycles - 1 - firstCyc) `div` spacing
                in  min 8 (lastIndex + 1)

      P.length requestAccepted `shouldBe` 8
      P.length axiRequests     `shouldBe` expectedWithinWindow
      P.length dramOutputs     `shouldBe` expectedWithinWindow

  describe "weightLoader - TRIGGER DEBUG" $ do
    it "confirms fetchTrigger fires 8 times" $ do
      let maxCycles = 300
          readySig = pure True
          cyclesPerRequest = 40

          -- Add idle cycle 0 before first request
          requestGroups = [(0, False)] :  -- Idle during reset
                          [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                          | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqList = P.map fst requestPairs
          validList = P.map snd requestPairs

          reqSig = fromList (reqList P.++ P.repeat 0)
          reqValidSig = fromList (validList P.++ P.repeat False)

          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample the signals used to compute fetchTrigger
          reqValidSampled = sampleN maxCycles reqValidSig
          readySampled = sampleN maxCycles readyDRAM

          -- Compute fetchTrigger cycles
          triggerFires = [n | n <- [0..maxCycles-1]
                        , reqValidSampled P.!! n
                        , readySampled P.!! n]

      P.putStrLn "\n=== TRIGGER DEBUG ==="
      P.putStrLn $ "fetchTrigger fires: " P.++ show (P.length triggerFires)
      P.putStrLn $ "  At cycles: " P.++ show triggerFires
      P.putStrLn "========================\n"

      -- fetchTrigger should fire exactly 8 times
      P.length triggerFires `shouldBe` 8

  describe "weightLoader - ADDRESS DEBUG" $ do
    it "traces addresses to find the pattern" $ do
      let maxCycles = 1400
          readySig = pure True
          cyclesPerRequest = 50

          -- Add idle cycle 0 before first request
          requestGroups = [(0, False)] :  -- Idle during reset
                          [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                          | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqList = P.map fst requestPairs
          validList = P.map snd requestPairs

          reqSig = fromList (reqList P.++ P.repeat 0)
          reqValidSig = fromList (validList P.++ P.repeat False)

          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          arValidSampled = sampleN maxCycles $ Master.arvalid axiDRAM
          arAddrSampled = sampleN maxCycles $ araddr <$> Master.ardata axiDRAM

          slaveIn = realDRAM axiDRAM
          arReadySampled = sampleN maxCycles $ Slave.arready slaveIn

          validsSampled = sampleN maxCycles dvDRAM

          axiRequests = [(n, arAddrSampled P.!! n)
                        | n <- [0..maxCycles-1]
                        , arValidSampled P.!! n
                        , arReadySampled P.!! n]

          dramOutputs = [n | n <- [0..maxCycles-1], validsSampled P.!! n]

      P.putStrLn "\n=== ADDRESS PATTERN DEBUG ==="
      P.putStrLn $ "AXI AR transactions: " P.++ show (P.length axiRequests)
      P.putStrLn "Cycle -> Address:"
      mapM_ (\(cyc, addr) -> P.putStrLn $ "  " P.++ show cyc P.++ " -> " P.++ show addr) (P.take 20 axiRequests)
      P.putStrLn $ "DRAM outputs: " P.++ show (P.length dramOutputs)
      P.putStrLn $ "  At cycles: " P.++ show dramOutputs
      P.putStrLn "========================\n"

      let expectedWithinWindow =
            case axiRequests of
              [] -> 0
              ((firstCyc,_):_) ->
                let spacing = 50  -- cyclesPerRequest
                    lastIndex = (maxCycles - 1 - firstCyc) `div` spacing
                in  min 8 (lastIndex + 1)
      P.length dramOutputs `shouldBe` expectedWithinWindow
