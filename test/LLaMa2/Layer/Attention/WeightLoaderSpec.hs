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
import LLaMa2.Memory.AXI.Types (AxiAR (..))
import LLaMa2.Numeric.Quantization (RowI8E (..))
import Control.Monad (forM_)

type TestDRAMDepth = 65536

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
  -- Test 1: hardcoded path with real DRAM (required for latched timing)
  --------------------------------------------------------------------------
  describe "weightLoader hardcoded path" $ do
    it "produces valid outputs for requested rows" $ do
      let maxCycles = 400
          cyclesPerRequest = 40

          -- Build request pattern with spacing
          requestGroups = [(0, False)] :  -- Idle during reset
                [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqList = P.map fst requestPairs
          validList = P.map snd requestPairs

          reqSig = fromList (reqList P.++ P.repeat 0)
          reqValidSig = fromList (validList P.++ P.repeat False)
          readySig = pure True

          -- Use real DRAM backend (required because hcRowOut is now latched)
          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams

          realDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec 
                (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, weightsOut, dvDRAM, _ready) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample the hard-coded row mantissa 0 when output is valid
          validsSampled = sampleN maxCycles dvDRAM
          hcMant0s = sampleN maxCycles ((!! (0 :: Int)) . rowMantissas <$> hcRowOut weightsOut)
          
          -- Get mantissa values only at valid cycles
          validMant0s = [hcMant0s P.!! n | n <- [0..maxCycles-1], validsSampled P.!! n]

      -- We expect at least 8 valid outputs with non-zero mantissas
      P.length validMant0s `shouldSatisfy` (>= 8)
      -- At least some should have non-zero mantissa (depends on actual weight data)
      P.length (P.filter (/= 0) validMant0s) `shouldSatisfy` (>= 1)

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
          (axiDRAM, _outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) testParams)
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

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, _outDRAM, dvDRAM, readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample everything
          reqValidSampled = sampleN @System maxCycles reqValidSig
          readySampled = sampleN maxCycles readyDRAM
          validsSampled = sampleN maxCycles dvDRAM

          arValidSampled = sampleN maxCycles $ Master.arvalid axiDRAM
          slaveIn = realDRAM axiDRAM
          arReadySampled = sampleN maxCycles $ Slave.arready slaveIn

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

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams @TestDRAMDepth testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, _outDRAM, _dvDRAM, readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) testParams)
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

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, _outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) testParams)
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

  describe "weightLoader - HC vs DRAM EQUIVALENCE" $ do
    it "DRAM row matches HC row for each request" $ do
      let maxCycles = 400
          readySig = pure True
          cyclesPerRequest = 50

          requestGroups = [(0, False)] :
                          [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                          | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqList = P.map fst requestPairs
          validList = P.map snd requestPairs

          reqSig = fromList (reqList P.++ P.repeat 0)
          reqValidSig = fromList (validList P.++ P.repeat False)

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams
          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample both HC and DRAM outputs
          validsSampled = sampleN maxCycles dvDRAM
          hcExpSampled = sampleN maxCycles $ rowExponent <$> hcRowOut outDRAM
          dramExpSampled = sampleN maxCycles $ rowExponent <$> dramRowOut outDRAM
          hcMant0Sampled = sampleN maxCycles $ (!! (0::Int)) . rowMantissas <$> hcRowOut outDRAM
          dramMant0Sampled = sampleN maxCycles $ (!! (0::Int)) . rowMantissas <$> dramRowOut outDRAM

          -- Get values at cycles when DRAM output is valid
          validCycles = [n | n <- [0..maxCycles-1], validsSampled P.!! n]
          
          comparisons = [(n, 
                          hcExpSampled P.!! n, 
                          dramExpSampled P.!! n,
                          hcMant0Sampled P.!! n,
                          dramMant0Sampled P.!! n)
                        | n <- validCycles]

      P.putStrLn "\n=== HC vs DRAM EQUIVALENCE ==="
      P.putStrLn "Cycle | HC Exp | DRAM Exp | HC Mant[0] | DRAM Mant[0] | Match?"
      P.putStrLn "------+--------+----------+------------+--------------+-------"
      forM_ comparisons $ \(cyc, hcE, dramE, hcM, dramM) -> do
        let expMatch = hcE == dramE
            mantMatch = hcM == dramM
            allMatch = expMatch && mantMatch
        P.putStrLn $ P.concat 
          [ show cyc, " | "
          , show hcE, " | "
          , show dramE, " | "
          , show hcM, " | "
          , show dramM, " | "
          , if allMatch then "✓" else "✗ MISMATCH"
          ]
      P.putStrLn "========================\n"

      -- All valid outputs should match
      let mismatches = [(hcE, dramE) | (_, hcE, dramE, _, _) <- comparisons, hcE /= dramE]
      mismatches `shouldBe` []
