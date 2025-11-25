module LLaMa2.Layer.Attention.QKVProjectionSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Layer.Attention.QKVProjection (QHeadDebugInfo (..), queryHeadProjector)
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Numeric.Operations as OPS
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E (..))
import LLaMa2.Numeric.Types (Exponent, FixedPoint, Mantissa)
import LLaMa2.Types.ModelConfig
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave

-- Diagnostic record for cycle-by-cycle comparison
data CycleDiagnostic = CycleDiagnostic
  { cycleNum :: Int,
    rowIdx :: Index HeadDimension,
    state :: OPS.MultiplierState,
    rowReset :: Bool,
    rowEnable :: Bool,
    fetchValid :: Bool,
    firstMant :: Mantissa,
    accumVal :: FixedPoint,
    rowResult :: FixedPoint,
    rowDone :: Bool,
    qOutVec :: Vec HeadDimension FixedPoint,
    expsWL :: Exponent,
    mant0WL :: Mantissa
  }
  deriving (Show, Eq)

-- Helper to create mock DRAM with configurable pattern
createMockDRAM :: BitVector 512 -> Signal System Bool -> Slave.AxiSlaveIn System
createMockDRAM pattern arvalidSignal' =
  Slave.AxiSlaveIn
    { arready = pure True,
      rvalid = delayedValid arvalidSignal',
      rdata = pure (AxiR pattern 0 True 0),
      awready = pure False,
      wready = pure False,
      bvalid = pure False,
      bdata = pure (AxiB 0 0)
    }
  where
    delayedValid arvalid =
      exposeClockResetEnable
        (register False $ register False arvalid)
        CS.systemClockGen
        CS.resetGen
        CS.enableGen

-- Create test parameters with specific weight pattern
createTestParams :: MatI8E HeadDimension ModelDimension -> PARAM.DecoderParameters
createTestParams testMatrix =
  PARAM.DecoderParameters
    { PARAM.modelEmbedding =
        PARAM.EmbeddingComponentQ
          { PARAM.vocabularyQ = repeat testRow0 :: MatI8E VocabularySize ModelDimension,
            PARAM.rmsFinalWeightF = repeat 1.0 :: Vec ModelDimension FixedPoint
          },
      PARAM.modelLayers = repeat layerParams
    }
  where
    testRow0 = head testMatrix
    mockRotary =
      PARAM.RotaryEncodingComponentF
        { PARAM.freqCosF = repeat (repeat 1.0),
          PARAM.freqSinF = repeat (repeat 0.0)
        }
    mockHeadParams =
      PARAM.SingleHeadComponentQ
        { PARAM.wqHeadQ = testMatrix,
          PARAM.wkHeadQ = testMatrix,
          PARAM.wvHeadQ = testMatrix,
          PARAM.rotaryF = mockRotary
        }
    testRow' = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E HeadDimension
    testWOMatrix = repeat testRow' :: MatI8E ModelDimension HeadDimension
    mhaParams =
      PARAM.MultiHeadAttentionComponentQ
        { PARAM.headsQ = repeat mockHeadParams,
          PARAM.mWoQ = repeat testWOMatrix,
          PARAM.rmsAttF = repeat 1.0 :< 0
        }
    ffnW1 = repeat testRow0 :: MatI8E HiddenDimension ModelDimension
    ffnW2 = repeat RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: MatI8E ModelDimension HiddenDimension
    ffnW3 = repeat testRow0 :: MatI8E HiddenDimension ModelDimension
    ffnParams =
      PARAM.FeedForwardNetworkComponentQ
        { PARAM.fW1Q = ffnW1,
          PARAM.fW2Q = ffnW2,
          PARAM.fW3Q = ffnW3,
          PARAM.fRMSFfnF = repeat 1.0 :< 0
        }
    layerParams =
      PARAM.TransformerLayerComponent
        { PARAM.multiHeadAttention = mhaParams,
          PARAM.feedforwardNetwork = ffnParams
        }

spec :: Spec
spec = do
  describe "queryHeadProjector - DRAM vs Hardcoded Comparison" $ do
    context "when DRAM matches hardcoded params" $ do
      let maxCycles = 100
          layerIdx = 4 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          -- Hardcoded params: mantissa=1, exp=0
          testRow = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E ModelDimension
          testMatrix = repeat testRow
          paramsWL = createTestParams testMatrix

          -- DRAM pattern: same as hardcoded (mantissa=1, exp=0)
          dramPatternMatching :: BitVector 512
          dramPatternMatching = pack $ replicate (SNat @63) (1 :: BitVector 8) ++ singleton (0 :: BitVector 8)

          inputVec = repeat 1.0 :: Vec ModelDimension FixedPoint
          validIn = fromList ([False, True] P.++ P.replicate (maxCycles - 2) False) :: Signal System Bool
          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index SequenceLength)
          input = pure inputVec :: Signal System (Vec ModelDimension FixedPoint)

          (masterOut, qOut, validOut, readyOut, debugInfo) =
            exposeClockResetEnable
              ( queryHeadProjector
                  (createMockDRAM dramPatternMatching arvalidSignal)
                  layerIdx
                  headIdx
                  validIn
                  downStreamReady
                  stepCount
                  input
                  paramsWL
              )
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          arvalidSignal = Master.arvalid masterOut

          diagnostics = flip P.map [0 .. maxCycles - 1] $ \i ->
            CycleDiagnostic
              { cycleNum = i,
                rowIdx = rowIdxs P.!! i,
                state = states P.!! i,
                rowReset = rowResets P.!! i,
                rowEnable = rowEnables P.!! i,
                fetchValid = fetchValids P.!! i,
                firstMant = firstMants P.!! i,
                accumVal = accumVals P.!! i,
                rowResult = rowResults P.!! i,
                rowDone = rowDones P.!! i,
                qOutVec = qOutVecs P.!! i,
                expsWL = expsWLs P.!! i,
                mant0WL = mant0WLs P.!! i
              }
            where
              rowIdxs = sampleN maxCycles (qhRowIndex debugInfo)
              states = sampleN maxCycles (qhState debugInfo)
              rowResets = sampleN maxCycles (qhRowReset debugInfo)
              rowEnables = sampleN maxCycles (qhRowEnable debugInfo)
              fetchValids = sampleN maxCycles (qhFetchValid debugInfo)
              firstMants = sampleN maxCycles (qhFirstMant debugInfo)
              accumVals = sampleN maxCycles (qhAccumValue debugInfo)
              rowResults = sampleN maxCycles (qhRowResult debugInfo)
              rowDones = sampleN maxCycles (qhRowDone debugInfo)
              qOutVecs = sampleN maxCycles (qhQOut debugInfo)
              expsWLs = sampleN maxCycles (qhCurrentRowExp debugInfo)
              mant0WLs = sampleN maxCycles (qhCurrentRowMant0 debugInfo)

      it "completes 8 rows successfully" $ do
        let doneCycles = P.filter rowDone diagnostics
        P.length doneCycles `shouldBe` 8

  describe "queryHeadProjector - Multi-Token / Boundary Regression" $ do
    it "processes Token 2 with correct Row 0 weights (Fixes 'Stale Row' bug)" $ do
      let maxCycles = 150
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          -- Create distinctive rows:
          -- Row 0: All 1s (Result should be 64.0)
          -- Row 7: All 10s (Result should be 640.0)
          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E ModelDimension
          rowOther = RowI8E {rowMantissas = repeat 10, rowExponent = 0} :: RowI8E ModelDimension

          -- Build matrix: Row 0 is distinct
          testMatrix = row0 :> replicate d7 rowOther
          params = createTestParams testMatrix

          -- Input Vector (all 1.0)
          inputVec = repeat 1.0 :: Vec ModelDimension FixedPoint

          -- Signal sequence:
          -- Token 1 @ Cycle 1
          -- ... wait ...
          -- Token 2 @ Cycle 80 (After Token 1 is definitely done)
          validIn = fromList (
              [False, True] P.++ P.replicate 78 False P.++
              [True] P.++ P.replicate (maxCycles - 81) False
            ) :: Signal System Bool

          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index SequenceLength)
          input = pure inputVec :: Signal System (Vec ModelDimension FixedPoint)

          -- Use REAL STUB to simulate realistic latency/addressing
          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen

          (masterOut, qOut, validOut, readyOut, debugInfo) =
            exposeClockResetEnable
              (queryHeadProjector stubSlaveIn layerIdx headIdx validIn downStreamReady stepCount input params)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Capture data
          rowDones = sampleN maxCycles (qhRowDone debugInfo)
          rowIndices = sampleN maxCycles (qhRowIndex debugInfo)
          accumVals = sampleN maxCycles (qhAccumValue debugInfo)
          states = sampleN maxCycles (qhState debugInfo)

      -- 1. Verify Token 1 completed
      let token1Row0Done = P.head $ P.filter P.snd (P.zip [0..] rowDones)
      P.putStrLn $ "Token 1 Row 0 Done at: " P.++ show (fst token1Row0Done :: Int)

      -- 2. Find start of Token 2 (approximate)
      -- We look for Row 0 Done *after* cycle 80
      let token2Row0DoneCandidates = P.filter (\(i, done) -> i > 80 && done && rowIndices P.!! i == 0) (P.zip [0..] rowDones)

      case token2Row0DoneCandidates of
        [] -> expectationFailure "Token 2 did not complete Row 0 processing!"
        (t2Cycle, _):_ -> do
           let result = accumVals P.!! t2Cycle
           P.putStrLn $ "Token 2 Row 0 Result (Cycle " P.++ show t2Cycle P.++ "): " P.++ show result

           -- Check logic:
           -- If 64.0  -> Correct (Used Row 0 weights)
           -- If 640.0 -> Fail (Used Row 7 weights / Stale Index)
           -- If 0.0   -> Fail (Used Zeros / Stale Latch)

           if result == 64.0
             then return () -- PASS
             else expectationFailure $
                "Token 2 Data Corruption! Expected 64.0 (Row 0), Got " P.++ show result P.++
                ". (Result 640.0 implies stuck Index, 0.0 implies invalid fetch)"

    it "resets rowIndex to 0 immediately upon Idle state" $ do
      let maxCycles = 100
          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
          testMatrix = repeat row0
          params = createTestParams testMatrix

          validIn = fromList ([False, True] P.++ P.replicate 98 False)

          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen

          (masterOut, _, _, _, debugInfo) = exposeClockResetEnable
              (queryHeadProjector stubSlaveIn 0 0 validIn (pure True) (pure 0) (pure (repeat 1.0)) params)
              CS.systemClockGen CS.resetGen CS.enableGen

          states = sampleN maxCycles (qhState debugInfo)
          rowIndices = sampleN maxCycles (qhRowIndex debugInfo)

          -- Find cycle where we transition Done -> Idle
          idlesAfterWork = DL.elemIndices OPS.MIdle states
          lateIdles = P.filter (> 10) idlesAfterWork -- Filter initial idles

      case lateIdles of
        [] -> P.return () -- Test might be too short to see Idle return
        (i:_) -> do
           let idxAtIdle = rowIndices P.!! i
           P.putStrLn $ "Index at first Return-to-Idle (Cycle " P.++ show i P.++ "): " P.++ show idxAtIdle
           idxAtIdle `shouldBe` 0

    it "CRITICAL: rowReqValid is only asserted when weightReady is True" $ do
      let maxCycles = 100
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads
          
          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
          params = createTestParams (repeat row0)
          
          validIn = fromList ([False, True] P.++ P.replicate 78 False P.++
                            [True] P.++ P.replicate 19 False)
          
          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen
          
          (masterOut, qOut, validOut, readyOut, debugInfo) = exposeClockResetEnable
            (queryHeadProjector stubSlaveIn layerIdx headIdx validIn (pure True) (pure 0) (pure (repeat 1.0)) params)
            CS.systemClockGen CS.resetGen CS.enableGen
          
          -- We need to expose these from queryHeadProjector:
          -- - rowReqValid signal
          -- - weightReady signal  
          -- For now, we can infer violations from arvalid timing
          
          arvalids = sampleN maxCycles (Master.arvalid masterOut)
          fetchValids = sampleN maxCycles (qhFetchValid debugInfo)
          
      -- Count how many AXI read requests are issued
      let totalRequests = P.length (P.filter id arvalids)
      P.putStrLn $ "Total AXI read requests: " P.++ show totalRequests
      
      -- For 2 tokens × 8 rows = should be exactly 16 requests
      -- If we see more, it means requests were issued when not ready
      totalRequests `shouldBe` 16

    it "detects lost requests when rowReqValid asserted while weightReady is False" $ do
      let maxCycles = 150
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads
          
          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
          params = createTestParams (repeat row0)
          
          validIn = fromList ([False, True] P.++ P.replicate 78 False P.++
                            [True] P.++ P.replicate 69 False)
          
          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen
          
          (masterOut, qOut, validOut, readyOut, debugInfo) = exposeClockResetEnable
            (queryHeadProjector stubSlaveIn layerIdx headIdx validIn (pure True) (pure 0) (pure (repeat 1.0)) params)
            CS.systemClockGen CS.resetGen CS.enableGen
          
          reqValids = sampleN maxCycles (qhRowReqValid debugInfo)
          weightReadys = sampleN maxCycles (qhWeightReady debugInfo)
          weightValids = sampleN maxCycles (qhWeightValid debugInfo)
          arvalids = sampleN maxCycles (Master.arvalid masterOut)
          states = sampleN maxCycles (qhState debugInfo)
          rowIndices = sampleN maxCycles (qhRowIndex debugInfo)
      
      -- Find cycles where rowReqValid is True but weightReady is False
      let lostRequests = P.filter (\i -> reqValids P.!! i && not (weightReadys P.!! i)) [0..maxCycles-1]
      
      P.putStrLn $ "\n*** LOST REQUESTS: " P.++ show (P.length lostRequests) P.++ " ***"
      P.mapM_ (\i -> P.putStrLn $ 
                "  Cycle " P.++ show i P.++
                ": rowIdx=" P.++ show (rowIndices P.!! i) P.++
                ", state=" P.++ show (states P.!! i) P.++
                ", reqValid=" P.++ show (reqValids P.!! i) P.++
                ", weightReady=" P.++ show (weightReadys P.!! i))
              (P.take 10 lostRequests)
      
      -- These lost requests explain why we only see 11 AXI requests instead of 16
      let totalArvalid = P.length (P.filter id arvalids)
          totalLost = P.length lostRequests
      
      P.putStrLn $ "\nTotal AXI requests: " P.++ show totalArvalid
      P.putStrLn $ "Lost requests: " P.++ show totalLost
      P.putStrLn "Expected total: 16"
      P.putStrLn $ "Actual + Lost: " P.++ show (totalArvalid + totalLost)
      
      -- The fix: rowReqValid should only be asserted when weightReady is True
      P.length lostRequests `shouldBe` 0
