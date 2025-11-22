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
    expsHC :: Exponent,
    expsDRAM :: Exponent,
    mant0HC :: Mantissa,
    mant0DRAM :: Mantissa
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
createTestParams :: RowI8E ModelDimension -> PARAM.DecoderParameters
createTestParams testRow =
  PARAM.DecoderParameters
    { PARAM.modelEmbedding =
        PARAM.EmbeddingComponentQ
          { PARAM.vocabularyQ = repeat testRow :: MatI8E VocabularySize ModelDimension,
            PARAM.rmsFinalWeightF = repeat 1.0 :: Vec ModelDimension FixedPoint
          },
      PARAM.modelLayers = repeat layerParams
    }
  where
    testMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
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
    ffnW1 = repeat testRow :: MatI8E HiddenDimension ModelDimension
    ffnW2 = repeat RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: MatI8E ModelDimension HiddenDimension
    ffnW3 = repeat testRow :: MatI8E HiddenDimension ModelDimension
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
      let maxCycles = 30
          layerIdx = 4 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          -- Hardcoded params: mantissa=1, exp=0
          testRowHC = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E ModelDimension
          paramsHC = createTestParams testRowHC

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
                  paramsHC
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
                expsHC = expsHCs P.!! i,
                expsDRAM = expsDRAMs P.!! i,
                mant0HC = mant0HCs P.!! i,
                mant0DRAM = mant0DRAMs P.!! i
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
              expsHCs = sampleN maxCycles (qhCurrentRowExp debugInfo)
              expsDRAMs = sampleN maxCycles (qhCurrentRow'Exp debugInfo)
              mant0HCs = sampleN maxCycles (qhCurrentRowMant0 debugInfo)
              mant0DRAMs = sampleN maxCycles (qhCurrentRow'Mant0 debugInfo)

      it "both sources show identical data in all cycles" $ do
        let mismatches = P.filter (\d -> mant0HC d /= mant0DRAM d || expsHC d /= expsDRAM d) diagnostics
        case mismatches of
          [] -> P.return ()
          (d:_) -> expectationFailure $ 
            "Data sources diverged at cycle " P.++ show (cycleNum d) 
            P.++ ": HC[mant=" P.++ show (mant0HC d) P.++ ",exp=" P.++ show (expsHC d) 
            P.++ "] vs DRAM[mant=" P.++ show (mant0DRAM d) P.++ ",exp=" P.++ show (expsDRAM d) P.++ "]"

      it "DRAM fetches occur at expected intervals" $ do
        let fetchCycles = P.map cycleNum $ P.filter fetchValid diagnostics
            expectedPattern = [5, 11, 17, 23] -- Every 6 cycles after first completion
        fetchCycles `shouldBe` expectedPattern

      it "completes 8 rows successfully" $ do
        let doneCycles = P.filter rowDone diagnostics
        P.length doneCycles `shouldBe` 8

      it "produces valid output exactly once" $ do
        let valids = sampleN maxCycles validOut
            validCount = P.length $ P.filter id valids
        validCount `shouldBe` 1

    context "when DRAM differs from hardcoded params" $ do
      let maxCycles = 30
          layerIdx = 4 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          -- Hardcoded params: mantissa=1, exp=0
          testRowHC = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E ModelDimension
          paramsHC = createTestParams testRowHC

          -- DRAM pattern: DIFFERENT (mantissa=2, exp=1)
          dramPatternDifferent :: BitVector 512
          dramPatternDifferent = pack $ replicate (SNat @63) (2 :: BitVector 8) ++ singleton (1 :: BitVector 8)

          inputVec = repeat 1.0 :: Vec ModelDimension FixedPoint
          validIn = fromList ([False, True] P.++ P.replicate (maxCycles - 2) False) :: Signal System Bool
          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index SequenceLength)
          input = pure inputVec :: Signal System (Vec ModelDimension FixedPoint)

          (masterOut, qOut, validOut, readyOut, debugInfo) =
            exposeClockResetEnable
              ( queryHeadProjector
                  (createMockDRAM dramPatternDifferent arvalidSignal)
                  layerIdx
                  headIdx
                  validIn
                  downStreamReady
                  stepCount
                  input
                  paramsHC
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
                expsHC = expsHCs P.!! i,
                expsDRAM = expsDRAMs P.!! i,
                mant0HC = mant0HCs P.!! i,
                mant0DRAM = mant0DRAMs P.!! i
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
              expsHCs = sampleN maxCycles (qhCurrentRowExp debugInfo)
              expsDRAMs = sampleN maxCycles (qhCurrentRow'Exp debugInfo)
              mant0HCs = sampleN maxCycles (qhCurrentRowMant0 debugInfo)
              mant0DRAMs = sampleN maxCycles (qhCurrentRow'Mant0 debugInfo)

      it "hardcoded source remains constant" $ do
        let activeCycles = P.filter (\d -> rowIdx d > 0) diagnostics
            allHCMants = P.map mant0HC activeCycles
            allHCExps = P.map expsHC activeCycles
        P.all (== 1) allHCMants `shouldBe` True
        P.all (== 0) allHCExps `shouldBe` True

      it "DRAM source shows different values after fetch" $ do
        let postFetchCycles = P.filter (\d -> cycleNum d >= 5 && rowIdx d > 0) diagnostics
            allDRAMMants = P.map mant0DRAM postFetchCycles
            allDRAMExps = P.map expsDRAM postFetchCycles
        -- After first fetch, DRAM should have different values
        P.all (== 2) allDRAMMants `shouldBe` True
        P.all (== 1) allDRAMExps `shouldBe` True

      it "computation uses hardcoded source (not DRAM)" $ do
        let finalQOut = P.last (sampleN maxCycles qOut)
            actualResult = P.head $ toList finalQOut
            expectedHC = 64.0 -- Using HC: 64 * 1.0 * 1 * 2^0
            expectedDRAM = 128.0 -- Using DRAM: 64 * 1.0 * 2 * 2^1
        actualResult `shouldBe` expectedHC
        actualResult `shouldNotBe` expectedDRAM

    context "FSM and timing verification" $ do
      let maxCycles = 30
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads
          testRowHC = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E ModelDimension
          paramsHC = createTestParams testRowHC
          dramPattern = pack $ replicate (SNat @63) (1 :: BitVector 8) ++ singleton (0 :: BitVector 8)

          inputVec = repeat 1.0 :: Vec ModelDimension FixedPoint
          validIn = fromList ([False, True] P.++ P.replicate (maxCycles - 2) False) :: Signal System Bool
          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index SequenceLength)
          input = pure inputVec :: Signal System (Vec ModelDimension FixedPoint)

          (masterOut, qOut, validOut, readyOut, debugInfo) =
            exposeClockResetEnable
              ( queryHeadProjector
                  (createMockDRAM dramPattern arvalidSignal)
                  layerIdx
                  headIdx
                  validIn
                  downStreamReady
                  stepCount
                  input
                  paramsHC
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
                expsHC = expsHCs P.!! i,
                expsDRAM = expsDRAMs P.!! i,
                mant0HC = mant0HCs P.!! i,
                mant0DRAM = mant0DRAMs P.!! i
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
              expsHCs = sampleN maxCycles (qhCurrentRowExp debugInfo)
              expsDRAMs = sampleN maxCycles (qhCurrentRow'Exp debugInfo)
              mant0HCs = sampleN maxCycles (qhCurrentRowMant0 debugInfo)
              mant0DRAMs = sampleN maxCycles (qhCurrentRow'Mant0 debugInfo)

      it "FSM transitions: MIdle -> MReset -> MProcessing" $ do
        let stateTransitions = P.zip (P.map state diagnostics) (P.tail $ P.map state diagnostics)
            hasIdleToReset = P.any (\(s1, s2) -> s1 == OPS.MIdle && s2 == OPS.MReset) stateTransitions
            hasResetToProcessing = P.any (\(s1, s2) -> s1 == OPS.MReset && s2 == OPS.MProcessing) stateTransitions
        hasIdleToReset `shouldBe` True
        hasResetToProcessing `shouldBe` True

      it "each row takes exactly 3 cycles (Reset + Processing + Done)" $ do
        let rowCompletions = P.filter rowDone diagnostics
            rowCycles = P.map cycleNum rowCompletions
            -- Expected: 4, 7, 10, 13, 16, 19, 22, 25 (every 3 cycles)
            expectedCycles = [4, 7, 10, 13, 16, 19, 22, 25]
        rowCycles `shouldBe` expectedCycles

      it "rowReset fires exactly once per row" $ do
        let resetCycles = P.filter rowReset diagnostics
        P.length resetCycles `shouldBe` 8

      it "accumulator resets to 0 after each row" $ do
        let resetCycles = P.filter rowReset diagnostics
            accumsAfterReset = P.map (\d -> accumVal $ diagnostics P.!! (cycleNum d + 1)) resetCycles
        P.all (== 0.0) accumsAfterReset `shouldBe` True

      it "final accumulator value equals 64.0 for each row" $ do
        let doneCycles = P.filter rowDone diagnostics
            finalAccums = P.map accumVal doneCycles
        P.all (== 64.0) finalAccums `shouldBe` True

      it "CRITICAL BUG: row 0 processes before DRAM fetch completes" $ do
        let row0ProcessingCycles = P.filter (\d -> rowIdx d == 0 && rowEnable d) diagnostics
            firstFetchCycle = P.head $ P.map cycleNum $ P.filter fetchValid diagnostics
            row0EnableCycle = P.head $ P.map cycleNum row0ProcessingCycles
        
        P.putStrLn $ "\nRow 0 processing starts at cycle: " P.++ show row0EnableCycle
        P.putStrLn $ "First DRAM fetch completes at cycle: " P.++ show firstFetchCycle
        
        -- This SHOULD fail (and currently does) - processing happens before data arrives
        row0EnableCycle `shouldSatisfy` (< firstFetchCycle)
        -- This documents the bug
        expectationFailure $ "Row 0 processes at cycle " P.++ show row0EnableCycle 
          P.++ " but DRAM data not valid until cycle " P.++ show firstFetchCycle

      it "shows which row gets corrupted data" $ do
        let rowResults = P.map (\i -> 
              let rowCycles = P.filter (\d -> rowIdx d == i) diagnostics
                  finalCycle = P.last rowCycles
              in (i, P.head $ toList $ qOutVec finalCycle, cycleNum finalCycle)
              ) [0..7]
        
        P.putStrLn "\nFinal qOut values per row:"
        mapM_ (\(row, val, cyc) -> 
          P.putStrLn $ "  Row " P.++ show row P.++ " @ cycle " P.++ show cyc P.++ ": " P.++ show val
          ) rowResults
        
        -- Expect all to be 128.0 if using DRAM correctly, but likely row 0 will be wrong
        let wrongRows = P.filter (\(_, val, _) -> val /= 128.0) rowResults
        P.length wrongRows `shouldSatisfy` (> 0)
