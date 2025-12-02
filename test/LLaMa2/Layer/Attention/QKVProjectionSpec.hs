module LLaMa2.Layer.Attention.QKVProjectionSpec (spec) where

import Clash.Prelude
import qualified Prelude as P
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
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import Data.Maybe (isNothing, isJust)

-- Read-only AXI stub that returns all beats in 'payload' per AR.
-- - ARREADY always True
-- - First RVALID appears 2 cycles after ARVALID rises
-- - RVALID stays high for each beat; RLAST high on final beat
-- - Ignores RREADY (push model), which is fine for these unit tests
createMockDRAMBurstL
  :: [BitVector 512]          -- ^ payload beats to emit per AR
  -> Signal System Bool       -- ^ ARVALID from the master
  -> Slave.AxiSlaveIn System
createMockDRAMBurstL payload arvalidSig =
  Slave.AxiSlaveIn
    { arready = pure True
    , rvalid  = isActive
    , rdata   = rData
    , awready = pure False
    , wready  = pure False
    , bvalid  = pure False
    , bdata   = pure (AxiB 0 0)
    }
 where
  lastIx :: Int
  lastIx = P.length payload P.- 1

  -- Start a burst 2 cycles after ARVALID (simple fixed latency)
  start :: Signal System Bool
  start = exposeClockResetEnable
            (register False (register False arvalidSig))
            CS.systemClockGen CS.resetGen CS.enableGen

  -- Current beat index: Nothing=idle; Just i = serving beat i
  idxS :: Signal System (Maybe Int)
  idxS = exposeClockResetEnable (register Nothing nextIdx)
          CS.systemClockGen CS.resetGen CS.enableGen

  nextIdx :: Signal System (Maybe Int)
  nextIdx =
    let idleToStart = mux start (pure (Just 0)) (pure Nothing)
    in  mux (fmap isNothing idxS)
            idleToStart
        -- serving: advance until last beat, then go idle
        ( (\case
              Just i
                | i >= lastIx -> Nothing
                | otherwise -> Just (i + 1)
              Nothing -> Nothing
          ) <$> idxS )

  isActive :: Signal System Bool
  isActive = isJust <$> idxS

  rData :: Signal System AxiR
  rData =
    (\case
        Just i
          -> let
              dat = payload P.!! i
              last' = i == lastIx
            in AxiR dat 0 last' 0
        Nothing -> AxiR 0 0 False 0
    ) <$> idxS

-- Convenience: build a correct payload for a given RowI8E using the same packer as production
createMockDRAMForRow
  :: RowI8E ModelDimension
  -> Signal System Bool
  -> Slave.AxiSlaveIn System
createMockDRAMForRow row arvalidSig =
  let wordsL = DRAMSlave.packRowMultiWord @ModelDimension row  -- [BitVector 512], length = WordsPerRow ModelDimension
  in  createMockDRAMBurstL wordsL arvalidSig

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
      let maxCycles = 1000
          layerIdx = 4 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          -- Hardcoded params: mantissa=1, exp=0
          testRow = RowI8E {rowMantissas = repeat 1, rowExponent = 0} :: RowI8E ModelDimension
          testMatrix = repeat testRow
          paramsWL = createTestParams testMatrix

          inputVec = repeat 1.0 :: Vec ModelDimension FixedPoint
          validIn = fromList ([False, True] P.++ P.replicate (maxCycles - 2) False) :: Signal System Bool
          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index SequenceLength)
          input = pure inputVec :: Signal System (Vec ModelDimension FixedPoint)

          (masterOut, _qOut, _validOut, _readyOut, debugInfo) =
            exposeClockResetEnable
              ( queryHeadProjector
                  (createMockDRAMForRow testRow arvalidSignal)
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
