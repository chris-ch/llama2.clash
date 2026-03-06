module LLaMa2.Embedding.OutputProjection
 ( logitsProjector
) where
import Clash.Prelude

import qualified Prelude as P

import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM (DecoderParameters (..), EmbeddingComponentQ (..))
import LLaMa2.Types.ModelConfig (ModelDimension, VocabularySize, NumLayers, NumQueryHeads)

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified LLaMa2.Memory.FPVecLoader as FPVec
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator as OutputAccumulator
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- Row Result Checker
-- Asserts DRAM and HC row results match when rowDone fires.
--------------------------------------------------------------------------------
rowResultChecker :: forall dom numRows. (HiddenClockResetEnable dom, KnownNat numRows)
  => Signal dom Bool
  -> Signal dom (Index numRows)
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
rowResultChecker rowDone rowIdx dramResult hcResult = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 nextTokenCnt
    nextTokenCnt = mux (rowDone .&&. (rowIdx .==. pure maxBound)) (tokenCnt + 1) tokenCnt

    result = mux rowDone
                 (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult)
                 dramResult

    check tok ri dr hr
      | dr P.== hr = dr
      | otherwise  = P.error $ "Logits row result mismatch at token " P.++ show tok
                    P.++ " row " P.++ show ri P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr

--------------------------------------------------------------------------------
-- Output Checker
-- Compares final logits output vectors between DRAM and HC paths.
-- Rising-edge capture avoids stale-data bug.
--------------------------------------------------------------------------------
logitsOutputChecker :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
logitsOutputChecker outputValid dramOut hcOut = result
  where
    risingEdge  = outputValid .&&. (not <$> register False outputValid)

    dramSampled = register (repeat 0) (mux risingEdge dramOut dramSampled)
    hcSampled   = register (repeat 0) (mux risingEdge hcOut  hcSampled)

    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 (mux risingEdge (tokenCnt + 1) tokenCnt)

    checkTrigger = register False risingEdge

    result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

    checkPure tok dr hr =
      let pairs      = P.zip [0..] (P.zip (toList dr) (toList hr))
          mismatches = P.filter (\(_, (d, h)) -> d P./= h) pairs
      in if P.null mismatches then dr
         else let (i, (d, h)) = P.head mismatches
              in P.error $ "Logits output mismatch at token " P.++ show tok
                        P.++ ": index " P.++ show (i :: Int)
                        P.++ " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")"
                        P.++ " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

--------------------------------------------------------------------------------
-- logitsProjector
--
-- DRAM-backed output projection (vocabulary classifier).
-- Multiplies RMS-normalised token vector (Vec ModelDimension FixedPoint) by
-- the vocabulary embedding matrix (MatI8E VocabularySize ModelDimension) to
-- produce logits (Vec VocabularySize FixedPoint).
--
-- numRows = VocabularySize (rows to schedule, one DRAM fetch each)
-- numCols = ModelDimension (columns per row)
--------------------------------------------------------------------------------
logitsProjector :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                        -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                            -- ^ DRAM
  -> Signal dom Bool                                 -- ^ inputValid (lastLayerComplete)
  -> Signal dom Bool                                 -- ^ downStreamReady
  -> Signal dom Bool                                 -- ^ consumeSignal
  -> PARAM.DecoderParameters
  -> Signal dom (Vec ModelDimension FixedPoint)      -- ^ token embedding vector
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec VocabularySize FixedPoint)    -- ^ logits output
     , Signal dom Bool                               -- ^ outputValid
     )
logitsProjector cycleCounter dramSlaveIn inputValid downStreamReady consumeSignal params tokenVecSig =
  (axiMasterOut, logitsOutFinal, outputValid)
 where
  emb = PARAM.modelEmbedding params

  --------------------------------------------------------------------------
  -- DRAM-backed rmsFinalWeightF fetch (once per inputValid; gates logits start)
  --------------------------------------------------------------------------
  -- Rising edge: fpVecLoader requires a one-shot trigger; inputValid may be
  -- level-triggered from the decoder state machine.
  inputValidRise :: Signal dom Bool
  inputValidRise = inputValid .&&. (not <$> register False inputValid)

  (rmsFinalAxiMaster, rmsFinalVec, rmsFinalValid, rmsFinalBusy) =
    FPVec.fpVecLoader cycleCounter dramSlaveIn
      inputValidRise
      (pure Layout.rmsFinalAddress)
      (PARAM.rmsFinalWeightF emb)
      "[RmsFinal] "

  -- Hold inputValidRise latch until rmsFinal fetch completes
  pendingInput :: Signal dom Bool
  pendingInput = register False nextPendingInput
   where
    nextPendingInput =
      mux (pendingInput .&&. rmsFinalValid) (pure False) $
      mux inputValidRise (pure True)
      pendingInput

  effectiveInputOuter :: Signal dom Bool
  effectiveInputOuter = pendingInput .&&. rmsFinalValid

  -- Pre-normalise with DRAM-fetched RMS weights
  tokenWithRms :: Signal dom (Vec ModelDimension FixedPoint)
  tokenWithRms = rmsNormFwFix <$> tokenVecSig <*> rmsFinalVec

  tag :: String
  tag = "[LOGITS] "

  -- Fixed indices used only for sub-module trace tags
  layerIdx :: Index NumLayers
  layerIdx = 0

  headIdx :: Index NumQueryHeads
  headIdx = 0

  -------------------------------------------------------------------------
  -- Row Index Register (VocabularySize rows)
  -------------------------------------------------------------------------
  rowIndex :: Signal dom (Index VocabularySize)
  rowIndex = register 0 nextRowIndex

  rsIn :: RowScheduler.RowSchedulerIn dom VocabularySize
  rsIn = RowScheduler.RowSchedulerIn
    { rsRowDone       = rowDone
    , rsOutputValid   = OutputTransactionController.otcOutputValid outputTxn
    , rsConsumeSignal = consumeSignal
    , rsCurrentIndex  = rowIndex
    }

  rowSched :: RowScheduler.RowSchedulerOut dom VocabularySize
  rowSched     = RowScheduler.rowScheduler rsIn
  nextRowIndex = RowScheduler.rsNextRowIndex rowSched

  -------------------------------------------------------------------------
  -- Input / Output Transaction Controllers
  -------------------------------------------------------------------------
  inputTxn = InputTransactionController.inputTransactionController
    cycleCounter layerIdx headIdx
    InputTransactionController.InputTransactionIn
      { itcInputValid      = effectiveInputOuter
      , itcOutputValid     = OutputTransactionController.otcOutputValid outputTxn
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  inputValidLatched = InputTransactionController.itcLatchedValid inputTxn

  outputTxn = OutputTransactionController.outputTransactionController
    cycleCounter layerIdx headIdx
    OutputTransactionController.OutputTransactionIn
      { otcAllDone       = RowComputeUnit.rcAllDone compute
      , otcConsumeSignal = consumeSignal
      }

  -------------------------------------------------------------------------
  -- Effective row index (resets combinationally on consume)
  -------------------------------------------------------------------------
  effectiveRowIndex :: Signal dom (Index VocabularySize)
  effectiveRowIndex = mux
    (RowScheduler.rsOutputValid rsIn .&&. RowScheduler.rsConsumeSignal rsIn)
    (pure 0)
    rowIndex

  -------------------------------------------------------------------------
  -- Row request pulse logic
  -------------------------------------------------------------------------
  loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
  rowReqValidGated = RowComputeUnit.rcFetchReq compute .&&. weightReady
  prevRowReqValid  = register False $ mux loaderBecameIdle (pure False) rowReqValidGated
  rowReqRise       = rowReqValidGated .&&. (not <$> prevRowReqValid)
  prevRowIndex     = register 0 effectiveRowIndex
  rowIndexChanged  = effectiveRowIndex ./=. prevRowIndex
  rowReqPulse      = traceEdgeC cycleCounter (tag P.++ "reqPulse") $
                       rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

  -------------------------------------------------------------------------
  -- Embedding Weight Loader (DRAM-backed)
  -------------------------------------------------------------------------
  (logitsAxiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
    LOADER.embWeightLoader cycleCounter dramSlaveIn
      effectiveRowIndex rowReqPulse (pure True) (RowComputeUnit.rcRowDone compute) params

  weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
  weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw

  -- rmsFinal fetch has priority (finishes before logits mat-mul starts)
  axiMasterOut :: Master.AxiMasterOut dom
  axiMasterOut = Master.axiMasterMux rmsFinalBusy rmsFinalAxiMaster logitsAxiMaster

  currentRowDram = LOADER.assertRowStable weightValid (LOADER.dramRowOut weightLoaderOut)
  currentRowHC   = LOADER.assertRowStable weightValid (LOADER.hcRowOut   weightLoaderOut)

  -------------------------------------------------------------------------
  -- Prevent restart immediately after consume
  -------------------------------------------------------------------------
  justConsumed :: Signal dom Bool
  justConsumed = register False consumeSignal

  effectiveInputValid = inputValidLatched
    .&&. (not <$> OutputTransactionController.otcOutputValid outputTxn)
    .&&. (not <$> justConsumed)

  -------------------------------------------------------------------------
  -- Row Compute Unit
  -- numRows = VocabularySize, numCols = ModelDimension
  -------------------------------------------------------------------------
  compute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = effectiveInputValid
      , rcWeightValid     = weightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = rowIndex
      , rcWeightHC        = currentRowHC
      , rcWeightDram      = currentRowDram
      , rcColumn          = tokenWithRms
      }

  rowDone = traceEdgeC cycleCounter (tag P.++ "rowDone") $ RowComputeUnit.rcRowDone compute

  -------------------------------------------------------------------------
  -- Row Result Checker (DRAM vs HC per row)
  -------------------------------------------------------------------------
  dramRowResultChecked = rowResultChecker
    (RowComputeUnit.rcRowDone compute) rowIndex
    (RowComputeUnit.rcResult   compute)
    (RowComputeUnit.rcResultHC compute)

  -------------------------------------------------------------------------
  -- Output Accumulator (accumulates VocabularySize results)
  -------------------------------------------------------------------------
  outputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx headIdx
    OutputAccumulator.OutputAccumIn
      { oaRowDone     = RowComputeUnit.rcRowDone compute
      , oaRowIndex    = rowIndex
      , oaRowResult   = dramRowResultChecked
      , oaRowResultHC = RowComputeUnit.rcResultHC compute
      }

  outputValid = OutputTransactionController.otcOutputValid outputTxn

  logitsOutFinal = logitsOutputChecker outputValid
                     (OutputAccumulator.oaOutput   outputAccum)
                     (OutputAccumulator.oaOutputHC outputAccum)
