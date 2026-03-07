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
  (axiMasterOut, logitsOut, outputValid)
 where
  emb = PARAM.modelEmbedding params

  inputValidRise :: Signal dom Bool
  inputValidRise = inputValid .&&. (not <$> register False inputValid)

  (rmsFinalAxiMaster, rmsFinalVec, rmsFinalValid, rmsFinalBusy) =
    FPVec.fpVecLoader cycleCounter dramSlaveIn
      inputValidRise
      (pure Layout.rmsFinalAddress)
      (PARAM.rmsFinalWeightF emb)
      "[RmsFinal] "

  pendingInput :: Signal dom Bool
  pendingInput = register False nextPendingInput
   where
    nextPendingInput =
      mux (pendingInput .&&. rmsFinalValid) (pure False) $
      mux inputValidRise (pure True)
      pendingInput

  effectiveInputOuter :: Signal dom Bool
  effectiveInputOuter = pendingInput .&&. rmsFinalValid

  tokenWithRms :: Signal dom (Vec ModelDimension FixedPoint)
  tokenWithRms = rmsNormFwFix <$> tokenVecSig <*> rmsFinalVec

  tag :: String
  tag = "[LOGITS] "

  layerIdx :: Index NumLayers
  layerIdx = 0

  headIdx :: Index NumQueryHeads
  headIdx = 0

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

  effectiveRowIndex :: Signal dom (Index VocabularySize)
  effectiveRowIndex = mux
    (RowScheduler.rsOutputValid rsIn .&&. RowScheduler.rsConsumeSignal rsIn)
    (pure 0)
    rowIndex

  loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
  rowReqValidGated = RowComputeUnit.rcFetchReq compute .&&. weightReady
  prevRowReqValid  = register False $ mux loaderBecameIdle (pure False) rowReqValidGated
  rowReqRise       = rowReqValidGated .&&. (not <$> prevRowReqValid)
  prevRowIndex     = register 0 effectiveRowIndex
  rowIndexChanged  = effectiveRowIndex ./=. prevRowIndex
  rowReqPulse      = traceEdgeC cycleCounter (tag P.++ "reqPulse") $
                       rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

  (logitsAxiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
    LOADER.embWeightLoader cycleCounter dramSlaveIn
      effectiveRowIndex rowReqPulse (pure True) (RowComputeUnit.rcRowDone compute) params

  weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
  weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw

  axiMasterOut :: Master.AxiMasterOut dom
  axiMasterOut = Master.axiMasterMux rmsFinalBusy rmsFinalAxiMaster logitsAxiMaster

  currentRowDram = LOADER.assertRowStable weightValid (LOADER.dramRowOut weightLoaderOut)

  justConsumed :: Signal dom Bool
  justConsumed = register False consumeSignal

  effectiveInputValid = inputValidLatched
    .&&. (not <$> OutputTransactionController.otcOutputValid outputTxn)
    .&&. (not <$> justConsumed)

  compute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = effectiveInputValid
      , rcWeightValid     = weightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = rowIndex
      , rcWeightDram      = currentRowDram
      , rcColumn          = tokenWithRms
      }

  rowDone = traceEdgeC cycleCounter (tag P.++ "rowDone") $ RowComputeUnit.rcRowDone compute

  outputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx headIdx
    OutputAccumulator.OutputAccumIn
      { oaRowDone   = RowComputeUnit.rcRowDone compute
      , oaRowIndex  = rowIndex
      , oaRowResult = RowComputeUnit.rcResult compute
      }

  outputValid = OutputTransactionController.otcOutputValid outputTxn
  logitsOut   = OutputAccumulator.oaOutput outputAccum
