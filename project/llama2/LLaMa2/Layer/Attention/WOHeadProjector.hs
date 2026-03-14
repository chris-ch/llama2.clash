module LLaMa2.Layer.Attention.WOHeadProjector
  ( woHeadProjector
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumLayers, NumQueryHeads )
import LLaMa2.Numeric.Types (FixedPoint)

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator as OutputAccumulator
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

woHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal (coordinated)
  -> Signal dom (Vec HeadDimension FixedPoint)    -- per-head attention output
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec ModelDimension FixedPoint) -- WO projected output
     , Signal dom Bool                            -- outputValid
     , Signal dom Bool                            -- readyForInput
     )
woHeadProjector cycleCounter dramSlaveIn layerIdx headIdx
  inputValid downStreamReady consumeSignal headVec =
  (axiMaster, woOut, outputValid, readyForInput)
 where
  rowIndex :: Signal dom (Index ModelDimension)
  rowIndex = register 0 nextRowIndex

  rsIn :: RowScheduler.RowSchedulerIn dom ModelDimension
  rsIn = RowScheduler.RowSchedulerIn
    { rsRowDone       = rowDone
    , rsOutputValid   = OutputTransactionController.otcOutputValid outputTxn
    , rsConsumeSignal = consumeSignal
    , rsCurrentIndex  = rowIndex
    }

  rowSched :: RowScheduler.RowSchedulerOut dom ModelDimension
  rowSched     = RowScheduler.rowScheduler rsIn
  nextRowIndex = RowScheduler.rsNextRowIndex rowSched

  inputTxn = InputTransactionController.inputTransactionController
    cycleCounter headIdx
    InputTransactionController.InputTransactionIn
      { itcInputValid      = inputValid
      , itcOutputValid     = OutputTransactionController.otcOutputValid outputTxn
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  inputValidLatched = InputTransactionController.itcLatchedValid inputTxn

  outputTxn = OutputTransactionController.outputTransactionController
    cycleCounter headIdx
    OutputTransactionController.OutputTransactionIn
      { otcAllDone       = RowComputeUnit.rcAllDone compute
      , otcConsumeSignal = consumeSignal
      }

  effectiveRowIndex :: Signal dom (Index ModelDimension)
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
  rowReqPulse      = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

  (axiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
    LOADER.woWeightLoader cycleCounter dramSlaveIn layerIdx headIdx
      effectiveRowIndex rowReqPulse (pure True) (RowComputeUnit.rcRowDone compute)

  weightValid = weightValidRaw
  weightReady = weightReadyRaw

  currentRowDram = LOADER.dramRowOut weightLoaderOut

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
      , rcColumn          = headVec
      }

  readyForInput = RowComputeUnit.rcIdleReady compute .&&. weightReady

  rowDone = RowComputeUnit.rcRowDone compute

  outputAccum = OutputAccumulator.outputAccumulator cycleCounter headIdx
    OutputAccumulator.OutputAccumIn
      { oaRowDone   = RowComputeUnit.rcRowDone compute
      , oaRowIndex  = rowIndex
      , oaRowResult = RowComputeUnit.rcResult compute
      }

  outputValid = OutputTransactionController.otcOutputValid outputTxn
  woOut       = OutputAccumulator.oaOutput outputAccum
