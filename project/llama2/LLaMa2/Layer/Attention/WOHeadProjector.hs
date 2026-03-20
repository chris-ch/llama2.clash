module LLaMa2.Layer.Attention.WOHeadProjector
  ( woHeadProjector
  ) where

import Clash.Prelude
import qualified GHC.TypeNats as TN

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumLayers, NumQueryHeads )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData (ActivationBramAddr)

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

-- Base address of slot 2 (attentionOutput) in the activation BRAM.
slot2BramBase :: ActivationBramAddr
slot2BramBase = natToNum @(2 TN.* ModelDimension)

woHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- ^ inputValid
  -> Signal dom Bool                              -- ^ downStreamReady
  -> Signal dom Bool                              -- ^ consumeSignal
  -> Signal dom FixedPoint                        -- ^ bramRdData (activation BRAM, 1-cycle latency)
  -> Signal dom ActivationBramAddr                -- ^ rdBase (slot0 for head 0, slot2 otherwise)
  -> Signal dom (Vec HeadDimension FixedPoint)    -- ^ per-head attention output
  -> ( Master.AxiMasterOut dom
     , Signal dom ActivationBramAddr              -- ^ bramRdAddr (to activation BRAM read port)
     , Signal dom (Maybe (ActivationBramAddr, FixedPoint))  -- ^ bramWrite
     , Signal dom Bool                            -- ^ outputValid (one-cycle pulse: last row written)
     , Signal dom Bool                            -- ^ readyForInput
     )
woHeadProjector cycleCounter dramSlaveIn layerIdx headIdx
  inputValid downStreamReady consumeSignal bramRdData rdBase headVec =
  (axiMaster, bramRdAddr, bramWrite, outputValid, readyForInput)
 where
  rowIndex :: Signal dom (Index ModelDimension)
  rowIndex = register 0 nextRowIndex

  -- outputDone: True from rcAllDone until consumeSignal resets it.
  -- Replaces OutputTransactionController.otcOutputValid.
  outputDone = register False $
    mux (RowComputeUnit.rcAllDone compute) (pure True) $
    mux consumeSignal                      (pure False)
    outputDone

  rsIn :: RowScheduler.RowSchedulerIn dom ModelDimension
  rsIn = RowScheduler.RowSchedulerIn
    { rsRowDone       = rowDone
    , rsOutputValid   = outputDone
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
      , itcOutputValid     = outputDone
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  inputValidLatched = InputTransactionController.itcLatchedValid inputTxn

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
    .&&. (not <$> outputDone)
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

  --------------------------------------------------------------------------
  -- Per-row BRAM write logic.
  -- Cycle T: rowDone fires for row I.
  --   bramRdAddr = rdBase + I  (issues activation BRAM read, 1-cycle latency)
  --   Latch rowResult and rowIndex.
  -- Cycle T+1: bramRdData = residual[I].
  --   bramWrite = Just (slot2[I], bramRdData + rowResultLatch)
  --------------------------------------------------------------------------
  rowDonePrev = register False rowDone

  rowResultLatch = regEn 0 rowDone (RowComputeUnit.rcResult compute)

  rowIndexLatch = regEn (0 :: Index ModelDimension) rowDone rowIndex

  -- Issue BRAM read on rowDone; idle otherwise.
  bramRdAddr = mux rowDone
    (liftA2 (+) rdBase (fromIntegral <$> rowIndex))
    (pure 0)

  -- Write one cycle after rowDone (BRAM read data now valid).
  bramWrite = mux rowDonePrev
    (Just <$> ((,)
        <$> ((slot2BramBase +) . fromIntegral <$> rowIndexLatch)
        <*> ((+) <$> bramRdData <*> rowResultLatch)))
    (pure Nothing)

  -- outputValid fires one cycle after rcAllDone, coinciding with the last BRAM write.
  outputValid = register False (RowComputeUnit.rcAllDone compute)
