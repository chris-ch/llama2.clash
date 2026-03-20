module LLaMa2.Embedding.OutputProjection
 ( logitsProjector, logitsProjectorTop
) where
import Clash.Prelude
import qualified GHC.TypeNats as TN

import LLaMa2.Numeric.RmsNormSeq (rmsNormSeq)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.ModelConfig (ModelDimension, VocabularySize, NumQueryHeads)
import LLaMa2.Types.LayerData (ActivationBramAddr)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified LLaMa2.Memory.FPVecLoader as FPVec
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

-- Base address of slot 3 (ffnOutput) in the activation BRAM.
slot3BramBase :: ActivationBramAddr
slot3BramBase = natToNum @(3 TN.* ModelDimension)

{-# NOINLINE logitsProjector #-}
logitsProjector :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                        -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                            -- ^ DRAM
  -> Signal dom Bool                                 -- ^ inputValid (lastLayerComplete)
  -> Signal dom Bool                                 -- ^ downStreamReady
  -> Signal dom Bool                                 -- ^ consumeSignal
  -> Signal dom FixedPoint                           -- ^ bramRdData (activation BRAM slot 3, 1-cycle latency)
  -> ( Master.AxiMasterOut dom
     , Signal dom ActivationBramAddr                 -- ^ bramRdAddr (pre-issue address for BRAM slot 3)
     , Signal dom (Index VocabularySize)             -- ^ logit index  (streaming, valid when logitValid)
     , Signal dom FixedPoint                         -- ^ logit value  (streaming, valid when logitValid)
     , Signal dom Bool                               -- ^ logitValid   (one pulse per completed row)
     , Signal dom Bool                               -- ^ logitsAllDone (all VocabularySize rows done)
     )
logitsProjector cycleCounter dramSlaveIn inputValid downStreamReady consumeSignal bramRdData =
  (axiMasterOut, bramRdAddr, rowIndex, RowComputeUnit.rcResult compute, rowDone, outputValid)
 where
  inputValidRise :: Signal dom Bool
  inputValidRise = inputValid .&&. (not <$> register False inputValid)

  (rmsFinalAxiMaster, rmsFinalVec, rmsFinalValid, rmsFinalBusy) =
    FPVec.fpVecLoader cycleCounter dramSlaveIn
      inputValidRise
      (pure Layout.rmsFinalAddress)

  -- Sequential rmsNorm: triggered on the rising edge of rmsFinalValid.
  -- xi comes from activation BRAM slot 3 (1-cycle latency via bramRdData).
  -- rdNext drives bramRdAddr one cycle ahead so the data arrives aligned.
  rmsFinalDone :: Signal dom Bool
  rmsFinalDone = rmsFinalValid .&&. (not <$> register False rmsFinalValid)

  (rmsNormValid, tokenWithRms, _, rdNext) = rmsNormSeq rmsFinalDone bramRdData rmsFinalVec

  -- Pre-issue activation BRAM read address one cycle ahead of when data is needed.
  bramRdAddr = (slot3BramBase +) . fromIntegral <$> rdNext

  -- effectiveInputOuter: fires once on rising edge of rmsNormValid while pending.
  effectiveInputOuter :: Signal dom Bool
  effectiveInputOuter = pendingInput .&&. rmsNormValid .&&. (not <$> register False rmsNormValid)

  pendingInput :: Signal dom Bool
  pendingInput = register False nextPendingInput
   where
    nextPendingInput =
      mux effectiveInputOuter (pure False) $
      mux inputValidRise (pure True)
      pendingInput

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
    cycleCounter headIdx
    InputTransactionController.InputTransactionIn
      { itcInputValid      = effectiveInputOuter
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
  rowReqPulse      = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

  (logitsAxiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
    LOADER.embWeightLoader cycleCounter dramSlaveIn
      effectiveRowIndex rowReqPulse (pure True) (RowComputeUnit.rcRowDone compute)

  weightValid = weightValidRaw
  weightReady = weightReadyRaw

  axiMasterOut :: Master.AxiMasterOut dom
  axiMasterOut = Master.axiMasterMux rmsFinalBusy rmsFinalAxiMaster logitsAxiMaster

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
      , rcColumn          = tokenWithRms
      }

  rowDone = RowComputeUnit.rcRowDone compute

  outputValid = OutputTransactionController.otcOutputValid outputTxn

{-# ANN logitsProjectorTop
  (Synthesize
    { t_name   = "logits_projector"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "cycle_counter"
        , PortProduct "dram" []
        , PortName "input_valid"
        , PortName "downstream_ready"
        , PortName "consume_signal"
        , PortName "bram_rd_data"
        ]
    , t_output = PortProduct ""
        [ PortProduct "axi_out" []
        , PortName "bram_rd_addr"
        , PortName "logit_idx"
        , PortName "logit_value"
        , PortName "logit_valid"
        , PortName "logits_all_done"
        ]
    }) #-}
logitsProjectorTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 32)
  -> Slave.AxiSlaveIn System
  -> Signal System Bool
  -> Signal System Bool
  -> Signal System Bool
  -> Signal System FixedPoint
  -> ( Master.AxiMasterOut System
     , Signal System ActivationBramAddr
     , Signal System (Index VocabularySize)
     , Signal System FixedPoint
     , Signal System Bool
     , Signal System Bool
     )
logitsProjectorTop = exposeClockResetEnable logitsProjector
