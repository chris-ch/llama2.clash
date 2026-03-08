module LLaMa2.Layer.Attention.KeyValueHeadProjector
  ( keyValueHeadProjector
  ) where

import Clash.Prelude

import qualified Prelude as P

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumLayers, NumQueryHeads
    , NumKeyValueHeads, RotaryPositionalEmbeddingDimension )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryPositionEncoder)

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator as OutputAccumulator
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

import TraceUtils (traceChangeC, traceEdgeC)

keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom                          -- K DRAM slave
  -> Slave.AxiSlaveIn dom                          -- V DRAM slave
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Bool                               -- inputValid
  -> Signal dom Bool                               -- downStreamReady
  -> Signal dom Bool                               -- consumeSignal (coordinated)
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint) -- cosVec
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint) -- sinVec
  -> Signal dom (Vec ModelDimension FixedPoint)                     -- xHat
  -> ( Master.AxiMasterOut dom                     -- K AXI master
     , Master.AxiMasterOut dom                     -- V AXI master
     , Signal dom (Vec HeadDimension FixedPoint)   -- K output (with rotary)
     , Signal dom (Vec HeadDimension FixedPoint)   -- V output
     , Signal dom Bool                             -- outputValid
     , Signal dom Bool                             -- readyForInput
     )
keyValueHeadProjector cycleCounter kDramSlaveIn vDramSlaveIn layerIdx kvHeadIdx
  inputValid downStreamReady consumeSignal cosVec sinVec xHat =
  (kAxiMaster, vAxiMaster, kRoOut, vOut, outputValid, readyForInput)
 where
  qTag :: Index NumQueryHeads
  qTag = fromIntegral kvHeadIdx

  justConsumed :: Signal dom Bool
  justConsumed = register False consumeSignal

  -------------------------------------------------------------------------
  -- K PATH
  -------------------------------------------------------------------------
  kTag :: String
  kTag = "[KHP L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

  kRowIndex :: Signal dom (Index HeadDimension)
  kRowIndex = traceChangeC cycleCounter (kTag P.++ "rowIndex") $ register 0 kNextRowIndex

  kRsIn :: RowScheduler.RowSchedulerIn dom HeadDimension
  kRsIn = RowScheduler.RowSchedulerIn
    { rsRowDone       = kRowDone
    , rsOutputValid   = OutputTransactionController.otcOutputValid kOutputTxn
    , rsConsumeSignal = consumeSignal
    , rsCurrentIndex  = kRowIndex
    }

  kRowSched     = RowScheduler.rowScheduler kRsIn
  kNextRowIndex = RowScheduler.rsNextRowIndex kRowSched

  kInputTxn = InputTransactionController.inputTransactionController
    cycleCounter layerIdx qTag
    InputTransactionController.InputTransactionIn
      { itcInputValid      = inputValid
      , itcOutputValid     = OutputTransactionController.otcOutputValid kOutputTxn
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  kInputValidLatched = InputTransactionController.itcLatchedValid kInputTxn

  kOutputTxn = OutputTransactionController.outputTransactionController
    cycleCounter layerIdx qTag
    OutputTransactionController.OutputTransactionIn
      { otcAllDone       = RowComputeUnit.rcAllDone kCompute
      , otcConsumeSignal = consumeSignal
      }

  kEffectiveRowIndex :: Signal dom (Index HeadDimension)
  kEffectiveRowIndex = mux
    (RowScheduler.rsOutputValid kRsIn .&&. RowScheduler.rsConsumeSignal kRsIn)
    (pure 0)
    kRowIndex

  kLoaderBecameIdle = kWeightReady .&&. (not <$> register False kWeightReady)
  kRowReqValidGated = RowComputeUnit.rcFetchReq kCompute .&&. kWeightReady
  kPrevRowReqValid  = register False $ mux kLoaderBecameIdle (pure False) kRowReqValidGated
  kRowReqRise       = kRowReqValidGated .&&. (not <$> kPrevRowReqValid)
  kPrevRowIndex     = register 0 kEffectiveRowIndex
  kRowIndexChanged  = kEffectiveRowIndex ./=. kPrevRowIndex
  kRowReqPulse      = traceEdgeC cycleCounter (kTag P.++ "reqPulse") $
                        kRowReqRise .||. (kRowReqValidGated .&&. kRowIndexChanged)

  (kAxiMaster, kWeightLoaderOut, kWeightValidRaw, kWeightReadyRaw) =
    LOADER.kWeightLoader cycleCounter kDramSlaveIn layerIdx kvHeadIdx
      kEffectiveRowIndex kRowReqPulse (pure True) (RowComputeUnit.rcRowDone kCompute)

  kWeightValid = traceEdgeC cycleCounter (kTag P.++ "weightValid") kWeightValidRaw
  kWeightReady = traceEdgeC cycleCounter (kTag P.++ "weightReady") kWeightReadyRaw

  kCurrentRowDram = LOADER.assertRowStable kWeightValid (LOADER.dramRowOut kWeightLoaderOut)

  kEffectiveInputValid = kInputValidLatched
    .&&. (not <$> OutputTransactionController.otcOutputValid kOutputTxn)
    .&&. (not <$> justConsumed)

  kCompute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = kEffectiveInputValid
      , rcWeightValid     = kWeightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = kRowIndex
      , rcWeightDram      = kCurrentRowDram
      , rcColumn          = xHat
      }

  kReadyForInput = RowComputeUnit.rcIdleReady kCompute .&&. kWeightReady

  kRowDone = traceEdgeC cycleCounter (kTag P.++ "rowDone") $ RowComputeUnit.rcRowDone kCompute

  kOutputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx qTag
    OutputAccumulator.OutputAccumIn
      { oaRowDone   = RowComputeUnit.rcRowDone kCompute
      , oaRowIndex  = kRowIndex
      , oaRowResult = RowComputeUnit.rcResult kCompute
      }

  kOutputValid = OutputTransactionController.otcOutputValid kOutputTxn
  kRoOut = rotaryPositionEncoder <$> OutputAccumulator.oaOutput kOutputAccum <*> cosVec <*> sinVec

  -------------------------------------------------------------------------
  -- V PATH
  -------------------------------------------------------------------------
  vTag :: String
  vTag = "[VHP L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

  vRowIndex :: Signal dom (Index HeadDimension)
  vRowIndex = traceChangeC cycleCounter (vTag P.++ "rowIndex") $ register 0 vNextRowIndex

  vRsIn :: RowScheduler.RowSchedulerIn dom HeadDimension
  vRsIn = RowScheduler.RowSchedulerIn
    { rsRowDone       = vRowDone
    , rsOutputValid   = OutputTransactionController.otcOutputValid vOutputTxn
    , rsConsumeSignal = consumeSignal
    , rsCurrentIndex  = vRowIndex
    }

  vRowSched     = RowScheduler.rowScheduler vRsIn
  vNextRowIndex = RowScheduler.rsNextRowIndex vRowSched

  vInputTxn = InputTransactionController.inputTransactionController
    cycleCounter layerIdx qTag
    InputTransactionController.InputTransactionIn
      { itcInputValid      = inputValid
      , itcOutputValid     = OutputTransactionController.otcOutputValid vOutputTxn
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  vInputValidLatched = InputTransactionController.itcLatchedValid vInputTxn

  vOutputTxn = OutputTransactionController.outputTransactionController
    cycleCounter layerIdx qTag
    OutputTransactionController.OutputTransactionIn
      { otcAllDone       = RowComputeUnit.rcAllDone vCompute
      , otcConsumeSignal = consumeSignal
      }

  vEffectiveRowIndex :: Signal dom (Index HeadDimension)
  vEffectiveRowIndex = mux
    (RowScheduler.rsOutputValid vRsIn .&&. RowScheduler.rsConsumeSignal vRsIn)
    (pure 0)
    vRowIndex

  vLoaderBecameIdle = vWeightReady .&&. (not <$> register False vWeightReady)
  vRowReqValidGated = RowComputeUnit.rcFetchReq vCompute .&&. vWeightReady
  vPrevRowReqValid  = register False $ mux vLoaderBecameIdle (pure False) vRowReqValidGated
  vRowReqRise       = vRowReqValidGated .&&. (not <$> vPrevRowReqValid)
  vPrevRowIndex     = register 0 vEffectiveRowIndex
  vRowIndexChanged  = vEffectiveRowIndex ./=. vPrevRowIndex
  vRowReqPulse      = traceEdgeC cycleCounter (vTag P.++ "reqPulse") $
                        vRowReqRise .||. (vRowReqValidGated .&&. vRowIndexChanged)

  (vAxiMaster, vWeightLoaderOut, vWeightValidRaw, vWeightReadyRaw) =
    LOADER.vWeightLoader cycleCounter vDramSlaveIn layerIdx kvHeadIdx
      vEffectiveRowIndex vRowReqPulse (pure True) (RowComputeUnit.rcRowDone vCompute)

  vWeightValid = traceEdgeC cycleCounter (vTag P.++ "weightValid") vWeightValidRaw
  vWeightReady = traceEdgeC cycleCounter (vTag P.++ "weightReady") vWeightReadyRaw

  vCurrentRowDram = LOADER.assertRowStable vWeightValid (LOADER.dramRowOut vWeightLoaderOut)

  vEffectiveInputValid = vInputValidLatched
    .&&. (not <$> OutputTransactionController.otcOutputValid vOutputTxn)
    .&&. (not <$> justConsumed)

  vCompute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = vEffectiveInputValid
      , rcWeightValid     = vWeightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = vRowIndex
      , rcWeightDram      = vCurrentRowDram
      , rcColumn          = xHat
      }

  vReadyForInput = RowComputeUnit.rcIdleReady vCompute .&&. vWeightReady

  vRowDone = traceEdgeC cycleCounter (vTag P.++ "rowDone") $ RowComputeUnit.rcRowDone vCompute

  vOutputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx qTag
    OutputAccumulator.OutputAccumIn
      { oaRowDone   = RowComputeUnit.rcRowDone vCompute
      , oaRowIndex  = vRowIndex
      , oaRowResult = RowComputeUnit.rcResult vCompute
      }

  vOutputValid = OutputTransactionController.otcOutputValid vOutputTxn
  vOut         = OutputAccumulator.oaOutput vOutputAccum

  -------------------------------------------------------------------------
  -- COMBINED OUTPUTS
  -------------------------------------------------------------------------
  outputValid   = kOutputValid .&&. vOutputValid
  readyForInput = kReadyForInput .&&. vReadyForInput
