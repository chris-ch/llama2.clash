module LLaMa2.Layer.Attention.KeyValueHeadProjector
  ( keyValueHeadProjector
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumLayers, NumQueryHeads
    , NumKeyValueHeads, RotaryPositionalEmbeddingDimension )
import LLaMa2.Numeric.Types (FixedPoint)

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom                          -- K DRAM slave
  -> Slave.AxiSlaveIn dom                          -- V DRAM slave
  -> Signal dom (Index NumLayers)
  -> Index NumKeyValueHeads
  -> Signal dom Bool                               -- inputValid
  -> Signal dom Bool                               -- downStreamReady
  -> Signal dom Bool                               -- consumeSignal (coordinated)
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint) -- cosVec
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint) -- sinVec
  -> Signal dom (Vec ModelDimension FixedPoint)                     -- xHat
  -> ( Master.AxiMasterOut dom                                      -- K AXI master
     , Master.AxiMasterOut dom                                      -- V AXI master
     , Signal dom (Maybe (Index HeadDimension, FixedPoint))         -- K writes (with rotary)
     , Signal dom (Maybe (Index HeadDimension, FixedPoint))         -- V writes (raw)
     , Signal dom Bool                                              -- outputValid
     , Signal dom Bool                                              -- readyForInput
     )
keyValueHeadProjector cycleCounter kDramSlaveIn vDramSlaveIn layerIdx kvHeadIdx
  inputValid downStreamReady consumeSignal cosVec sinVec xHat =
  (kAxiMaster, vAxiMaster, kWrite, vWrite, outputValid, readyForInput)
 where
  qTag :: Index NumQueryHeads
  qTag = fromIntegral kvHeadIdx

  justConsumed :: Signal dom Bool
  justConsumed = register False consumeSignal

  -------------------------------------------------------------------------
  -- K PATH
  -------------------------------------------------------------------------
  kRowIndex :: Signal dom (Index HeadDimension)
  kRowIndex = register 0 kNextRowIndex

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
    cycleCounter qTag
    InputTransactionController.InputTransactionIn
      { itcInputValid      = inputValid
      , itcOutputValid     = OutputTransactionController.otcOutputValid kOutputTxn
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  kInputValidLatched = InputTransactionController.itcLatchedValid kInputTxn

  kOutputTxn = OutputTransactionController.outputTransactionController
    cycleCounter qTag
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
  kRowReqPulse      = kRowReqRise .||. (kRowReqValidGated .&&. kRowIndexChanged)

  (kAxiMaster, kWeightLoaderOut, kWeightValidRaw, kWeightReadyRaw) =
    LOADER.kWeightLoader cycleCounter kDramSlaveIn layerIdx kvHeadIdx
      kEffectiveRowIndex kRowReqPulse (pure True) (RowComputeUnit.rcRowDone kCompute)

  kWeightValid = kWeightValidRaw
  kWeightReady = kWeightReadyRaw

  kCurrentRowDram = LOADER.dramRowOut kWeightLoaderOut

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

  kRowDone   = RowComputeUnit.rcRowDone kCompute
  kRowResult = RowComputeUnit.rcResult kCompute

  -- -----------------------------------------------------------------------
  -- K: pair-wise rotary encoding → streaming BRAM writes
  -- Mirrors queryHeadCore in QueryHeadProjector.hs.
  -- Row 2i (even): latch result; Row 2i+1 (odd): compute rotary pair and emit.
  -- -----------------------------------------------------------------------
  kIsOddRow :: Signal dom Bool
  kIsOddRow = odd . fromEnum <$> kRowIndex

  kEvenLatched :: Signal dom FixedPoint
  kEvenLatched = regEn 0 (kRowDone .&&. (not <$> kIsOddRow)) kRowResult

  kPairIdx :: Signal dom (Index RotaryPositionalEmbeddingDimension)
  kPairIdx = (\r -> toEnum (fromEnum r `div` 2)) <$> kRowIndex

  kCosVal :: Signal dom FixedPoint
  kCosVal = (!!) <$> cosVec <*> kPairIdx

  kSinVal :: Signal dom FixedPoint
  kSinVal = (!!) <$> sinVec <*> kPairIdx

  -- k'[2i]   = k[2i]*cos - k[2i+1]*sin
  kEvenRotated :: Signal dom FixedPoint
  kEvenRotated = liftA2 (-) (liftA2 (*) kEvenLatched kCosVal)
                             (liftA2 (*) kRowResult   kSinVal)

  -- k'[2i+1] = k[2i]*sin + k[2i+1]*cos
  kOddRotated :: Signal dom FixedPoint
  kOddRotated  = liftA2 (+) (liftA2 (*) kEvenLatched kSinVal)
                             (liftA2 (*) kRowResult   kCosVal)

  -- On odd rowDone: write k'[2i] immediately (even address = kRowIndex - 1)
  kWriteImm :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
  kWriteImm = mux (kRowDone .&&. kIsOddRow)
    ((curry Just . subtract 1 <$> kRowIndex) <*> kEvenRotated)
    (pure Nothing)

  -- One cycle later: write k'[2i+1] (odd address = kRowIndex)
  kWritePending :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
  kWritePending = register Nothing $
    mux (kRowDone .&&. kIsOddRow)
      (curry Just <$> kRowIndex <*> kOddRotated)
      (pure Nothing)

  kWrite :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
  kWrite = mux (kRowDone .&&. kIsOddRow) kWriteImm kWritePending

  -------------------------------------------------------------------------
  -- V PATH
  -------------------------------------------------------------------------
  vRowIndex :: Signal dom (Index HeadDimension)
  vRowIndex = register 0 vNextRowIndex

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
    cycleCounter qTag
    InputTransactionController.InputTransactionIn
      { itcInputValid      = inputValid
      , itcOutputValid     = OutputTransactionController.otcOutputValid vOutputTxn
      , itcDownStreamReady = downStreamReady
      , itcConsumeSignal   = consumeSignal
      }

  vInputValidLatched = InputTransactionController.itcLatchedValid vInputTxn

  vOutputTxn = OutputTransactionController.outputTransactionController
    cycleCounter qTag
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
  vRowReqPulse      = vRowReqRise .||. (vRowReqValidGated .&&. vRowIndexChanged)

  (vAxiMaster, vWeightLoaderOut, vWeightValidRaw, vWeightReadyRaw) =
    LOADER.vWeightLoader cycleCounter vDramSlaveIn layerIdx kvHeadIdx
      vEffectiveRowIndex vRowReqPulse (pure True) (RowComputeUnit.rcRowDone vCompute)

  vWeightValid = vWeightValidRaw
  vWeightReady = vWeightReadyRaw

  vCurrentRowDram = LOADER.dramRowOut vWeightLoaderOut

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

  vRowDone = RowComputeUnit.rcRowDone vCompute

  -- V: no rotary encoding — stream each row result directly as it completes.
  vWrite :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
  vWrite = mux vRowDone
    (curry Just <$> vRowIndex <*> RowComputeUnit.rcResult vCompute)
    (pure Nothing)

  -------------------------------------------------------------------------
  -- COMBINED OUTPUTS
  -------------------------------------------------------------------------
  kOutputValid  = OutputTransactionController.otcOutputValid kOutputTxn
  vOutputValid  = OutputTransactionController.otcOutputValid vOutputTxn
  outputValid   = kOutputValid .&&. vOutputValid
  readyForInput = kReadyForInput .&&. vReadyForInput
