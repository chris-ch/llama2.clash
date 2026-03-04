module LLaMa2.Layer.Attention.KeyValueHeadProjector
  ( keyValueHeadProjector
  ) where

import Clash.Prelude

import qualified Prelude as P
import Clash.Debug (trace)

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumLayers, NumQueryHeads
    , NumKeyValueHeads, SequenceLength )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator as OutputAccumulator
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler

import TraceUtils (traceChangeC, traceEdgeC)

--------------------------------------------------------------------------------
-- KV Head Projector
--
-- Implements DRAM-backed weight loading for K and V, following the same FSM
-- pattern as QueryHeadProjector. K and V are completely independent compute
-- paths, each mirroring queryHeadCore. Rotary encoding is applied to K only.
--
-- 'consumeSignal' is provided by the parent (QKVProjection) as:
--   consumeSignal = outputValid .&&. downStreamReady
-- where outputValid is the AND of all head valids (Q, K, V), ensuring
-- coordinated clearing: no head restarts until ALL heads are done.
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- Row Result Checker
-- Asserts DRAM and HC row results match when rowDone fires
--------------------------------------------------------------------------------
rowResultChecker :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index HeadDimension)
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
      | otherwise  = P.error $ "KV Row result mismatch at token " P.++ show tok
                    P.++ " row " P.++ show ri P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr

--------------------------------------------------------------------------------
-- Output Checker
-- Compares final output vectors between DRAM and HC paths
--------------------------------------------------------------------------------
kvOutputChecker :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
kvOutputChecker outputValid dramOut hcOut = result
  where
    -- Capture on rising edge of outputValid (first cycle it fires).
    -- Using prevOutputValid.&&.everValid caused a one-cycle late capture:
    -- for the last head (V3, master 15 in round-robin), outputValid fires
    -- exactly at consumeSignal, so checkTrigger would fire at T_all+1 when
    -- qkvDone fires — but dramSampled was still stale (repeat 0 or previous
    -- token's values) because it hadn't been captured yet. That returned
    -- wrong values to the KV cache write.
    risingEdge   = outputValid .&&. (not <$> register False outputValid)

    -- Latch values on the FIRST cycle outputValid is True.
    dramSampled = register (repeat 0) (mux risingEdge dramOut dramSampled)
    hcSampled   = register (repeat 0) (mux risingEdge hcOut   hcSampled)

    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 (mux risingEdge (tokenCnt + 1) tokenCnt)

    -- checkTrigger fires ONE cycle after capture (when dramSampled is valid).
    checkTrigger = register False risingEdge

    result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

    checkPure tok dr hr =
      let pairs      = P.zip [0..] (P.zip (toList dr) (toList hr))
          mismatches = P.filter (\(_, (d, h)) -> d P./= h) pairs
      in if P.null mismatches then dr
         else let (i, (d, h)) = P.head mismatches
              in P.error $ "KVHead output mismatch at token " P.++ show tok
                        P.++ ": index " P.++ show (i :: Int)
                        P.++ " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")"
                        P.++ " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

--------------------------------------------------------------------------------
-- Weight Mismatch Checker (trace-only, mirrors WeightFetchUnit)
--------------------------------------------------------------------------------
weightMismatchChecker :: Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
weightMismatchChecker _cycleCounter layerIdx valid dram hc = result
  where
    result = check <$> valid <*> dram <*> hc
    check v d h
      | v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h) =
          trace ("[KV WFU L" P.++ show layerIdx P.++ "] WEIGHT_MISMATCH exp_d="
                 P.++ show (rowExponent d) P.++ " exp_h=" P.++ show (rowExponent h)) d
      | otherwise = d

--------------------------------------------------------------------------------
-- keyValueHeadProjector
--
-- K and V are completely independent DRAM-backed compute paths.
-- Each mirrors queryHeadCore from QueryHeadProjector.
--------------------------------------------------------------------------------
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
  -> Signal dom (Index SequenceLength)             -- stepCount
  -> Signal dom (Vec ModelDimension FixedPoint)    -- xHat
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom                     -- K AXI master
     , Master.AxiMasterOut dom                     -- V AXI master
     , Signal dom (Vec HeadDimension FixedPoint)   -- K output (with rotary)
     , Signal dom (Vec HeadDimension FixedPoint)   -- V output
     , Signal dom Bool                             -- outputValid
     , Signal dom Bool                             -- readyForInput
     )
keyValueHeadProjector cycleCounter kDramSlaveIn vDramSlaveIn layerIdx kvHeadIdx
  inputValid downStreamReady consumeSignal stepCount xHat params =
  (kAxiMaster, vAxiMaster, kRoOut, vOut, outputValid, readyForInput)
 where
  -- Cast KV head index to QueryHead index for sub-module trace tags only.
  -- Safe: NumKeyValueHeads = 4 <= NumQueryHeads = 8.
  qTag :: Index NumQueryHeads
  qTag = fromIntegral kvHeadIdx

  -- Prevent restart immediately after consume (shared by K and V)
  justConsumed :: Signal dom Bool
  justConsumed = register False consumeSignal

  -------------------------------------------------------------------------
  -- K PATH: mirrors queryHeadCore with kWeightLoader
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

  kRowSched :: RowScheduler.RowSchedulerOut dom HeadDimension
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

  -- Effective row index resets combinationally when output is consumed
  kEffectiveRowIndex :: Signal dom (Index HeadDimension)
  kEffectiveRowIndex = mux
    (RowScheduler.rsOutputValid kRsIn .&&. RowScheduler.rsConsumeSignal kRsIn)
    (pure 0)
    kRowIndex

  -- K row request pulse logic (mirrors WeightFetchUnit)
  kLoaderBecameIdle = kWeightReady .&&. (not <$> register False kWeightReady)
  kRowReqValidGated = RowComputeUnit.rcFetchReq kCompute .&&. kWeightReady
  kPrevRowReqValid  = register False $ mux kLoaderBecameIdle (pure False) kRowReqValidGated
  kRowReqRise       = kRowReqValidGated .&&. (not <$> kPrevRowReqValid)
  kPrevRowIndex     = register 0 kEffectiveRowIndex
  kRowIndexChanged  = kEffectiveRowIndex ./=. kPrevRowIndex
  kRowReqPulse      = traceEdgeC cycleCounter (kTag P.++ "reqPulse") $
                        kRowReqRise .||. (kRowReqValidGated .&&. kRowIndexChanged)

  -- K weight loader
  (kAxiMaster, kWeightLoaderOut, kWeightValidRaw, kWeightReadyRaw) =
    LOADER.kWeightLoader cycleCounter kDramSlaveIn layerIdx kvHeadIdx
      kEffectiveRowIndex kRowReqPulse (pure True) (RowComputeUnit.rcRowDone kCompute) params

  kWeightValid = traceEdgeC cycleCounter (kTag P.++ "weightValid") kWeightValidRaw
  kWeightReady = traceEdgeC cycleCounter (kTag P.++ "weightReady") kWeightReadyRaw

  kCurrentRowDram = LOADER.assertRowStable kWeightValid (LOADER.dramRowOut kWeightLoaderOut)
  kCurrentRowHC   = LOADER.assertRowStable kWeightValid (LOADER.hcRowOut   kWeightLoaderOut)
  kCurrentRowDramChecked = weightMismatchChecker cycleCounter layerIdx kWeightValid
                             kCurrentRowDram kCurrentRowHC

  kEffectiveInputValid = kInputValidLatched
    .&&. (not <$> OutputTransactionController.otcOutputValid kOutputTxn)
    .&&. (not <$> justConsumed)

  kCompute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = kEffectiveInputValid
      , rcWeightValid     = kWeightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = kRowIndex
      , rcWeightHC        = kCurrentRowHC
      , rcWeightDram      = kCurrentRowDramChecked
      , rcColumn          = xHat
      }

  kReadyForInput = RowComputeUnit.rcIdleReady kCompute .&&. kWeightReady

  kRowDone = traceEdgeC cycleCounter (kTag P.++ "rowDone") $ RowComputeUnit.rcRowDone kCompute

  kDramRowResultChecked = rowResultChecker
    (RowComputeUnit.rcRowDone kCompute) kRowIndex
    (RowComputeUnit.rcResult   kCompute)
    (RowComputeUnit.rcResultHC kCompute)

  kOutputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx qTag
    OutputAccumulator.OutputAccumIn
      { oaRowDone     = RowComputeUnit.rcRowDone kCompute
      , oaRowIndex    = kRowIndex
      , oaRowResult   = kDramRowResultChecked
      , oaRowResultHC = RowComputeUnit.rcResultHC kCompute
      }

  kOutputValid = OutputTransactionController.otcOutputValid kOutputTxn
  kOutFinal    = kvOutputChecker kOutputValid
                   (OutputAccumulator.oaOutput   kOutputAccum)
                   (OutputAccumulator.oaOutputHC kOutputAccum)

  -- Apply rotary encoding to K
  kRoOut = (rotaryEncoder (PARAM.rotaryEncoding params) <$> stepCount) <*> kOutFinal

  -------------------------------------------------------------------------
  -- V PATH: mirrors queryHeadCore with vWeightLoader (no rotary)
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

  vRowSched :: RowScheduler.RowSchedulerOut dom HeadDimension
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

  -- V row request pulse logic (mirrors WeightFetchUnit)
  vLoaderBecameIdle = vWeightReady .&&. (not <$> register False vWeightReady)
  vRowReqValidGated = RowComputeUnit.rcFetchReq vCompute .&&. vWeightReady
  vPrevRowReqValid  = register False $ mux vLoaderBecameIdle (pure False) vRowReqValidGated
  vRowReqRise       = vRowReqValidGated .&&. (not <$> vPrevRowReqValid)
  vPrevRowIndex     = register 0 vEffectiveRowIndex
  vRowIndexChanged  = vEffectiveRowIndex ./=. vPrevRowIndex
  vRowReqPulse      = traceEdgeC cycleCounter (vTag P.++ "reqPulse") $
                        vRowReqRise .||. (vRowReqValidGated .&&. vRowIndexChanged)

  -- V weight loader
  (vAxiMaster, vWeightLoaderOut, vWeightValidRaw, vWeightReadyRaw) =
    LOADER.vWeightLoader cycleCounter vDramSlaveIn layerIdx kvHeadIdx
      vEffectiveRowIndex vRowReqPulse (pure True) (RowComputeUnit.rcRowDone vCompute) params

  vWeightValid = traceEdgeC cycleCounter (vTag P.++ "weightValid") vWeightValidRaw
  vWeightReady = traceEdgeC cycleCounter (vTag P.++ "weightReady") vWeightReadyRaw

  vCurrentRowDram = LOADER.assertRowStable vWeightValid (LOADER.dramRowOut vWeightLoaderOut)
  vCurrentRowHC   = LOADER.assertRowStable vWeightValid (LOADER.hcRowOut   vWeightLoaderOut)
  vCurrentRowDramChecked = weightMismatchChecker cycleCounter layerIdx vWeightValid
                             vCurrentRowDram vCurrentRowHC

  vEffectiveInputValid = vInputValidLatched
    .&&. (not <$> OutputTransactionController.otcOutputValid vOutputTxn)
    .&&. (not <$> justConsumed)

  vCompute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = vEffectiveInputValid
      , rcWeightValid     = vWeightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = vRowIndex
      , rcWeightHC        = vCurrentRowHC
      , rcWeightDram      = vCurrentRowDramChecked
      , rcColumn          = xHat
      }

  vReadyForInput = RowComputeUnit.rcIdleReady vCompute .&&. vWeightReady

  vRowDone = traceEdgeC cycleCounter (vTag P.++ "rowDone") $ RowComputeUnit.rcRowDone vCompute

  vDramRowResultChecked = rowResultChecker
    (RowComputeUnit.rcRowDone vCompute) vRowIndex
    (RowComputeUnit.rcResult   vCompute)
    (RowComputeUnit.rcResultHC vCompute)

  vOutputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx qTag
    OutputAccumulator.OutputAccumIn
      { oaRowDone     = RowComputeUnit.rcRowDone vCompute
      , oaRowIndex    = vRowIndex
      , oaRowResult   = vDramRowResultChecked
      , oaRowResultHC = RowComputeUnit.rcResultHC vCompute
      }

  vOutputValid = OutputTransactionController.otcOutputValid vOutputTxn
  vOut         = kvOutputChecker vOutputValid
                   (OutputAccumulator.oaOutput   vOutputAccum)
                   (OutputAccumulator.oaOutputHC vOutputAccum)

  -------------------------------------------------------------------------
  -- COMBINED OUTPUTS
  -------------------------------------------------------------------------
  outputValid   = kOutputValid .&&. vOutputValid
  readyForInput = kReadyForInput .&&. vReadyForInput
