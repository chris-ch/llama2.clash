module LLaMa2.Layer.Attention.WOHeadProjector
  ( woHeadProjector
  ) where

import Clash.Prelude

import qualified Prelude as P

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumLayers, NumQueryHeads )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
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
-- Generic in numRows so it works for WO (ModelDimension rows).
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
      | otherwise  = P.error $ "WO row result mismatch at token " P.++ show tok
                    P.++ " row " P.++ show ri P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr

--------------------------------------------------------------------------------
-- Output Checker
-- Compares final WO output vectors between DRAM and HC paths.
-- Rising-edge capture avoids stale-data bug on last head in round-robin.
--------------------------------------------------------------------------------
woOutputChecker :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
woOutputChecker outputValid dramOut hcOut = result
  where
    risingEdge   = outputValid .&&. (not <$> register False outputValid)

    dramSampled = register (repeat 0) (mux risingEdge dramOut dramSampled)
    hcSampled   = register (repeat 0) (mux risingEdge hcOut   hcSampled)

    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 (mux risingEdge (tokenCnt + 1) tokenCnt)

    checkTrigger = register False risingEdge

    result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

    checkPure tok dr hr =
      let pairs      = P.zip [0..] (P.zip (toList dr) (toList hr))
          mismatches = P.filter (\(_, (d, h)) -> d P./= h) pairs
      in if P.null mismatches then dr
         else let (i, (d, h)) = P.head mismatches
              in P.error $ "WOHead output mismatch at token " P.++ show tok
                        P.++ ": index " P.++ show (i :: Int)
                        P.++ " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")"
                        P.++ " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

--------------------------------------------------------------------------------
-- woHeadProjector
--
-- One WO output-projection head. Multiplies the per-head attention output
-- (Vec HeadDimension FixedPoint) by the WO weight matrix
-- (MatI8E ModelDimension HeadDimension) to produce a
-- (Vec ModelDimension FixedPoint) partial sum.
--
-- numRows = ModelDimension (64 rows to schedule)
-- numCols = HeadDimension  (8 columns per row)
--------------------------------------------------------------------------------
woHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal (coordinated)
  -> Signal dom (Vec HeadDimension FixedPoint)    -- per-head attention output
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec ModelDimension FixedPoint) -- WO projected output
     , Signal dom Bool                            -- outputValid
     , Signal dom Bool                            -- readyForInput
     )
woHeadProjector cycleCounter dramSlaveIn layerIdx headIdx
  inputValid downStreamReady consumeSignal headVec params =
  (axiMaster, woOutFinal, outputValid, readyForInput)
 where
  tag :: String
  tag = "[WOHP L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

  -------------------------------------------------------------------------
  -- Row Index Register (ModelDimension rows for WO)
  -------------------------------------------------------------------------
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
  rowSched      = RowScheduler.rowScheduler rsIn
  nextRowIndex  = RowScheduler.rsNextRowIndex rowSched

  -------------------------------------------------------------------------
  -- Input / Output Transaction Controllers
  -------------------------------------------------------------------------
  inputTxn = InputTransactionController.inputTransactionController
    cycleCounter layerIdx headIdx
    InputTransactionController.InputTransactionIn
      { itcInputValid      = inputValid
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
  effectiveRowIndex :: Signal dom (Index ModelDimension)
  effectiveRowIndex = mux
    (RowScheduler.rsOutputValid rsIn .&&. RowScheduler.rsConsumeSignal rsIn)
    (pure 0)
    rowIndex

  -------------------------------------------------------------------------
  -- Row request pulse logic (mirrors KeyValueHeadProjector pattern)
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
  -- WO Weight Loader (DRAM-backed)
  -------------------------------------------------------------------------
  (axiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
    LOADER.woWeightLoader cycleCounter dramSlaveIn layerIdx headIdx
      effectiveRowIndex rowReqPulse (pure True) (RowComputeUnit.rcRowDone compute) params

  weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
  weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw

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
  -- numRows = ModelDimension, numCols = HeadDimension
  -------------------------------------------------------------------------
  compute = RowComputeUnit.rowComputeUnit cycleCounter
    RowComputeUnit.RowComputeIn
      { rcInputValid      = effectiveInputValid
      , rcWeightValid     = weightValid
      , rcDownStreamReady = downStreamReady
      , rcRowIndex        = rowIndex
      , rcWeightHC        = currentRowHC
      , rcWeightDram      = currentRowDram
      , rcColumn          = headVec
      }

  readyForInput = RowComputeUnit.rcIdleReady compute .&&. weightReady

  rowDone = traceEdgeC cycleCounter (tag P.++ "rowDone") $ RowComputeUnit.rcRowDone compute

  -------------------------------------------------------------------------
  -- Row Result Checker (DRAM vs HC per row)
  -------------------------------------------------------------------------
  dramRowResultChecked = rowResultChecker
    (RowComputeUnit.rcRowDone compute) rowIndex
    (RowComputeUnit.rcResult   compute)
    (RowComputeUnit.rcResultHC compute)

  -------------------------------------------------------------------------
  -- Output Accumulator (accumulates ModelDimension results)
  -------------------------------------------------------------------------
  outputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx headIdx
    OutputAccumulator.OutputAccumIn
      { oaRowDone     = RowComputeUnit.rcRowDone compute
      , oaRowIndex    = rowIndex
      , oaRowResult   = dramRowResultChecked
      , oaRowResultHC = RowComputeUnit.rcResultHC compute
      }

  outputValid = OutputTransactionController.otcOutputValid outputTxn

  woOutFinal = woOutputChecker outputValid
                 (OutputAccumulator.oaOutput   outputAccum)
                 (OutputAccumulator.oaOutputHC outputAccum)
