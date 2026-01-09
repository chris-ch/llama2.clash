module LLaMa2.Layer.Attention.QueryHeadProjector
  ( queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import qualified Prelude as P

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator as OutputAccumulator
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.WeightFetchUnit as WeightFetchUnit

import TraceUtils (traceChangeC, traceEdgeC, traceChangeC)

--------------------------------------------------------------------------------
-- Debug Info Record
--------------------------------------------------------------------------------
data QHeadDebugInfo dom = QHeadDebugInfo
  { qhRowIndex        :: Signal dom (Index HeadDimension)
  , qhState           :: Signal dom OPS.MultiplierState
  , qhFirstMant       :: Signal dom Mantissa
  , qhRowResult       :: Signal dom FixedPoint
  , qhRowDone         :: Signal dom Bool
  , qhFetchValid      :: Signal dom Bool
  , qhFetchedWord     :: Signal dom (BitVector 512)
  , qhRowReset        :: Signal dom Bool
  , qhRowEnable       :: Signal dom Bool
  , qhAccumValue      :: Signal dom FixedPoint
  , qhQOut            :: Signal dom (Vec HeadDimension FixedPoint)
  , qhCurrentRowExp   :: Signal dom Exponent
  , qhCurrentRowMant0 :: Signal dom Mantissa
  , qhRowReqValid     :: Signal dom Bool
  , qhWeightReady     :: Signal dom Bool
  , qhWeightValid     :: Signal dom Bool
  } deriving (Generic)

--------------------------------------------------------------------------------
-- BLOCK: RowResultChecker
-- Asserts DRAM and HC row results match when rowDone fires
--------------------------------------------------------------------------------
rowResultChecker :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom FixedPoint
rowResultChecker rowDone rowIdx dramResult hcResult dramWeights hcWeights = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 nextTokenCnt
    nextTokenCnt = mux (rowDone .&&. (rowIdx .==. pure maxBound)) (tokenCnt + 1) tokenCnt

    result = mux rowDone
                 (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult <*> dramWeights <*> hcWeights)
                 dramResult

    check tok ri dr hr dramW hcW
      | dr P.== hr = dr
      | otherwise  = P.error $ "Row result mismatch at token " P.++ show tok
                    P.++ " row " P.++ show ri P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr
                    P.++ "\n  DRAM weight exp=" P.++ show (rowExponent dramW)
                    P.++ " mant[0]=" P.++ show (P.head (toList (rowMantissas dramW)))
                    P.++ "\n  HC weight exp=" P.++ show (rowExponent hcW)
                    P.++ " mant[0]=" P.++ show (P.head (toList (rowMantissas hcW)))
                    P.++ "\n  weights match=" P.++ show (rowExponent dramW P.== rowExponent hcW
                                                         P.&& rowMantissas dramW P.== rowMantissas hcW)

--------------------------------------------------------------------------------
-- BLOCK: QOutputChecker
-- Compares final Q output vectors (X-safe version)
--------------------------------------------------------------------------------
qOutputChecker :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
qOutputChecker outputValid dramOut hcOut = result
  where
    everValid = register False (everValid .||. outputValid)
    prevOutputValid = register False outputValid
    checkTrigger = prevOutputValid .&&. everValid

    dramSampled = register (repeat 0) (mux checkTrigger dramOut dramSampled)
    hcSampled   = register (repeat 0) (mux checkTrigger hcOut hcSampled)

    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 (mux checkTrigger (tokenCnt + 1) tokenCnt)

    result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

    checkPure tok dr hr =
      let pairs = P.zip [0..] (P.zip (toList dr) (toList hr))
          mismatches = P.filter (\(_, (d,h)) -> d P./= h) pairs
      in if P.null mismatches then dr
         else let (i, (d, h)) = P.head mismatches
              in P.error $ "QHead output mismatch at token " P.++ show tok
                        P.++ ": index " P.++ show (i :: Int)
                        P.++ " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")"
                        P.++ " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

--------------------------------------------------------------------------------
-- QueryHeadCore
--------------------------------------------------------------------------------
data QueryHeadCoreOut dom = QueryHeadCoreOut
  { qhcAxiMaster   :: Master.AxiMasterOut dom
  , qhcResult      :: Signal dom (Vec HeadDimension FixedPoint)
  , qhcOutputValid :: Signal dom Bool
  , qhcReady       :: Signal dom Bool
  , qhcDebug       :: QHeadDebugInfo dom
  }

queryHeadCore :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal
  -> Signal dom (Vec ModelDimension FixedPoint)   -- xHat
  -> PARAM.DecoderParameters
  -> QueryHeadCoreOut dom
queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat params =
  QueryHeadCoreOut
    { qhcAxiMaster   = WeightFetchUnit.wfAxiMaster weightFetch
    , qhcResult      = qOutFinal
    , qhcOutputValid = OutputTransactionController.otcOutputValid outputTxn
    , qhcReady       = readyForInput
    , qhcDebug       = debugInfo
    }
  where
    -- Trace tag for this instance
    tag = "[QHP L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

    ----------------------------------------------------------------------------
    -- Row Index Register
    ----------------------------------------------------------------------------
    rowIndex :: Signal dom (Index HeadDimension)
    rowIndex = traceChangeC cycleCounter (tag P.++ "rowIndex") $ register 0 nextRowIndex

    -- RowScheduler computes next index (combinatorial)
    rsIn = RowScheduler.RowSchedulerIn
                   { rsRowDone       = rowDone
                   , rsOutputValid   = OutputTransactionController.otcOutputValid outputTxn
                   , rsConsumeSignal = consumeSignal
                   , rsCurrentIndex  = rowIndex
                   }
    
    rowSched = RowScheduler.rowScheduler rsIn
                 
    nextRowIndex = RowScheduler.rsNextRowIndex rowSched

    inputTxn = InputTransactionController.inputTransactionController cycleCounter layerIdx headIdx rowIndex
                 InputTransactionController.InputTransactionIn
                   { itcInputValid      = inputValid
                   , itcOutputValid     = OutputTransactionController.otcOutputValid outputTxn
                   , itcDownStreamReady = downStreamReady
                   }

    inputValidLatched = InputTransactionController.itcLatchedValid inputTxn

    ----------------------------------------------------------------------------
    -- Output Valid Latch
    -- CLR has priority over SET (critical for correct handshake)
    -- Uses consumeSignal for coordinated multi-head clearing
    ----------------------------------------------------------------------------
    outputTxn = OutputTransactionController.outputTransactionController cycleCounter layerIdx headIdx rowIndex downStreamReady
                  OutputTransactionController.OutputTransactionIn
                    { otcAllDone       = RowComputeUnit.rcAllDone compute
                    , otcConsumeSignal = consumeSignal
                    }

    -- Compute effective row index that resets combinationally
    effectiveRowIndex :: Signal dom (Index HeadDimension)
    effectiveRowIndex = mux (RowScheduler.rsOutputValid rsIn .&&. RowScheduler.rsConsumeSignal rsIn) 
                            (pure 0) 
                            rowIndex

    ----------------------------------------------------------------------------
    -- Weight Loader
    ----------------------------------------------------------------------------
    weightFetch = WeightFetchUnit.weightFetchUnit cycleCounter dramSlaveIn layerIdx headIdx params
                    WeightFetchUnit.WeightFetchIn
                      { wfRowIndex      = effectiveRowIndex
                      , wfRowReqValid   = RowComputeUnit.rcFetchReq compute
                      , wfConsumeSignal = consumeSignal
                      , wfRowDone       = RowComputeUnit.rcRowDone compute
                      , wfInputValid    = inputValid
                      }

    currentRowDram = WeightFetchUnit.wfWeightDram weightFetch
    currentRowHC   = WeightFetchUnit.wfWeightHC weightFetch
    weightValid    = WeightFetchUnit.wfWeightValid weightFetch
    weightReady    = WeightFetchUnit.wfIdleReady weightFetch

    ----------------------------------------------------------------------------
    -- Row Multiplier (FSM + parallel processor)
    ----------------------------------------------------------------------------
    -- Don't allow compute to restart while outputValid is high
    effectiveInputValid = inputValidLatched .&&. 
                          (not <$> OutputTransactionController.otcOutputValid outputTxn)

    compute = RowComputeUnit.rowComputeUnit cycleCounter
            RowComputeUnit.RowComputeIn
              { rcInputValid      = effectiveInputValid  -- ← Changed from inputValidLatched
              , rcWeightValid     = weightValid
              , rcDownStreamReady = downStreamReady
              , rcRowIndex        = rowIndex
              , rcWeightHC        = currentRowHC
              , rcWeightDram      = currentRowDram
              , rcColumn          = xHat
              }
    
    readyForInput = RowComputeUnit.rcIdleReady compute .&&. weightReady

    ----------------------------------------------------------------------------
    -- Row Done - simple edge trace
    ----------------------------------------------------------------------------
    rowDone = traceEdgeC cycleCounter (tag P.++ "rowDone") $ RowComputeUnit.rcRowDone compute

    ----------------------------------------------------------------------------
    -- Row Result Checker
    -- Compares DRAM-computed vs HC-computed results for live verification
    ----------------------------------------------------------------------------
    dramRowResultChecked = rowResultChecker
      (RowComputeUnit.rcRowDone compute) rowIndex 
      (RowComputeUnit.rcResult compute)    -- DRAM result
      (RowComputeUnit.rcResultHC compute)  -- HC result
      currentRowHC currentRowHC

    ----------------------------------------------------------------------------
    -- Result Accumulator
    ----------------------------------------------------------------------------
    outputAccum = OutputAccumulator.outputAccumulator cycleCounter layerIdx headIdx
                    OutputAccumulator.OutputAccumIn
                      { oaRowDone     = RowComputeUnit.rcRowDone compute
                      , oaRowIndex    = rowIndex
                      , oaRowResult   = dramRowResultChecked   -- Checked DRAM result
                      , oaRowResultHC = RowComputeUnit.rcResultHC compute
                      }

    qOut   = OutputAccumulator.oaOutput outputAccum
    qOutHC = OutputAccumulator.oaOutputHC outputAccum

    qOutFinal = qOutputChecker (OutputTransactionController.otcOutputValid outputTxn) qOut qOutHC

    ----------------------------------------------------------------------------
    -- Debug Info
    ----------------------------------------------------------------------------
    debugInfo = QHeadDebugInfo
      { qhRowIndex        = rowIndex
      , qhState           = RowComputeUnit.rcMultState compute
      , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowDram)
      , qhRowResult       = register 0 (RowComputeUnit.rcResult compute)
      , qhRowDone         = RowComputeUnit.rcRowDone compute
      , qhFetchValid      = weightValid
      , qhFetchedWord     = pure 0
      , qhRowReset        = RowComputeUnit.rmdRowReset (RowComputeUnit.rcDebug compute)
      , qhRowEnable       = RowComputeUnit.rmdRowEnable (RowComputeUnit.rcDebug compute)
      , qhAccumValue      = RowComputeUnit.rmdAccValue (RowComputeUnit.rcDebug compute)
      , qhQOut            = qOut
      , qhCurrentRowExp   = register 0 (rowExponent <$> currentRowDram)
      , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRowDram)
      , qhRowReqValid     = RowComputeUnit.rcFetchReq compute
      , qhWeightReady     = weightReady
      , qhWeightValid     = weightValid
      }

--------------------------------------------------------------------------------
-- Top Level: queryHeadProjector
-- Wraps QueryHeadCore and applies rotary encoding
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal
  -> Signal dom (Index SequenceLength)            -- stepCount
  -> Signal dom (Vec ModelDimension FixedPoint)   -- xHat
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
queryHeadProjector cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal stepCount xHat params =
  ( qhcAxiMaster core
  , qWithRotary
  , qhcOutputValid core
  , qhcReady core
  , qhcDebug core
  )
  where
    core = queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat params

    -- Apply rotary encoding to output
    qWithRotary = (rotaryEncoder (PARAM.rotaryEncoding params) <$> stepCount) <*> qhcResult core
