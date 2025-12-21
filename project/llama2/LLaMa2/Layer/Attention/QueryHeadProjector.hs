module LLaMa2.Layer.Attention.QueryHeadProjector
  ( queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified Prelude as P
import Clash.Debug (trace)
import qualified LLaMa2.Layer.Attention.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.OutputAccumulator as OutputAccumulator
import qualified LLaMa2.Layer.Attention.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.RowScheduler as RowScheduler

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
-- BLOCK: Tracing Utilities
-- Debug trace functions for signal monitoring
--------------------------------------------------------------------------------

-- | Trace row index changes
traceRowIndexChange :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension) -> Signal dom (Index HeadDimension)
  -> Signal dom Bool -> Signal dom Bool -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
traceRowIndexChange layerIdx headIdx current prev done ovl dsr = traced
  where
    traced = go <$> current <*> prev <*> done <*> ovl <*> dsr
    go curr p d o ds
      | curr /= p = trace (prefix P.++ "RI_CHANGE " P.++ show p P.++ "->" P.++ show curr 
                          P.++ " done=" P.++ show d P.++ " ovl=" P.++ show o P.++ " dsr=" P.++ show ds) curr
      | otherwise = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

-- | Trace row computation results
traceRowDone :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom (Index HeadDimension) -> Signal dom FixedPoint
  -> Signal dom Bool
traceRowDone layerIdx headIdx rowDone ri result = traced
  where
    traced = go <$> rowDone <*> ri <*> result
    go rd ridx res
      | rd        = trace (prefix P.++ "row=" P.++ show ridx P.++ " result=" P.++ show res) rd
      | otherwise = rd
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

-- | Trace INPUT_VALID signal
traceInputValidSignal :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
traceInputValidSignal layerIdx headIdx sig inputValid weightValid ri = traced
  where
    traced = go <$> sig <*> inputValid <*> weightValid <*> ri
    go req iv wv ridx
      | iv        = trace (prefix P.++ "INPUT_VALID wv=" P.++ show wv P.++ " ri=" P.++ show ridx) req
      | otherwise = req
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

-- | Trace multiplier state on rowDone
traceMultiplierState :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom OPS.MultiplierState -> Signal dom Bool -> Signal dom (Index HeadDimension)
  -> Signal dom Bool -> Signal dom Bool
  -> Signal dom a -> Signal dom a
traceMultiplierState layerIdx headIdx state ov ri rd dsr out = traced
  where
    traced = go <$> state <*> ov <*> ri <*> rd <*> dsr <*> out
    go st outputValid ridx rowDone downReady val
      | rowDone   = trace (prefix P.++ "st=" P.++ show st P.++ " ov=" P.++ show outputValid
                          P.++ " ri=" P.++ show ridx P.++ " rd=" P.++ show rowDone
                          P.++ " dsr=" P.++ show downReady) val
      | otherwise = val
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

--------------------------------------------------------------------------------
-- BLOCK: WeightMismatchTracer
-- Traces when DRAM and HC weights differ
--------------------------------------------------------------------------------
weightMismatchTracer :: Index NumLayers
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
weightMismatchTracer layerIdx valid dram hc = result
  where
    result = check <$> valid <*> dram <*> hc
    check v d h
      | v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h) =
          trace ("L" P.++ show layerIdx P.++ " WEIGHT_MISMATCH exp_d=" P.++ show (rowExponent d)
                P.++ " exp_h=" P.++ show (rowExponent h)) d
      | otherwise = d

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
-- BLOCK: QueryHeadCore
-- Complete query head matrix multiplier with input/output latching
--
-- Coordinates: InputValidLatch, OutputValidLatch, RowIndex, 
--              WeightLoader, RowMultiplier, ResultAccumulator
--------------------------------------------------------------------------------
data QueryHeadCoreOut dom = QueryHeadCoreOut
  { qhcAxiMaster   :: Master.AxiMasterOut dom
  , qhcResult      :: Signal dom (Vec HeadDimension FixedPoint)
  , qhcOutputValid :: Signal dom Bool
  , qhcReady       :: Signal dom Bool
  , qhcDebug       :: QHeadDebugInfo dom
  } deriving (Generic)

queryHeadCore :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal
  -> Signal dom (Vec ModelDimension FixedPoint)   -- xHat
  -> PARAM.DecoderParameters
  -> QueryHeadCoreOut dom
queryHeadCore dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat params =
  QueryHeadCoreOut
    { qhcAxiMaster   = axiMaster
    , qhcResult      = qOutFinal
    , qhcOutputValid = OutputTransactionController.otcOutputValid outputTxn
    , qhcReady       = readyForInput
    , qhcDebug       = debugInfo
    }
  where
    ----------------------------------------------------------------------------
    -- Row Index Counter
    -- Increments on rowDone (except at max), resets on consume
    ----------------------------------------------------------------------------
    rowIndex :: Signal dom (Index HeadDimension)
    rowIndex = register 0 nextRowIndex

    rowIndexTraced = traceRowIndexChange layerIdx headIdx 
                       rowIndex (register 0 rowIndex)
                       (RowComputeUnit.rcRowDone compute) (OutputTransactionController.otcOutputValid outputTxn) downStreamReady

    -- RowScheduler computes next index (combinatorial)
    rowSched = RowScheduler.rowScheduler
                 RowScheduler.RowSchedulerIn
                   { rsRowDone       = rowDoneTraced
                   , rsOutputValid   = OutputTransactionController.otcOutputValid outputTxn
                   , rsConsumeSignal = consumeSignal
                   , rsCurrentIndex  = rowIndexTraced  -- Feed current registered value
                   }
    
    nextRowIndex = RowScheduler.rsNextRowIndex rowSched  -- Get next combinatorial value

    inputTxn = InputTransactionController.inputTransactionController layerIdx headIdx rowIndex
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
    outputTxn = OutputTransactionController.outputTransactionController layerIdx headIdx rowIndex downStreamReady
                  OutputTransactionController.OutputTransactionIn
                    { otcAllDone       = RowComputeUnit.rcAllDone compute
                    , otcConsumeSignal = consumeSignal
                    }

    ----------------------------------------------------------------------------
    -- Weight Loader
    ----------------------------------------------------------------------------
    (axiMaster, weightLoaderOut, weightValid, weightReady) =
      LOADER.weightLoader dramSlaveIn layerIdx headIdx
                          rowIndex rowReqValidTraced consumeSignal
                          (RowComputeUnit.rcRowDone compute)
                          params

    currentRowDramRaw = LOADER.dramRowOut weightLoaderOut
    currentRowHCRaw   = LOADER.hcRowOut weightLoaderOut

    -- Ensure rows don't change while valid
    currentRowDram = LOADER.assertRowStable weightValid currentRowDramRaw
    currentRowHC   = LOADER.assertRowStable weightValid currentRowHCRaw

    currentRowDramTraced = weightMismatchTracer layerIdx weightValid currentRowDram currentRowHC

    ----------------------------------------------------------------------------
    -- Row Multiplier (FSM + parallel processor)
    ----------------------------------------------------------------------------
    compute = RowComputeUnit.rowComputeUnit
                RowComputeUnit.RowComputeIn
                  { rcInputValid      = inputValidLatched  -- FIX: use directly (already traced)
                  , rcWeightValid     = weightValid
                  , rcDownStreamReady = downStreamReady
                  , rcRowIndex        = rowIndexTraced
                  , rcWeight          = currentRowHC  -- Use HC for now
                  , rcColumn          = xHat
                  }

    rowReqValidGated = RowComputeUnit.rcFetchReq compute .&&. weightReady
    rowReqValidTraced = traceInputValidSignal layerIdx headIdx 
                          rowReqValidGated inputValid weightValid rowIndex

    readyForInput = RowComputeUnit.rcIdleReady compute .&&. weightReady

    ----------------------------------------------------------------------------
    -- Row Done Tracing
    ----------------------------------------------------------------------------
    rowDoneTraced = traceRowDone layerIdx headIdx (RowComputeUnit.rcRowDone compute) rowIndex (RowComputeUnit.rcResult compute)

    ----------------------------------------------------------------------------
    -- Row Result Checker
    ----------------------------------------------------------------------------
    dramRowResultChecked = rowResultChecker
      (RowComputeUnit.rcRowDone compute) rowIndex (RowComputeUnit.rcResult compute) (RowComputeUnit.rcResultHC compute)
      currentRowHC currentRowHC  -- TODO: use currentRowDram when enabled

    ----------------------------------------------------------------------------
    -- Result Accumulator
    -- Stores row results into output vector
    ----------------------------------------------------------------------------
    outputAccum = OutputAccumulator.outputAccumulator layerIdx headIdx
                    OutputAccumulator.OutputAccumIn
                      { oaRowDone     = RowComputeUnit.rcRowDone compute
                      , oaRowIndex    = rowIndex
                      , oaRowResult   = dramRowResultChecked
                      , oaRowResultHC = RowComputeUnit.rcResultHC compute
                      }

    qOut   = OutputAccumulator.oaOutput outputAccum
    qOutHC = OutputAccumulator.oaOutputHC outputAccum
    ----------------------------------------------------------------------------

    qOutChecked = qOutputChecker (OutputTransactionController.otcOutputValid outputTxn) qOut qOutHC
    qOutFinal = traceMultiplierState layerIdx headIdx
                  (RowComputeUnit.rcMultState compute) (OutputTransactionController.otcOutputValid outputTxn) rowIndex (RowComputeUnit.rcRowDone compute) downStreamReady
                  qOutChecked

    ----------------------------------------------------------------------------
    -- Debug Info
    ----------------------------------------------------------------------------
    debugInfo = QHeadDebugInfo
      { qhRowIndex        = rowIndex
      , qhState           = RowComputeUnit.rcMultState compute
      , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowHC)
      , qhRowResult       = register 0 (RowComputeUnit.rcResult compute)
      , qhRowDone         = RowComputeUnit.rcRowDone compute
      , qhFetchValid      = weightValid
      , qhFetchedWord     = pure 0
      , qhRowReset        = RowComputeUnit.rmdRowReset (RowComputeUnit.rcDebug compute)
      , qhRowEnable       = RowComputeUnit.rmdRowEnable (RowComputeUnit.rcDebug compute)
      , qhAccumValue      = RowComputeUnit.rmdAccValue (RowComputeUnit.rcDebug compute)
      , qhQOut            = qOut
      , qhCurrentRowExp   = register 0 (rowExponent <$> currentRowDramTraced)
      , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRowDramTraced)
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
  => Slave.AxiSlaveIn dom
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
queryHeadProjector dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal stepCount xHat params =
  ( qhcAxiMaster core
  , qWithRotary
  , qhcOutputValid core
  , qhcReady core
  , qhcDebug core
  )
  where
    core = queryHeadCore dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat params

    -- Apply rotary encoding to output
    qWithRotary = (rotaryEncoder (PARAM.rotaryEncoding params) <$> stepCount) <*> qhcResult core
    