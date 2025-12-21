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
-- BLOCK: RowMultiplier
-- Bundles FSM controller with parallel64RowProcessor
--
-- Inputs:
--   column      - input vector to multiply
--   row         - current row weights
--   colValid    - start signal (latched externally)
--   rowValid    - weights ready signal
--   downReady   - downstream ready
--   rowIndex    - current row (0..HeadDimension-1)
--
-- Outputs:
--   result, rowDone, state, fetchReq, allDone, idleReady, debug signals
--------------------------------------------------------------------------------
data RowMultiplierDebug dom = RowMultiplierDebug
  { rmdAccValue  :: Signal dom FixedPoint
  , rmdRowReset  :: Signal dom Bool
  , rmdRowEnable :: Signal dom Bool
  } deriving (Generic)

data RowMultiplierOut dom = RowMultiplierOut
  { rmoResult     :: Signal dom FixedPoint
  , rmoRowDone    :: Signal dom Bool
  , rmoState      :: Signal dom OPS.MultiplierState
  , rmoFetchReq   :: Signal dom Bool
  , rmoAllDone    :: Signal dom Bool
  , rmoIdleReady  :: Signal dom Bool
  , rmoDebug      :: RowMultiplierDebug dom
  } deriving (Generic)

rowMultiplier :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> RowMultiplierOut dom
rowMultiplier column row colValid rowValid downReady rowIndex =
  RowMultiplierOut
    { rmoResult     = rowResult
    , rmoRowDone    = rowDone
    , rmoState      = state
    , rmoFetchReq   = fetchReq
    , rmoAllDone    = allDone
    , rmoIdleReady  = idleReady
    , rmoDebug      = RowMultiplierDebug accValue rowReset rowEnable
    }
  where
    -- Detect rowValid rising edge for debug
    rowValidRise = rowValid .&&. (not <$> register False rowValid)

    colValidTraced = go <$> rowValidRise <*> colValid
      where
        go True cv = trace ("MULT: rowValid ROSE, colValid=" P.++ show cv) cv
        go False cv = cv

    -- Core computation
    (rowResult, rowDone, accValue) =
      OPS.parallel64RowProcessor rowReset rowEnable row column

    -- FSM control
    (state, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      OPS.matrixMultiplierStateMachine colValidTraced rowValid downReady rowDone rowIndex

--------------------------------------------------------------------------------
-- BLOCK: Tracing Utilities
-- Debug trace functions for signal monitoring
--------------------------------------------------------------------------------

-- | Trace latch state changes (rise/fall edges)
traceLatchEdges :: Index NumLayers -> Index NumQueryHeads -> P.String
  -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension) 
  -> Signal dom Bool
traceLatchEdges layerIdx headIdx name current prev ri = traced
  where
    traced = go <$> current <*> prev <*> ri
    go curr p ridx
      | curr && not p = trace (prefix P.++ name P.++ "_RISE ri=" P.++ show ridx) curr
      | not curr && p = trace (prefix P.++ name P.++ "_FALL ri=" P.++ show ridx) curr
      | otherwise     = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

-- | Trace output valid latch with downstream ready status
traceOutputLatchEdges :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension) -> Signal dom Bool
  -> Signal dom Bool
traceOutputLatchEdges layerIdx headIdx current prev ri dsr = traced
  where
    traced = go <$> current <*> prev <*> ri <*> dsr
    go curr p ridx downReady
      | curr && not p = trace (prefix P.++ "OVL_RISE ri=" P.++ show ridx) curr
      | not curr && p = trace (prefix P.++ "OVL_FALL ri=" P.++ show ridx P.++ " dsr=" P.++ show downReady) curr
      | otherwise     = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

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

-- | Trace accumulator updates
traceAccumUpdate :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom (Index HeadDimension) -> Signal dom FixedPoint
  -> Signal dom a -> Signal dom a
traceAccumUpdate layerIdx headIdx done ri value current = traced
  where
    traced = go <$> done <*> ri <*> value <*> current
    go d ridx val curr
      | d         = trace (prefix P.++ "QOUT_ACCUM ri=" P.++ show ridx P.++ " val=" P.++ show val) curr
      | otherwise = curr
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
    , qhcOutputValid = outputValidTraced
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

    nextRowIndex = mux (rowDoneTraced .&&. (rowIndex ./=. pure maxBound)) (rowIndex + 1)
                 $ mux (outputValidLatch .&&. consumeSignal) (pure 0)
                   rowIndex

    rowIndexTraced = traceRowIndexChange layerIdx headIdx 
                       rowIndex (register 0 rowIndex)
                       (rmoRowDone mult) outputValidLatchTraced downStreamReady

    ----------------------------------------------------------------------------
    -- Input Valid Latch
    -- SET on inputValid when not already latched
    -- CLR when this head finishes AND downstream ready
    ----------------------------------------------------------------------------
    inputValidLatched :: Signal dom Bool
    inputValidLatched = register False nextInputValidLatched

    nextInputValidLatched =
      mux (inputValid .&&. (not <$> inputValidLatched)) (pure True)
      $ mux (outputValidLatch .&&. downStreamReady) (pure False)
        inputValidLatched

    inputValidLatchedTraced = traceLatchEdges layerIdx headIdx "IVL"
                                inputValidLatched (register False inputValidLatched) rowIndex

    ----------------------------------------------------------------------------
    -- Output Valid Latch
    -- CLR has priority over SET (critical for correct handshake)
    -- Uses consumeSignal for coordinated multi-head clearing
    ----------------------------------------------------------------------------
    outputValidLatch :: Signal dom Bool
    outputValidLatch = register False nextOutputValidLatch

    nextOutputValidLatch =
      mux (outputValidLatch .&&. consumeSignal) (pure False)   -- CLR first (priority)
      $ mux (rmoAllDone mult) (pure True)                       -- SET second
        outputValidLatch                                        -- HOLD

    outputValidLatchTraced = traceOutputLatchEdges layerIdx headIdx
                               outputValidLatch (register False outputValidLatch)
                               rowIndex downStreamReady

    -- Exported output valid (traced version)
    outputValidTraced = outputValidLatchTraced

    ----------------------------------------------------------------------------
    -- Weight Loader
    ----------------------------------------------------------------------------
    (axiMaster, weightLoaderOut, weightValid, weightReady) =
      LOADER.weightLoader dramSlaveIn layerIdx headIdx
                          rowIndex rowReqValidTraced consumeSignal
                          (rmoRowDone mult)
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
    mult = rowMultiplier xHat currentRowHC inputValidLatchedTraced weightValid downStreamReady rowIndexTraced

    rowReqValidGated = rmoFetchReq mult .&&. weightReady
    rowReqValidTraced = traceInputValidSignal layerIdx headIdx 
                          rowReqValidGated inputValid weightValid rowIndex

    readyForInput = rmoIdleReady mult .&&. weightReady

    ----------------------------------------------------------------------------
    -- HC Reference Path (for validation)
    ----------------------------------------------------------------------------
    (hcRowResult, _, _) =
      OPS.parallel64RowProcessor
        (rmdRowReset (rmoDebug mult))
        (rmdRowEnable (rmoDebug mult))
        currentRowHC
        xHat

    ----------------------------------------------------------------------------
    -- Row Done Tracing
    ----------------------------------------------------------------------------
    rowDoneTraced = traceRowDone layerIdx headIdx (rmoRowDone mult) rowIndex (rmoResult mult)

    ----------------------------------------------------------------------------
    -- Row Result Checker
    ----------------------------------------------------------------------------
    dramRowResultChecked = rowResultChecker
      (rmoRowDone mult) rowIndex (rmoResult mult) hcRowResult
      currentRowHC currentRowHC  -- TODO: use currentRowDram when enabled

    ----------------------------------------------------------------------------
    -- Result Accumulator
    -- Stores row results into output vector
    ----------------------------------------------------------------------------
    qOut :: Signal dom (Vec HeadDimension FixedPoint)
    qOut = register (repeat 0) nextOutput

    qOutTraced = traceAccumUpdate layerIdx headIdx 
                   (rmoRowDone mult) rowIndex dramRowResultChecked qOut

    nextOutput = mux (rmoRowDone mult)
                     (replace <$> rowIndex <*> dramRowResultChecked <*> qOutTraced)
                     qOut

    -- HC reference accumulator
    qOutHC :: Signal dom (Vec HeadDimension FixedPoint)
    qOutHC = register (repeat 0) nextOutputHC

    nextOutputHC = mux (rmoRowDone mult)
                       (replace <$> rowIndex <*> hcRowResult <*> qOutHC)
                       qOutHC

    qOutChecked = qOutputChecker outputValidTraced qOut qOutHC

    qOutFinal = traceMultiplierState layerIdx headIdx
                  (rmoState mult) outputValidTraced rowIndex (rmoRowDone mult) downStreamReady
                  qOutChecked

    ----------------------------------------------------------------------------
    -- Debug Info
    ----------------------------------------------------------------------------
    debugInfo = QHeadDebugInfo
      { qhRowIndex        = rowIndex
      , qhState           = rmoState mult
      , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowHC)
      , qhRowResult       = register 0 (rmoResult mult)
      , qhRowDone         = rmoRowDone mult
      , qhFetchValid      = weightValid
      , qhFetchedWord     = pure 0
      , qhRowReset        = rmdRowReset (rmoDebug mult)
      , qhRowEnable       = rmdRowEnable (rmoDebug mult)
      , qhAccumValue      = rmdAccValue (rmoDebug mult)
      , qhQOut            = qOut
      , qhCurrentRowExp   = register 0 (rowExponent <$> currentRowDramTraced)
      , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRowDramTraced)
      , qhRowReqValid     = rmoFetchReq mult
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
    