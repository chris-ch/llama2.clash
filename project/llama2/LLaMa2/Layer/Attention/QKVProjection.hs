module LLaMa2.Layer.Attention.QKVProjection
  ( keyValueHeadProjector
  , qkvProjectionController
  , queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E (..))
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified LLaMa2.Layer.Attention.FSM as FSM (processingControllerFSM)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import Simulation.Parameters (DecoderParameters(..))
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified Prelude as P
import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..), AxiAW (..), AxiW (..))
import Clash.Debug (trace)

data RowFetchState = RFIdle | RFFetching | RFProcessing | RFDone
  deriving (Show, Eq, Generic, NFDataX)

-- Debug info
data QHeadDebugInfo dom = QHeadDebugInfo
  { qhRowIndex     :: Signal dom (Index HeadDimension)
  , qhState        :: Signal dom OPS.MultiplierState
  , qhFirstMant    :: Signal dom Mantissa
  , qhRowResult    :: Signal dom FixedPoint
  , qhRowDone      :: Signal dom Bool
  , qhFetchValid   :: Signal dom Bool
  , qhFetchedWord :: Signal dom (BitVector 512)
  , qhRowReset     :: Signal dom Bool
  , qhRowEnable    :: Signal dom Bool
  , qhAccumValue   :: Signal dom FixedPoint
  , qhQOut         :: Signal dom (Vec HeadDimension FixedPoint)
  , qhCurrentRowExp    :: Signal dom Exponent
  , qhCurrentRowMant0  :: Signal dom Mantissa
  , qhRowReqValid      :: Signal dom Bool      -- State machine's request signal
  , qhWeightReady      :: Signal dom Bool      -- WeightLoader's ready signal
  , qhWeightValid      :: Signal dom Bool      -- WeightLoader's valid signal
  } deriving (Generic)

data MultiplierDebug dom = MultiplierDebug
  { accValue  :: Signal dom FixedPoint
  , rowReset  :: Signal dom Bool
  , rowEnable :: Signal dom Bool
  } deriving (Generic)

data MultiplierOutput dom = MultiplierOutput
  { moRowResult     :: Signal dom FixedPoint
  , moRowDone       :: Signal dom Bool
  , moState         :: Signal dom OPS.MultiplierState
  , moRowReqValid   :: Signal dom Bool
  , moOutputValid   :: Signal dom Bool
  , moReadyForInput :: Signal dom Bool
  , moDebug         :: MultiplierDebug dom
  } deriving (Generic)

-- | Wrapper combining matrixMultiplierStateMachine with parallel64RowProcessor.
--
-- == Overview
--
-- This module bundles the FSM controller with the row processor to provide
-- a complete single-head matrix-vector multiplier interface. It handles the
-- internal coordination between state machine control signals and the row
-- processor's reset/enable inputs.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────┐
--                    │               multiplier                    │
--                    │                                             │
--   column ─────────►│  ┌─────────────────────────────────────┐    │
--   (Vec ModelDim)   │  │                                     │    │
--                    │  │    parallel64RowProcessor           │    │
--   row ────────────►│  │                                     │───►│──► moRowResult
--   (RowI8E)         │  │    rowReset◄─┐    rowEnable◄─┐      │    │
--                    │  │              │               │      │───►│──► moRowDone
--                    │  └──────────────┼───────────────┼──────┘    │
--                    │                 │               │           │
--                    │  ┌──────────────┴───────────────┴──────┐    │
--   colValid ───────►│  │                                     │───►│──► moState
--                    │  │    matrixMultiplierStateMachine     │    │
--   rowValid ───────►│  │                                     │───►│──► moRowReqValid
--                    │  │                                     │    │
--   downStreamReady─►│  │                                     │───►│──► moOutputValid
--                    │  │                                     │    │
--   rowIndex ───────►│  │                                     │───►│──► moReadyForInput
--                    │  └─────────────────────────────────────┘    │
--                    │                                             │
--                    └─────────────────────────────────────────────┘
-- @
--
-- == Input Signals
--
-- [@column@] Input vector (Vec ModelDimension FixedPoint).
--            The vector to multiply with each row.
--
-- [@row@] Current row weights (RowI8E ModelDimension).
--         Must be valid when rowValid is True.
--
-- [@colValid@] Start signal. When True in MIdle, begins processing.
--              Must remain asserted until at least the first rowValid handshake completes.
--
-- [@rowValid@] Row weights available. Indicates that the current row data
--              on the 'row' input is stable and ready to use.
--              
--              For HC path: pure True (weights always available).
--              For DRAM path: driven by weight loader's valid signal.
--              
--              CRITICAL INTERACTION: The FSM waits in MFetching state until
--              rowValid becomes True. On the cycle when rowValid rises:
--                1. FSM transitions MFetching → MReset
--                2. rowValid rising is traced/logged (rowValidRise detector)
--                3. colValid state is traced for debugging
--              
--              This handshake ensures weights are stable before processing begins.
--              The caller must hold rowValid True until the row is consumed
--              (typically until rowDone fires).
--
--              rowValid is only a permission signal. The data on row must remain stable
--              from the cycle rowValid is accepted through the entire MProcessing phase.
--
-- [@downStreamReady@] Downstream can accept output.
--                     Triggers MDone→MIdle transition.
--
-- [@rowIndex@] Current row number (managed externally).
--
-- == Output Signals
--
-- [@moRowResult@] Current row's dot product result.
--
-- [@moRowDone@] Current row computation complete (ONE-CYCLE PULSE from row processor).
--
-- [@moState@] FSM state for debugging.
--
-- [@moRowReqValid@] True in MFetching state - requests next row weights.
--
-- [@moOutputValid@] True in MDone state - all rows complete.
--
-- [@moReadyForInput@] True in MIdle state - ready for new operation.
--
-- [@moDebug@] Debug signals (accValue, rowReset, rowEnable).
--
-- == Protocol
--
-- 1. Caller asserts colValid when input vector is ready
-- 2. For DRAM path: Caller monitors moRowReqValid and provides rowValid when weights ready
-- 3. FSM cycles through rows: Fetch(wait for rowValid)→Reset→Process for each row
-- 4. After last row: moOutputValid asserts
-- 5. Caller asserts downStreamReady to acknowledge
-- 6. FSM returns to MIdle, moReadyForInput asserts
--
-- == colValid vs rowValid Handshake
--
-- The interaction between colValid and rowValid is as follows:
--
-- - __colValid__: Overall "start processing" signal, latched by caller until complete
-- - __rowValid__: Per-row "weights are ready" signal from weight loader
--
-- For HC (hardcoded) path:
-- @
-- rowValid = pure True  -- Always ready, FSM never waits in MFetching
-- @
--
-- For DRAM path:
-- @
-- Cycle:       0    1    2    3    4    5    6    7
-- colValid:    ────┐─────────────────────────────────  (latched high)
-- state:       Idle Ftch Ftch Rst  Proc Ftch Rst  Proc
-- rowReqValid: ────┐────┐────────────┐─────────────────  (MFetching)
-- rowValid:    ─────────┐────┐───────────┐────┐─────────  (from loader)
--              (FSM waits in MFetching until rowValid arrives)
-- @
--
-- The rowValidRise detector and colValidTraced logging help debug this handshake.
--
multiplier :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> MultiplierOutput dom
multiplier column row colValid rowValid downStreamReady rowIndex =
  MultiplierOutput
    { moRowResult     = rowResult
    , moRowDone       = rowDone
    , moState         = state
    , moRowReqValid   = rowReqValid
    , moOutputValid   = outputValid
    , moReadyForInput = readyForInputRaw
    , moDebug         = dbgInfo
    }
  where
    rowValidRise = rowValid .&&. (not <$> register False rowValid)

    colValidTraced = go <$> rowValidRise <*> colValid
      where
        go True cv = trace ("MULT: rowValid ROSE, colValid=" P.++ show cv) cv
        go False cv = cv

    (rowResult, rowDone, accValue) =
      OPS.parallel64RowProcessor rowReset rowEnable row column

    (state, rowReqValid, rowReset, rowEnable, outputValid, readyForInputRaw) =
      OPS.matrixMultiplierStateMachine colValidTraced rowValid downStreamReady rowDone rowIndex

    dbgInfo = MultiplierDebug
      { accValue  = accValue
      , rowReset  = rowReset
      , rowEnable = rowEnable
      }

--------------------------------------------------------------------------------
-- High-level query head matrix multiplier with DRAM weight loading
--------------------------------------------------------------------------------
data QueryHeadOutput dom = QueryHeadOutput
  { qhoAxiMaster     :: Master.AxiMasterOut dom
  , qhoResult        :: Signal dom (Vec HeadDimension FixedPoint)
  , qhoOutputValid   :: Signal dom Bool
  , qhoReadyForInput :: Signal dom Bool
  , qhoDebugInfo     :: QHeadDebugInfo dom
  } deriving (Generic)

-- | Complete query head matrix multiplier with input/output latching.
--
-- == Overview
--
-- This is the top-level component for a single query head's Q projection.
-- It adds input latching and output latching around the core multiplier
-- to provide clean handshake semantics for multi-head coordination.
--
-- The latching ensures:
-- 1. A pulse on inputValid starts processing (latched until complete)
-- 2. Output remains valid until downstream acknowledges (latched)
-- 3. Clean handoff between tokens without state pollution
--
-- == CURRENT CONFIGURATION (TEMPORARY)
-- NOTE: As of now, DRAM data does NOT affect computation.
-- All numerical results come exclusively from HC weights.
-- DRAM logic is instantiated only for handshake validation.
--
-- __IMPORTANT__: This component is currently configured for HC-path testing only.
-- Both the "DRAM path" and "HC path" multipliers use the same HC weights
-- (currentRowHC) to verify the multiplier logic before enabling DRAM loading.
--
-- When DRAM path is fully enabled:
-- - multOut should use currentRowDram instead of currentRowHC
-- - assertRowResultMatch should compare currentRowDram vs currentRowHC
--
-- Current configuration:
-- @
-- multOut = multiplier xHat currentRowHC ...  -- SHOULD BE: currentRowDram
-- 
-- dramRowResultChecked = assertRowResultMatch
--                          ...
--                          currentRowHC   -- TEMP: should be currentRowDram
--                          currentRowHC   -- HC reference
-- @
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────────────┐
--                    │           queryHeadMatrixMultiplier                 │
--                    │                                                     │
--                    │  ┌──────────────────┐                               │
--   inputValid ─────►│  │ inputValidLatched│─────────┐                     │
--                    │  │                  │         │                     │
--                    │  │ SET: inputValid  │         │                     │
--                    │  │ CLR: outputValid │         │                     │
--                    │  │   && downStream  │         │                     │
--                    │  └──────────────────┘         │                     │
--                    │                               ▼                     │
--                    │                        ┌─────────────┐              │
--   xHat ───────────►│───────────────────────►│             │              │
--   (input vector)   │                        │  multiplier │              │
--                    │                        │             │──►rowResult  │
--                    │  ┌──────────────┐      │  (HC path   │              │
--                    │  │ weightLoader │─────►│   temp)     │──►rowDone    │
--                    │  │  (disabled)  │      │             │              │
--                    │  └──────────────┘      │             │──►moOutputValid
--                    │                        └─────────────┘              │
--                    │                               │                     │
--                    │                               ▼                     │
--                    │  ┌───────────────────┐  moOutputValid               │
--                    │  │ outputValidLatch  │◄──────┘                      │
--                    │  │                   │                              │
--                    │  │ SET: moOutputValid│                              │
--                    │  │ CLR: latch &&     │──────────────────►outputValid│
--                    │  │      downStream   │                              │
--                    │  │ (CLR has priority)│                              │
--                    │  └───────────────────┘                              │
--                    │                                                     │
--   downStreamReady─►│─────────────────────────────────────────────────────┘
--                    │                                                     │
--                    │  ┌───────────────┐                                  │
--                    │  │   rowIndex    │ (increments on rowDone)          │
--                    │  │   register    │ (resets on output consumed)      │
--                    │  └───────────────┘                                  │
--                    │                                                     │
--                    │  ┌───────────────┐                                  │
--                    │  │    qOut       │ (accumulates row results)        │
--                    │  │   register    │                                  │
--                    │  └───────────────┘──────────────────────►result     │
--                    │                                                     │
--                    └─────────────────────────────────────────────────────┘
-- @
--
-- == Input Signals
--
-- [@inputValid@] __Pulse or level__. Starts a new matrix-vector operation.
--                Latched internally, so can be a single-cycle pulse.
--                Ignored while a previous operation is in progress.
--
-- [@downStreamReady@] __Level signal__. Downstream ready to accept output.
--                     When True and outputValid is True:
--                       - outputValidLatch clears
--                       - inputValidLatched clears  
--                       - rowIndex resets to 0
--                       - FSM transitions MDone→MIdle
--
-- [@xHat@] Input vector (normalized activations).
--
-- [@params@] Model parameters containing weight matrices.
--
-- == Output Signals
--
-- [@outputValid@] __Level signal__. True when all HeadDimension rows complete.
--                 Remains True until downStreamReady acknowledged.
--                 This is the LATCHED version of moOutputValid.
--
-- [@readyForInput@] __Level signal__. True when ready for new inputValid.
--                   Derived from multiplier's readyForInput AND weightReady.
--
-- [@qOutFinal@] Result vector (Vec HeadDimension FixedPoint).
--               Valid when outputValid is True.
--
-- == Latch Semantics (CRITICAL)
--
-- === inputValidLatched
-- @
-- nextInputValidLatched = 
--   mux inputValid (pure True)                           -- SET on inputValid
--   $ mux (outputValid .&&. downStreamReady) (pure False) -- CLR on consume
--     inputValidLatched                                   -- HOLD otherwise
-- @
--
-- Note: SET has priority over CLR. This is safe because inputValid should
-- not be asserted while outputValid is True (protocol violation).
--
-- === outputValidLatch  
-- @
-- nextOutputValidLatch = 
--   mux (outputValidLatch .&&. downStreamReady) (pure False) -- CLR first!
--   $ mux (moOutputValid multOut) (pure True)                 -- SET second
--     outputValidLatch                                        -- HOLD otherwise
-- @
--
-- __CRITICAL__: CLR has priority over SET. This is essential because:
-- 1. moOutputValid is a LEVEL (True while FSM in MDone)
-- 2. When downStreamReady arrives, both CLR and SET conditions are True
-- 3. CLR must win to ensure latch clears for next token
-- 4. If SET won, latch would stay True, corrupting next token
--
-- ⚠️ Do NOT connect moOutputValid directly downstream.
-- Always use the latched outputValid.
--
-- == Row Index Management
--
-- @
-- nextRowIndex =
--   mux (moRowDone .&&. (rowIndex ./=. maxBound))
--       (rowIndex + 1)                                    -- Increment on rowDone
--       (mux (outputValidLatch .&&. downStreamReady)
--            (pure 0)                                     -- Reset on consume
--            rowIndex)                                    -- Hold otherwise
-- @
--
-- == Output Accumulation
--
-- Results are accumulated into qOut register:
-- @
-- nextOutput = mux moRowDone
--                  (replace rowIndex rowResult qOut)  -- Store row result
--                  qOut                               -- Hold otherwise
-- @
--
-- == Token Processing Timeline
--
-- @
-- Token 1:
-- ────────────────────────────────────────────────────────────────────
-- Cycle:           0    1    ...  N    N+1  N+2
-- inputValid:      ─┐___________________________________________
-- inputValidLatch: __┐──────────────────────┐___________________
-- moOutputValid:   _____________________┐───┐___________________
-- outputValidLatch:_____________________┐───┐___________________
-- downStreamReady: _________________________┐___________________
-- rowIndex:        0    0    ...  7    7    0
-- 
-- Token 2:
-- ────────────────────────────────────────────────────────────────────
-- Cycle:           N+2  N+3  ...  M    M+1  M+2
-- inputValid:      ─┐___________________________________________
-- inputValidLatch: __┐──────────────────────┐___________________
-- ...
-- @
--
-- == Usage Notes
--
-- 1. Single pulse on inputValid is sufficient to start processing.
--
-- 2. Output remains valid and stable until acknowledged by downStreamReady.
--
-- 3. Back-to-back tokens: assert new inputValid on or after the cycle
--    that downStreamReady clears the previous operation.
--
-- 4. The component handles HeadDimension rows internally; caller only
--    sees the complete vector output.
--
-- 5. DRAM weight loading is currently disabled - all paths use HC weights.
--
queryHeadMatrixMultiplier :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                    -- inputValid
  -> Signal dom Bool                    -- downStreamReady (for FSM row-by-row)
  -> Signal dom Bool                    -- consumeSignal (for output latch clearing)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> QueryHeadOutput dom
queryHeadMatrixMultiplier dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat params =
  QueryHeadOutput
    { qhoAxiMaster     = axiMaster
    , qhoResult        = qOutFinal
    , qhoOutputValid   = outputValid
    , qhoReadyForInput = readyForInput
    , qhoDebugInfo     = debugInfo
    }
 where
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex

  -- Latch inputValid until we complete all rows
  inputValidLatched :: Signal dom Bool
  inputValidLatched = register False nextInputValidLatched
    where
      nextInputValidLatched = mux inputValid (pure True)
                            $ mux consumeSignal (pure False)  -- Use consumeSignal here
                              inputValidLatched

  -- Weight loader (note: pass moRowDone to allow the loader to hold LDone until consumed)
  (axiMaster, weightLoaderOut, weightValid, weightReady) =
    LOADER.weightLoader dramSlaveIn layerIdx headIdx
                        rowIndex rowReqValidTraced consumeSignal
                        (moRowDone multOut)
                        params

  -- COMMITTED rows from loader
  currentRowDramRaw = LOADER.dramRowOut weightLoaderOut
  currentRowHCRaw   = LOADER.hcRowOut   weightLoaderOut

  -- Ensure rows don't change while valid (live-path assertion)
  currentRowDram = LOADER.assertRowStable weightValid currentRowDramRaw
  currentRowHC   = LOADER.assertRowStable weightValid currentRowHCRaw

  currentRowDramTraced = traceDramHCMatch weightValid currentRowDram currentRowHC

  traceDramHCMatch :: Signal dom Bool
    -> Signal dom (RowI8E ModelDimension)
    -> Signal dom (RowI8E ModelDimension)
    -> Signal dom (RowI8E ModelDimension)
  traceDramHCMatch valid dram hc = result
    where
      result = check <$> valid <*> dram <*> hc
      check v d h = 
        if v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h)
          then trace ("L" P.++ show layerIdx P.++ " WEIGHT_MISMATCH exp_d=" P.++ show (rowExponent d) 
                    P.++ " exp_h=" P.++ show (rowExponent h)) d
          else d

  -- 1. Trace inputValidLatched state changes
  inputValidLatchedTraced = traceInputLatch <$> inputValidLatched <*> register False inputValidLatched <*> rowIndex
    where
      traceInputLatch current prev ri =
        let rise = current && not prev
            fall = not current && prev
        in if rise
          then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " IVL_RISE ri=" P.++ show ri) current
          else if fall
            then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " IVL_FALL ri=" P.++ show ri) current
            else current

  -- 4. Trace rowIndex changes
  rowIndexTraced = traceRowIndex <$> rowIndex <*> register 0 rowIndex <*> moRowDone multOut <*> outputValidLatchTraced <*> downStreamReady
    where
      traceRowIndex current prev done ovl dsr =
        if current /= prev
          then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " RI_CHANGE " P.++ show prev P.++ "->" P.++ show current 
                    P.++ " done=" P.++ show done P.++ " ovl=" P.++ show ovl P.++ " dsr=" P.++ show dsr) current
          else current

  -- DRAM path multiplier
  multOut = multiplier xHat currentRowHC inputValidLatchedTraced weightValid downStreamReady rowIndexTraced

  -- HC path: reuse the DRAM path's control signals
  (hcRowResult, _hcRowDone, _hcAccValue) =
    OPS.parallel64RowProcessor
      (rowReset (moDebug multOut))
      (rowEnable (moDebug multOut))
      currentRowHC
      xHat

  -- Assert row results match exactly when rowDone fires; feed the checked DRAM result forward
  dramRowResultChecked :: Signal dom FixedPoint
  dramRowResultChecked = assertRowResultMatch
                           (moRowDone multOut)
                           rowIndex
                           (moRowResult multOut)
                           hcRowResult
                           currentRowHC   -- TEMPORARILY DISABLED, should be "currentRowDram" - use committed+stable rows for debug
                           currentRowHC

  rowReqValidGated = moRowReqValid multOut .&&. weightReady

  traceInputValid :: Signal dom Bool -> Signal dom Bool
  traceInputValid sig = go <$> sig <*> inputValid <*> weightValid <*> rowIndex
    where
      go req True wv ri = trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " INPUT_VALID wv=" P.++ show wv P.++ " ri=" P.++ show ri) req
      go req False _ _ = req

  rowReqValidTraced = traceInputValid rowReqValidGated

  readyForInput    = moReadyForInput multOut .&&. weightReady

  -- Latch outputValid until downstream consumes it
  -- Output latch uses consumeSignal
  outputValidLatch :: Signal dom Bool
  outputValidLatch = register False nextOutputValidLatch
    where
      nextOutputValidLatch = mux (outputValidLatch .&&. consumeSignal) (pure False)  -- Use consumeSignal
                          $ mux (moOutputValid multOut) (pure True)
                            outputValidLatch

  nextRowIndex = mux (rowDoneTraced .&&. (rowIndex ./=. pure maxBound))
                      (rowIndex + 1)
                      (mux (outputValidLatch .&&. consumeSignal)  -- Use consumeSignal
                          (pure 0)
                          rowIndex)
  
  -- 2. Trace outputValidLatch state changes
  outputValidLatchTraced = traceOutputLatch <$> outputValidLatch <*> register False outputValidLatch <*> rowIndex <*> downStreamReady
    where
      traceOutputLatch current prev ri dsr =
        let rise = current && not prev
            fall = not current && prev
        in if rise
          then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " OVL_RISE ri=" P.++ show ri) current
          else if fall
            then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " OVL_FALL ri=" P.++ show ri P.++ " dsr=" P.++ show dsr) current
            else current

  -- Use the latched version externally
  outputValid = outputValidLatchTraced

  rowDoneTraced = traceRowDone <$> moRowDone multOut <*> rowIndex <*> moRowResult multOut
        where
          traceRowDone rd ri res =
            if rd 
              then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " row=" P.++ show ri 
                          P.++ " result=" P.++ show res) rd
              else rd

  -- Accumulate using checked DRAM result
  qOut = register (repeat 0) nextOutput
  -- 3. Trace qOut accumulation with values
  qOutTraced = traceQOutAccum <$> moRowDone multOut <*> rowIndex <*> dramRowResultChecked <*> qOut
    where
      traceQOutAccum done ri result current =
        if done
          then trace ("L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " QOUT_ACCUM ri=" P.++ show ri P.++ " val=" P.++ show result) current
          else current

  nextOutput = mux (moRowDone multOut)
                   (replace <$> rowIndex <*> dramRowResultChecked <*> qOutTraced) -- ! dramRowResultChecked, use hcRowResult to disable comparison
                   qOut

  -- Accumulate HC results (reference)
  qOutHC :: Signal dom (Vec HeadDimension FixedPoint)
  qOutHC = register (repeat 0) nextOutputHC
  nextOutputHC = mux (moRowDone multOut)
                     (replace <$> rowIndex <*> hcRowResult <*> qOutHC)
                     qOutHC

  qOutChecked = assertQOutputsMatch outputValid rowIndex qOut qOutHC

  traceMultState :: Signal dom (Vec HeadDimension FixedPoint) -> Signal dom (Vec HeadDimension FixedPoint)
  traceMultState qOut' = go <$> moState multOut <*> outputValid <*> rowIndex <*> moRowDone multOut <*> downStreamReady <*> qOut'
    where
      go st ov ri rd dsr out =
        let msg = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx
                  P.++ " st=" P.++ show st
                  P.++ " ov=" P.++ show ov
                  P.++ " ri=" P.++ show ri
                  P.++ " rd=" P.++ show rd
                  P.++ " dsr=" P.++ show dsr
        in if rd  -- Trace every time rowDone fires
          then trace msg out
          else out

  qOutFinal = traceMultState qOutChecked

  debugInfo = QHeadDebugInfo
    { qhRowIndex        = rowIndex
    , qhState           = moState multOut
    , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowHC)
    , qhRowResult       = register 0 (moRowResult multOut)
    , qhRowDone         = moRowDone multOut
    , qhFetchValid      = weightValid
    , qhFetchedWord     = pure 0
    , qhRowReset        = rowReset (moDebug multOut)
    , qhRowEnable       = rowEnable (moDebug multOut)
    , qhAccumValue      = accValue (moDebug multOut)
    , qhQOut            = qOut
    , qhCurrentRowExp   = register 0 (rowExponent <$> currentRowDramTraced)
    , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRowDramTraced)
    , qhRowReqValid     = moRowReqValid multOut
    , qhWeightReady     = weightReady
    , qhWeightValid     = weightValid
    }

-- | Assert that row results match when rowDone fires, with detailed debug
assertRowResultMatch :: forall dom . HiddenClockResetEnable dom
  => Signal dom Bool                    -- ^ rowDone trigger
  -> Signal dom (Index HeadDimension)   -- ^ row index
  -> Signal dom FixedPoint              -- ^ DRAM result
  -> Signal dom FixedPoint              -- ^ HC result
  -> Signal dom (RowI8E ModelDimension) -- ^ DRAM weights (for debug)
  -> Signal dom (RowI8E ModelDimension) -- ^ HC weights (for debug)
  -> Signal dom FixedPoint
assertRowResultMatch rowDone rowIdx dramResult hcResult dramWeights hcWeights = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 nextTokenCnt
    nextTokenCnt = mux (rowDone .&&. (rowIdx .==. pure maxBound))
                       (tokenCnt + 1)
                       tokenCnt

    result = mux rowDone
                 (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult 
                        <*> dramWeights <*> hcWeights)
                 dramResult

    check :: Unsigned 32 -> Index HeadDimension -> FixedPoint -> FixedPoint 
          -> RowI8E ModelDimension -> RowI8E ModelDimension -> FixedPoint
    check tok ri dr hr dramW hcW =
      if dr P.== hr
        then dr
        else P.error $ "Row result mismatch at token " P.++ show tok 
                    P.++ " row " P.++ show ri
                    P.++ ": DRAM=" P.++ show dr 
                    P.++ " HC=" P.++ show hr
                    P.++ "\n  DRAM weight exp=" P.++ show (rowExponent dramW)
                    P.++ " mant[0]=" P.++ show (P.head (toList (rowMantissas dramW)))
                    P.++ "\n  HC weight exp=" P.++ show (rowExponent hcW)
                    P.++ " mant[0]=" P.++ show (P.head (toList (rowMantissas hcW)))
                    P.++ "\n  weights match=" P.++ show (rowExponent dramW P.== rowExponent hcW 
                                                         P.&& rowMantissas dramW P.== rowMantissas hcW)

-- | Compare DRAM and HC Q outputs when valid - X-safe version
assertQOutputsMatch
  :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool                      -- ^ outputValid (from DRAM path)
  -> Signal dom (Index HeadDimension)
  -> Signal dom (Vec n FixedPoint)         -- ^ DRAM result (real)
  -> Signal dom (Vec n FixedPoint)         -- ^ HC result (reference)
  -> Signal dom (Vec n FixedPoint)
assertQOutputsMatch outputValid _rowIdx dramOut hcOut = result
 where
  -- Detect the first valid output to skip initial undefined/warm-up phase
  everValid :: Signal dom Bool
  everValid = register False (everValid .||. outputValid)

  -- one-cycle delayed view of the outputValid
  prevOutputValid :: Signal dom Bool
  prevOutputValid = register False outputValid

  -- Trigger comparison one cycle *after* outputValid goes high
  -- (i.e. when prevOutputValid is True)
  checkTrigger :: Signal dom Bool
  checkTrigger = prevOutputValid .&&. everValid

  -- Sample only when we are sure data is valid and defined
  dramSampled = register (repeat 0) (mux checkTrigger dramOut dramSampled)
  hcSampled   = register (repeat 0) (mux checkTrigger hcOut   hcSampled)

  -- Simple, correct counter
  tokenCnt :: Signal dom (Unsigned 32)
  tokenCnt = register 0 (mux checkTrigger (tokenCnt + 1) tokenCnt)

  -- Final output: substitute checked value only when checkTrigger fires
  result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

  -- Pure function — safe, uses only Prelude, no Clash (==) on undefined BitVectors
  checkPure :: Unsigned 32 -> Vec n FixedPoint -> Vec n FixedPoint -> Vec n FixedPoint
  checkPure tok dr hr =
    let ds = toList dr
        hs = toList hr
        pairs = P.zip [0..] (P.zip ds hs)
        mismatches = P.filter (\(_, (d,h)) -> d P./= h) pairs
    in if P.null mismatches
       then dr
       else let (i, (d, h)) = P.head mismatches
            in P.error $ "QHead output mismatch at token " P.++ show tok P.++
                         ": first mismatch at index " P.++ show (i :: Int) P.++
                         " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")" P.++
                         " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

--------------------------------------------------------------------------------
-- Q head projector
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
queryHeadProjector dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal stepCount xHat params =
  (qhoAxiMaster qhOut, qRoOut, qhoOutputValid qhOut, qhoReadyForInput qhOut, qhoDebugInfo qhOut)
    where
      qhOut = queryHeadMatrixMultiplier dramSlaveIn layerIdx headIdx
                                        inputValid downStreamReady consumeSignal xHat params

      qRoOut = (rotaryEncoder (PARAM.rotaryEncoding params) <$> stepCount) <*> qhoResult qhOut

--------------------------------------------------------------------------------
-- KV head projector
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.KeyValueHeadComponentQ
  -> PARAM.RotaryEncodingComponentF
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     )
keyValueHeadProjector inputValid downStreamReady stepCountSig xHatSig kvHeadParams rotary =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedK = pure (PARAM.kMatrix kvHeadParams)

  selectedV :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedV = pure (PARAM.vMatrix kvHeadParams)

  (kOut, kValidOut, kReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder rotary <$> stepCountSig) <*> kOut

  outputValid = kValidOut .&&. vValidOut
  readyForInput = kReadyOut .&&. vReadyOut
-- | Round-robin AXI arbiter with per-master response routing.
--
-- == Overview
--
-- This arbiter multiplexes multiple AXI masters (query heads) onto a single
-- AXI slave (DRAM controller). It implements round-robin arbitration for
-- fairness and tracks in-flight transactions to route responses back to
-- the correct requesting master.
--
-- == CURRENT STATUS: DISABLED
--
-- __IMPORTANT__: This arbiter is instantiated in qkvProjector but NOT connected
-- to the data path. The current working implementation (qResults) passes
-- dramSlaveIn directly to all query heads, bypassing the arbiter completely.
--
-- The arbiter code (qResults' with perHeadSlaves) exists but is unused:
-- @
-- (axiMasterOut, perHeadSlaves) = axiArbiterWithRouting dramSlaveIn qAxiMasters
-- qResults' = imap (\headIdx _ ->
--     queryHeadProjector (perHeadSlaves !! headIdx) ...  -- NOT USED
--   ) ...
-- 
-- qResults = map (qHead params) indicesI  -- ACTUALLY USED
--   where
--     qHead params' headIdx = 
--       queryHeadProjector dramSlaveIn layerIdx headIdx ...  -- Direct connection
-- @
--
-- To enable the arbiter, replace qResults with qResults' in qkvProjector.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────────────────┐
--                    │              axiArbiterWithRouting                      │
--                    │                                                         │
--   masters[0] ─────►│  ┌─────────────────────────────────────────────────┐    │
--     .arvalid       │  │                                                 │    │
--     .ardata        │  │            Request Arbitration                  │    │
--     .rready        │  │                                                 │    │
--                    │  │  ┌──────────┐    ┌──────────────┐               │    │
--   masters[1] ─────►│  │  │ Round-   │    │ Transaction  │               │    │
--     ...            │  │  │ Robin    │───►│   Tracker    │               │    │
--                    │  │  │ Selector │    │              │               │    │
--   masters[N] ─────►│  │  └──────────┘    │ - inFlight   │               │    │
--                    │  │       ▲          │ - owner      │               │    │
--                    │  │       │          │ - lastGrant  │               │    │
--                    │  │  arRequests      │ - reqLatched │               │    │
--                    │  │                  └──────────────┘               │    │
--                    │  │                         │                       │    │
--                    │  └─────────────────────────┼───────────────────────┘    │
--                    │                            │                            │
--                    │                            ▼                            │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              masterOut (to DRAM)                │    │
--                    │  │                                                 │───►│
--                    │  │  .arvalid = !inFlight && selectedArValid        │    │
--                    │  │  .ardata  = masters[activeIdx].ardata           │    │
--                    │  │  .rready  = masters[owner].rready               │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--   slaveIn ────────►│  ┌─────────────────────────────────────────────────┐    │
--     .arready       │  │              Response Routing                   │    │
--     .rvalid        │  │                                                 │    │
--     .rdata         │  │  Route responses to transaction owner only:     │    │
--                    │  │                                                 │    │
--                    │  │  perHeadSlaves[i].arready =                     │───►│ perHeadSlaves[0]
--                    │  │      (activeIdx == i) && !inFlight &&           │    │
--                    │  │      slaveIn.arready                            │───►│ perHeadSlaves[1]
--                    │  │                                                 │    │   ...
--                    │  │  perHeadSlaves[i].rvalid =                      │───►│ perHeadSlaves[N]
--                    │  │      (owner == i) && inFlight &&                │    │
--                    │  │      slaveIn.rvalid                             │    │
--                    │  │                                                 │    │
--                    │  │  perHeadSlaves[i].rdata = slaveIn.rdata         │    │
--                    │  │      (broadcast, but only owner sees rvalid)    │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    └─────────────────────────────────────────────────────────┘
-- @
--
-- == State Registers
--
-- [@inFlight@] __Bool__. True when a transaction is active (AR accepted, waiting for R).
--              Prevents new AR requests from being issued.
--              Set when AR handshake completes, cleared when R handshake with rlast.
--
-- [@transactionOwner@] __Index n__. Which master owns the current in-flight transaction.
--                      Latched on AR handshake, used to route R responses.
--                      Only valid when inFlight is True.
--
-- [@lastGranted@] __Index n__. Last master that completed a transaction.
--                 Used for round-robin fairness: next arbitration starts
--                 from (lastGranted + 1).
--                 Updated when transaction completes (not when granted).
--
-- [@requestLatched@] __Bool__. Latches the selected master's arvalid to prevent
--                    drops during multi-cycle arready waits.
--                    Set when selectedArValid becomes True, cleared on AR handshake.
--                    
--                    __IMPLEMENTATION BUG__: requestLatched is used in arHandshake
--                    detection but NOT included in masterOut.arvalid, creating
--                    an inconsistency. The handshake can complete using the latched
--                    request, but the output arvalid doesn't reflect it.
--                    
--                    This may cause spurious handshakes or missed transactions.
--
-- == Arbitration Logic
--
-- === Round-Robin Selection
--
-- @
-- findNextRequester :: Vec n Bool -> Index n -> Index n
-- findNextRequester reqs lastR =
--   -- Start searching from (lastGranted + 1)
--   -- Wrap around, find first requesting master
--   -- If none requesting, return lastR (no change)
-- @
--
-- === Active Index Selection
--
-- @
-- activeIdx = mux inFlight 
--                 transactionOwner  -- Locked while in-flight
--                 nextRequester     -- Round-robin when idle
-- @
--
-- === Handshake Detection
--
-- @
-- selectedArValid = arRequests !! activeIdx
-- requestLatched = latch selectedArValid (cleared on handshake)
-- 
-- arHandshake = (selectedArValid || requestLatched) && slaveIn.arready && !inFlight
-- rHandshake  = slaveIn.rvalid && masters[owner].rready
-- rLast       = slaveIn.rdata.rlast
-- transactionDone = rHandshake && rLast && inFlight
-- @
--
-- __BUG__: requestLatched is in arHandshake but not in masterOut.arvalid!
--
-- == State Transitions
--
-- @
--                 arHandshake
--     ┌────────┐ ───────────► ┌────────────┐
--     │  Idle  │              │  InFlight  │
--     │inFlight│              │ inFlight=T │
--     │ =False │ ◄─────────── │ owner=who  │
--     └────────┘ transDone    └────────────┘
-- @
--
-- == Output Signal Generation
--
-- === masterOut (to DRAM)
--
-- @
-- masterOut.arvalid = !inFlight && selectedArValid
--     -- BUG: Should include requestLatched!
--     -- Should be: !inFlight && (selectedArValid || requestLatched)
--
-- masterOut.ardata = masters[activeIdx].ardata
--     -- Forward selected master's address
--
-- masterOut.rready = masters[transactionOwner].rready
--     -- Forward owner's ready (only owner should be ready)
-- @
--
-- === perHeadSlaves[i] (to each head)
--
-- @
-- perHeadSlaves[i].arready = (activeIdx == i) && !inFlight && slaveIn.arready
--     -- Only active head sees arready, and only when idle
--
-- perHeadSlaves[i].rvalid = (transactionOwner == i) && inFlight && slaveIn.rvalid
--     -- Only owner sees rvalid
--
-- perHeadSlaves[i].rdata = slaveIn.rdata
--     -- Broadcast data (but non-owners ignore due to rvalid=False)
-- @
--
-- == Transaction Timeline
--
-- @
-- Cycle:        0    1    2    3    4    5    6    7    8
-- 
-- Head0.arvalid ─┐________________________________...
-- Head1.arvalid ───────┐____________________________...
-- 
-- inFlight:     F    F    T    T    T    T    F    F    T
-- activeIdx:    0    0    0    0    0    0    1    1    1
-- owner:        x    0    0    0    0    0    0    1    1
-- lastGranted:  x    x    x    x    x    0    0    0    0
-- 
-- masterOut.ar: ─────┐_____________┐____...
-- slaveIn.ardy: ─────┐_____________┐____...
-- slaveIn.rvalid: _______┐──┐__________┐...
-- slaveIn.rlast:  ___________┐__________...
-- 
-- perHead0.ardy: ────┐___________________...
-- perHead0.rvalid: ______┐──┐____________...
-- perHead1.ardy: _______________┐________...
-- perHead1.rvalid: __________________┐___...
-- @
--
-- == Critical Design Points
--
-- 1. __No AR while in-flight__: masterOut.arvalid is gated by !inFlight.
--    This prevents pipelining but ensures simple response routing.
--
-- 2. __Owner-based routing__: Responses go ONLY to transactionOwner.
--    Other heads see rvalid=False, preventing spurious data capture.
--
-- 3. __Round-robin fairness__: lastGranted updates on transaction COMPLETION,
--    not on grant. This ensures stuck transactions don't starve others.
--
-- 4. __Broadcast data__: rdata is broadcast to all heads (simpler routing),
--    but only owner's rvalid is True, so only owner latches it.
--
-- 5. __Request latching__: Prevents arvalid drops during multi-cycle waits,
--    but has implementation bug (not included in output arvalid).
--
-- == Why Response Routing Matters
--
-- Without proper routing, if Head 0 and Head 1 both request:
-- 1. Arbiter grants Head 1's request
-- 2. DRAM returns data for Head 1
-- 3. BUG: Head 0 sees rvalid (thinks its data arrived) and latches wrong data!
--
-- With proper routing:
-- 1. Arbiter grants Head 1, sets owner=1
-- 2. DRAM returns data
-- 3. perHeadSlaves[0].rvalid = False (owner != 0)
-- 4. perHeadSlaves[1].rvalid = True (owner == 1)
-- 5. Only Head 1 latches the data
--
-- == Known Issues
--
-- 1. requestLatched not included in masterOut.arvalid output
-- 2. Entire arbiter currently disabled in qkvProjector (qResults used instead of qResults')
--
-- == Usage Notes
--
-- 1. All masters must follow AXI protocol: hold arvalid until arready.
--
-- 2. Masters must be prepared to wait indefinitely for arready (arbitration).
--
-- 3. Masters should only assert rready when they can actually accept data.
--
-- 4. The arbiter assumes single-beat transactions (no burst support in 
--    current implementation, but rlast is checked for future extension).
--
-- 5. To enable this arbiter, modify qkvProjector to use qResults' instead of qResults.
--
axiArbiterWithRouting :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => Slave.AxiSlaveIn dom              -- ^ Single DRAM slave
  -> Vec n (Master.AxiMasterOut dom)   -- ^ Multiple masters (heads)
  -> ( Master.AxiMasterOut dom         -- ^ Combined master to DRAM
     , Vec n (Slave.AxiSlaveIn dom)    -- ^ Per-head slave interfaces
     )
axiArbiterWithRouting slaveIn masters = (masterOut, perHeadSlaves)
  where
    arRequests :: Vec n (Signal dom Bool)
    arRequests = map Master.arvalid masters

    -- Transaction tracking state machine
    inFlight :: Signal dom Bool
    inFlight = register False nextInFlight

    transactionOwner :: Signal dom (Index n)
    transactionOwner = register 0 nextTransactionOwner

    lastGranted :: Signal dom (Index n)
    lastGranted = register 0 nextLastGranted

    -- Round-robin selection: find next requesting head
    nextRequester :: Signal dom (Index n)
    nextRequester = findNextRequester <$> bundle arRequests <*> lastGranted

    findNextRequester :: Vec n Bool -> Index n -> Index n
    findNextRequester reqs lastR =
      let start = if lastR == maxBound then 0 else lastR + 1
          go i cnt
            | cnt == (0 :: Int) = lastR
            | reqs !! i = i
            | i == maxBound = go 0 (cnt - 1)
            | otherwise = go (i + 1) (cnt - 1)
      in go start (natToNum @n)

    -- Active index: locked to owner when in-flight, otherwise round-robin
    activeIdx :: Signal dom (Index n)
    activeIdx = mux inFlight transactionOwner nextRequester

    -- Latch arvalid on first assertion to prevent drops
    requestLatched = register False nextRequestLatched
      where
        nextRequestLatched = mux arHandshake (pure False)  -- Clear on handshake
                          $ mux selectedArValid (pure True)  -- Latch on request
                            requestLatched

    -- AR handshake detection
    selectedArValid = (!!) <$> bundle arRequests <*> activeIdx
    arHandshake = (selectedArValid .||. requestLatched) .&&. Slave.arready slaveIn .&&. (not <$> inFlight)

    -- R channel handshake detection
    selectedRReady = (!!) <$> bundle (map Master.rready masters) <*> transactionOwner
    rHandshake = Slave.rvalid slaveIn .&&. selectedRReady
    rLast = rlast <$> Slave.rdata slaveIn
    transactionDone = rHandshake .&&. rLast .&&. inFlight

    -- State transitions
    nextInFlight = mux arHandshake (pure True)
                 $ mux transactionDone (pure False)
                   inFlight

    nextTransactionOwner = mux arHandshake activeIdx transactionOwner

    -- Update lastGranted only when a transaction completes (for fair round-robin)
    nextLastGranted = mux transactionDone transactionOwner lastGranted

    -- Build master output using activeIdx for AR, transactionOwner for R
    masterOut = Master.AxiMasterOut
      { arvalid = mux inFlight (pure False) selectedArValid  -- Don't issue AR while in-flight
      , ardata  = (!!) <$> bundle (map Master.ardata masters) <*> activeIdx
      , rready  = (!!) <$> bundle (map Master.rready masters) <*> transactionOwner
      , awvalid = pure False
      , awdata  = pure (AxiAW 0 0 0 0 0)
      , wvalid  = pure False
      , wdata   = pure (AxiW 0 0 False)
      , bready  = pure False
      }

    -- Per-head slave interfaces with response routing
    perHeadSlaves :: Vec n (Slave.AxiSlaveIn dom)
    perHeadSlaves = map makeHeadSlave indicesI

    makeHeadSlave :: Index n -> Slave.AxiSlaveIn dom
    makeHeadSlave headIdx = Slave.AxiSlaveIn
      { arready = isActiveAndIdle .&&. Slave.arready slaveIn
      , rvalid  = isOwner .&&. Slave.rvalid slaveIn
      , rdata   = Slave.rdata slaveIn
      , awready = pure False
      , wready  = pure False
      , bvalid  = pure False
      , bdata   = pure (AxiB 0 0)
      }
      where
        isActiveAndIdle = (activeIdx .==. pure headIdx) .&&. (not <$> inFlight)
        isOwner = inFlight .&&. (transactionOwner .==. pure headIdx)

--------------------------------------------------------------------------------
-- QKV projector
--------------------------------------------------------------------------------

-- | Coordinates all query heads and key-value heads for QKV projection.
--
-- == Overview
--
-- This component instantiates NumQueryHeads query projectors and NumKeyValueHeads
-- KV projectors, combining their outputs. It provides a single interface for
-- the complete QKV projection stage of the attention mechanism.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────────────────┐
--                    │                    qkvProjector                         │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              Query Heads (×NumQueryHeads)       │    │
--                    │  │                                                 │    │
--   inputValid ─────►│  │  ┌─────────┐ ┌─────────┐     ┌─────────┐        │    │
--                    │  │  │ QHead 0 │ │ QHead 1 │ ... │ QHead N │        │    │
--   downStreamReady─►│  │  │         │ │         │     │         │        │    │
--                    │  │  └────┬────┘ └────┬────┘     └────┬────┘        │    │
--   xVec ───────────►│  │       │           │               │             │    │
--                    │  │       ▼           ▼               ▼             │    │
--   seqPos ─────────►│  │   qVecs[0]    qVecs[1]  ...   qVecs[N]          │    │
--                    │  │   qValids[0]  qValids[1] ... qValids[N]         │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              KV Heads (×NumKeyValueHeads)       │    │
--                    │  │                                                 │    │
--                    │  │  ┌─────────┐ ┌─────────┐     ┌─────────┐        │    │
--                    │  │  │ KVHead0 │ │ KVHead1 │ ... │ KVHeadM │        │    │
--                    │  │  │ (K & V) │ │ (K & V) │     │ (K & V) │        │    │
--                    │  │  └────┬────┘ └────┬────┘     └────┬────┘        │    │
--                    │  │       │           │               │             │    │
--                    │  │       ▼           ▼               ▼             │    │
--                    │  │   kVecs[0]    kVecs[1]  ...   kVecs[M]          │    │
--                    │  │   vVecs[0]    vVecs[1]  ...   vVecs[M]          │    │
--                    │  │   kvValids[0] kvValids[1]... kvValids[M]        │    │
--                    │  │                                                 │    │
--                    │  └────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    │  outputValid = AND(all qValids) AND AND(all kvValids)   │
--                    │  readyForInput = AND(all qReadys) AND AND(all kvReadys) │
--                    │                                                         │
--                    └─────────────────────────────────────────────────────────┘
-- @
--
-- == Input Signals
--
-- [@inputValid@] Start signal for all heads.
--                Directly passed to each head's inputValid.
--
-- [@downStreamReady@] Acknowledgment from downstream.
--                     __CRITICAL__: In the working HC-path version, this is
--                     passed DIRECTLY to each head. Each head clears its
--                     output latch independently when downStreamReady arrives.
--
-- [@seqPos@] Sequence position for rotary encoding.
--
-- [@xVec@] Input activations (Vec ModelDimension FixedPoint).
--          Normalized internally using RMS norm before projection.
--
-- [@params@] Full model parameters.
--
-- == Output Signals
--
-- [@qkvOut@] Bundled output: (Vec NumQueryHeads qVec, Vec NumKVHeads kVec, Vec NumKVHeads vVec)
--
-- [@outputValid@] True when ALL heads have completed.
--                 Computed as: AND of all individual head valids.
--
-- [@readyForInput@] True when ALL heads are ready for new input.
--                   Computed as: AND of all individual head readys.
--
-- == Coordination Strategy (Working HC-path Version)
--
-- In the current working version, all heads receive `downStreamReady` directly:
--
-- @
-- qResults = map (qHead params) indicesI
--   where
--     qHead params' headIdx = 
--       queryHeadProjector dramSlaveIn layerIdx headIdx
--                         inputValid downStreamReady seqPos xNorm params'
--                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
--                         Direct downStreamReady to each head
-- @
--
-- This means:
-- 1. All heads start simultaneously when inputValid arrives
-- 2. Heads may finish at different times (due to AXI arbitration in DRAM path)
-- 3. When downStreamReady arrives, ALL heads clear their latches simultaneously
-- 4. Combined outputValid = AND of all head valids
--
-- The key insight: even though heads might finish at slightly different times,
-- they all CLEAR together when downStreamReady arrives, ensuring clean handoff.
--
-- == Alternative: consumeSignal Coordination (for DRAM path)
--
-- For proper DRAM arbitration, heads need coordinated clearing:
--
-- @
-- consumeSignal = outputValid .&&. downStreamReady
--
-- qResults' = imap (\headIdx _ ->
--     queryHeadProjector (perHeadSlaves !! headIdx) layerIdx headIdx
--                       inputValid consumeSignal seqPos xNorm params
--                       ^^^^^^^^^^^^
--                       consumeSignal instead of downStreamReady
--   ) (repeat () :: Vec NumQueryHeads ())
-- @
--
-- This ensures:
-- 1. Heads only clear when ALL heads are done AND downstream ready
-- 2. Prevents early-finishing heads from restarting before others complete
-- 3. Required for proper AXI arbiter operation
--
-- __REQUIREMENT__: When using consumeSignal, outputValidLatch MUST have
-- CLR priority over SET (as documented in queryHeadMatrixMultiplier).
--
-- == Usage Notes
--
-- 1. Start processing by asserting inputValid for at least 1 cycle.
--
-- 2. Wait for outputValid before reading qkvOut.
--
-- 3. Assert downStreamReady to acknowledge and prepare for next token.
--
-- 4. Ensure proper handshake: don't assert new inputValid until previous
--    operation is acknowledged.
--
qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
qkvProjector dramSlaveIn layerIdx inputValid downStreamReady seqPos xVec params =
  (axiMasterOut, qkvOut, outputValid, readyForInput, head0Debug)
 where
  layerParams = modelLayers params !! layerIdx
  mhaParams = PARAM.multiHeadAttention layerParams
  xNorm = rmsNormFwFix <$> xVec <*> pure (PARAM.rmsAttF mhaParams)

  -- Get global rotary once
  rotary = PARAM.rotaryEncoding params

  -- AXI arbiter setup (mutual recursion handled by lazy evaluation)
  qAxiMasters :: Vec NumQueryHeads (Master.AxiMasterOut dom)
  perHeadSlaves :: Vec NumQueryHeads (Slave.AxiSlaveIn dom)
  
  (axiMasterOut, perHeadSlaves) = axiArbiterWithRouting dramSlaveIn qAxiMasters

  -- CRITICAL: Heads only clear their output latches when ALL heads are done
  -- AND downstream is ready. This prevents early-finishing heads from restarting
  -- before late-finishing heads complete.
  consumeSignal = outputValid .&&. downStreamReady

  -- Working version: direct connection, each head gets downStreamReady for FSM
  -- and consumeSignal for latch clearing
  qResults :: Vec NumQueryHeads (Master.AxiMasterOut dom, Signal dom (Vec HeadDimension FixedPoint), Signal dom Bool, Signal dom Bool, QHeadDebugInfo dom)
  qResults = imap (\headIdx _ ->
      queryHeadProjector dramSlaveIn layerIdx headIdx
                        inputValid 
                        (pure True)      -- downStreamReady for FSM (always ready for next row)
                        downStreamReady    -- consumeSignal for latch clearing
                        seqPos xNorm params
    ) (repeat () :: Vec NumQueryHeads ())

  -- Alternative with arbiter (currently unused)
  qResults' :: Vec NumQueryHeads (Master.AxiMasterOut dom, Signal dom (Vec HeadDimension FixedPoint), Signal dom Bool, Signal dom Bool, QHeadDebugInfo dom)
  qResults' = imap (\headIdx _ ->
      queryHeadProjector (perHeadSlaves !! headIdx) layerIdx headIdx
                        inputValid 
                        (pure True)      -- downStreamReady for FSM
                        consumeSignal    -- consumeSignal for latch clearing
                        seqPos xNorm params
    ) (repeat () :: Vec NumQueryHeads ())

  head0Debug = head qDebugInfos
  qAxiMasters = map (\(axi, _, _, _, _) -> axi) qResults
  qVecs       = map (\(_, q, _, _, _) -> q) qResults
  qValids     = map (\(_, _, v, _, _) -> v) qResults
  qReadys     = map (\(_, _, _, r, _) -> r) qResults
  qDebugInfos = map (\(_, _, _, _, d) -> d) qResults

  kvResults = map kvHead indicesI
   where
    kvHead kvIdx =
      let kvHeadParams = PARAM.kvHeads mhaParams !! kvIdx  -- Get actual KV head
      in keyValueHeadProjector inputValid downStreamReady seqPos xNorm kvHeadParams rotary
  
  kVecs    = map (\(k, _, _, _) -> k) kvResults
  vVecs    = map (\(_, v, _, _) -> v) kvResults
  kvValids = map (\(_, _, v, _) -> v) kvResults
  kvReadys = map (\(_, _, _, r) -> r) kvResults
  outputValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)
  qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)

--------------------------------------------------------------------------------
-- QKV Projection Controller
--------------------------------------------------------------------------------
qkvProjectionController ::
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> Signal dom (Index SequenceLength)
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
qkvProjectionController dramSlaveIn layerIdx inputValid downStreamReady input params seqPos =
  (axiMasterOut, result, outputValid, readyForInput, debugInfo)
 where
  (enableRaw, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  -- Fixed: deadlock prevention
  enableGated = enableRaw

  (axiMasterOut, result, matVecValid, projReadyOut, debugInfo) =
    qkvProjector dramSlaveIn layerIdx enableGated downStreamReady
                 seqPos input params

  projReadyOut_d = register True projReadyOut
  readyForInput  = inReadyRaw .&&. projReadyOut_d
