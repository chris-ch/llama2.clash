module LLaMa2.Layer.Attention.QueryHeadProjector
  (
    queryHeadProjector
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
      nextInputValidLatched = 
        mux (inputValid .&&. (not <$> inputValidLatched)) (pure True)  -- SET gated by LOCAL inputValidLatched
        $ mux (outputValidLatch .&&. downStreamReady) (pure False)    -- CLR when THIS head finishes AND downstream ready
          inputValidLatched                                            -- HOLD

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
