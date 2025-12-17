module LLaMa2.Numeric.Operations
  ( accumulator
  , MultiplierState(..)
  , matrixMultiplierStateMachine
  , parallelRowMatrixMultiplierDyn
  , parallel64RowProcessor
  , parallelRowMatrixMultiplier -- to be migrated to dynamic version
  , cyclicalCounter64 -- should be internal only
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E (..))

parallelRowMatrixMultiplier :: forall dom rows cols .
 ( HiddenClockResetEnable dom
 , KnownNat cols, KnownNat rows
 )
 => Signal dom Bool -- ^ validIn input from the upstream producer, indicating that the input vector is valid.
 -> Signal dom Bool -- ^ readyIn from downstream consumer, indicating it can accept new data
 -> MatI8E rows cols -- ^ matrix as row vectors.
 -> Signal dom (Vec cols FixedPoint) -- ^ input vector.
 -> ( Signal dom (Vec rows FixedPoint) -- ^ output vector.
 , Signal dom Bool -- ^ validOut indicating downstream consumer that the output vector is valid.
 , Signal dom Bool -- ^ readyOut input to the producer, indicating that the multiplier is ready to accept new input.
 )
parallelRowMatrixMultiplier = parallel64RowMatrixMultiplier
--parallelRowMatrixMultiplier = Simulation.MatVecSim.matrixMultiplierStub

-- | An accumulator for sequentially summing FixedPoint values in a matrix-vector
-- multiplication process. It accumulates products when enabled, can be reset,
-- and holds its value otherwise. This function is critical for computing the dot product
-- of a row vector with an input vector in a cycle-by-cycle manner.
--
-- Inputs:
--   - reset: A signal that, when True, resets the accumulator to zero.
--   - enable: A signal that, when True, enables accumulation of the input value.
--   - input: A signal carrying the FixedPoint value to be accumulated.
--
-- Output:
--   - A signal carrying the current accumulated sum as a FixedPoint value.
--
-- The accumulator updates the sum based on the enable and reset signals.
accumulator :: HiddenClockResetEnable dom
  => Signal dom Bool        -- reset
  -> Signal dom Bool        -- enable
  -> Signal dom FixedPoint  -- input
  -> Signal dom FixedPoint
accumulator reset enable input = acc
  where
    -- Compute next value *combinationally* from current and inputs
    next = mux reset 0 $
           mux enable (acc + input) acc

    -- Register holds state; output is current (updated) value
    acc = register 0 next

-- | Sequential matrix-vector multiplication processor that computes the product
-- of a quantized matrix with an input vector in a row-by-row, column-by-column manner.
-- The function processes one column per cycle, completing one row before moving to the next.

-- | State for the matrix multiplier state machine
data MultiplierState = MIdle | MFetching | MReset | MProcessing | MDone
  deriving (Show, Eq, Generic, NFDataX)

-- | Finite-state controller for sequential row-by-row matrix-vector multiplication.
--
-- == Overview
--
-- This FSM coordinates the processing of a matrix-vector product one row at a time.
-- It manages the lifecycle of fetching row weights, resetting the row processor,
-- enabling computation, and signaling completion.
--
-- == State Diagram
--
-- @
--                    ┌─────────────────────────────────────────────┐
--                    │                                             │
--                    ▼                                             │
--               ┌─────────┐                                        │
--        ┌─────►│  MIdle  │◄──────────────────────────┐            │
--        │      └────┬────┘                           │            │
--        │           │ colValid                       │            │
--        │           ▼                                │            │
--        │      ┌──────────┐                          │            │
--        │      │MFetching │─────────────────────┐    │            │
--        │      └────┬─────┘                     │    │            │
--        │           │ rowValid                  │    │            │
--        │           ▼                           │    │            │
--        │      ┌─────────┐                      │    │            │
--        │      │ MReset  │                      │    │            │
--        │      └────┬────┘                      │    │            │
--        │           │ (always, 1 cycle)         │    │            │
--        │           ▼                           │    │            │
--        │      ┌────────────┐                   │    │            │
--        │      │MProcessing │───────────────────┤    │            │
--        │      └─────┬──────┘                   │    │            │
--        │            │ rowDone                  │    │            │
--        │            ├──────────────────────────┘    │            │
--        │            │ (rowIdx /= maxBound)          │            │
--        │            │                               │            │
--        │            │ rowDone && rowIdx == maxBound │            │
--        │            ▼                               │            │
--        │      ┌─────────┐                           │            │
--        │      │  MDone  │───────────────────────────┘            │
--        │      └────┬────┘  downStreamReady                       │
--        │           │                                             │
--        │           │ (wait for consumer)                         │
--        └───────────┴─────────────────────────────────────────────┘
-- @
--
-- == Input Signals (directly accent or DIRECTLY connected to inputs)
--
-- [@colValid@] __Level signal__. When True in MIdle state, starts processing.
--              Typically driven by a latched version of upstream's inputValid.
--              The FSM samples this only in MIdle; once processing starts,
--              changes to colValid are ignored until the FSM returns to MIdle.
--
-- [@rowValid@] __Level signal__. Indicates row weights are available.
--              For constant weights (HC path): always @pure True@.
--              For DRAM path: driven by weight loader's valid signal.
--              The FSM waits in MFetching until rowValid becomes True.
--
-- [@downStreamReady@] __Level signal__. Indicates downstream consumer can accept output.
--                     The FSM waits in MDone until this becomes True.
--                     When True in MDone, transitions to MIdle on next cycle.
--
-- [@rowDone@] __Level signal__ (but typically True for only 1 cycle).
--             Asserted by the row processor when current row computation completes.
--             When True in MProcessing:
--               - If rowIndex == maxBound: transition to MDone
--               - Otherwise: transition to MFetching for next row
--
-- [@rowIndex@] __Data signal__. Current row being processed (0 to rows-1).
--              Used to detect when the last row has completed.
--              Managed externally; FSM only reads it.
--
-- == Output Signals
--
-- [@state@] Current FSM state, exposed for debugging.
--
-- [@fetchTrigger@] __Level signal__. True while state == MFetching.
--                  Used to trigger weight loading from memory.
--                  For HC path, this is typically ignored.
--
-- [@rowReset@] __Level signal__. True while state == MReset (exactly 1 cycle).
--              Connected to row processor's reset input.
--              Clears accumulator and column counter for new row.
--
-- [@rowEnable@] __Level signal__. True while state == MProcessing AND rowDone == False.
--               Connected to row processor's enable input.
--               Advances column counter and accumulates products.
--               Note: Deasserts on the cycle rowDone becomes True to prevent
--               extra accumulation.
--
-- [@outputValid@] __Level signal__. True while state == MDone.
--                 Indicates all rows complete, output vector is valid and stable.
--                 Remains True until downStreamReady causes transition to MIdle.
--
-- [@readyForInput@] __Level signal__. True while state == MIdle.
--                   Indicates FSM can accept a new matrix-vector operation.
--
-- == Timing Example (3 rows, 64 columns each = 1 cycle per row)
--
-- @
-- Cycle:    0    1    2    3    4    5    6    7    8    9   10   11   12
-- colValid: ─────┐________________________________________________________
--                └────────────────────────────────────────────────────────
-- state:    Idle Idle Ftch Rst  Proc Ftch Rst  Proc Ftch Rst  Proc Done Idle
-- rowIndex: 0    0    0    0    0    1    1    1    2    2    2    2    0
-- rowReset: ___________┐____┐___________┐____┐___________┐____┐____________
-- rowEnable:________________┐____┐___________┐____┐___________┐____┐_______
-- rowDone:  __________________┐________________┐________________┐__________
-- outValid: _______________________________________________________┐______
-- downReady:________________________________________________________┐_____
-- @
--
-- == Usage Notes
--
-- 1. The caller must manage rowIndex externally, incrementing it when rowDone
--    fires (except on the last row).
--
-- 2. The FSM is purely reactive to levels. No edge detection required.
--
-- 3. For back-to-back operations: after downStreamReady clears MDone,
--    the FSM returns to MIdle and will immediately start again if colValid
--    is still True. To prevent this, ensure colValid is pulsed or cleared.
--
-- 4. The rowEnable output is gated by (not rowDone) to ensure the row
--    processor doesn't accumulate an extra cycle after completion.
--
matrixMultiplierStateMachine :: forall dom rows .
  (HiddenClockResetEnable dom, KnownNat rows)  =>
  Signal dom Bool ->  -- colValid
  Signal dom Bool ->  -- rowValid
  Signal dom Bool ->  -- downStreamReady  
  Signal dom Bool ->  -- rowDone
  Signal dom (Index rows) ->  -- rowIndex
  ( Signal dom MultiplierState
  , Signal dom Bool  -- fetchTrigger
  , Signal dom Bool  -- rowReset
  , Signal dom Bool  -- rowEnable
  , Signal dom Bool  -- outputValid
  , Signal dom Bool  -- readyForInput
  )
matrixMultiplierStateMachine colValid rowValid downStreamReady rowDone rowIndex =
  (state, fetchTrigger, rowReset, rowEnable, outputValid, readyForInput)
  where
    state = register MIdle nextState
    
    -- Pure function for state transitions
    stateTransition :: MultiplierState -> Bool -> Bool -> Bool -> Bool -> Index rows -> MultiplierState
    stateTransition currentState colVld rowVld downReady done idx =
      case currentState of
        MIdle -> if colVld then MFetching else MIdle
        MFetching -> if rowVld then MReset else MFetching  -- Wait for fetch!
        MReset -> MProcessing
        MProcessing -> if done 
                       then (if idx == maxBound then MDone else MFetching)
                       else MProcessing
        MDone -> if downReady then MIdle else MDone
    
    -- Apply the pure function to the signals
    nextState = stateTransition <$> state <*> colValid <*> rowValid <*> downStreamReady 
                                 <*> rowDone <*> rowIndex
    
    -- Clear output signals based on state
    fetchTrigger = (== MFetching) <$> state
    rowReset = (== MReset) <$> state
    rowEnable = (== MProcessing) <$> state .&&. (not <$> rowDone)
    outputValid = (== MDone) <$> state
    readyForInput = (== MIdle) <$> state

-- | Process a single column of a row (one lane)
singleLaneProcessor :: forall dom . Signal dom (Signed 8)           -- mantissa element
  -> Signal dom FixedPoint           -- column element
  -> Signal dom FixedPoint           -- product
singleLaneProcessor mantissa columnComponent = inputValue
  where
    mantissaFP = fromIntegral <$> mantissa :: Signal dom FixedPoint
    inputValue = mantissaFP * columnComponent

-- | Helper: safely compute index with offset, clamping to valid range
-- Fixed to avoid out-of-bounds errors by checking before calling toEnum
addOffset :: forall size . KnownNat size => Index size -> Int -> Index size
addOffset idx offset =
  let idxInt = fromEnum idx
      newIdx = idxInt + offset
      maxIdx = fromEnum (maxBound :: Index size)
  in toEnum (min newIdx maxIdx)  -- Clamp BEFORE calling toEnum

-- | Parallel 64-lane dot product engine for a single matrix row.
--
-- == Overview
--
-- This unit computes the dot product of a quantized row vector (RowI8E format)
-- with an input vector of FixedPoint values. It processes 64 elements per cycle,
-- making it efficient for large vectors while maintaining reasonable hardware cost.
--
-- The row format uses I8E quantization: 8-bit signed mantissas with a shared
-- exponent per row. The final result is scaled by 2^exponent.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────┐
--                    │           parallel64RowProcessor            │
--                    │                                             │
--   reset ──────────►│  ┌─────────────┐                            │
--                    │  │ Column      │                            │
--   enable ─────────►│  │ Counter     │──► columnIndex             │
--                    │  │ (+64/cycle) │                            │
--                    │  └─────────────┘                            │
--                    │         │                                   │
--   row ────────────►│  ┌──────▼──────┐                            │
--   (RowI8E)         │  │ 64 Parallel │                            │
--                    │  │   Lanes     │                            │
--   columnVec ──────►│  │ mant[i]*col │                            │
--   (Vec n FP)       │  │    [i]      │                            │
--                    │  └──────┬──────┘                            │
--                    │         │ (masked for overflow)             │
--                    │  ┌──────▼──────┐                            │
--                    │  │  Tree Sum   │                            │
--                    │  │  (64→1)     │                            │
--                    │  └──────┬──────┘                            │
--                    │         │                                   │
--                    │  ┌──────▼──────┐                            │
--                    │  │ Accumulator │──────────────────►acc      │
--                    │  └──────┬──────┘                            │
--                    │         │                                   │
--                    │  ┌──────▼──────┐                            │
--                    │  │Scale by 2^e │──────────────────►output   │
--                    │  └─────────────┘                            │
--                    │                                             │
--                    │  ┌─────────────┐                            │
--                    │  │ Done Detect │──────────────────►rowDone  │
--                    │  │ (edge det.) │                            │
--                    │  └─────────────┘                            │
--                    └─────────────────────────────────────────────┘
-- @
--
-- == Input Signals
--
-- [@reset@] __Level signal__. Synchronous reset for new row.
--           When True:
--             - Column counter resets to 0
--             - Accumulator resets to 0  
--             - rowDone flag preparation resets
--           Must be asserted for exactly 1 cycle before processing begins.
--           Typically driven by FSM's rowReset output.
--
-- [@enable@] __Level signal__. Advances computation when True.
--            Each cycle with enable=True:
--              - Column counter advances by 64 (clamped to maxBound)
--              - 64 products computed and summed
--              - Sum added to accumulator
--            When False, all state holds (no progress).
--            Typically driven by FSM's rowEnable output.
--
-- [@row@] __Data signal__. The quantized row vector (RowI8E format).
--         Contains: rowMantissas (Vec n (Signed 8)) and rowExponent (Signed 16).
--         Must remain stable while enable=True.
--         For HC path: constant from parameters.
--         For DRAM path: loaded from weight loader.
--
-- [@columnVec@] __Data signal__. The input vector to multiply with.
--               Must remain stable while enable=True.
--               Typically the normalized input (xNorm) for attention.
--
-- == Output Signals
--
-- [@output@] __Data signal__. The scaled dot product result.
--            Computed as: accumulator * 2^(rowExponent)
--            Valid when rowDone=True; stable until next reset.
--
-- [@rowDone@] __Pulse signal__ (one-cycle high). Asserted for exactly ONE cycle
--             when the last column group has been processed.
--             Implementation uses edge detection: fires on the cycle when
--             (columnIndex + 63 >= maxBound) transitions from False to True
--             while enable is True.
--             
--             CRITICAL: This is a PULSE, not a level. It is high for exactly
--             one cycle. The FSM must be designed to respond to this single-cycle
--             pulse, not wait for a level to clear.
--
-- [@acc@] __Data signal__. Raw accumulator value before scaling.
--         Exposed for debugging/verification.
--
-- == Processing Timeline (for n=64, i.e., 1 cycle to complete)
--
-- @
-- Cycle:     0      1      2      3
-- reset:     True   False  False  False
-- enable:    False  True   False  False
-- colIndex:  0      0      63     63
-- acc:       0      0      sum    sum
-- rowDone:   False  False  True   False  ← ONE CYCLE PULSE
-- output:    0      0      scaled scaled
-- @
--
-- == Processing Timeline (for n=128, i.e., 2 cycles to complete)
--
-- @
-- Cycle:     0      1      2      3      4
-- reset:     True   False  False  False  False
-- enable:    False  True   True   False  False
-- colIndex:  0      0      64     127    127
-- acc:       0      0      sum1   sum2   sum2
-- rowDone:   False  False  False  True   False  ← ONE CYCLE PULSE
-- output:    0      0      x      scaled scaled
-- @
--
-- == Implementation Notes
--
-- 1. The 64-lane parallelism is fixed. For vectors smaller than 64 elements,
--    lanes beyond the vector size are masked to 0.
--
-- 2. Column index is clamped to maxBound, not wrapped. This ensures
--    out-of-bounds accesses read the last element (masked anyway).
--
-- 3. Edge detection for rowDone:
--    @
--    lastColumnFlag = (columnIndex + 63 >= maxBound) && enable
--    rowDoneRaw = lastColumnFlag && !register(lastColumnFlag)  -- Rising edge
--    rowDone = register rowDoneRaw                              -- Delay by 1 cycle
--    @
--    This ensures rowDone pulses exactly once per row completion.
--
-- 4. The accumulator receives 0 (via mux) during the cycle when rowDone is True,
--    which doesn't affect the result since enable should be False after completion.
--
-- 5. Tree reduction for 64 lanes uses Clash's built-in 'sum' which
--    synthesizes to an efficient adder tree.
--
parallel64RowProcessor :: forall dom size.
  ( HiddenClockResetEnable dom
  , KnownNat size)
  => Signal dom Bool                           -- ^ reset for new row
  -> Signal dom Bool                           -- ^ enable
  -> Signal dom (RowI8E size)                  -- ^ input row
  -> Signal dom (Vec size FixedPoint)          -- ^ input column
  -> ( Signal dom FixedPoint                   -- ^ output scalar
     , Signal dom Bool                         -- ^ done flag
     , Signal dom FixedPoint
  )
parallel64RowProcessor reset enable row columnVec = (output, rowDone, acc)
  where
    mant = rowMantissas <$> row
    expon = rowExponent <$> row

    -- Column index advances by 64 each cycle
    columnIndex :: Signal dom (Index size)
    columnIndex = cyclicalCounter64 reset enable

    -- Extract mantissas and columns for 64 lanes
    lanes = iterateI @64 (+1) 0
    mantissas = (\i -> (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure i)) <$> lanes
    columns   = (\i -> (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure i)) <$> lanes

    -- Compute products for all 64 lanes
    products = zipWith singleLaneProcessor mantissas columns

    -- Check validity for each lane
    isValid i = (\idx -> fromEnum idx + i <= fromEnum (maxBound :: Index size)) <$> columnIndex
    validities = isValid <$> lanes

    -- Mask invalid lanes
    maskedProducts = zipWith3 mux validities products (pure 0 :: Vec 64 (Signal dom FixedPoint))

    -- Tree reduction sum
    laneSum = sum maskedProducts

    -- Accumulate sum
    accInput = mux rowDone 0 laneSum

    acc :: Signal dom FixedPoint
    acc = accumulator reset enable accInput

    output :: Signal dom FixedPoint
    output = scalePow2F <$> expon <*> acc

    -- Done when index + 63 >= maxBound
    lastColumnFlag = ((\idx -> fromEnum idx + 63>= fromEnum (maxBound :: Index size)) <$> columnIndex) .&&. enable

    -- Combinational, true only when condition first met
    rowDoneRaw = lastColumnFlag .&&. (not <$> register False lastColumnFlag)
    rowDone = register False rowDoneRaw

cyclicalCounter64 :: forall dom size . (HiddenClockResetEnable dom, KnownNat size)
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index size)
cyclicalCounter64 reset enable = index
  where
    maxBoundVal = pure (fromEnum (maxBound :: Index size)) :: Signal dom Int
    indexInt = fromIntegral <$> index :: Signal dom Int
    nextIndexInt = mux enable
                       (mux (indexInt + 64 .>. maxBoundVal)
                            maxBoundVal
                            (indexInt + 64))
                       indexInt
    nextIndex = toEnum <$> mux reset (pure 0) nextIndexInt :: Signal dom (Index size)
    index = register 0 nextIndex

-- | Parallel 64-lane matrix-vector multiplication processor
-- Uses parallel64RowProcessor to compute 64 columns per cycle
parallel64RowMatrixMultiplier :: forall dom rows cols.
  ( HiddenClockResetEnable dom
  , KnownNat cols
  , KnownNat rows
  )
  => Signal dom Bool        -- ^ validIn
  -> Signal dom Bool        -- ^ readyIn (downstream)
  -> MatI8E rows cols       -- ^ input matrix
  -> Signal dom (Vec cols FixedPoint) -- ^ input vector
  -> ( Signal dom (Vec rows FixedPoint) -- ^ output vector
     , Signal dom Bool      -- ^ validOut
     , Signal dom Bool      -- ^ readyOut
     )
parallel64RowMatrixMultiplier validIn readyIn rowVectors inputVector =
  (outputVector, validOut, readyOut)
  where
    -- Row counter
    rowIndex = register (0 :: Index rows) nextRowIndex
    currentRow = (!!) rowVectors <$> rowIndex

    -- Parallel 64-lane row processor
    (rowResult, rowDone, _) = parallel64RowProcessor rowReset rowEnable currentRow inputVector

    -- State machine controls the protocol
    -- For non-DRAM: fetchDone is always True (data immediately available)
    (state, _fetchTrigger, rowReset, rowEnable, validOut, readyOut) =
      matrixMultiplierStateMachine validIn (pure True) readyIn rowDone rowIndex
      -- Note: we ignore fetchTrigger since we don't fetch anything

    -- Increment row index when row completes, reset after last row
    nextRowIndex = mux (rowDone .&&. (rowIndex ./=. pure maxBound))
                       (rowIndex + 1)
                       (mux ((state .==. pure MDone) .&&. readyIn)
                            (pure 0)
                            rowIndex)

    -- Accumulate results into output vector
    outputVector = register (repeat 0) nextOutput
    nextOutput = mux rowDone
                     (replace <$> rowIndex <*> rowResult <*> outputVector)
                     outputVector

--------------------------------------------------------------------------------
-- Dynamic matrix-vector multiplier: accepts Signal dom (MatI8E rows cols)
-- Mirrors parallel64RowMatrixMultiplier but with runtime-selectable matrix.
--------------------------------------------------------------------------------
parallelRowMatrixMultiplierDyn :: forall dom rows cols.
  ( HiddenClockResetEnable dom
  , KnownNat rows, KnownNat cols
  )
  => Signal dom Bool                      -- ^ inputValid
  -> Signal dom Bool                      -- ^ downStreamReady
  -> Signal dom (MatI8E rows cols)        -- ^ matrix (runtime, from RAM or const)
  -> Signal dom (Vec cols FixedPoint)     -- ^ input vector
  -> ( Signal dom (Vec rows FixedPoint)   -- ^ output vector
     , Signal dom Bool                    -- ^ outputValid
     , Signal dom Bool                    -- ^ readyForInput
     )
parallelRowMatrixMultiplierDyn inputValid downStreamReady matSig inputVector =
  (outputVector, outputValid, readyForInput)
 where
  -- Row counter
  rowIndex :: Signal dom (Index rows)
  rowIndex = register 0 nextRowIndex

  -- Fetch current row from the runtime matrix
  currentRow :: Signal dom (RowI8E cols)
  currentRow = (!!) <$> matSig <*> rowIndex

  initialRow :: RowI8E cols
  initialRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0}

  -- Parallel row engine
  currentRowReg :: Signal dom (RowI8E cols)
  currentRowReg = register initialRow currentRow

  (rowResult, rowDone, _accValue) =
      parallel64RowProcessor rowReset rowEnable currentRowReg inputVector

  -- Protocol FSM
  -- For non-DRAM: fetchDone is always True (data immediately available)
  (state, _fetchTrigger, rowReset, rowEnable, outputValid, readyForInput) =
    matrixMultiplierStateMachine inputValid (pure True) downStreamReady rowDone rowIndex
    -- Note: we ignore fetchTrigger since we don't fetch anything

  -- Row index sequencing
  nextRowIndex =
    mux (rowDone .&&. (rowIndex ./=. pure maxBound))
        (rowIndex + 1)
        (mux ((state .==. pure MDone) .&&. downStreamReady)
             (pure 0)
             rowIndex)

  -- Accumulate per-row results into the output vector
  outputVector = register (repeat 0) nextOutput
  nextOutput   = mux rowDone
                   (replace <$> rowIndex <*> rowResult <*> outputVector)
                   outputVector

