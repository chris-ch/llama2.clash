module LLaMa2.Numeric.Operations
  ( accumulator
  , MultiplierState(..)
  , matrixMultiplierStateMachine
  , parallel64RowMatrixMultiplier
  , cyclicalCounter64
  , parallel64RowProcessor
  , parallelRowMatrixMultiplier
  , parallelRowMatrixMultiplierDyn
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
accumulator
  :: HiddenClockResetEnable dom
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

-- | Finite-state controller for the sequential /parallel-row/ engine.
--
-- This FSM coordinates:
--
--   * accepting a new input vector,
--   * fetching (or latching) a new row,
--   * resetting and enabling the row-processor,
--   * detecting per-row completion,
--   * signalling final completion of the whole matrix-vector product.
--
-- The FSM is entirely **level-sensitive**: it does not watch edges or
-- pulses.  All control signals are sampled synchronously on each clock
-- edge based on their instantaneous Boolean level.
--
-- = Inputs
--
--   * **inputValid :: Signal dom Bool**  
--     Level-true indicates that the upstream producer is offering a new
--     input vector.  
--     The FSM may accept it only while in state 'MIdle'.  
--     Hold high until 'readyForInput' becomes True.
--
--   * **downStreamReady :: Signal dom Bool**  
--     Level-true indicates that the downstream consumer is ready to accept
--     the completed output vector.  
--     When the FSM reaches 'MDone', it will remain there until this signal
--     goes high.
--
--   * **rowDone :: Signal dom Bool**  
--     Level-true for one cycle when the row processor has finished the
--     current row.  
--     The FSM uses this to determine whether to:
--       - fetch the next row, or
--       - transition to 'MDone' if the last row has completed.
--
--   * **rowValid :: Signal dom Bool**  
--     Level-true indicates that the row data is available (after fetch).  
--     For systems without memory latency, this is simply 'pure True'.
--
--   * **rowIndex :: Signal dom (Index rows)**  
--     Indicates which row is currently being processed.  
--     Used to detect when the last row has been reached.
--
-- = Outputs (control lines)
--
--   * **fetchTrigger :: Signal dom Bool**  
--     Level-true while in 'MFetching'.  
--     Used to initiate a row fetch or latch.  
--     The FSM will not advance until 'rowValid' is True.
--
--   * **rowReset :: Signal dom Bool**  
--     Level-true while in 'MReset'.  
--     Must be interpreted by the row-processor as a synchronous “start
--     new row” reset.  
--     Hold high for exactly one cycle.
--
--   * **rowEnable :: Signal dom Bool**  
--     Level-true while in 'MProcessing'.  
--     Causes the row-processor to advance its 64-lane step.  
--     If 'rowDone' becomes True, the FSM transitions to a fetch of the
--     next row or to 'MDone'.
--
--   * **outputValid :: Signal dom Bool**  
--     Level-true while in 'MDone'.  
--     Indicates to the downstream consumer that the entire output vector
--     is ready and stable.
--
--   * **readyForInput :: Signal dom Bool**  
--     Level-true while in 'MIdle'.  
--     Indicates to the upstream producer that a new input vector may be
--     provided.
--
-- = Protocol summary
--
--   **1. Upstream provides an input vector**  
--      Hold 'inputValid' = True until 'readyForInput' = True.  
--      FSM transitions: @MIdle → MFetching@.
--
--   **2. Row fetch**  
--      FSM asserts 'fetchTrigger' (level).  
--      Once 'rowValid' = True, FSM moves @MFetching → MReset@.
--
--   **3. Row reset**  
--      FSM asserts 'rowReset' for one cycle: @MReset → MProcessing@.
--
--   **4. Row processing**  
--      FSM holds 'rowEnable' = True until 'rowDone' = True.  
--      If not the last row: @MProcessing → MFetching@.  
--      If last row:          @MProcessing → MDone@.
--
--   **5. Output ready**  
--      FSM asserts 'outputValid' while in 'MDone'.  
--      Waits for 'downStreamReady' = True: @MDone → MIdle@.
--
-- The protocol uses **levels only**. No pulse generation or edge detection
-- is required from the caller: simply hold the control signals high or low
-- for as long as the FSM requires them.
matrixMultiplierStateMachine :: forall dom rows .
  (HiddenClockResetEnable dom, KnownNat rows)  =>
  Signal dom Bool ->  -- inputValid
  Signal dom Bool ->  -- downStreamReady  
  Signal dom Bool ->  -- rowDone
  Signal dom Bool ->  -- rowValid
  Signal dom (Index rows) ->  -- rowIndex
  ( Signal dom MultiplierState
  , Signal dom Bool  -- fetchTrigger
  , Signal dom Bool  -- rowReset
  , Signal dom Bool  -- rowEnable
  , Signal dom Bool  -- outputValid
  , Signal dom Bool  -- readyForInput
  )
matrixMultiplierStateMachine inputValid downStreamReady rowDone rowValid rowIndex =
  (state, fetchTrigger, rowReset, rowEnable, outputValid, readyForInput)
  where
    state = register MIdle nextState
    
    -- Pure function for state transitions
    stateTransition :: MultiplierState -> Bool -> Bool -> Bool -> Bool -> Index rows -> MultiplierState
    stateTransition currentState inValid downReady done fetched idx =
      case currentState of
        MIdle -> if inValid then MFetching else MIdle
        MFetching -> if fetched then MReset else MFetching  -- Wait for fetch!
        MReset -> MProcessing
        MProcessing -> if done 
                       then (if idx == maxBound then MDone else MFetching)
                       else MProcessing
        MDone -> if downReady then MIdle else MDone
    
    -- Apply the pure function to the signals
    nextState = stateTransition <$> state <*> inputValid <*> downStreamReady 
                                 <*> rowDone <*> rowValid <*> rowIndex
    
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

-- | Parallel 64-lane row processor.
--
-- This unit consumes one row of quantized mantissas and a full input vector,
-- and produces /one scalar dot-product result/ per row by processing
-- 64 columns per cycle. It internally keeps track of the current
-- column index and accumulates partial sums until the entire row
-- has been consumed.
--
-- = Control protocol
--
-- The control signals of this block are **level-sensitive**, not edge-sensitive:
--
--   * **reset :: Signal dom Bool**  
--     A /level/ signal.  
--     When 'True', the internal accumulator, the row-completion flag,
--     and the 64-column cyclical counter are synchronously reset to zero
--     on the next clock edge.  
--     It must be held high for exactly one cycle to start a new row.
--     (Holding it longer will simply keep the internal state cleared.)
--
--   * **enable :: Signal dom Bool**  
--     A /level/ signal.  
--     When 'True', the 64-column step is executed:
--       - the cyclical counter advances by 64,
--       - the 64 mantissa/column lane products are computed,
--       - the masked lane-sum is added into the accumulator (unless the last
--         column has been reached, in which case 0 is injected instead).
--     When 'False', the processor holds all internal registers (no progress).
--
-- Neither 'reset' nor 'enable' detect edges. The block reacts to their
-- instantaneous Boolean level at each cycle.
--
-- = Output protocol
--
--   * **output :: Signal dom FixedPoint**  
--     The scaled, accumulated result for the current row.  
--     Meaningful only when 'rowDone' is asserted.
--
--   * **rowDone :: Signal dom Bool**  
--     A /level/ signal that becomes 'True' for one cycle
--     when the processor has consumed the last column of the row.  
--     It is internally registered: asserted on the cycle *after* the
--     final 64-lane chunk completes.  
--     Automatically cleared on the next 'reset'.
--
--   * **acc :: Signal dom FixedPoint**  
--     Internal accumulator state, exposed for observability.
--
-- = Usage notes
--
-- * To start a new row, assert 'reset' = True for one cycle while keeping
--   'enable' = False.  
--   On the next cycle, deassert 'reset' and assert 'enable' to begin
--   processing.
--
-- * Keep 'enable' = True for as many cycles as needed. The block will
--   automatically detect when the last column group is processed.
--
-- * When 'rowDone' becomes True, the caller may read the final 'output'
--   and should assert 'reset' in the next cycle to prepare for the next row.
--
-- * Inputs ('row' and 'columnVec') are sampled combinationally each cycle.
--   They must stay valid while 'enable' is asserted.
--
-- This processor is purely synchronous with explicit state.
-- No edge-detection or hand-shake pulses are required: simply hold the
-- control lines at the appropriate levels each cycle.
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
    rowDone = register False nextRowDone
      where
        nextRowDone = mux reset (pure False) lastColumnFlag

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
      matrixMultiplierStateMachine validIn readyIn rowDone (pure True) rowIndex
      --                                                    ^^^^^^^^^^ Always ready
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
    matrixMultiplierStateMachine inputValid downStreamReady rowDone (pure True) rowIndex
    --                                                              ^^^^^^^^^^ Always ready
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

