module LLaMa2.Helpers.MatVecI8E
  ( matrixVectorMult
  , matrixMultiplierStub
  , matrixMultiplier
  , singleRowProcessor
  , cyclicalCounter
  , accumulator
  , MultiplierState(..)
  , matrixMultiplierStateMachine
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.ParamPack (MatI8E, RowI8E, dequantRowToF)
import LLaMa2.Helpers.FixedPoint (dotProductF)

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotProductRowI8E :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotProductRowI8E row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMult
  :: (KnownNat cols)
  => MatI8E rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMult byRows xF =
  map (`dotProductRowI8E` xF) byRows


-- Ready/Valid sequential façade for matrix-vector multiplication (STUB)
-- This stub adds proper ready/valid handshaking even though computation is combinational
matrixMultiplierStub
  :: forall dom rows cols .
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
matrixMultiplierStub validIn readyIn rowsQ vecIn = (outVec, validOut, readyOut)
  where
    -- Combinational result
    resultComb :: Signal dom (Vec rows FixedPoint)
    resultComb = matrixVectorMult rowsQ <$> vecIn

    -- Latch output when we accept input
    outVec :: Signal dom (Vec rows FixedPoint)
    outVec = regEn (repeat 0) (validIn .&&. readyOut) resultComb

    -- Valid out: high while output is waiting for downstream
    validOut :: Signal dom Bool
    validOut = register False $
          mux validOut (not <$> readyIn)  -- stay busy until downstream consumes
                    (validIn .&&. readyOut)  -- start busy on new input

    -- Ready out: can accept new input when idle
    readyOut :: Signal dom Bool
    readyOut = not <$> validOut

-- ============================================================================
-- SEQUENTIAL IMPLEMENTATION
-- ============================================================================

-- | A stateful column counter for tracking the current column index in a sequential
-- matrix-vector multiplication process. The counter cyclicylly increments when enabled
-- unless it gets reset, operating within a bounded range defined by the type parameter 'size'.
-- This function is used to select components from a row vector during sequential processing.
--
-- Inputs:
--   - reset: A signal that, when True, forces the counter to 0.
--   - enable: A signal that, when True, increments the counter cyclically.
--
-- Output:
--   - A signal carrying the current column index as an 'Index n' value.
--
cyclicalCounter :: (HiddenClockResetEnable dom, KnownNat size)
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index size)
cyclicalCounter reset enable = index
  where
    -- Compute next value *combinationally* from current and inputs
    nextIndex = mux (enable .&&. (index ./=. pure maxBound)) (index + 1) index
    next = mux (reset .||. (index .==. pure maxBound)) 0 nextIndex
    -- Register holds state; output is current (updated) value
    index = register 0 next

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

singleRowProcessor :: forall dom size.
  ( HiddenClockResetEnable dom
  , KnownNat size )
  => Signal dom Bool                           -- ^ reset for new row
  -> Signal dom Bool                           -- ^ enable
  -> Signal dom (RowI8E size)                  -- ^ input row
  -> Signal dom (Vec size FixedPoint)          -- ^ input column
  -> ( Signal dom FixedPoint                   -- ^ output scalar
     , Signal dom Bool                         -- ^ done flag
  )
singleRowProcessor reset enable row columnVec = (output, rowDone)
  where
    mant = fst <$> row
    expon = snd <$> row

    columnIndex = cyclicalCounter reset enable
    mantissa = (!!) <$> mant <*> columnIndex
    columnComponent = (!!) <$> columnVec <*> columnIndex  -- Select component using counter

    mantissaFP = fromIntegral <$> mantissa :: Signal dom FixedPoint
    inputValue = mantissaFP * columnComponent
    acc = accumulator reset enable inputValue
    output = liftA2 scalePow2F expon acc

    lastColumnFlag = (columnIndex .==. pure (maxBound :: Index size)) .&&. enable
    rowDone       = register False lastColumnFlag

-- | Sequential matrix-vector multiplication processor that computes the product
-- of a quantized matrix with an input vector in a row-by-row, column-by-column manner.
-- The function processes one column per cycle, completing one row before moving to the next.
--
-- Inputs:
-- - matrix: A quantized 2D array (QArray2D) containing the matrix in I8E format
-- - validIn: A signal indicating when the input vector is valid and processing should begin
-- - inputVec: A signal carrying the input vector of FixedPoint values
--
-- Outputs:
-- - outputVec: A signal carrying the resulting vector after matrix-vector multiplication
-- - validOut: A signal indicating when the output vector is valid (all rows processed)
-- - readyOut: A signal indicating when the processor is ready to accept new input
--
-- The processor operates as a finite state machine that:
-- 1. Waits for validIn to be asserted (IDLE state)
-- 2. Processes each row sequentially using singleRowProcessor (PROCESSING state)
-- 3. Outputs the complete result vector and asserts validOut (DONE state)
-- 4. Returns to IDLE state

-- | State for the matrix multiplier state machine
data MultiplierState = MIdle | MReset | MProcessing | MDone
  deriving (Show, Eq, Generic, NFDataX)

-- | State machine for matrix multiplier
-- Manages state transitions and control signals
matrixMultiplierStateMachine  :: forall dom rows .
  (HiddenClockResetEnable dom, KnownNat rows)
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index rows)
  -> (Signal dom MultiplierState, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)
matrixMultiplierStateMachine enable rowDone currentRow =
  (state, rowReset, rowEnable, validOut, readyIn)
  where
    state = register MIdle nextState

    lastRow = currentRow .==. pure (maxBound :: Index rows)

    nextState = mux (state .==. pure MIdle .&&. enable)
                    (pure MReset)
                    (mux (state .==. pure MReset)
                         (pure MProcessing)
                         (mux (state .==. pure MProcessing .&&. rowDone .&&. lastRow)
                              (pure MDone)
                              (mux (state .==. pure MProcessing .&&. rowDone .&&. (not <$> lastRow))
                                   (pure MReset)
                                   (mux (state .==. pure MDone)
                                        (pure MIdle)
                                        state))))

    rowReset = state .==. pure MReset
    -- Disable enable on the cycle when rowDone arrives
    rowEnable = (state .==. pure MProcessing) .&&. (not <$> rowDone) .&&. (not <$> rowReset)
    validOut = state .==. pure MDone
    readyIn = state .==. pure MIdle

-- | Sequential matrix-vector multiplication processor
matrixMultiplier
  :: forall dom rows cols .
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
matrixMultiplier validIn readyIn rowVectors inputVector = (outputVector, validOut, readyOut)
  where
    -- Row counter to track which row we're processing
    rowIndex = register (0 :: Index rows) nextRowIndex

    -- Current row from the matrix
    currentRow = (!!) rowVectors <$> rowIndex

    -- Single row processor
    (rowResult, rowDone) = singleRowProcessor rowReset rowEnable currentRow inputVector

    -- State machine controls the protocol
    (state, rowReset, rowEnable, validOut, readyOut) =
      matrixMultiplierStateMachine validIn rowDone rowIndex

    -- Increment row index when a row completes, but not on the last row
    nextRowIndex = mux (rowDone .&&. (rowIndex ./=. pure maxBound))
                       (rowIndex + 1)
                       (mux (state .==. pure MDone)
                            (pure 0)
                            rowIndex)

    -- Accumulate results into output vector
    -- When rowDone is high, store the result at the current row index
    outputVector = register (repeat 0) nextOutput
    nextOutput = mux rowDone
                     (replace <$> rowIndex <*> rowResult <*> outputVector)
                     outputVector
