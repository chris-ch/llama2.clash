module Model.Helpers.MatVecI8E
  ( matrixVectorMult
  , sequentialMatVecStub
  , matrixMultiplier
  , singleRowProcessor
  , columnComponentCounter
  , accumulator
  , rowStateMachine
  ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, scalePow2F)
import Model.Numeric.ParamPack (QArray2D(..), RowI8E, dequantRowToF)
import Model.Helpers.FixedPoint (dotProductF)

import Data.Proxy (Proxy(..))           -- for type-level Proxy

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotProductRowI8E :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotProductRowI8E row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMult
  :: (KnownNat cols)
  => QArray2D rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMult (QArray2D byRows) xF =
  map (`dotProductRowI8E` xF) byRows

-- Ready/Valid sequential façade for matrix-vector multiplication (STUB)
-- This stub adds proper ready/valid handshaking even though computation is combinational
sequentialMatVecStub
  :: forall dom rows cols .
     ( HiddenClockResetEnable dom
     , KnownNat cols, KnownNat rows
     )
  => QArray2D rows cols
  -> Signal dom (Bool, Vec cols FixedPoint)       -- ^ (validIn, inputVec)
  -> ( Signal dom (Vec rows FixedPoint)           -- ^ outputVec
     , Signal dom Bool                            -- ^ validOut
     , Signal dom Bool                            -- ^ readyOut
     )
sequentialMatVecStub (QArray2D rowsQ) inSig = (outVec, validOut, readyOut)
  where
    (validIn, vecIn) = unbundle inSig

    -- Compute result combinationally
    resultComb :: Signal dom (Vec rows FixedPoint)
    resultComb = matrixVectorMult (QArray2D rowsQ) <$> vecIn

    -- Latch input when we accept it
    outVec :: Signal dom (Vec rows FixedPoint)
    outVec = regEn (repeat 0) validIn resultComb

    -- ValidOut pulses one cycle after validIn (when computation "completes")
    validOut :: Signal dom Bool
    validOut = register False validIn

    -- State machine should ensure busy period
    state :: Signal dom Bool  -- False = ready/idle, True = busy
    state = register False $
      mux (validIn .&&. (not <$> state)) (pure True) $   -- Start only when idle and validIn
      mux state (pure False) state                        -- Complete next cycle

    -- Ready only when truly idle
    readyOut :: Signal dom Bool
    readyOut = register True $ not <$> (validIn .||. state)

-- ============================================================================
-- SEQUENTIAL IMPLEMENTATION
-- ============================================================================

-- | A stateful column counter for tracking the current column index in a sequential
-- matrix-vector multiplication process. The counter increments when enabled unless it gets reset,
-- operating within a bounded range defined by the type parameter 'n'.
-- This function is used to select components from a row vector during sequential processing.
--
-- Inputs:
--   - reset: A signal that, when True, forces the counter to 0.
--   - enable: A signal that, when True, increments the counter if it is below its maximum value.
--
-- Output:
--   - A signal carrying the current column index as an 'Index n' value.
--
-- The counter updates the index based on the
-- reset and enable signals. It ensures the index does not exceed the maximum bound defined
-- by 'n', making it suitable for iterating over columns in a matrix row.
columnComponentCounter :: (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index n)
columnComponentCounter reset enable = index
  where
    -- Compute next value *combinationally* from current and inputs
    incrementFlag = enable .&&. (index .<. pure maxBound)
    nextIndex = mux incrementFlag (index + 1) index
    next = mux reset 0 nextIndex

    -- Register holds state; output is current (updated) value
    index = register 0 next

-- | An accumulator for sequentially summing FixedPoint values in a matrix-vector
-- multiplication process. It accumulates products when enabled, can be reset,
-- and holds its value otherwise. This function is critical for computing the dot product
-- of a row vector with an input vector in a cycle-by-cycle manner.
--
-- Inputs:
--   - reset: A signal that, when True, resets the accumulator to the input value.
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
    next = mux reset input $
           mux enable (acc + input) acc

    -- Register holds state; output is current (updated) value
    acc = register 0 next

singleRowProcessor :: forall dom cols .
  ( HiddenClockResetEnable dom
  , KnownNat cols )
  => Signal dom Bool                           -- ^ reset for new row
  -> Signal dom Bool                           -- ^ enable
  -> Signal dom (RowI8E cols)                  -- ^ input row
  -> Signal dom (Vec cols FixedPoint)          -- ^ input column
  -> ( Signal dom FixedPoint                   -- ^ output scalar
     , Signal dom Bool )                       -- ^ done flag
singleRowProcessor reset enable row columnVec = (output, rowDone)
  where
    mant = fst <$> row
    expon = snd <$> row
    
    columnIndex = columnComponentCounter reset enable
    mantissa = (!!) <$> mant <*> columnIndex
    columnComponent = (!!) <$> columnVec <*> columnIndex  -- Select component using counter
    
    mantissaFP = fromIntegral <$> mantissa :: Signal dom FixedPoint
    inputValue = mantissaFP * columnComponent
    acc = accumulator reset enable inputValue
    output = liftA2 scalePow2F expon acc

    lastColumnFlag = (columnIndex .==. pure (maxBound :: Index cols)) .&&. enable
    rowDone       = register False lastColumnFlag

-- ============================================================================
-- Row State Machine
-- ============================================================================

data RowState rows
  = RowIdle
  | RowProcessing (Index rows)
  deriving (Generic, NFDataX, Show, Eq)

rowStateMachine ::
  forall dom rows .
  ( HiddenClockResetEnable dom
  , KnownNat rows
  ) =>
  Signal dom Bool           -- ^ validIn: start signal
  -> Signal dom Bool        -- ^ rowDone: completion signal for current row
  -> ( Signal dom Bool       -- ^ busy
     , Signal dom (Index rows) -- ^ rowIdx
     , Signal dom Bool       -- ^ clearRow pulse
     )
rowStateMachine validIn rowDone = (busy, rowIdx, clearRow)
  where
    maxRow :: Index rows
    maxRow = fromIntegral (natVal (Proxy :: Proxy rows) - 1)

    -- Mealy step function: returns next state + output
    step :: RowState rows -> (Bool, Bool) -> (RowState rows, (Bool, Index rows, Bool))
    step s (vIn, rDone) = case s of
      RowIdle ->
        if vIn
          then (RowProcessing 0, (True, 0, True))  -- Start processing, pulse clearRow
          else (RowIdle, (False, 0, True))         -- Idle, keep clearRow True
      RowProcessing n ->
        if rDone
          then if n == maxRow
                 then (RowIdle, (True, n, True))  -- Done with last row, stay busy one more cycle
                 else (RowProcessing (n+1), (True, n, True)) -- Advance to next row, output current rowIdx
          else (RowProcessing n, (True, n, False)) -- Stay in current row, no pulse

    -- FSM output signal
    out :: Signal dom (Bool, Index rows, Bool)
    out = mealy step RowIdle (bundle (validIn, rowDone))

    -- Extract outputs
    busy     = (\(b,_,_) -> b) <$> out
    rowIdx   = (\(_,r,_) -> r) <$> out
    clearRow = (\(_,_,c) -> c) <$> out

-- ============================================================================
-- Main Sequential Matrix-Vector Multiplication
-- ============================================================================

type State rows = (Bool, Index rows)
    
matrixMultiplier
  :: forall dom rows cols .
     ( HiddenClockResetEnable dom
     , KnownNat rows
     , KnownNat cols
     )
  => QArray2D rows cols                    -- ^ matrix
  -> Signal dom Bool                       -- ^ enable)
  ->  Signal dom (Vec cols FixedPoint)     -- ^ input vector
  -> ( Signal dom (Vec rows FixedPoint)    -- ^ output vector
     , Signal dom Bool                     -- ^ validOut
     , Signal dom Bool                     -- ^ readyOut
     , Signal dom FixedPoint
     , Signal dom Bool
     , Signal dom (State rows)
     )
matrixMultiplier (QArray2D rowsQ) enable xVec = (outVec, validOut, readyOut, yOutRow, doneCurrentRow, stateForDebug)
  where

    -- Use the cleaner row state machine
    (busy, rowIdx, clearRow) = rowStateMachine enable doneCurrentRow

    -- Current row being processed
    currentRow :: Signal dom (RowI8E cols)
    currentRow = (!!) rowsQ <$> rowIdx

    -- Only enable processing when busy
    enableProcessing :: Signal dom Bool
    enableProcessing = busy .&&. enable

    -- Sequentially compute dot product for the current row
    (yOutRow, doneCurrentRow) = singleRowProcessor clearRow enableProcessing currentRow xVec

    -- Output vector register: store each row result as it completes
    outVec :: Signal dom (Vec rows FixedPoint)
    outVec = regEn (repeat 0) doneCurrentRow outVecNext

    -- Build next output vector by updating current row
    outVecNext :: Signal dom (Vec rows FixedPoint)
    outVecNext = liftA3 (\vec idx y -> replace idx y vec) outVec rowIdx yOutRow

    -- Done when last row finishes
    doneAllRows :: Signal dom Bool
    doneAllRows = liftA2 (\idx doneRow -> doneRow && idx == maxBound) rowIdx doneCurrentRow

    -- Valid pulses when the full matrix is done
    validOut :: Signal dom Bool
    validOut = register False doneAllRows

    -- Ready signal: high when not busy
    readyOut :: Signal dom Bool
    readyOut = not <$> busy

    -- For debug compatibility
    stateForDebug = bundle (busy, rowIdx)
