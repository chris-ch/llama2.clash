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
  resultComb = matrixVectorMult rowsQ <$> vecIn
  
  -- State: busy when we have valid output waiting for downstream
  busy = register False nextBusy
  
  -- Accept new input when not busy and validIn is high
  acceptInput = validIn .&&. (not <$> busy)
  
  -- Output consumed when busy and downstream is ready
  outputConsumed = busy .&&. readyIn
  
  -- Next busy state: become busy on new input, stay busy until consumed
  nextBusy = mux acceptInput
                 (pure True)                    -- become busy on new input
                 (mux outputConsumed
                      (pure False)              -- become idle when output consumed
                      busy)                     -- otherwise maintain state
  
  -- Latch output when we accept input
  outVec :: Signal dom (Vec rows FixedPoint)
  outVec = regEn (repeat 0) acceptInput resultComb
  
  -- Valid out: high when we have data waiting for downstream
  validOut :: Signal dom Bool
  validOut = busy
  
  -- Ready out: can accept new input when not busy
  readyOut :: Signal dom Bool
  readyOut = not <$> busy

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

-- | State for the matrix multiplier state machine
data MultiplierState = MIdle | MReset | MProcessing | MDone
  deriving (Show, Eq, Generic, NFDataX)

-- | State machine for matrix multiplier
-- Manages state transitions and control signals
matrixMultiplierStateMachine  :: forall dom rows .
  (HiddenClockResetEnable dom, KnownNat rows)
  => Signal dom Bool
  -> Signal dom Bool -- readyIn from downstream
  -> Signal dom Bool  
  -> Signal dom (Index rows)
  -> (Signal dom MultiplierState, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)
matrixMultiplierStateMachine validIn readyInDownstream rowDone currentRow =
  (state, rowReset, rowEnable, validOut, readyOut)
  where
    state = register MIdle nextState

    lastRow = currentRow .==. pure (maxBound :: Index rows)

    -- Accept input when idle and validIn is high
    acceptInput = (state .==. pure MIdle) .&&. validIn

    -- Move to next state when done and downstream is ready
    outputAccepted = (state .==. pure MDone) .&&. readyInDownstream

    nextState = mux acceptInput
                    (pure MReset)
                    (mux (state .==. pure MReset)
                         (pure MProcessing)
                         (mux (state .==. pure MProcessing .&&. rowDone .&&. lastRow)
                              (pure MDone)
                              (mux (state .==. pure MProcessing .&&. rowDone .&&. (not <$> lastRow))
                                   (pure MReset)
                                   (mux outputAccepted
                                        (pure MIdle)
                                        state))))

    rowReset = state .==. pure MReset
    rowEnable = (state .==. pure MProcessing) .&&. (not <$> rowDone)
    validOut = state .==. pure MDone

    -- Ready to accept new input when idle
    readyOut = state .==. pure MIdle

-- | Sequential matrix-vector multiplication processor
matrixMultiplier
  :: forall dom rows cols .
     ( HiddenClockResetEnable dom
     , KnownNat cols, KnownNat rows
     )
  => Signal dom Bool        -- validIn
  -> Signal dom Bool        -- readyIn (downstream)
  -> MatI8E rows cols
  -> Signal dom (Vec cols FixedPoint)
  -> ( Signal dom (Vec rows FixedPoint)
     , Signal dom Bool      -- validOut
     , Signal dom Bool      -- readyOut
     )
matrixMultiplier validIn readyInDownstream rowVectors inputVector = (outputVector, validOut, readyOut)
  where
    -- Row counter
    rowIndex = register (0 :: Index rows) nextRowIndex
    currentRow = (!!) rowVectors <$> rowIndex

    -- Single-row processor
    (rowResult, rowDone) = singleRowProcessor rowReset rowEnable currentRow inputVector

    -- State machine controls the protocol
    (state, rowReset, rowEnable, validOut, readyOut) =
      matrixMultiplierStateMachine validIn readyInDownstream rowDone rowIndex

    -- Increment row index when row completes, reset after last row
    nextRowIndex = mux (rowDone .&&. (rowIndex ./=. pure maxBound))
                       (rowIndex + 1)
                       (mux ((state .==. pure MDone) .&&. readyInDownstream)
                            (pure 0)
                            rowIndex)

    -- Accumulate results into output vector
    outputVector = register (repeat 0) nextOutput
    nextOutput = mux rowDone
                     (replace <$> rowIndex <*> rowResult <*> outputVector)
                     outputVector
