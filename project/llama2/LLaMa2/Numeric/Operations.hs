module LLaMa2.Numeric.Operations
  ( accumulator
  , MultiplierState(..)
  , matrixMultiplierStateMachine
  , parallel64RowMatrixMultiplier
  , cyclicalCounter64
  , parallel64RowProcessor
  , parallelRowMatrixMultiplier
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import qualified Simulation.MatVecSim (matrixMultiplierStub)

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

-- | Parallel row processor with hardcoded 64 lanes
-- Processes 64 columns per cycle
parallel64RowProcessor :: forall dom size.
  ( HiddenClockResetEnable dom
  , KnownNat size)
  => Signal dom Bool                           -- ^ reset for new row
  -> Signal dom Bool                           -- ^ enable
  -> Signal dom (RowI8E size)                  -- ^ input row
  -> Signal dom (Vec size FixedPoint)          -- ^ input column
  -> ( Signal dom FixedPoint                   -- ^ output scalar
     , Signal dom Bool                         -- ^ done flag
  )
parallel64RowProcessor reset enable row columnVec = (output, rowDone)
  where
    mant = fst <$> row
    expon = snd <$> row

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

    acc = accumulator reset enable accInput
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
parallel64RowMatrixMultiplier validIn readyInDownstream rowVectors inputVector =
  (outputVector, validOut, readyOut)
  where
    -- Row counter
    rowIndex = register (0 :: Index rows) nextRowIndex
    currentRow = (!!) rowVectors <$> rowIndex

    -- Parallel 64-lane row processor
    (rowResult, rowDone) = parallel64RowProcessor rowReset rowEnable currentRow inputVector

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
