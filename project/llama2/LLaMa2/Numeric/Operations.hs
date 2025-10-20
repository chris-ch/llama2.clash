module LLaMa2.Numeric.Operations
  (
    matrixMultiplier
  , singleRowProcessor
  , cyclicalCounter
  , accumulator
  , MultiplierState(..)
  , matrixMultiplierStateMachine
  , parallel32RowMatrixMultiplier
  , cyclicalCounter32
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import qualified Simulation.MatVecSim

parallel32RowMatrixMultiplier' :: forall dom rows cols .
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
parallel32RowMatrixMultiplier' = Simulation.MatVecSim.matrixMultiplierStub

matrixMultiplier' :: forall dom rows cols .
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
matrixMultiplier' = Simulation.MatVecSim.matrixMultiplierStub

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
    output = scalePow2F <$> expon <*> acc

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
-- Handshaking via ready/valid signals
--
--          | ----validIn---> |            | ----validOut---> |
-- Upstream |                 | Multiplier |                  | Downstream
--          | <---readyOut--- |            | <---readyIn----- |
--
matrixMultiplier :: forall dom rows cols .
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

-- | Process a single column of a row (one lane)
singleLaneProcessor :: forall dom . Signal dom (Signed 8)           -- mantissa element
  -> Signal dom FixedPoint           -- column element
  -> Signal dom FixedPoint           -- product
singleLaneProcessor mantissa columnComponent = inputValue
  where
    mantissaFP = fromIntegral <$> mantissa :: Signal dom FixedPoint
    inputValue = mantissaFP * columnComponent

-- | Parallel row processor with hardcoded 32 lanes
-- Processes 32 columns per cycle
parallel32RowProcessor :: forall dom size.
  ( HiddenClockResetEnable dom
  , KnownNat size)
  => Signal dom Bool                           -- ^ reset for new row
  -> Signal dom Bool                           -- ^ enable
  -> Signal dom (RowI8E size)                  -- ^ input row
  -> Signal dom (Vec size FixedPoint)          -- ^ input column
  -> ( Signal dom FixedPoint                   -- ^ output scalar
     , Signal dom Bool                         -- ^ done flag
  )
parallel32RowProcessor reset enable row columnVec = (output, rowDone)
  where
    mant = fst <$> row
    expon = snd <$> row

    -- Column index advances by 32 each cycle
    columnIndex :: Signal dom (Index size)
    columnIndex = cyclicalCounter32 reset enable

    -- Extract mantissas for 32 lanes
    m0  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 0)
    m1  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 1)
    m2  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 2)
    m3  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 3)
    m4  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 4)
    m5  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 5)
    m6  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 6)
    m7  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 7)
    m8  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 8)
    m9  = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 9)
    m10 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 10)
    m11 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 11)
    m12 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 12)
    m13 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 13)
    m14 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 14)
    m15 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 15)
    m16 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 16)
    m17 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 17)
    m18 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 18)
    m19 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 19)
    m20 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 20)
    m21 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 21)
    m22 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 22)
    m23 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 23)
    m24 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 24)
    m25 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 25)
    m26 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 26)
    m27 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 27)
    m28 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 28)
    m29 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 29)
    m30 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 30)
    m31 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 31)

    -- Extract column components for 32 lanes
    c0  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 0)
    c1  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 1)
    c2  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 2)
    c3  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 3)
    c4  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 4)
    c5  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 5)
    c6  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 6)
    c7  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 7)
    c8  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 8)
    c9  = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 9)
    c10 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 10)
    c11 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 11)
    c12 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 12)
    c13 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 13)
    c14 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 14)
    c15 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 15)
    c16 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 16)
    c17 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 17)
    c18 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 18)
    c19 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 19)
    c20 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 20)
    c21 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 21)
    c22 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 22)
    c23 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 23)
    c24 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 24)
    c25 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 25)
    c26 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 26)
    c27 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 27)
    c28 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 28)
    c29 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 29)
    c30 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 30)
    c31 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 31)

    -- Compute products for all 32 lanes
    p0  = singleLaneProcessor m0 c0
    p1  = singleLaneProcessor m1 c1
    p2  = singleLaneProcessor m2 c2
    p3  = singleLaneProcessor m3 c3
    p4  = singleLaneProcessor m4 c4
    p5  = singleLaneProcessor m5 c5
    p6  = singleLaneProcessor m6 c6
    p7  = singleLaneProcessor m7 c7
    p8  = singleLaneProcessor m8 c8
    p9  = singleLaneProcessor m9 c9
    p10 = singleLaneProcessor m10 c10
    p11 = singleLaneProcessor m11 c11
    p12 = singleLaneProcessor m12 c12
    p13 = singleLaneProcessor m13 c13
    p14 = singleLaneProcessor m14 c14
    p15 = singleLaneProcessor m15 c15
    p16 = singleLaneProcessor m16 c16
    p17 = singleLaneProcessor m17 c17
    p18 = singleLaneProcessor m18 c18
    p19 = singleLaneProcessor m19 c19
    p20 = singleLaneProcessor m20 c20
    p21 = singleLaneProcessor m21 c21
    p22 = singleLaneProcessor m22 c22
    p23 = singleLaneProcessor m23 c23
    p24 = singleLaneProcessor m24 c24
    p25 = singleLaneProcessor m25 c25
    p26 = singleLaneProcessor m26 c26
    p27 = singleLaneProcessor m27 c27
    p28 = singleLaneProcessor m28 c28
    p29 = singleLaneProcessor m29 c29
    p30 = singleLaneProcessor m30 c30
    p31 = singleLaneProcessor m31 c31

    -- Check validity for each lane (for partial last iteration)
    isValid :: Int -> Signal dom Bool
    isValid offset = (\idx -> fromEnum idx + offset <= fromEnum (maxBound :: Index size)) <$> columnIndex

    v0  = isValid 0
    v1  = isValid 1
    v2  = isValid 2
    v3  = isValid 3
    v4  = isValid 4
    v5  = isValid 5
    v6  = isValid 6
    v7  = isValid 7
    v8  = isValid 8
    v9  = isValid 9
    v10 = isValid 10
    v11 = isValid 11
    v12 = isValid 12
    v13 = isValid 13
    v14 = isValid 14
    v15 = isValid 15
    v16 = isValid 16
    v17 = isValid 17
    v18 = isValid 18
    v19 = isValid 19
    v20 = isValid 20
    v21 = isValid 21
    v22 = isValid 22
    v23 = isValid 23
    v24 = isValid 24
    v25 = isValid 25
    v26 = isValid 26
    v27 = isValid 27
    v28 = isValid 28
    v29 = isValid 29
    v30 = isValid 30
    v31 = isValid 31

    -- Mask invalid lanes
    mp0  = mux v0 p0 0
    mp1  = mux v1 p1 0
    mp2  = mux v2 p2 0
    mp3  = mux v3 p3 0
    mp4  = mux v4 p4 0
    mp5  = mux v5 p5 0
    mp6  = mux v6 p6 0
    mp7  = mux v7 p7 0
    mp8  = mux v8 p8 0
    mp9  = mux v9 p9 0
    mp10 = mux v10 p10 0
    mp11 = mux v11 p11 0
    mp12 = mux v12 p12 0
    mp13 = mux v13 p13 0
    mp14 = mux v14 p14 0
    mp15 = mux v15 p15 0
    mp16 = mux v16 p16 0
    mp17 = mux v17 p17 0
    mp18 = mux v18 p18 0
    mp19 = mux v19 p19 0
    mp20 = mux v20 p20 0
    mp21 = mux v21 p21 0
    mp22 = mux v22 p22 0
    mp23 = mux v23 p23 0
    mp24 = mux v24 p24 0
    mp25 = mux v25 p25 0
    mp26 = mux v26 p26 0
    mp27 = mux v27 p27 0
    mp28 = mux v28 p28 0
    mp29 = mux v29 p29 0
    mp30 = mux v30 p30 0
    mp31 = mux v31 p31 0

    -- Tree reduction sum (5 levels for 32 inputs)
    -- Level 1: 32 -> 16
    s1_0  = mp0 + mp1
    s1_1  = mp2 + mp3
    s1_2  = mp4 + mp5
    s1_3  = mp6 + mp7
    s1_4  = mp8 + mp9
    s1_5  = mp10 + mp11
    s1_6  = mp12 + mp13
    s1_7  = mp14 + mp15
    s1_8  = mp16 + mp17
    s1_9  = mp18 + mp19
    s1_10 = mp20 + mp21
    s1_11 = mp22 + mp23
    s1_12 = mp24 + mp25
    s1_13 = mp26 + mp27
    s1_14 = mp28 + mp29
    s1_15 = mp30 + mp31

    -- Level 2: 16 -> 8
    s2_0 = s1_0 + s1_1
    s2_1 = s1_2 + s1_3
    s2_2 = s1_4 + s1_5
    s2_3 = s1_6 + s1_7
    s2_4 = s1_8 + s1_9
    s2_5 = s1_10 + s1_11
    s2_6 = s1_12 + s1_13
    s2_7 = s1_14 + s1_15

    -- Level 3: 8 -> 4
    s3_0 = s2_0 + s2_1
    s3_1 = s2_2 + s2_3
    s3_2 = s2_4 + s2_5
    s3_3 = s2_6 + s2_7

    -- Level 4: 4 -> 2
    s4_0 = s3_0 + s3_1
    s4_1 = s3_2 + s3_3

    -- Level 5: 2 -> 1
    laneSum = s4_0 + s4_1

    -- Accumulate sum
    acc = accumulator reset enable laneSum
    output = scalePow2F <$> expon <*> acc

    -- Done when index + 31 >= maxBound
    lastColumnFlag = ((\idx -> fromEnum idx + 31 >= fromEnum (maxBound :: Index size)) <$> columnIndex) .&&. enable
    rowDone = register False lastColumnFlag

cyclicalCounter32 :: forall dom size . (HiddenClockResetEnable dom, KnownNat size)
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index size)
cyclicalCounter32 reset enable = index
  where
    maxBoundVal = pure (fromEnum (maxBound :: Index size)) :: Signal dom Int
    indexInt = fromIntegral <$> index :: Signal dom Int
    nextIndexInt = mux enable
                       (mux (indexInt + 32 .>. maxBoundVal)
                            maxBoundVal
                            (indexInt + 32))
                       indexInt
    nextIndex = toEnum <$> mux reset (pure 0) nextIndexInt :: Signal dom (Index size)
    index = register 0 nextIndex

-- | Parallel 32-lane matrix-vector multiplication processor
-- Uses parallel32RowProcessor to compute 32 columns per cycle
-- 
-- Handshaking via ready/valid signals
--
--          | ----validIn---> |            | ----validOut---> |
-- Upstream |                 | Multiplier |                  | Downstream
--          | <---readyOut--- |            | <---readyIn----- |
--
parallel32RowMatrixMultiplier :: forall dom rows cols .
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
parallel32RowMatrixMultiplier validIn readyInDownstream rowVectors inputVector = 
  (outputVector, validOut, readyOut)
  where
    -- Row counter
    rowIndex = register (0 :: Index rows) nextRowIndex
    currentRow = (!!) rowVectors <$> rowIndex

    -- Parallel 32-lane row processor
    (rowResult, rowDone) = parallel32RowProcessor rowReset rowEnable currentRow inputVector

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
            
-- | Helper: safely compute index with offset, clamping to valid range
-- Fixed to avoid out-of-bounds errors by checking before calling toEnum
addOffset :: forall size . KnownNat size => Index size -> Int -> Index size
addOffset idx offset =
  let idxInt = fromEnum idx
      newIdx = idxInt + offset
      maxIdx = fromEnum (maxBound :: Index size)
  in toEnum (min newIdx maxIdx)  -- Clamp BEFORE calling toEnum
