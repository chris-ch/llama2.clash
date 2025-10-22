module LLaMa2.Numeric.Operations
  ( accumulator
  , MultiplierState(..)
  , matrixMultiplierStateMachine
  , parallel64RowMatrixMultiplier
  , parallel32RowMatrixMultiplier
  , cyclicalCounter64
  , parallel64RowProcessor
  , parallel32RowProcessor
  , parallelRowMatrixMultiplier
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import qualified Simulation.MatVecSim

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

parallelRowMatrixMultiplier' :: forall dom rows cols .
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
parallelRowMatrixMultiplier' = Simulation.MatVecSim.matrixMultiplierStub

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
    accInput = mux rowDone 0 laneSum

    acc = accumulator reset enable accInput
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

    -- Extract mantissas for 64 lanes
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
    m32 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 32)
    m33 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 33)
    m34 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 34)
    m35 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 35)
    m36 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 36)
    m37 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 37)
    m38 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 38)
    m39 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 39)
    m40 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 40)
    m41 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 41)
    m42 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 42)
    m43 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 43)
    m44 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 44)
    m45 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 45)
    m46 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 46)
    m47 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 47)
    m48 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 48)
    m49 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 49)
    m50 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 50)
    m51 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 51)
    m52 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 52)
    m53 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 53)
    m54 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 54)
    m55 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 55)
    m56 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 56)
    m57 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 57)
    m58 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 58)
    m59 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 59)
    m60 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 60)
    m61 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 61)
    m62 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 62)
    m63 = (!!) <$> mant <*> (addOffset <$> columnIndex <*> pure 63)

    -- Extract column components for 64 lanes
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
    c32 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 32)
    c33 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 33)
    c34 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 34)
    c35 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 35)
    c36 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 36)
    c37 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 37)
    c38 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 38)
    c39 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 39)
    c40 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 40)
    c41 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 41)
    c42 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 42)
    c43 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 43)
    c44 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 44)
    c45 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 45)
    c46 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 46)
    c47 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 47)
    c48 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 48)
    c49 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 49)
    c50 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 50)
    c51 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 51)
    c52 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 52)
    c53 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 53)
    c54 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 54)
    c55 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 55)
    c56 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 56)
    c57 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 57)
    c58 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 58)
    c59 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 59)
    c60 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 60)
    c61 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 61)
    c62 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 62)
    c63 = (!!) <$> columnVec <*> (addOffset <$> columnIndex <*> pure 63)

    -- Compute products for all 64 lanes
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
    p32 = singleLaneProcessor m32 c32
    p33 = singleLaneProcessor m33 c33
    p34 = singleLaneProcessor m34 c34
    p35 = singleLaneProcessor m35 c35
    p36 = singleLaneProcessor m36 c36
    p37 = singleLaneProcessor m37 c37
    p38 = singleLaneProcessor m38 c38
    p39 = singleLaneProcessor m39 c39
    p40 = singleLaneProcessor m40 c40
    p41 = singleLaneProcessor m41 c41
    p42 = singleLaneProcessor m42 c42
    p43 = singleLaneProcessor m43 c43
    p44 = singleLaneProcessor m44 c44
    p45 = singleLaneProcessor m45 c45
    p46 = singleLaneProcessor m46 c46
    p47 = singleLaneProcessor m47 c47
    p48 = singleLaneProcessor m48 c48
    p49 = singleLaneProcessor m49 c49
    p50 = singleLaneProcessor m50 c50
    p51 = singleLaneProcessor m51 c51
    p52 = singleLaneProcessor m52 c52
    p53 = singleLaneProcessor m53 c53
    p54 = singleLaneProcessor m54 c54
    p55 = singleLaneProcessor m55 c55
    p56 = singleLaneProcessor m56 c56
    p57 = singleLaneProcessor m57 c57
    p58 = singleLaneProcessor m58 c58
    p59 = singleLaneProcessor m59 c59
    p60 = singleLaneProcessor m60 c60
    p61 = singleLaneProcessor m61 c61
    p62 = singleLaneProcessor m62 c62
    p63 = singleLaneProcessor m63 c63

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
    v32 = isValid 32
    v33 = isValid 33
    v34 = isValid 34
    v35 = isValid 35
    v36 = isValid 36
    v37 = isValid 37
    v38 = isValid 38
    v39 = isValid 39
    v40 = isValid 40
    v41 = isValid 41
    v42 = isValid 42
    v43 = isValid 43
    v44 = isValid 44
    v45 = isValid 45
    v46 = isValid 46
    v47 = isValid 47
    v48 = isValid 48
    v49 = isValid 49
    v50 = isValid 50
    v51 = isValid 51
    v52 = isValid 52
    v53 = isValid 53
    v54 = isValid 54
    v55 = isValid 55
    v56 = isValid 56
    v57 = isValid 57
    v58 = isValid 58
    v59 = isValid 59
    v60 = isValid 60
    v61 = isValid 61
    v62 = isValid 62
    v63 = isValid 63

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
    mp32 = mux v32 p32 0
    mp33 = mux v33 p33 0
    mp34 = mux v34 p34 0
    mp35 = mux v35 p35 0
    mp36 = mux v36 p36 0
    mp37 = mux v37 p37 0
    mp38 = mux v38 p38 0
    mp39 = mux v39 p39 0
    mp40 = mux v40 p40 0
    mp41 = mux v41 p41 0
    mp42 = mux v42 p42 0
    mp43 = mux v43 p43 0
    mp44 = mux v44 p44 0
    mp45 = mux v45 p45 0
    mp46 = mux v46 p46 0
    mp47 = mux v47 p47 0
    mp48 = mux v48 p48 0
    mp49 = mux v49 p49 0
    mp50 = mux v50 p50 0
    mp51 = mux v51 p51 0
    mp52 = mux v52 p52 0
    mp53 = mux v53 p53 0
    mp54 = mux v54 p54 0
    mp55 = mux v55 p55 0
    mp56 = mux v56 p56 0
    mp57 = mux v57 p57 0
    mp58 = mux v58 p58 0
    mp59 = mux v59 p59 0
    mp60 = mux v60 p60 0
    mp61 = mux v61 p61 0
    mp62 = mux v62 p62 0
    mp63 = mux v63 p63 0

    -- Tree reduction sum (6 levels for 64 inputs)
    -- Level 1: 64 -> 32
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
    s1_16 = mp32 + mp33
    s1_17 = mp34 + mp35
    s1_18 = mp36 + mp37
    s1_19 = mp38 + mp39
    s1_20 = mp40 + mp41
    s1_21 = mp42 + mp43
    s1_22 = mp44 + mp45
    s1_23 = mp46 + mp47
    s1_24 = mp48 + mp49
    s1_25 = mp50 + mp51
    s1_26 = mp52 + mp53
    s1_27 = mp54 + mp55
    s1_28 = mp56 + mp57
    s1_29 = mp58 + mp59
    s1_30 = mp60 + mp61
    s1_31 = mp62 + mp63

    -- Level 2: 32 -> 16
    s2_0  = s1_0 + s1_1
    s2_1  = s1_2 + s1_3
    s2_2  = s1_4 + s1_5
    s2_3  = s1_6 + s1_7
    s2_4  = s1_8 + s1_9
    s2_5  = s1_10 + s1_11
    s2_6  = s1_12 + s1_13
    s2_7  = s1_14 + s1_15
    s2_8  = s1_16 + s1_17
    s2_9  = s1_18 + s1_19
    s2_10 = s1_20 + s1_21
    s2_11 = s1_22 + s1_23
    s2_12 = s1_24 + s1_25
    s2_13 = s1_26 + s1_27
    s2_14 = s1_28 + s1_29
    s2_15 = s1_30 + s1_31

    -- Level 3: 16 -> 8
    s3_0 = s2_0 + s2_1
    s3_1 = s2_2 + s2_3
    s3_2 = s2_4 + s2_5
    s3_3 = s2_6 + s2_7
    s3_4 = s2_8 + s2_9
    s3_5 = s2_10 + s2_11
    s3_6 = s2_12 + s2_13
    s3_7 = s2_14 + s2_15

    -- Level 4: 8 -> 4
    s4_0 = s3_0 + s3_1
    s4_1 = s3_2 + s3_3
    s4_2 = s3_4 + s3_5
    s4_3 = s3_6 + s3_7

    -- Level 5: 4 -> 2
    s5_0 = s4_0 + s4_1
    s5_1 = s4_2 + s4_3

    -- Level 6: 2 -> 1
    laneSum = s5_0 + s5_1

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
