module Model.Helpers.MatVecI8E
  ( matrixVectorMult
  , sequentialMatVecStub
  , sequentialMatVec
  , sequentialMatVecOneRow
  , matVecRowSeq
  ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, scalePow2F, Mantissa)
import Model.Numeric.ParamPack (QArray2D(..), RowI8E, dequantRowToF)
import Model.Helpers.FixedPoint (dotProductF)

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotRowI8E_Fixed :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotRowI8E_Fixed row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMult
  :: (KnownNat cols)
  => QArray2D rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMult (QArray2D rowsQ) xF =
  map (`dotRowI8E_Fixed` xF) rowsQ

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
    
    -- State machine: IDLE -> COMPUTING -> DONE
    -- We simulate a 1-cycle computation delay for proper handshaking
    state :: Signal dom Bool  -- False = ready/idle, True = busy
    state = register False $ 
      mux validIn (pure True) $           -- Start on validIn
      mux state (pure False) state        -- Complete next cycle
    
    -- Latch input when we accept it
    outVec :: Signal dom (Vec rows FixedPoint)
    outVec = regEn (repeat 0) validIn resultComb
    
    -- ValidOut pulses one cycle after validIn (when computation "completes")
    validOut :: Signal dom Bool
    validOut = register False validIn
    
    -- Ready when idle (not busy)
    readyOut :: Signal dom Bool
    readyOut = not <$> state

-- ============================================================================
-- SEQUENTIAL IMPLEMENTATION
-- ============================================================================

-- | Row engine: processes one row sequentially, column by column
--   Protocol:
--   - clear: pulse to reset internal state for new row
--   - en: one step (consumes one column) when True
--   - lastCol: asserted with en on the final column; 'done' pulses next cycle
--   - row: static I8E row for this computation
--   - x: stream of FixedPoint activations (aligned to en)
matVecRowSeq
  :: forall dom n
   . (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool                           -- ^ clear
  -> Signal dom Bool                           -- ^ en
  -> Signal dom Bool                           -- ^ lastCol
  -> Signal dom (RowI8E n)                     -- ^ (mant, e) -- static over the row
  -> Signal dom FixedPoint                     -- ^ x stream (aligned to en)
  -> ( Signal dom FixedPoint                   -- ^ result (valid when done=1)
     , Signal dom Bool )                       -- ^ done (one-cycle pulse)
matVecRowSeq clear en lastCol rowSig xSig = (yOut, done)
 where
  -- Unpack row (assume static over a row operation)
  mant = fst <$> rowSig
  eRow = snd <$> rowSig

  -- Column index
  col :: Signal dom (Index n)
  col = mealy (\i (cl, step) ->
                 let i0 = if cl then 0 else i
                     iN = if step then (if i0 == maxBound then i0 else succ i0) else i0
                 in (iN, i0))
              0
              (bundle (clear, en))

  -- Read mantissa at current column
  qNow :: Signal dom Mantissa
  qNow = (!!) <$> mant <*> col

  -- Accumulator in FixedPoint domain
  acc :: Signal dom FixedPoint
  acc = regEn 0 (en .||. clear)
        (mux clear 0 (acc + (fromIntegral <$> qNow) * xSig))

  done = regEn False en lastCol
  
  -- Fuse the row exponent once at the end
  yOut = regEn 0 done (scalePow2F <$> eRow <*> acc)

-- | One row sequential matvec (streamed xVec, one element per cycle)
sequentialMatVecOneRow ::
  forall dom cols .
  ( HiddenClockResetEnable dom
  , KnownNat cols )
  => Signal dom (RowI8E cols)                  -- ^ row as a signal
  -> Signal dom (Bool, Vec cols FixedPoint)    -- ^ (validIn, input vector)
  -> Signal dom Bool                           -- ^ clear (reset for new row)
  -> ( Signal dom FixedPoint                   -- ^ output scalar
     , Signal dom Bool )                       -- ^ done pulse
sequentialMatVecOneRow rowSig inSig clearSig = (yOut, done)
 where
  (validIn, xVec) = unbundle inSig

  -- Column index sequencer with clear
  colIdx :: Signal dom (Index cols)
  colIdx = mealy step 0 (bundle (validIn, clearSig))
    where
      step i (vIn, clr) =
        let i0
              | clr = 0                         -- Reset on clear
              | vIn && i == maxBound = 0        -- Wrap at end
              | vIn = succ i                    -- Advance
              | otherwise = i                   -- Hold
        in  (i0, i)

  -- Current input element from xVec
  xElem = (!!) <$> xVec <*> colIdx

  -- Control signals for row engine
  enSig      = validIn
  lastColSig = (colIdx .==. pure (maxBound :: Index cols)) .&&. validIn
  clearRowSig = clearSig

  -- Call the row engine
  (yOut, done) = matVecRowSeq clearRowSig enSig lastColSig rowSig xElem

-- | Sequential matrix-vector multiply over all rows
--   Produces full output vector, one row at a time
--   Compatible with the stub's interface (ready/valid handshake)
sequentialMatVec
  :: forall dom rows cols .
     ( HiddenClockResetEnable dom
     , KnownNat rows
     , KnownNat cols 
     )
  => QArray2D rows cols
  -> Signal dom (Bool, Vec cols FixedPoint)    -- ^ (validIn, input vector)
  -> ( Signal dom (Vec rows FixedPoint)        -- ^ output vector
     , Signal dom Bool                         -- ^ validOut (one-cycle pulse)
     , Signal dom Bool                         -- ^ readyOut
     )
sequentialMatVec (QArray2D rowsQ) inSig = (outVec, validOut, readyOut)
 where
  (validIn, xVec) = unbundle inSig

  -- Row index state machine with active flag
  (rowIdx, isActive) = unbundle $ mealy stepRow (0, False) (bundle (validIn, rowDone))
    where
      stepRow (i, active) (start, done) =
        let 
            -- Start on validIn pulse
            newActive = if start then True
                       else if done && i == maxBound then False
                       else active
            
            -- Row counter logic
            i'
              | start = 0                              -- Reset on start
              | active && done && i /= maxBound = succ i  -- Advance when row completes
              | otherwise = i
        in ((i', newActive), (i, newActive))

  -- Current row from matrix
  curRow :: Signal dom (RowI8E cols)
  curRow = (!!) rowsQ <$> rowIdx

  -- Generate clear pulse when starting a new row
  rowIdxPrev = register 0 rowIdx
  clearRow = (rowIdx ./=. rowIdxPrev) .||. validIn

  -- Latch input vector when starting computation
  xVecLatched :: Signal dom (Vec cols FixedPoint)
  xVecLatched = regEn (repeat 0) validIn xVec

  -- One-row sequential matvec (use LATCHED input with clear signal)
  (yRow, rowDone) = sequentialMatVecOneRow curRow (bundle (isActive, xVecLatched)) clearRow

  -- Accumulate results into output vector
  outMem :: Signal dom (Vec rows FixedPoint)
  outMem = regEn (repeat 0) rowDone
             (replace <$> rowIdx <*> yRow <*> outMem)

  outVec = outMem

  -- Pulse when last row has finished
  validOut = rowDone .&&. (rowIdx .==. pure (maxBound :: Index rows))
  
  -- For now, always ready (could add backpressure logic later)
  readyOut = pure True