module Model.Helpers.MatVecI8E_Seq
  ( matVecRowSeq ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, Activation, scalePow2F)
import Model.Numeric.ParamPack (RowI8E)

-- Protocol:
--  - clear: pulse to reset internal state for new row
--  - en: one step (consumes one column) when True
--  - lastCol: asserted with en on the final column; 'done' pulses next cycle
--  - row: static I8E row for this computation
--  - x: stream of FixedPoint activations (aligned to en)
-- Result is in FixedPoint with per-row exponent fused at the end.
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
  qNow :: Signal dom Activation
  qNow = (!!) <$> mant <*> col

  -- Accumulator in a wider integer domain (Signed 32)
  -- INT8×FixedPoint: convert x to, e.g., Signed 16 (truncate) or keep FixedPoint and use DSP.
  -- Here we keep FixedPoint and accumulate in FixedPoint to stay in your F-domain.
  acc :: Signal dom FixedPoint
  acc = regEn 0 (en .||. clear)
        (mux clear 0 (acc + (fromIntegral <$> qNow) * xSig))

  done = regEn False en lastCol
  -- Fuse the row exponent once at the end
  yOut = regEn 0 done (scalePow2F <$> eRow <*> acc)
