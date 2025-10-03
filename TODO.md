# LLaMA‑2 Decoder (Clash) — Plan and Tasks

Scope
- Target: synthesizable, timing‑clean hardware decoder for LLaMA‑2 style transformer in Clash.
- Goal: keep current API/structure where possible; replace impractical blocks with streaming/pipelined variants.

# Design choices and constraints

## Toolchain and scope
 - Clash 1.8.2, GHC 9.2.8. Synthesizable only; no floating point in the final design.
 - Target device flexible; prioritize simplicity and timing robustness over maximum clock.
 - Goal: LLaMA2/3-class decoder, context length > 16k.

# GPT-5 Notes

Part 2 — Architectural gaps to make the design realistic at scale

A) Matrix-vector kernels are fully combinational

    Where: Mat-vec in projectQKV, WO projection, FFN, and logits (vocabulary mat-vec).
    Issue: matrixVectorMult (quantized rows dequantized to FixedPoint, then a pure dot product) unrolls everything. For realistic sizes (e.g., ModelDimension 288–512; Vocabulary 32k), this explodes LUT/DSP counts and timing.
    Action:
        Replace matrixVectorMult with time-multiplexed MAC engines (INT8×INT16→INT32 accumulators; scale/fuse shifts at row end; output recast to FixedPoint only at the boundary).
        Store weights in BRAM (or URAM on large FPGAs). Stream activations from small register files or BRAM.
        Provide a row engine interface: start/rowId/xVec -> done/result. Instantiate one engine per concurrent row you can afford; schedule rows over cycles.

Sketch: single-row sequential I8E×F -> F engine (copy-pasteable, standalone)

module Model.Helpers.MatVecI8E_Seq
  ( matVecRowSeq ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, Activation, Exponent, scalePow2F)
import Model.Numeric.ParamPack (RowI8E)

-- Sequential dot for one row: consumes one (mant, x) per cycle.
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

  done = regEn False en (lastCol)
  -- Fuse the row exponent once at the end
  yOut = regEn 0 done (scalePow2F <$> eRow <*> acc)

Notes:

    This engine is per row. Build a small farm if you want some parallelism (e.g., 4–8 rows at once).
    For best DSP mapping and determinism: consider accumulating in a dedicated Accumulator = Signed 32 and only convert to FixedPoint at the very end. That requires defining the column scaling convention (PoT) for x and using a shared shift schedule.
