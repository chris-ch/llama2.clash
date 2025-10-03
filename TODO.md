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

Reasoning, assumptions, and approach
I reviewed your current Clash codebase against the stated design goals: FP-free runtime, INT8×INT16→INT32 MACs, PoT (power-of-two) scales, sequential attention over a quantized K/V cache, very long context, and “configuration-agnostic” leaf modules. The code mostly compiles conceptually, but several parts still block realistic synthesis or violate the FP-free + INT8/INT32 constraints. Below I list the remaining work in priority order and include copy-paste-ready fixes for the immediate synthesis blockers (Integer/Float leakage in datapaths, missing liftA4, a partial pattern, and the seed type). For the larger architectural items (quantized KV cache and sequential INT8 MACs), I outline the changes needed and propose concrete module and interface shapes you can implement next.

What still remains (prioritized)
Hard synthesis blockers to fix now
1) Integer and Float in runtime datapaths
- Problem: expF, invSqrtF, and log2Decompose use floor/ceiling into Integer and fromEnum/Int in hot paths. Clash cannot synthesize Integer/Int in circuits. You also compute a LUT index via floor :: FixedPoint -> Integer, then fromInteger, which is the same issue.
- Fix: keep results in bounded Integral types (Unsigned/Signed). Replace Integer/Int with Signed 7 / Unsigned 8, etc. See “Model.Numeric.Fixed: drop Integer/Int at runtime” code below.

2) Runtime Float in rotary encoding
- Problem: applyRotation converts Float→FixedPoint at runtime (arrCos !! stepCount |> map realToFrac). This pulls Float into the datapath.
- Fix: convert all rotary sin/cos to FixedPoint at elaboration (like you already do for weights). Replace RotaryEncodingComponent (Float tables) with a FixedPoint carrier used by the Q components. See “Rotary conversion: Float-free runtime” code below.

3) Missing Helpers.liftA4
- Problem: several modules import Helpers.liftA4, but the module is not present.
- Fix: add a tiny Helpers module. See code below.

4) Partial pattern / error in applyRotaryPositionEncoding
- Problem: error "Unexpected vector structure..." is unsynthesizable and unnecessary; the pattern is total for Vec 2.
- Fix: replace with a total pattern. Included below.

5) PRNG seed type
- Problem: tokenSampler’s seed argument is typed as Token, not Seed (even though both are Unsigned 32). This is confusing and easy to misuse.
- Fix: adjust the signature to Seed. Included below.

Architectural work required for a realistic design (next steps)
6) Quantized K/V cache (I8 mantissas + grouped exponents)
- Current code writes whole unquantized Vec HeadDimension FixedPoint rows into a BRAM word. That creates very wide RAMs (e.g., 48×(12+20) ≈ 1536-bit) and defeats your I8E plan. You also only produce a write “done” pulse; there’s no mantissa streaming or exponent grouping logic as your comment promises.
- What to implement:
  - A bank format for K/V rows with:
    - Mantissa RAM: depth = SequenceLength × HeadDimension, payload = Signed 8
    - Exponent RAM: depth = SequenceLength × KVExpGroups, payload = Signed 7 (grouped PoT scale)
  - A write path that quantizes one (K or V) head-dim element per cycle, asserts exp-write once per group, and asserts bank-done at the tail.
  - A read path that streams one mant per cycle with the correct exp for its group and feeds either:
    a) an INT8×INT16→INT32 sequential MAC that accumulates in 32 bits and fuses shifts, or
    b) a small on-the-fly dequantizer to FixedPoint if you temporarily keep the F path for validation (but that breaks the “FP-free” goal).
- Interface sketch you’ll likely want (new module, not a drop-in replacement):
  - kvWriteI8E: (en, rowVecF) -> (mantEn, mantAddr, mantData, expEn, expAddr, expData, done)
  - kvReadI8E: (row, step) -> (mant, exp, lastT)
  - You can keep the attentionRowSequencer you already wrote and just change payloads.

7) Sequential INT8 MACs and mat-vec kernels
- Current matrixVectorMult dequantizes to FixedPoint and does a full combinational dot product. That is: FP multiplies, big adders, no INT8×INT16→INT32, and no pipelining.
- What to implement:
  - A sequential dot with one INT8 mantissa per cycle and a pre-shifted INT16 activation (or vice-versa) so “scale fusion by shifts” holds. Accumulate in Signed 32.
  - A generic “mv kernel” wrapper that runs for “cols” cycles with a start/step/done handshake. Keep leaf modules Nat-agnostic: pass sizes through the canonical type aliases, as you already do.

8) Replace floating expSoftmax path with the fixed OnlineSoftmax already in the tree
- You’re on the right track with OnlineSoftmax. Make sure all attention paths use it and that expF is the FP-free version below. Also validate numerical ranges with your chosen SFixed format and clamp where needed.

9) External parameter loading at boot (no on-FPGA quantization)
- Currently, all params are elaboration constants. For realistic deployment and large models:
  - Add a ROM/BRAM initialization path (e.g., COE/MIF/HEX) or a loader (PCIe/JTAG/UART) that writes mantissa/exponent memories at boot.
  - Keep QArray2D only as a “compile-time packer” for small demos. For production, replace it with BRAM interfaces and loaders.

10) Timing/latency alignment around BRAMs
- Ensure 1-cycle BRAM read latency is fully aligned with step enables and “lastT”. Your attentionRowSequencer partly accounts for this; double-check the alignment of stepEnRow, kRowA/vRowA, and lastTRow with a simple testbench.

11) Minor cleanup
- Enable wrappers: avoid deepErrorX initial values in always-live registers (provide explicit reset values).
- Remove duplicate imports and Int/Integer conversions in library code.
- Verify Stage4 gating in PipelineController is exactly what you want (see comment in that section).

Copy-paste-ready fixes for the immediate blockers

A) Model.Numeric.Fixed: drop Integer/Int at runtime
Replace the definitions of exp2Frac, expF, log2Decompose, invSqrtF in project/llama2/Model/Numeric/Fixed.hs with the following. These avoid Integer/Int on the datapath and keep indices/exponents in Unsigned/Signed.

```haskell
module Model.Numeric.Fixed
  ( quantizeI8E
  , quantizeI8E_ceilSafe
  , quantizeI8E_best3_noClip
  , expF
  ) where

import Clash.Prelude
import Model.Numeric.Types
  ( FixedPoint, Activation, Exponent
  , scalePow2F, clampExp, satRoundToI8, epsF
  )

-- ... keep your quantizers as-is (they run at elaboration on constants) ...

ln2InvF :: FixedPoint
ln2InvF = realToFrac (1.4426950408889634 :: Double)

exp2FracLUT :: Vec 256 FixedPoint
exp2FracLUT =
  map (\(i :: Index 256) ->
         let k   = fromIntegral (fromEnum i) :: Double
             val = 2 ** (k / 256)
         in  realToFrac val)
      indicesI

-- 2^f with f in [0,1); LUT index stays in Unsigned 8 (no Integer on datapath)
exp2Frac :: FixedPoint -> FixedPoint
exp2Frac f =
  let fClamped = max 0 (min (1 - epsF) f)
      idx :: Unsigned 8
      idx = floor (fClamped * 256)  -- floor to Unsigned 8
  in exp2FracLUT !! idx

-- Decompose x = m * 2^e, with m in [1,2); avoid Int/Integer
log2Decompose :: FixedPoint -> (Exponent, FixedPoint)
log2Decompose xIn =
  let x = max xIn epsF
      pow2 :: Vec 64 (Exponent, FixedPoint)
      pow2 = map (\i ->
                    let e :: Exponent = clampExp (fromIntegral i - 32)
                    in (e, scalePow2F e 1))
                 indicesI
      (eBest, vBest) =
        foldl
          (\(be,bv) (e,v) -> if v <= x && v > bv then (e,v) else (be,bv))
          (minBound, 0)
          pow2
      m = x / scalePow2F eBest 1
  in (eBest, m)

invSqrtMantLUT :: Vec 256 FixedPoint
invSqrtMantLUT =
  map (\(i :: Index 256) ->
         let m   = 1.0 + (fromIntegral (fromEnum i) + 0.5) / 256.0 :: Double
             val = 1.0 / sqrt m
         in  realToFrac val)
      indicesI

invSqrt2 :: FixedPoint
invSqrt2 = realToFrac (1.0 / sqrt 2.0 :: Double)

nrImproveInvSqrt :: FixedPoint -> FixedPoint -> FixedPoint
nrImproveInvSqrt m y0 =
  let half      = (1 :: FixedPoint) / 2
      threeHalf = (3 :: FixedPoint) / 2
  in y0 * (threeHalf - half * m * y0 * y0)

invSqrtF :: FixedPoint -> FixedPoint
invSqrtF a0 =
  let a        = max a0 epsF
      (e, m)   = log2Decompose a
      idx :: Unsigned 8
      idx      = floor ((m - 1) * 256)
      seedMant = invSqrtMantLUT !! idx
      eEven    = testBit (pack e :: BitVector 7) 0 == False  -- cheap parity; or use even (as const)
      scalePow = if eEven then negate (shiftR e 1) else negate (shiftR (e - 1) 1)
      -- Since e :: Signed 7, shiftR keeps sign. scalePow2F handles Signed e.
      scale    = if eEven then scalePow2F scalePow 1
                          else scalePow2F scalePow 1 * invSqrt2
      y0       = seedMant * scale
  in nrImproveInvSqrt a y0

-- expF: x -> 2^(x/ln2) = 2^n * 2^f, all bounded ints on datapath
expF :: FixedPoint -> FixedPoint
expF x =
  let y  = x * ln2InvF
      nC :: Exponent
      nC = clampExp (floor y)
      f  = y - fromIntegral nC
      b  = exp2Frac f
  in scalePow2F nC b
```

Why it works
- floor is specialized to Unsigned/Signed targets, which Clash knows how to synthesize.
- No Integer/Int values flow through runtime circuits.
- log2Decompose works fully in bounded fixed and signed types.

B) Rotary conversion: Float-free runtime
Add a FixedPoint rotary carrier and converter, and use it in SingleHeadComponentQ. Replace the applyRotation that touched Float at runtime.

1) New Fixed rotary type and converter
Create a new file project/llama2/Model/Layers/Components/RotaryQ.hs:

```haskell
module Model.Layers.Components.RotaryQ
  ( RotaryEncodingComponentF(..)
  , quantizeRotary
  ) where

import Clash.Prelude
import Model.Core.Types (CArray2D(..))
import Model.Config (SequenceLength, RotaryPositionalEmbeddingDimension)
import Model.Numeric.Types (FixedPoint)

data RotaryEncodingComponentF = RotaryEncodingComponentF
  { freqCosF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  , freqSinF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  } deriving (Generic, NFDataX, Show, Eq)

quantizeRotary :: (KnownNat SequenceLength, KnownNat RotaryPositionalEmbeddingDimension)
               => (CArray2D SequenceLength RotaryPositionalEmbeddingDimension,
                   CArray2D SequenceLength RotaryPositionalEmbeddingDimension)
               -> RotaryEncodingComponentF
quantizeRotary (CArray2D cosF, CArray2D sinF) =
  RotaryEncodingComponentF
    { freqCosF = map (map realToFrac) cosF
    , freqSinF = map (map realToFrac) sinF
    }
```

2) Use it in the quantized single head (change record field)
In project/llama2/Model/Layers/Components/Quantized.hs, adjust SingleHeadComponentQ and quantizeSingleHead:

```haskell
-- add import:
import Model.Layers.Components.RotaryQ (RotaryEncodingComponentF(..), quantizeRotary)

data SingleHeadComponentQ = SingleHeadComponentQ
  { wqHeadQ :: QArray2D HeadDimension ModelDimension
  , wkHeadQ :: QArray2D HeadDimension ModelDimension
  , wvHeadQ :: QArray2D HeadDimension ModelDimension
  , rotaryQ :: RotaryEncodingComponentF
  } deriving (Generic, Show, Eq)

quantizeSingleHead :: SingleHeadComponent -> SingleHeadComponentQ
quantizeSingleHead sh =
  SingleHeadComponentQ
    { wqHeadQ = quantizeMatI8E (wqHead sh)
    , wkHeadQ = quantizeMatI8E (wkHead sh)
    , wvHeadQ = quantizeMatI8E (wvHead sh)
    , rotaryQ = quantizeRotary (freqCos (rotary sh), freqSin (rotary sh))
    }
```

3) Apply rotation without Float at runtime
Replace applyRotation and applyRotaryPositionEncoding in project/llama2/Model/Layers/Attention/MultiHeadAttention/Internal.hs with:

```haskell
module Model.Layers.Attention.MultiHeadAttention.Internal (
  applyRotaryPositionEncoding
  , computeHeadQ
  , computeHeadKV
  , applyRotation
) where

import Clash.Prelude
import Model.Config
  ( ModelDimension, HeadDimension, RotaryPositionalEmbeddingDimension, SequenceLength )
import Model.Numeric.Types (FixedPoint)
import Model.Layers.Components.Quantized (SingleHeadComponentQ(..))
import Model.Layers.Components.RotaryQ (RotaryEncodingComponentF(..))
import Model.Helpers.MatVecI8E (matrixVectorMult)

applyRotaryPositionEncoding
  :: Vec HeadDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotaryPositionEncoding inputVec cosVecF sinVecF =
  concat (imap rotatePair (unconcat d2 inputVec))
 where
  rotatePair :: Index RotaryPositionalEmbeddingDimension -> Vec 2 FixedPoint -> Vec 2 FixedPoint
  rotatePair i (realC :> imagC :> Nil) =
    let c = cosVecF !! i
        s = sinVecF !! i
        r = realC * c - imagC * s
        im = realC * s + imagC * c
    in  r :> im :> Nil

applyRotation
  :: RotaryEncodingComponentF
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotation rot step tokenVec =
  let cosF = freqCosF rot !! step
      sinF = freqSinF rot !! step
  in  applyRotaryPositionEncoding tokenVec cosF sinF

-- computeHeadQ / computeHeadKV unchanged except the new rotaryQ type
```

Why it works
- The Float→Fixed conversion runs at elaboration (realToFrac on constants).
- No runtime Float values or conversions remain in the datapath.

C) Helpers.liftA4
Add project/llama2/Helpers.hs:

```haskell
module Helpers (liftA4) where

import Clash.Prelude

liftA4 :: Applicative f => (a -> b -> c -> d -> e)
       -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = f <$> a <*> b <*> c <*> d
```

D) PRNG: fix the seed type
Replace the tokenSampler signature header in project/llama2/Model/Embedding/PRNG.hs with:

```haskell
tokenSampler :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> TransformerLayer.TransformerDecoderComponent
  -> Signal dom LayerData
  -> Signal dom Token
```

E) Optional: safer enable wrapper
If you use holdWhen in active logic, prefer an explicit reset value to avoid deepErrorX fan-out. Add alongside your current function:

```haskell
module Model.Helpers.EnableWrappers (holdWhen, holdWhenInit) where

import Clash.Prelude

-- Existing (be careful: X-init fanout)
holdWhen :: (HiddenClockResetEnable dom, NFDataX a)
         => Signal dom Bool -> Signal dom a -> Signal dom a
holdWhen = regEn (deepErrorX "holdWhen init")

-- Safer variant with explicit init
holdWhenInit :: HiddenClockResetEnable dom
             => a -> Signal dom Bool -> Signal dom a -> Signal dom a
holdWhenInit initV en x = regEn initV en x
```

Notes and guidance on the larger items you’ll implement next
- KV cache I8E:
  - Add a new module, e.g., Model.Memory.KVCacheI8E, that exposes:
    - write path: quantize stream with (mantEn, mantAddr :: BankAddress, mantData :: Signed 8) and (expEn, expAddr :: Index (SequenceLength*KVExpGroups), expData :: Signed 7), plus a done pulse.
    - read path: for Stage 3, given row position and step enable, produce (mant :: Signed 8, exp :: Signed 7, lastT).
  - Replace the current trueDualPortBlockRam of Vec HeadDimension FixedPoint with two narrow TDP memories (mant and exp). Your existing attentionRowSequencer already fits the “row scan” contract.

- Sequential INT8×INT16→INT32 MAC:
  - Build a small Mealy core:
    - State: accumulator :: Signed 32, colIdx :: Index HeadDimension.
    - Inputs per cycle: qMant :: Signed 8, kMant :: Signed 8, fuseShift :: Signed 6..7 (from exponents and model-scale fusion).
    - Multiply to Signed 16, shift/align, add to acc.
    - Assert done when colIdx == last.
  - Wrap a mat-vec around it by iterating colIdx and feeding mantissas from RAM and the activation vector; keep leaf kernels Nat-agnostic by taking sizes from type aliases in Model.Config.

- PipelineController Stage4 gating:
  - Today Stage4_FeedForward uses (not <$> readyPulseRaw) as stageFinished. If you intend FFN to take multiple cycles or to handshake, replace this with the FFN “done” pulse. Otherwise, for 1-cycle FFN, this is fine, but document it and test with multiple layers.

- Parameter load at boot:
  - For a realistic large model, integrate a loader (e.g., simple AXI-lite or UART FSM) that fills mantissa/exponent BRAMs before the first token. Provide a done flag gating the very first Stage1 (layer 0, pos 0).

Tooling and verification tips
- Use GHDL/ModelSim and Clash’s simulate/trace to check Stage 2→3 handshakes and BRAM latency alignment.
- For arithmetic range: QuickCheck or small testbenches over HeadDimension=8/16 to validate OnlineSoftmax stability, invSqrtF, and expF saturation behavior with SFixed 12 20.
- Synthesis sanity: run yosys/nextpnr or Vivado on small configs (e.g., MODEL_260K) to ensure BRAM inference (narrow data widths), control fan-out, and Fmax with single-cycle LUT indexing.

References / further reading
- Clash Prelude and FixedPoint conversions (Clash Users Guide) for synthesizable RealFrac/Integral specializations.
- “Online softmax” derivations for numerical stability.
- Vendor RAM inference guides (Xilinx/Intel) for true dual-port BRAMs with narrow data widths.

If you want, I can next:
- Draft the KV I8E bank module with concrete ports and a tested write/read sequencer, and
- Provide a sequential INT8 MAC core (mealy) plus a simple wrapper to replace matrixVectorMult for both MHA and FFN paths.