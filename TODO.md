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
- Goal: make the design realistically synthesizable on an FPGA in Clash 1.8.2 / GHC 9.6.7, fixed-point only on datapath. Your current code is functionally clean, but several blocks are still “algorithmic” (combinational Vec math over large dimensions) and/or memory-unrealistic (Vec ROMs feeding huge muxes).
- I audited every module you posted with these questions in mind:
  1) Will Clash infer BRAMs and small SRAMs instead of giant mux nets?
  2) Are long combinational paths broken up or time-multiplexed?
  3) Does every multi-cycle stage have a “done” handshake so the pipeline controller can actually wait?
  4) Are all registers initialized to defined values?
  5) Is the top entity elaboration-friendly (no higher-order args, deterministically monomorphic, annotated)?
- Summary: you are close on the sequential attention and KV-cache pieces, but Stage1 (QKV projection), vocab embedding and the FFN are still fully combinational. Also, the top-level needs a synthesizable wrapper, a Stage1 “done” handshake in the pipeline controller, and a few utilities need minor fixes for reset semantics and write-pulsing.

What remains to be done (blockers vs improvements)
Blockers (must-fix to build a real bitstream)
1) Top entity: remove higher‑order parameter at the hardware boundary and add a Synthesize annotation.
   - Today: topEntity :: TransformerDecoderComponent -> Signal … -> …
   - For synthesis, the decoder (weights, LUTs) must be a constant bound inside the top or provided via an external loader, not a port. Provide a new top that is fully applied and monomorphic.

2) Use true ROM/BRAM for the vocabulary table (and large parameter tables).
   - Model.Core.Embedding.embedder currently indexes a Vec with a runtime Int/Token. That becomes a giant mux and will not map to BRAM for large VocabularySize (e.g., 32k). Use rom/romPow2 (1‑cycle) or blockRam (1‑cycle) to force BRAM.

3) Stage1 (QKV projection) and FFN are purely combinational mat‑vec on full ModelDimension. That won’t close timing and creates huge multiplier farms.
   - Replace matrixVectorMult (combinational) with a sequential mat‑vec engine (you already have MatVecI8E_Seq.matVecRowSeq as a kernel) and add a Stage1 “done” handshake.
   - Do the same for FFN’s three mat‑vecs (W1, W3, W2). A single time‑multiplexed engine with a small controller can run all three passes.

4) Pipeline controller needs a Stage1 done input.
   - runPipelineController currently advances from Stage1 to Stage2 unconditionally (except for the very first token). You must hold at Stage1 until “projQKVDoneThisLayer == 1”.

5) KV write in Stage2 repeatedly writes the same row every cycle while Stage2 is active.
   - Gate the write with a 1‑cycle rising‑edge pulse so you write exactly once, then assert “writeDone” without wasting cycles or energy.

6) Register init Xs.
   - holdWhen uses deepErrorX for init. You want a defined reset value for synthesis.

Strongly recommended improvements (to de-risk timing and area)
- Insert ROM/BRAM for all big constant matrices (Wq/Wk/Wv/Wo, W1/W2/W3, rotary tables), even if you still do combinational mat‑vec during bring-up. Otherwise Clash may build massive logic ROMs.
- Add basic pipelining (tree reductions or a staged accumulator) on fixed-point dot-products and softmax if you temporarily keep any combinational versions.
- Replace dynamic “divide by 2^n” in scalePow2F with arithmetic shifts on the fixed representation to avoid general division hardware (resource win).
- Confirm expF and invSqrtF are OK on your target frequency; pipeline them if needed (small 1–2 stage pipe is usually enough).
- Add a simple parameters module (generated) and a monomorphic top entity that imports constants (no big records as top ports).

Below are focused, drop-in pieces you can paste to unlock the blockers with minimal churn.


B) Embed vocabulary via ROM (1-cycle BRAM instead of giant mux)
Replace your embedding module with a BRAM-backed ROM and one-cycle latency.

haskell
module Model.Core.Embedding
  ( embedder ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, scalePow2F)
import Model.Numeric.ParamPack (QArray2D(..))
import Model.Core.Types (Token)
import Model.Config (ModelDimension, VocabularySize)

-- BRAM/ROM-backed dequantize-on-read. 1-cycle latency.
embedder
  :: HiddenClockResetEnable dom
  => QArray2D VocabularySize ModelDimension
  -> Signal dom Token
  -> Signal dom (Vec ModelDimension FixedPoint)
embedder (QArray2D table) tokSig =
  let
    -- Precompute dequantized rows at elaboration time; stored in ROM.
    deqRow (mant, e) =
      let s = scalePow2F e 1
      in map (\q -> fromIntegral q * s) mant
    romContent :: Vec VocabularySize (Vec ModelDimension FixedPoint)
    romContent = map deqRow table
    addr :: Signal dom (Index VocabularySize)
    addr = fromIntegral <$> tokSig
  in rom romContent addr

Why it works:
- rom infers a synchronous ROM with 1-cycle latency; Clash maps large Vec constants to BRAM.
- Interface now is Signal dom Token -> Signal dom (Vec … FixedPoint), so update call sites accordingly (see small change in Transformer below).

D) A single-cycle “write once” pulse for KV writes
Add a helper that generates a rising-edge pulse for Stage2, and use it to gate RAM writes to one cycle.

haskell
module Model.Memory.KVCacheBank (
  writeSequencer,
  writeOnce
) where

import Clash.Prelude
import Model.Config (HeadDimension)

-- Existing (unchanged) counter-based sequencer (if you still want it):
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
writeSequencer enSig = doneSig
 where
  dimCnt     = register (0 :: Index HeadDimension) nextDimCnt
  nextDimCnt = mux enSig (succ <$> dimCnt) (pure 0)
  atLastDim  = (== maxBound) <$> dimCnt
  doneSig    = (&&) <$> enSig <*> atLastDim

-- New: one-pulse generator (rising edge of 'en')
writeOnce
  :: HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ en (Level during Stage2)
  -> ( Signal dom Bool  -- ^ wrPulse (1 cycle on Stage2 entry)
     , Signal dom Bool) -- ^ donePulse (1 cycle, same as wrPulse by default)
writeOnce enSig =
  let enPrev   = register False enSig
      pulse    = enSig .&&. not <$> enPrev
  in (pulse, pulse)

Why it works:
- You write a row exactly once per Stage2 entry, then assert done for the pipeline to advance.

E) Gate KV writes and wire “writeDone” with the one-pulse
Drop-in replacement for fillOneBank (only this function) to avoid re-writing every cycle.

haskell
fillOneBank ::
  forall dom .
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBank layerIndex psSig idSig (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIndex)
             psSig (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    attnPrev = register False isStage3Attention
    clearS3  = liftA2 (\now prev -> now && not prev) isStage3Attention attnPrev

    seqPosSignal = sequencePosition <$> psSig

    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 = getQueryVector idSig qIdx0
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   idSig kvIx
    valueVec = getValueVector idSig kvIx

    -- Generate a single write pulse per Stage2, and use it for both K and V.
    (wrPulse, wrDonePulse) = Cache.writeOnce isStage2Write

    wrKVRowK = mux wrPulse (Just <$> bundle (seqPosSignal, keyVec))   (pure Nothing)
    wrKVRowV = mux wrPulse (Just <$> bundle (seqPosSignal, valueVec)) (pure Nothing)

    (kRowA, _kRowB) =
      trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                           (toRamOperation seqPosSignal wrKVRowK)

    (vRowA, _vRowB) =
      trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                           (toRamOperation seqPosSignal wrKVRowV)

    (tAddrRow, stepEnRow, lastTRow) =
      attentionRowSequencer clearS3 isStage3Attention seqPosSignal

    (out0_seqF, done0_seqF) = attendHeadSeq clearS3 stepEnRow query0 kRowA vRowA lastTRow
    (out1_seqF, done1_seqF) =
      if hasQ1 then attendHeadSeq clearS3 stepEnRow query1 kRowA vRowA lastTRow
               else (pure (repeat 0), pure False)

    headOutAcc0  = replace qIdx0 out0_seqF headOutAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1_seqF headOutAcc0 else headOutAcc0

    headDoneAcc0 = replace qIdx0 done0_seqF headDoneAcc
    headDoneAcc1 = if hasQ1 then replace qIdx1 done1_seqF headDoneAcc0 else headDoneAcc0

    writeDoneAcc1 = replace kvIx wrDonePulse writeDoneAcc

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1)

Why it works:
- BRAM port B sees a write exactly once per Stage2 (per bank). writeDone for this bank becomes the same pulse.

F) Add Stage1 done to the pipeline controller
Replace your PipelineController module with the following that includes s1DoneThisLayer.

haskell
module Model.Core.PipelineController
  ( PipelineOutputs(..)
  , runPipelineController
  ) where

import Clash.Prelude
import Model.Core.Types (ProcessingState(..), CycleStage(..))
import Model.Config (NumLayers, SequenceLength)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { processingStage  = Stage1_ProjectQKV
  , processingLayer  = 0
  , sequencePosition = 0
  }

nextProcessingState :: ProcessingState -> ProcessingState
nextProcessingState state = case processingStage state of
  Stage1_ProjectQKV -> state { processingStage = Stage2_WriteKV }
  Stage2_WriteKV    -> state { processingStage = Stage3_Attend }
  Stage3_Attend     -> state { processingStage = Stage4_FeedForward }
  Stage4_FeedForward ->
    if processingLayer state == maxBound
      then state { processingStage  = Stage5_Bookkeeping }
      else state { processingStage  = Stage1_ProjectQKV
                 , processingLayer  = succ (processingLayer state)
                 }
  Stage5_Bookkeeping ->
    state { processingStage  = Stage1_ProjectQKV
          , processingLayer  = 0
          , sequencePosition =
              if sequencePosition state == maxBound
                then 0 else succ (sequencePosition state)
          }

data PipelineOutputs dom = PipelineOutputs
  { processingState   :: Signal dom ProcessingState
  , stageSignal       :: Signal dom CycleStage
  , layerIndex        :: Signal dom (Index NumLayers)
  , seqPos            :: Signal dom (Index SequenceLength)
  , readyPulse        :: Signal dom Bool
  , stageFinished     :: Signal dom Bool
  }

-- Now includes 's1DoneThisLayer'
runPipelineController
  :: HiddenClockResetEnable dom
  => Signal dom Bool     -- ^ attnDoneThisLayer (Stage3)
  -> Signal dom Bool     -- ^ writeDoneThisLayer (Stage2)
  -> Signal dom Bool     -- ^ s1DoneThisLayer (Stage1)
  -> Signal dom Bool     -- ^ inputTokenValid (only used at very first (L0,P0))
  -> PipelineOutputs dom
runPipelineController attnDoneThisLayer writeDoneThisLayer s1DoneThisLayer inputTokenValid = outs
 where
  advance s done = if done then nextProcessingState s else s
  procState = register initialProcessingState (advance <$> procState <*> stageFinishedSig)

  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  -- readyPulse = rising edge when entering last-layer FFN
  isLastLayerFFN =
    liftA2 (\ps _ -> processingStage ps == Stage4_FeedForward
                  && processingLayer ps == maxBound)
           procState (pure ())
  readyPulseRaw =
    let rising now prev = now && not prev
    in  liftA2 rising isLastLayerFFN (register False isLastLayerFFN)

  atFirstStage1 =
    liftA2 (\ps _ -> processingStage ps == Stage1_ProjectQKV
                  && processingLayer ps == 0
                  && sequencePosition ps == 0)
           procState (pure ())

  isStage st = (== st) <$> stageSig
  stageFinishedSig =
    mux (isStage Stage1_ProjectQKV)
         (mux atFirstStage1 inputTokenValid s1DoneThisLayer) $
    mux (isStage Stage2_WriteKV)     writeDoneThisLayer      $
    mux (isStage Stage3_Attend)      attnDoneThisLayer       $
    mux (isStage Stage4_FeedForward) (not <$> readyPulseRaw) $
    mux (isStage Stage5_Bookkeeping) (pure True)             $
    pure False

  outs = PipelineOutputs
    { processingState = procState
    , stageSignal     = stageSig
    , layerIndex      = layerIx
    , seqPos          = posIx
    , readyPulse      = readyPulseRaw
    , stageFinished   = stageFinishedSig
    }

Why it works:
- The FSM now truly waits for Stage1 (QKV projection) to finish before moving on.

G) Wire the ROM-based embedding and Stage1 handshake in Transformer
Minimal changes you can apply now without having the sequential Stage1 engine yet:
- Call the ROM-based embedder (1-cycle latency).
- Pass a stub s1DoneThisLayer = True for bring-up, then replace it with the real done when you hook your sequential mat‑vec.

Drop-in replacements below for just the affected bindings in Model.Core.Transformer:

haskell
  -- Quantized embedding lookup via ROM (1-cycle)
  tokenEmbedding :: Signal dom (Vec ModelDimension FixedPoint)
  tokenEmbedding = Embedding.embedder vocabulary selectedToken

  -- Stage1 done handshake (stub True for now; replace with real signal)
  s1DoneThisLayer :: Signal dom Bool
  s1DoneThisLayer = pure True

  pipelineController :: PipelineController.PipelineOutputs dom
  pipelineController =
    PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer s1DoneThisLayer inputTokenValid

Why it works:
- Keeps your pipeline functional today; when you switch to sequential Stage1 you will replace s1DoneThisLayer with the engine’s done pulse.

H) Guidance for sequential QKV/FFN (next steps you’ll implement)
- Reuse your kernel matVecRowSeq for rows and add a small controller:
  - For each row: clear=1 for 1 cycle, then iterate columns with en=1 for HeadDimension cycles, assert lastCol on last column. Latch the row result and move to next row. Repeat for all rows of Wq, then Wk, then Wv.
  - Store Wq/Wk/Wv rows in ROM (Vec rows (RowI8E cols)) and stream the activation x via a small read-port FIFO or direct Signal.
  - Emit a single s1Done pulse when the last row of Wv is finished.
- Do the same three-pass schedule for FFN (W1, W3, W2), reusing the same engine.
- If you want 1 row/cycle throughput, instantiate three engines in parallel; otherwise, a single engine time-multiplexed keeps area in check.

I) Optional: shift-based scalePow2F (avoid generic division)
The current dynamic divide by 2^n is OK, but you can avoid division with bit-coercion shifts. One safe approach in Clash is to keep arithmetic in an integer-like container of the fixed representation. Sketch:

haskell
-- If you want this optimization later:
scalePow2F :: Exponent -> FixedPoint -> FixedPoint
scalePow2F n x =
  let nI = fromIntegral n :: Int
  in if nI >= 0 then shiftL x nI else shiftR x (negate nI)

Note: SFixed in Clash implements Bits, so shiftL/shiftR are available and synthesize to barrel shifters.

Checklist against your design constraints
- FP-free datapath: preserved. Any Double/Float use is elaboration only (LUTs, quantization).
- INT8×INT16→INT32 MACs: your sequential kernel keeps accumulation in FixedPoint today; for DSP mapping consider casting per-cycle product to a wider Accumulator (Signed 32) and only fuse the exponent at row end (you already do that).
- OnlineSoftmax: OK and already sequential.
- KV as FixedPoint in BRAM: OK after the single-pulse write gating.
- Context length > 16k: rely on BRAM depth. For >16k, ensure your vendor’s RAM primitive inference is triggered (romPow2/blockRam), not LUT ROM.
- No register mirrors: the ROM/BRAM interfaces respect that.

Tooling and debugging tips
- Synthesis hints: use rom/romPow2 for ROMs; trueDualPortBlockRam for KV. Check inferred memories in Clash’s -fclash-debug DebugSilent + vendor reports to ensure BRAM, not LUTs.
- Timing: if any block still uses combinational sum of large Vecs, switch to balanced tree (Clash’s fold may already be tree-like; otherwise use treeFold).
- Profiling: use clash --verilog -fclash-no-asserts for netlist gen. Simulate in GHDL/ModelSim to verify the 1-cycle ROM latency at embedding and that Stage2 writes only once.

References and further reading
- Clash Prelude docs: blockRam/trueDualPortBlockRam/rom/romPow2 and RamOp (Clash.Prelude)
- Clash 1.8 memory inference guide (ROM vs LUTs)
- LLM fixed-point softmax online algorithm: your implementation is consistent with the standard “running max/denom” streaming formulation

If you’d like, I can help you wire a first working sequential Stage1 (QKV) engine with its “done” pulse using your MatVecI8E_Seq kernel and show exactly how to store Wq/Wk/Wv in ROM and schedule the rows.
