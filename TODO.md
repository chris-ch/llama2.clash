# LLaMA‑2 Decoder (Clash) — Plan and Tasks

Scope
- Target: synthesizable, timing‑clean hardware decoder for LLaMA‑2 style transformer in Clash.
- Goal: keep current API/structure where possible; replace impractical blocks with streaming/pipelined variants.

# Design choices and constraints

## Toolchain and scope
 - Clash 1.8.2, GHC 9.6.7. Synthesizable only; no floating point in the final design.
 - Target device flexible; prioritize simplicity and timing robustness over maximum clock.
 - Goal: LLaMA2/3-class decoder, context length > 16k.

## Numeric formats
 - Scalar fixed-point for “real” math: F = SFixed 12 20 (signed; 12 integer, 20 fractional).
 - Quantized 8-bit with power-of-two (PoT) scaling: value ≈ mantissa(Act/Wgt :: Signed 8) × 2^Exponent, Exponent :: Signed 7 (clamped to [-64, 63]).
 - Accumulator for dot/MAC: Acc = Signed 32 with explicit saturating rounding when narrowing.

## MAC mapping
 - INT8 × INT16 → INT32 (promote one operand to 16 b), accumulate in 32 b, fuse PoT scales via shifts only.

## Scaling policy (PoT everywhere)
 - Weights: per-row static PoT exponent (one Exponent per weight row).
 - Activations: per-vector, per-timestep dynamic PoT exponent (one Exponent per produced activation vector).
 - Scale fusion via shifts at layer boundaries; no general multiplies for scaling.

## Attention and KV cache
 - Store K/V rows in BRAM as (Vec HeadDim Signed 8, Exponent).
 - Stage2 writes directly in quantized form; no float mirrors.
 - Stage3 uses streaming/sequential attention over BRAM (OnlineSoftmax), not a full-window register file; scales reconstruct with shifts.

## Math primitives (fixed-point)
 - exp(x): exp2 decomposition with 256-entry LUT for fractional part f in [0,1); result = 2^n × LUT[f]. Optional linear interpolation to reduce error.
 - Softmax: online, numerically stable variant in F using expF above.
 - RMSNorm: fixed-point mean-square, invsqrt via small LUT seed + one Newton–Raphson iteration; output renormalized in F and re-quantized to Act + Exponent.
 - RoPE: sine/cosine tables in F; all ops in fixed-point.

## Accuracy target
 - “As high as possible” under the above fixed-point constraints; PoT scaling chosen for hardware simplicity and consistent timing.
 - Validate with greedy (temp=0) and moderate temp sampling vs a float baseline; monitor top-k agreement and logit MSE.

## Performance/timing
 - Register-slice after matvec accumulation and after expF LUT.
 - Guard bits on accumulators; saturating rounding when converting back to Act.

## Build/runtime knobs
 - Prefer shift-based scale fusing; avoid general multipliers on hot paths.
 - Recommended flags: -O2, consider -fclash-inline-limit=64.

## Quantized constants offline
 - Offline, external binary loaded at boot (via PCIe/JTAG/UART)
 - Flow: store weights as packed bytes + exponents in external flash/DDR; on boot, DMA into BRAMs/URAMs

## 0 ) Moving to Fixed-Point from Float32

## 1) Synthesizability Assessment (Current Code)

What works as‑is (Clash‑synthesizable)
- Top and controller logic
  - Model.Top.topEntity
  - Model.Core.Transformer.*, Model.Core.PipelineController.runPipelineController
- KV cache write path
  - Model.Memory.KVCacheBank.writeSequencer
  - trueDualPortBlockRam with RamOps.toRamOperation
- Layer shell
  - Model.Layers.TransformerLayer.multiCycleTransformerLayer with stage predicates and gated write‑back
- PRNG and argmax
  - Model.Embedding.PRNG: xorshift, gating with readyPulse, argMax

Synthesizable but not practical at scale
- Attention as fully combinational across 0..pos and all HeadDimension, plus a register “mirror” of KV
- Fully parallel per‑head matvecs (Q/K/V projections and WO accumulation in one cycle)
- Full‑vocab logits and softmax (e.g., 32k wide) in one cycle

Practical issues
- Timing/area explode with SequenceLength and HeadDimension.
- Weight matrices inferred as distributed logic if left as big Vec literals.
- KV “mirror” uses O(SeqLen×HeadDim) registers per KV head per layer.

## 2) Architectural Recommendations (High Priority)

A. Stream the attention
- Replace KV “mirror” + combinational attendHead with a sequential kernel:
  - Read K/V from BRAM, compute dot q·k(t), update online softmax (running max/denom/numer).
  - When t == pos, emit attended vector and raise attnDone.

B. Time‑multiplex matvecs
- Share a MAC engine across Q/K/V projections and WO accumulation.
- Drive via ROM/BRAM weight reads; schedule per head/tile.

C. Stream logits
- Scan vocab rows from ROM/BRAM; compute greedy argmax (and later top‑k/topp) as a running scan.

D. Layer‑level pipelining (prefill mode)
- Replace the single global controller with per‑layer micro‑FSMs and valid/ready between layers.
- Use dual‑port KV RAM for Stage2 writes and Stage3 reads concurrently.
- Add 1–2 entry skid buffers to decouple bubbles.

## 3) Additional Tasks (Also High Priority Unless Noted)

- Weight storage/streaming
  - Put weights in ROM/BRAM (or external memory later); define a memory map and bandwidth matching the MAC schedule.
- KV RAM port semantics
  - Decide/read‑vs‑write behavior on same‑cycle same‑address; align schedule or add guards.
- Fixed interfaces for sequential attention
  - Start/clear, q latch, k/v stream, lastT, done, latency contract.
- Precise per‑layer handshake (prefill)
  - Layer l enters Stage1 for position p only after layer l‑1 finishes Stage4 for p (valid/ready).
- Activity gating (medium priority)
  - Add enables for heavy blocks (matvec, FFN, attention kernel, BRAM ports).
- Safety/cleanup (low priority)
  - Use bit‑accurate arithmetic for BankAddress; keep partial functions guarded.

## 4) Milestones and Acceptance Criteria

M1: Sequential attention kernel
- For a small config (SeqLen=16, HeadDim=8), the streamed kernel matches the existing combinational attendHead within FP tolerance for random inputs and all pos.
- No O(SeqLen×HeadDim) register “mirror” remains; KV maps to BRAM only.

M2: Time‑multiplexed matvecs
- Single shared MAC reused for Q/K/V/WO by schedule; meets target Fmax on the chosen FPGA family.
- Weight reads come from ROM/BRAM; no large LUT arrays for weights.

M3: Streamed logits
- Greedy argmax implemented as a row scan over vocab ROM; report cycles/token.
- Optionally, top‑k/topp nucleus sampling implemented as a streaming heap or thresholded scan.

M4: Prefill pipeline (layer wavefront)
- Multiple prompt tokens in flight across layers; measured throughput improvement versus the global FSM.
- No KV hazards; dual‑port behavior verified by assertions.

M5: Resource/timing report
- Post‑synth: KV and weights in BRAM/URAM; worst path no longer spans full‑window attention or full‑vocab softmax.

## 5) Integrating Implementation Helpers

A) Numerically stable online softmax (streaming reduction): Model.Layers.Attention.OnlineSoftmax

B) Sequential attention over one head: Model.Layers.Attention.AttendSequential

C) Stage enables and holding wrapper (activity gating): Model.Core.StageEnable, Helpers.EnableWrappers

D) Safer BankAddress arithmetic: refactor Model.Memory.Addressing

E) KV dual‑port mapping helper (read on A, write on B): Model.Memory.KVCacheBank.Ports

F) Per‑layer controller skeleton (for prefill wavefront): Model.Core.LayerController

## 6) Test & Verification Plan

- Equivalence tests
  - Compare sequential attention vs existing combinational attendHead for randomly generated q/k/v and all pos in a reduced config.
- Deterministic KV semantics
  - Constrain tests to avoid same‑addr read/write collisions, or assert expected read‑first/write‑first behavior and add a guard cycle in RTL.
- Throughput tests (prefill)
  - Drive a stream of prompt tokens and show multiple tokens in flight; measure cycles/token after pipeline fill.
- Resource/timing CI
  - Synthesize small and “15M” configs; track BRAM/URAM, FF/LUT, and Fmax regressions.

## 7) Targets (initial)

- Small config (sanity): SeqLen=16, ModelDim=64, Heads=4, HeadDim=16 at ≥200 MHz (modern FPGA).
- 15M config: post‑synth with sequential attention + time‑multiplexed matvecs, aiming ≥150–200 MHz, KV/weights in BRAM/URAM, no massive LUT memories.

## 8) Open Questions

- Speculative decoding support for multi‑token‑in‑flight during decoding (future work).


# GPT-5 Notes

Reasoning, assumptions, and approach
- Your current loop uses Clash’s pure simulator (simulate) around Top.topEntity.
  That makes internal nets invisible unless you explicitly surface them as ports.
- Easiest path (no switch of simulator): add a “debug” top that exposes per-layer lastKRowErr/lastVRowErr 
  (already computed inside the attention layer) and print them from Haskell as you simulate.
  This requires adding a debug variant of the layer and transformer that OR-reduces the per‑KV‑bank row-error
  flags to one Bool per layer. You keep Top.topEntity unchanged for synthesis; use TopDebug.topEntityDebug
  only for bring‑up.
- Alternative path (more visibility, zero code changes): generate HDL and use a waveform simulator (Verilator+GTKWave,
 Questa, GHDL). This is excellent once you want to inspect many signals and
 precise cycle alignment. But it’s a bigger tooling step than adding two debug ports.

Below I give copy‑pasteable modules/functions to:
1) Add a debug attention layer that returns two extra signals (K/V row-error flags per layer).
2) Add a debug transformer that threads those out as Vec NumLayers Bool.
3) Add a debug topEntity.
4) Show how to use them from your Haskell driver to print errors while generating tokens.

Code

1) Debug layer that surfaces lastKRowErr/lastVRowErr per layer
Put this new module alongside Model/Layers/TransformerLayer.hs. It is a minimally adapted copy that OR‑reduces the row error flags across KV banks and returns them. No changes to your non-debug layer are needed.

```haskell
-- project/llama2/Model/Layers/TransformerLayer/Debug.hs
module Model.Layers.TransformerLayer.Debug (
    multiCycleTransformerLayerDbg
  , TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
) where

import Clash.Prelude
import Model.Core.Types
  ( ProcessingState(..), LayerData(..), CycleStage(..) )
import Model.Config
  ( ModelDimension
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension, SequenceLength )
import qualified Model.Memory.KVCacheBank as Cache

import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention (projectQKV)
import Model.Layers.Components.Quantized
  ( FeedForwardNetworkComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , EmbeddingComponentQ(..)
  )

import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (computeFeedForward)
import Model.Helpers.MatVecI8E (matrixVectorMult)
import Model.Numeric.Types (Exponent, FixedPoint)
import Helpers (liftA4)
import Model.Layers.Attention.AttentionHead (attendHead)
import Model.Memory.KVCacheBank.RowStreamer (kvRowStreamer)
import Model.Layers.Attention.AttendSequential (attendHeadSeq)
import Model.Config.Debug (AttnMode(..), attnMode, attnEps)
import Model.Memory.KVCacheBank.RowFromRegs (rowsFromRegs)
import Model.Memory.RamOps (toRamOperation)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttentionComponentQ
  , feedforwardNetwork :: FeedForwardNetworkComponentQ
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponentQ
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

multiCycleTransformerLayerDbg
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> ( Signal dom LayerData
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom LayerData
     , Signal dom Bool  -- kRowErr (OR across KV banks), valid at end-of-stream
     , Signal dom Bool  -- vRowErr (OR across KV banks), valid at end-of-stream
     )
multiCycleTransformerLayerDbg layer kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
  ( nextLayerDataSignal
  , writeDoneThisLayerSignal
  , attentionDoneThisLayerSignal
  , commitCycle3Signal
  , kErrThisLayer
  , vErrThisLayer
  )
 where
  mhaQ = multiHeadAttention layer
  ffnQ = feedforwardNetwork layer

  -- Accumulators across KV banks (including error OR)
  initHeadOutputs = repeat (pure (repeat 0))
  initHeadDone    = repeat (pure False)
  initWriteDone   = repeat (pure False)
  initKErr        = pure False
  initVErr        = pure False

  ( perHeadOutputSignalsVec
  , perHeadDoneSignalsVec
  , perBankWriteDoneVec
  , kErrOR
  , vErrOR
  ) =
    foldl
      (fillOneBankDbg layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
      (initHeadOutputs, initHeadDone, initWriteDone, initKErr, initVErr)
      indicesI

  allHeadsDoneSignal     = fmap and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePrevSignal = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePrevSignal

  baseNextLayerDataSignal =
    liftA2 (processStage mhaQ ffnQ layerIndex) processingStateSignal intermediateDataSignal

  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\ps banksDone ->
           processingStage ps == Stage2_WriteKV
        && processingLayer ps == layerIndex
        && banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  perHeadProjectedSignalsVec =
    zipWith (\woQ hSig -> matrixVectorMult woQ <$> hSig) (mWoQ mhaQ) perHeadOutputSignalsVec

  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  woHeadsSignal          = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  xAfterAttnSignal =
      liftA2
        (\idata woHeads ->
          let xInput = inputVector idata
          in zipWith (+) xInput woHeads)
        intermediateDataSignal
        woHeadsSignal

  nextLayerDataSignal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal baseNextLayerDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  commitCycle3Signal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal intermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Expose OR of per-bank row errors; align it to the same pulse as attentionDone
  -- They are latched at lastT per bank; we OR and hold until Stage3 completes.
  kErrThisLayer =
    let hold = regEn False attentionDoneThisLayerSignal
    in  hold kErrOR
  vErrThisLayer =
    let hold = regEn False attentionDoneThisLayerSignal
    in  hold vErrOR

processStage
  :: MultiHeadAttentionComponentQ
  -> FeedForwardNetworkComponentQ
  -> Index NumLayers
  -> ProcessingState
  -> LayerData
  -> LayerData
processStage mhaQ ffnQ layerIndex ps idata
  | processingLayer ps /= layerIndex = idata
  | otherwise = case processingStage ps of
      Stage1_ProjectQKV ->
        let (qs, ks, vs) = MultiHeadAttention.projectQKV mhaQ (sequencePosition ps) (inputVector idata)
        in idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }
      Stage2_WriteKV     -> idata
      Stage3_Attend      -> idata
      Stage4_FeedForward ->
        let ffnOut = FeedForwardNetwork.computeFeedForward ffnQ (attentionOutput idata)
        in  idata { feedForwardOutput = ffnOut }
      Stage5_Bookkeeping -> idata

-- (Identical to your fillOneBank, but returns two extra OR‑able flags: lastKRowErr, lastVRowErr)
fillOneBankDbg :: HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom LayerData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool)
     , Signal dom Bool
     , Signal dom Bool )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool)
     , Signal dom Bool
     , Signal dom Bool )
fillOneBankDbg layerIx psSig kvOwner idSig
               (headOutAcc, headDoneAcc, writeDoneAcc, kErrAcc, vErrAcc) kvIx =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIx)
             psSig (pure ())
    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    attnPrev = register False isStage3Attention
    clearS3  = liftA2 (\now prev -> now && not prev) isStage3Attention attnPrev

    seqPosSignal = sequencePosition <$> psSig
    bank         = Cache.kvBanks kvOwner !! kvIx

    qHeadsPerKV  = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
    qIdx0        = toEnum (min (natToNum @NumQueryHeads - 1) (fromEnum kvIx * qHeadsPerKV))
    hasQ1        = qHeadsPerKV >= 2 && (fromEnum kvIx * qHeadsPerKV + 1 <= natToNum @NumQueryHeads - 1)
    qIdx1        = if hasQ1 then toEnum (fromEnum qIdx0 + 1) else qIdx0

    getQ i = (\d -> queryVectors d !! i) <$> idSig
    query0 = getQ qIdx0
    query1 = if hasQ1 then getQ qIdx1 else pure (repeat 0)

    keyVec   = (\d -> keyVectors   d !! kvIx) <$> idSig
    valueVec = (\d -> valueVectors d !! kvIx) <$> idSig

    (writeAddrSig, kMantWrRaw, kExpWrRaw, vMantWrRaw, vExpWrRaw, writeDoneThisBank) =
      Cache.writeSequencer isStage2Write seqPosSignal (bundle (keyVec, valueVec))
    wrAddrS = mux isStage2Write writeAddrSig (pure 0)
    kMantWr = mux isStage2Write kMantWrRaw (pure Nothing)
    vMantWr = mux isStage2Write vMantWrRaw (pure Nothing)
    kExpWr  = mux isStage2Write kExpWrRaw  (pure Nothing)
    vExpWr  = mux isStage2Write vExpWrRaw  (pure Nothing)

    -- Unquantized rows BRAM-F for progressive replacement
    (tAddrRow, stepEnRow, lastTRow) =
      unbundle $
        mealy
          (\(t, done) (cl, en, pos) ->
             let step  = en && not done
                 last  = step && t == pos
                 t'    = if not step || last then t else succ t
                 done' = (not cl && (done || last))
             in ((t', done'), (t, register False step, register False last)))
          (0 :: Index SequenceLength, False)
          (bundle (clearS3, isStage3Attention, seqPosSignal))

    wrKVRowF_K = mux isStage2Write (Just <$> bundle (seqPosSignal, keyVec))   (pure Nothing)
    wrKVRowF_V = mux isStage2Write (Just <$> bundle (seqPosSignal, valueVec)) (pure Nothing)

    (kRowF_A, _kRowF_B) =
      trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                           (toRamOperation seqPosSignal wrKVRowF_K)
    (vRowF_A, _vRowF_B) =
      trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                           (toRamOperation seqPosSignal wrKVRowF_V)

    kvKeysAll = mealy (\mem (we, p, rowK) -> let mem' = if we then replace p rowK mem else mem
                                             in (mem', mem'))
                      (repeat (repeat 0))
                      (bundle (isStage2Write, seqPosSignal, keyVec))
    kvValsAll = mealy (\mem (we, p, rowV) -> let mem' = if we then replace p rowV mem else mem
                                             in (mem', mem'))
                      (repeat (repeat 0))
                      (bundle (isStage2Write, seqPosSignal, valueVec))

    out0_baseline = liftA4 attendHead query0 kvKeysAll kvValsAll seqPosSignal
    out1_baseline = if hasQ1 then liftA4 attendHead query1 kvKeysAll kvValsAll seqPosSignal
                             else pure (repeat 0)
    doneBaseline = clearS3

    (kRowQ, vRowQ, rowValidQ, lastTQ) =
      kvRowStreamer bank clearS3 isStage3Attention seqPosSignal
                    wrAddrS kMantWr kExpWr vMantWr vExpWr
    (out0_seqQ, done0_seqQ) = attendHeadSeq clearS3 rowValidQ query0 kRowQ vRowQ lastTQ
    (out1_seqQ, done1_seqQ) = if hasQ1
                                then attendHeadSeq clearS3 rowValidQ query1 kRowQ vRowQ lastTQ
                                else (pure (repeat 0), pure False)

    (out0_seqF, done0_seqF) = attendHeadSeq clearS3 stepEnRow query0 kRowF_A vRowF_A lastTRow
    (out1_seqF, done1_seqF) = if hasQ1
                                then attendHeadSeq clearS3 stepEnRow query1 kRowF_A vRowF_A lastTRow
                                else (pure (repeat 0), pure False)

    (out0_sel, out1_sel, done0_sel, done1_sel) =
      case attnMode of
        AttnBaseline     -> (out0_baseline, out1_baseline, doneBaseline, doneBaseline)
        AttnShadowBRAM   -> (out0_baseline, out1_baseline, doneBaseline, doneBaseline)
        AttnReplaceBRAMF -> (out0_seqF,     out1_seqF,     done0_seqF,   done1_seqF)
        AttnReplaceBRAMQ -> (out0_seqQ,     out1_seqQ,     done0_seqQ,   done1_seqQ)

    -- Row error monitors (unchanged logic)
    maxAbs v = foldl max 0 (map abs v)
    diffTooBig a b = maxAbs (zipWith (-) a b) > attnEps

    kRowErr      = diffTooBig kRowQ kRowF_A
    vRowErr      = diffTooBig vRowQ vRowF_A
    lastKRowErr  = regEn False lastTQ kRowErr
    lastVRowErr  = regEn False lastTQ vRowErr
    !_probeK     = lastKRowErr
    !_probeV     = lastVRowErr

    headOutAcc0  = replace qIdx0 out0_sel headOutAcc
    headDoneAcc0 = replace qIdx0 done0_sel headDoneAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1_sel headOutAcc0 else headOutAcc0
    headDoneAcc1 = if hasQ1 then replace qIdx1 done1_sel headDoneAcc0 else headDoneAcc0
    writeDoneAcc1 = replace kvIx writeDoneThisBank writeDoneAcc

    kErrAcc1 = liftA2 (||) kErrAcc lastKRowErr
    vErrAcc1 = liftA2 (||) vErrAcc lastVRowErr

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1, kErrAcc1, vErrAcc1)
```

2) Debug transformer that threads the two vectors (one per layer)
```haskell
-- project/llama2/Model/Core/Transformer/Debug.hs
module Model.Core.Transformer.Debug (
    multiCycleTransformerDebug
) where

import Clash.Prelude
import Helpers (liftA4)
import Model.Core.Types
  ( LayerData(..)
  , ProcessingState(..)
  , CycleStage(..)
  , Temperature, Seed, Token )
import Model.Config ( NumLayers )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer
  ( TransformerDecoderComponent(..) )
import qualified Model.Layers.TransformerLayer.Debug as TLDbg
  ( TransformerLayerComponent(..)
  , multiCycleTransformerLayerDbg )
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController, PipelineOutputs(..) )
import qualified Model.Core.Transformer.Internal as internal
  ( outputTokenSignal, initialLayerData, embedFromComponent )

multiCycleTransformerDebug :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Vec NumLayers (Signal dom Bool)  -- kErr per layer (pulsed at end-of-attention)
     , Vec NumLayers (Signal dom Bool)  -- vErr per layer
     )
multiCycleTransformerDebug decoder cacheOwners inputTokenSignal inputTokenValid temperatureSignal seedSignal =
  ( selectedTokenSignal, PipelineController.readyPulse ctrl, kErrVec, vErrVec )
 where
  embeddingComponent = TransformerLayer.modelEmbedding decoder
  transformerLayers  = Model.Layers.TransformerLayer.modelLayers decoder

  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> PipelineController.layerIndex ctrl
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> PipelineController.layerIndex ctrl

  ctrl = PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer inputTokenValid

  feedbackTokenSignal =
    internal.outputTokenSignal (PipelineController.readyPulse ctrl) temperatureSignal seedSignal decoder nextLayerDataSignal

  selectedTokenSignal = mux inputTokenValid inputTokenSignal feedbackTokenSignal

  tokenEmbeddingSignal = internal.embedFromComponent embeddingComponent <$> selectedTokenSignal
  intermediateDataSignal = register internal.initialLayerData nextLayerDataSignal

  inputLoadedSignal =
    liftA3
      (\ps current tokenEmbedding ->
         if processingStage ps == Stage1_ProjectQKV
           then if processingLayer ps == 0
                  then current { inputVector = tokenEmbedding }
                  else current { inputVector = feedForwardOutput current }
           else current)
      (PipelineController.processingState ctrl) intermediateDataSignal tokenEmbeddingSignal

  layerStep
    :: ( Signal dom LayerData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       )
    -> (TLDbg.TransformerLayerComponent, Cache.KVRamOwner dom, Index NumLayers)
    -> ( Signal dom LayerData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       )
  layerStep (currData, wDoneVec, attnDoneVec, kErrAcc, vErrAcc)
            (layerComp, cacheOwner, lIx) =
    let
      (newData, wDone, attnDone, commitC3, kErr, vErr) =
        TLDbg.multiCycleTransformerLayerDbg layerComp cacheOwner lIx (PipelineController.processingState ctrl) currData
      selectedData =
        liftA4
          (\ps oldD newD c3D ->
             if processingLayer ps == lIx
                then if processingStage ps == Stage3_Attend then c3D else newD
                else oldD)
          (PipelineController.processingState ctrl) currData newData commitC3
    in  ( selectedData
        , replace lIx wDone    wDoneVec
        , replace lIx attnDone attnDoneVec
        , replace lIx kErr     kErrAcc
        , replace lIx vErr     vErrAcc
        )

  ( nextLayerDataSignal
  , writeDoneVector
  , attnDoneVector
  , kErrVec
  , vErrVec
  ) =
    foldl
      layerStep (inputLoadedSignal , repeat (pure False) , repeat (pure False), repeat (pure False), repeat (pure False))
      (zip3 transformerLayers cacheOwners indicesI)
```

3) Debug top-level that returns the two vectors
```haskell
-- project/llama2/Model/TopDebug.hs
module Model.TopDebug
  ( topEntityDebug
  ) where

import Clash.Prelude
import Model.Core.Types ( Temperature, Seed, Token )
import Model.Config ( NumLayers )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)
import qualified Model.Core.Transformer.Debug as TransformerDbg

topEntityDebug
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Vec NumLayers (Signal dom Bool)  -- K row error (per layer)
     , Vec NumLayers (Signal dom Bool)  -- V row error (per layer)
     )
topEntityDebug decoder =
  TransformerDbg.multiCycleTransformerDebug decoder (repeat Cache.makeRamOwnerKV)
```

4) Use it from your Haskell driver
Add a debug version of your bundler and a small print of flags. You don’t need to change your original function; add this variant to try first.

```haskell
-- In your driver module
import qualified Clash.Prelude as C
import qualified Clash.Explicit.Testbench as CS
import qualified Model.TopDebug as TopDebug
import           Model.Config (NumLayers)
import           Clash.Sized.Vector (toList)

bundledOutputsDebug
  :: TransformerDecoderComponent
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> C.Signal C.System (Token, Bool, Vec NumLayers Bool, Vec NumLayers Bool)
bundledOutputsDebug decoder bundledInputs =
  C.bundle $
    C.exposeClockResetEnable
      (TopDebug.topEntityDebug decoder token isValid temperature seed)
      CS.systemClockGen
      CS.resetGen
      CS.enableGen
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs

-- A debug generation routine that prints row errors when they occur.
generateTokensSimAutoregressiveDebug
  :: TransformerDecoderComponent
  -> T.Tokenizer
  -> Int
  -> [Token]
  -> Float
  -> Seed
  -> IO Int
generateTokensSimAutoregressiveDebug decoder tokenizer stepCount promptTokens temperature seed = do
  putStrLn $ "✅ Prompt: " ++ show promptTokens
  hFlush stdout
  let
    (firstToken, restPrompt) =
      case promptTokens of { (t:ts) -> (t, ts); [] -> (1, []) }

    advanceState :: (Token,[Token],Bool) -> (Bool,Token) -> (Token,[Token],Bool)
    advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
      | not isReady = (current, remPrompt, usingPrompt)
      | otherwise   = case remPrompt of
                        (p:ps) -> (p, ps, True)
                        []     -> (sampled, [], False)

    temperature' = realToFrac temperature :: FixedPoint

    outputsDbg :: [(Token, Bool, Vec NumLayers Bool, Vec NumLayers Bool)]
    outputsDbg =
      CS.simulate (bundledOutputsDebug decoder)
                  (DL.zip4 inputTokens inputValidFlags (repeat temperature') (repeat seed))

    (outputTokens, readyFlags, kErrVecs, vErrVecs) =
      unzip4 outputsDbg

    states :: [(Token,[Token],Bool)]
    states = scanl advanceState (firstToken, restPrompt, True)
                (zip (drop 1 readyFlags) (drop 1 outputTokens))

    inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
    inputValidFlags = True       : [ use | (_, _, use) <- states ]

    sampledTokens = [ tok | (tok, isReady) <- zip outputTokens readyFlags, isReady ]
    promptLength    = length promptTokens
    generatedTokens = take stepCount (drop promptLength sampledTokens)
    emittedTokens   = take (promptLength + stepCount) (promptTokens ++ generatedTokens)

    -- Side-effect: print any row errors per cycle
    printErr (cyc,(kE,vE)) =
      let ks = zip [0..] (toList kE)
          vs = zip [0..] (toList vE)
          badK = [ i | (i,True) <- ks ]
          badV = [ i | (i,True) <- vs ]
      in do
        unless (null badK && null badV) $
          putStrLn $ "⚠️  RowErr at cycle " ++ show cyc
                   ++ " K(layers)=" ++ show badK
                   ++ " V(layers)=" ++ show badV

  -- Print any errors observed
  mapM_ printErr (zip [0..] (zip kErrVecs vErrVecs))

  mapM_ (printToken tokenizer) emittedTokens
  putStrLn ""
  pure stepCount
```

How to run
- Keep your existing generateTokensSimAutoregressive for normal runs.
- For bring-up, call generateTokensSimAutoregressiveDebug with identical args. It will:
  - print tokens as before,
  - print a warning line whenever any layer’s end-of-stream row comparison flagged a mismatch (> attnEps).
- Start with:
  - Model.Config.Debug.attnMode = AttnShadowBRAM
  - Model.Config.Quant.quantModeKV = QuantCeilSafe
  - Small model (260K or 15M) and a short prompt.

When to switch to an HDL simulator
- Use the debug top if you only need: did any layer miscompare this position? That’s usually enough for the Step 1–3 validation.
- Switch to HDL waveforms when you need cycle‑accurate traces of:
  - t counters, dim counters, bank addresses, exp-bank addresses, rowValid/lastT pulses, write enables.
- Quick recipe:
  - clash --verilog -fclash-keep-attributes -fclash-no-prim-warn -fclash-hdldir build project/llama2/Model/Top.hs
  - Simulate with Verilator and dump VCD:
    - verilator -cc build/Model.Top/topEntity.v --trace
    - make -C obj_dir -f Vtop.mk
    - Write a tiny C++ main that wiggles clk/reset and drives token/isValid/temp/seed; dump VCD; open in GTKWave.
  - In waves, search for nets containing lastKRowErr/lastVRowErr, rowValidQ, lastTQ, addrExpA/addrExpB_K/addrExpB_V, rdAddr, dimCnt, tCnt.

Why this works
- In your code, lastKRowErr/lastVRowErr already latch the per‑row max-abs diff at the end of the streamed sequence (lastTQ). OR‑reducing them per layer and pulsing them at attentionDone gives you a clean, one‑cycle indication “this layer’s streamed BRAM rows matched the unquantized reference within attnEps.”
- Keeping QuantCeilSafe eliminates mantissa clipping; any mismatch then points to exp addressing or stream scheduling. If Shadow is clean, AttnReplaceBRAMQ should produce consistent text (modulo sampling). If it degrades at T=0, use AttnReplaceBRAMF to isolate quantization vs. streaming.

Tooling tips
- Haskell sim: prefer simulate for determinism; if lists get large, simulate_lazy reduces peak memory at the cost of lazier errors.
- HDL sim: Verilator is fast and integrates well with CI; GHDL/Questa provide richer GUI wave debug.
- Clash flags for better net names: -fclash-keep-attributes and compile with -O2 for the Haskell parts.

If you later want the t=0..2 details the advisor asked for, we can add two more debug ports per layer that expose the final t value at lastTQ and/or pipe out rdAddr/addrExpA at the lastTQ pulse. But the above should get you quick signal on “is exponent addressing/clipping still wrong?” without leaving Haskell simulation.
