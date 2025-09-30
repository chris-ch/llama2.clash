# LLaMA‑2 Decoder (Clash) — Plan and Tasks

Scope
- Target: synthesizable, timing‑clean hardware decoder for LLaMA‑2 style transformer in Clash.
- Floating‑point: out of scope for now (assume Float is a placeholder; quantization or fixed‑point/FPGA IP later).
- Goal: keep current API/structure where possible; replace impractical blocks with streaming/pipelined variants.

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

- Final numeric format (FXP vs FP IP) and corresponding scaling/activation tuning.
- Off‑chip weight/KV options if on‑chip BRAM is insufficient.
- Speculative decoding support for multi‑token‑in‑flight during decoding (future work).
