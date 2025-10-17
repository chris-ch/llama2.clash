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

1) Global build, config, and lint
- Task G1: Pin toolchain and Clash options.
  - Do: Use Clash 1.8.2, GHC 9.6.7. Build with -O2 -fclash-aggressive-x-optimization -fconstraint-solver-iterations=10.
  - DoD: clash --v shows 1.8.2; Netlist generation succeeds.

- Task G2: Select a small config first.
  - Do: Start with MODEL_260K or default 15M; verify ROM/BRAM sizes; only then scale up.
  - DoD: Synthesis completes and reports BRAM inference for vocabulary and KV banks.

- Task G3: Remove runtime Integer usage from datapath.
  - Done
  - DoD: Grep shows no Integer in synthesizable code paths; Clash reports no “Integer not supported” warnings.

- Task G4: Keep Float strictly at elaboration time.
  - Do: Ensure all Float → FixedPoint conversions are within constant generators (quantize*, ROM init). No Float appears in a Signal dom.
  - DoD: Grep shows Float only in Params/Decoder, Quantized converters, and LUT generators.

- Task G5: Identify and isolate sim-only modules.
  - Sim-only: LLaMa2.Layers.Attention.AttentionHead (pure combinational full-seq attention), any direct exp/log using Double for debug.
  - Do: Keep these compiled but unused by topEntity; do not import them on the synthesis path.
  - DoD: Netlist contains no references to AttentionHead.

2) Top level and pipeline controller
- Task T1: topEntity ports and reset semantics.
  - Do: Keep System domain; confirm Reset is synchronous (matches BRAM primitives). If your board requires async reset, wrap with resetSynchronizer.
  - DoD: Vendor synthesis shows synchronous BRAM; timing clean at target Fmax.

- Task T2: readyPulse contract.
  - Do: Verify readyPulse is exactly one-cycle pulse on entering last-layer FFN; it gates PRNG and feedback token register.
  - DoD: Formal/Sim: readyPulse rises once per generated token; no double pulses.

3) Numeric: fixed-point helpers
- Task N1: Replace scalePow2F with shift-only implementation.
  - Why: current version multiplies by fromInteger (1 `shiftL` n) at runtime, introducing Integer.
  - Action: Replace function with a Bits-based shifter. 
- Task N2: Accumulator width audit in dot products.
  - Do: Consider widening accumulators for long vectors (e.g., HeadDimension 64/128). Option: use SFixed (intBitsF+8) 20 in accumulation then truncate with saturation.
  - DoD: No overflow observed vs. golden (simulation), or document acceptable error.

4) Matrix-vector engines and handshakes
- Task M1: Done
  - DoD: Netlist shows no combinational full mat-vec for WO; only the sequential core and its FSM.

- Task M2: Make QKV projection sequential (optional first pass).
  - First pass: keep combinational projectQKV (synthesizable but large).
  - Second pass: Replace qkvProjectionController’s “compute in one cycle” with three sequential matrixMultiplier instances scheduled across heads.
  - DoD: For first pass, netlist produced; For second pass, resource usage drops; controller handshakes remain correct.

5) RAM/ROM usage and KV cache
- Task R1: Done
- Task R2: One-pulse KV write is correct but verify same-cycle R/W policy.
  - Do: Confirm vendor BRAM is write-first or no-collision on different ports (we write on Port B at pos and read Port A at t=0..pos during Stage3; Stage2 and Stage3 are mutually exclusive so safe).
  - DoD: Simulate a layer with pos=0 and pos>0; attendHeadSeq reads the just-written row next Stage3.

- Task R3: ROM for embedding and rotary
  - Do: Ensure rom is used (synchronous) and content fits BRAM; consider romPow2 or romFile for very large VocabularySize.
  - DoD: Vendor says ROMs inferred; read latency is 1 cycle and already accounted for.

6) Attention path (sequential)
- Task A1: Validate attendHeadSeq step enables and clear.
  - Do: attentionRowSequencer must:
    - issue clearS3 pulse on Stage3 entry,
    - step 0..pos inclusive with one element per cycle,
    - assert lastTRow aligned with the last key/value sample.
  - DoD: Sim proves softResult equals combinational attentionHead result for small heads/pos.

- Task A2: Online softmax stability on zero denominator.
  - Do: Code already guards d==0; keep as-is. Consider epsilon add if needed.
  - DoD: No X-propagation when all masked.

7) PRNG and sampling
  - Do: Done

8) Pipeline and stage control
- Task C1: Stage-finish generation in PipelineController
  - Do: Confirm s1DoneThisLayer = qkvDoneThisLayer is stable when Stage1 completes. The code wires this; leave as-is after M1/M2.
  - DoD: Sim traces show stageFinished pulses exactly once per stage transition.

- Task C2: Data selection and registers
  - Do: layerInputSelector: ensure tokenEmbedding latency is matched (rom is 1-cycle). You register layerData and use it consistently; ok.
  - DoD: No off-by-one at start of sequence; layer 0 uses token embed.

9) Cleanups, safety, and synthesis hygiene
- Task S1: Remove unused helpers and exports that can confuse synthesis (RamOp(..), toRamOperation).
  - Do: Delete RamOp and toRamOperation or move to a sim-only module.
  - DoD: No dead code warnings on synthesis build.

- Task S2: Avoid partial/unsafe on datapath.
  - Do: fromJustX is only used under mux with isJust; keep or refactor away with Maybe writes. No head/tail on empty Vec; (!!) uses Index so safe.
  - DoD: Clash reports no Partial selector in subject.

- Task S3: NOINLINE/INLINE pragmas for big combinational functions.
  - Do: Add {-# NOINLINE matrixVectorMult #-} to avoid unintended huge inlining when still used for small configs; or, conversely INLINE small helpers for Fmax.
  - DoD: Synthesis QoR stable between runs.

10) Definition of Done (netlist readiness)
- No Float or Integer on any Signal dom.
- No sim-only modules on synthesis path.
- All mat-vec ops on the critical path use sequential engines with RV handshake.
- BRAMs inferred for:
  - KV banks (2 per KV head).
  - Embedding ROM.
  - Optional: rotary tables.
- Ready/valid contracts:
  - singleHeadController: accepts a head when readyOut==1 and validIn==1; produces validOut for one cycle; back to IDLE after consumer handshake.
  - qkvProjectionController: either combinational (first pass) or sequential (second pass), but always produces a stable qkvDone signal used by PipelineController.
- topEntity generates without blackbox errors; gatelevel sim or vendor elaboration succeeds.

11) Post-synth performance (follow-up)
- Consider double-buffering per-head WO projection to overlap with attention row scan.
- Explore banking KV RAM further (one BRAM per head-dimension stripe) to raise read bandwidth and lower Fmax pressure.
- Add sfix widening in accumulators and saturating truncation to improve numerical stability for larger models.
- Vendor-specific RAM style attributes if inference needs help (Xilinx: ram_style = "block").

References and tips
- Clash Prelude, Block RAM/ROM: Clash.Prelude.BlockRam and Clash.Prelude.ROM.
- SFixed arithmetic and Bits: Clash docs for Fixed point types; shiftL/shiftR are synthesizable on SFixed.
- Handshake patterns in Clash: mealyB, regEn, and ready/valid recipes in Clash tutorials.

Why we sometimes don't use ready signals: they aren't needed in the current architecture.

The key insight is that the architecture uses **transaction-based control** rather than **streaming backpressure**:

- **One token processes completely** through all stages before the next token enters
- The **FSM controls when computations start and finish** - it doesn't ask the datapath "are you ready?", it just starts it and waits for completion
- **No overlap** between stages for the same layer, so no resource contention

The `readyIn` signals would only be needed if you wanted:
1. Multiple tokens in flight simultaneously (pipelined)
2. Shared resources that could be busy
3. Downstream buffers that could overflow

Since we have a single-token pipeline where each stage completes before the next starts, the FSM-level coordination (`qkvOutReady`, `matVecValid`) is sufficient.

The signals are present in the interfaces because they follow a standard ready/valid protocol pattern, which makes the code more composable and easier to extend later if you want to add pipelining. But for now, they can safely be tied to `(pure True)` or left unused.
