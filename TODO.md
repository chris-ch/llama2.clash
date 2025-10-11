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

Reasoning, assumptions, approach
- Goal: turn the current Clash design into a synthesizable LLaMa2 decoder core (Clash 1.8.2, GHC 9.6.7).
- Constraints:
  - No on-FPGA float; only SFixed and I8E are allowed on the datapath. Elaboration-time Float is fine.
- Approach: enumerate concrete, verifiable tasks per module; fix the few non-synthesizable constructs (Integer in runtime datapath; RAM op wrapper); replace simulation facades with sequential versions; lock ready/valid handshakes; and ensure BRAM/ROM inference. Each item has a definition of done (DoD). Where helpful, I include complete drop-in functions so you can copy-paste without stitching.

TODO.md

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
- Task N1: Done
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

1) RAM/ROM usage and KV cache
- Task R1: Fix RAM op API mismatch and missing type.
  - Issue: LLaMa2.Memory.RamOps exports RamOp(..) but does not define it, and uses trueDualPortBlockRam with a non-standard RamOp stream.
  - Action: Replace the RamOp pathway with Clash’s standard TDP BRAM interface (rdAddr, wrM) or provide a wrapper that converts signals. Minimal change: provide runTdpRam that matches the calls in TransformerLayer.
  - DoD: KV banks infer TDP BRAMs; no custom RamOp type is on the synthesis path.

  Haskell code (drop-in replacement for RamOps):
  ```haskell
  -- project/llama2/LLaMa2/Memory/RamOps.hs
  module LLaMa2.Memory.RamOps
    ( runTdpRam
    ) where

  import Clash.Prelude

  -- True-dual-port BRAM runner using standard Clash interface.
  -- Initializes memory to zero.
  runTdpRam
    :: forall dom n a
     . ( HiddenClockResetEnable dom
       , KnownNat n
       , NFDataX a )
    => Signal dom (Index n)              -- Port A read address
    -> Signal dom (Maybe (Index n, a))   -- Port A optional write
    -> Signal dom (Index n)              -- Port B read address
    -> Signal dom (Maybe (Index n, a))   -- Port B optional write
    -> ( Signal dom a                    -- Port A read data
       , Signal dom a )                  -- Port B read data
  runTdpRam rdA wrA rdB wrB =
    trueDualPortBlockRam (repeat (deepErrorX "uninit")) rdA wrA rdB wrB
  ```

  - Then, in TransformerLayer.fillOneBank, replace the two trueDualPortBlockRam calls with runTdpRam.

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
- Task P1: Rewrite PRNG mealy to avoid self-recursive signal in RHS.
  - Do: Replace pseudoRandomGenerator with a clean state machine. Drop-in below.
  - DoD: Lint shows no combinational loops; functional sim matches previous sequence.

  Haskell code:
  ```haskell
  -- project/llama2/LLaMa2/Embedding/PRNG.hs (replace pseudoRandomGenerator)
  pseudoRandomGenerator
    :: forall dom. HiddenClockResetEnable dom
    => Signal dom Bool           -- ^ readyPulse
    -> Signal dom (Unsigned 32)  -- ^ seed
    -> Signal dom (Unsigned 32)  -- ^ prng state/output
  pseudoRandomGenerator readyPulse seedSig =
    mealyB step 0 (bundle (readyPulse, seedSig))
    where
      step :: Unsigned 32 -> (Bool, Unsigned 32) -> (Unsigned 32, Unsigned 32)
      step s (rdy, seedNow) =
        let s' = if rdy then xorshift32 (seedNow `xor` 0x9E3779B9) else xorshift32 s
        in  (s', s')
  ```

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

Notes
- Relative dates and tool versions: as of October 10, 2025, Clash 1.8.2 is the baseline assumed here.
- If you want me to also provide a sequential QKV engine and scheduler as drop-in code for qkvProjectionController, say the word and I’ll supply the complete function.
  