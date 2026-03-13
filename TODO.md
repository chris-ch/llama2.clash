# To do

**Design choices and constraints**

- Toolchain: Clash 1.8.2, GHC 9.8.4
- Models: LLaMA 2 7B primary, 13B supported (70B excluded)
- Context: < 16k tokens
- AXI policy: streaming reads only for inference core; writes disabled in synthesized compute path (boot-time write path optional)
- Memory access: no register mirrors; strict streaming reads with burst support (≤256 beats)
- Boot flow: offline external binary; optional FP32→I8E conversion on first boot → persist I8E to eMMC

**Software implementation – AXI Interface & Weight Management**

- Weights (7–13 GB) stored on eMMC → active layer cached in DDR4 (~1.2 GB)
- Compute core performs row-wise AXI bursts from DDR4 (I8E → SFixed 12 20 dequantization inside core)
- Three main AXI masters:
  - Read from eMMC (boot/staging/conversion)
  - Read from DDR4 (inference – active weights)
  - Write to DDR4 (boot-time caching / conversion only; disabled in production inference build)

**Implementation Phases (AXI & weight path)**

1. AXI primitives (types, single-beat read, burst read ≤256, optional minimal write)
2. Throughput & robustness (burst rules, error handling, measurement)
3. Weight loading & format (boot-time FP32→I8E converter + persist, layer prefetch)
4. Decoder integration (remove const weights, add AXI row fetchers, layer FSM)

**Hardware implementation – parallel64 & resource estimates**

- parallel64: 64-wide matrix multiply → 64 DSPs per instance
- Required instances: ~8 active (QKV proj ×3, Attn out ×1, FFN ×3, vocab proj ×1)
- DSP usage: ~512 / 1728 (30%)
- LUT & BRAM usage comfortable for ZU9EG

**Memory Layout & Budget**

- Active layer (dequantized): ~1.2 GB
- KV cache (16k context):
  - 8-bit → ~4 GB
  - 16-bit → ~8 GB
- Intermediates + buffers: ~0.5–0.7 GB
- → Preference for 8-bit KV when targeting long context

**Final Checklist**

- 7B/13B only
- Context <16k, DDR budget met with 8-bit KV
- AXI bursts ≤256 beats per row (verified for both models)
- 32-bit address sufficient
- Timing closure @400 MHz feasible
- DDR bandwidth & prefetch overlap planned
- eMMC + first-boot conversion strategy viable

**Notes & Recommendations**

- Keep write path behind conditional compilation (inference = read-only)
- Use compile-time WordsPerRow assertions
- Add CI check for KV cache budget vs. DDR size
