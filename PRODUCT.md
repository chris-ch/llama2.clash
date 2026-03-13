
# Product

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

**LLaMA 7B/13B Accelerator – Final Spec (revised)**

- **Target performance**: ~5.2 tokens/s @ 7B (13B proportionally slower)
- **Retail price**: ~$1,799
- **Target cost (BOM)**: ~$1,220 per unit
- **NRE (one-time)**: $120K

**Bill of Materials**

- Xilinx Zynq UltraScale+ ZU9EG – $700
- 8GB DDR4-2666 (2×4GB) – $60
- 64GB eMMC 5.1 – $30
- 8-layer PCB + assembly – $120
- Power circuitry + 12V adapter – $55
- Heatsink + 40mm fan – $25
- USB 3.1 controller, LEDs, buttons, enclosure – $80
- Manufacturing & testing – $150
→ **Total BOM per unit**: $1,220  
→ **Suggested retail**: $1,799

**Physical Specifications**

- Dimensions: 12 cm × 10 cm × 3.5 cm
- Weight: 180 g
- Power: 12 V @ 3 A (30 W typical, 36 W peak)
- Connectors: USB-C 3.1 Gen 2 (10 Gbps), DC barrel jack (12 V)
- Indicators: Power (green), Activity (blue), Status (RGB)
- Cooling: Active, temperature-controlled

**Key Characteristics**

- Models supported: LLaMA 2 7B (primary), 13B (supported); 70B out of scope
- Context window: < 16k tokens (recommend 8-bit KV cache for longer contexts)
- Compute precision: SFixed 12.20 (during matrix operations)
- Weight storage: I8E format on eMMC → cached/dequantized in DDR4
- Interface: USB-C (data), barrel jack (power)
- Performance target: ~5.2 tok/s (7B model)

**Development Path**

- Phase 1: Validation on Kria KV260 → 3 months
- Phase 2: Custom ZU9EG board prototype (5 units) → 6 months, $120k NRE
- Phase 3: First production batch (100 units) → target retail $1,799, 12 months total timeline

**Quick Reference**

- ~5.2 tok/s (7B)
- 7B & 13B support
- Fixed-point compute (SFixed 12 20)
- I8E weights cached in DDR
- Context < 16k tokens (8-bit KV recommended for long context)
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

---