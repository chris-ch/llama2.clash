# Product

**Design choices and constraints**

- Toolchain: Clash 1.8.2, GHC 9.8.4
- Models: LLaMA 2 7B primary, 13B supported (70B excluded)
- Context: 4k tokens
- AXI policy: streaming reads for inference core; KV cache writes enabled for autoregressive decode
- Memory access: no register mirrors; strict streaming reads with burst support (≤256 beats)
- Boot flow: offline external binary; optional FP32→I8E conversion on first boot → persist I8E to eMMC

**Software implementation – AXI Interface & Weight Management**

- Weights (7–13 GB) stored on eMMC; streamed to compute via AXI in row-wise bursts
- Compute core performs row-wise AXI bursts from DRAM (I8E → SFixed 12.20 dequantization inside core)
- AXI masters:
  - Read path for model weights / embedding / rotary / RMS vectors
  - KV cache read path for attention
  - KV cache write path for K/V updates during token generation

**LLaMA 7B/13B Accelerator – Final Spec (revised)**

- **Target performance**: ~5.2 tokens/s @ 7B (13B proportionally slower)
- **Retail price**: ~$1,799
- **Target cost (BOM)**: ~$1,255 per unit
- **NRE (one-time)**: $120K
- **NRE breakeven**: 220 units ($544 margin/unit); first batch (100 units) does not recover NRE — profitability requires continued sales beyond Phase 3

**Bill of Materials**

- Xilinx Zynq UltraScale+ ZU9EG – $700
- 16GB DDR4-2666 (2×8GB) – $95 _(upgraded from 8GB; 13B KV cache requires 6.25 GB leaving insufficient headroom in 8 GB)_
- 64GB eMMC 5.1 – $30
- 8-layer PCB + assembly – $120
- Power circuitry + 12V adapter – $55
- Heatsink + 40mm fan – $25
- USB 3.1 controller, LEDs, buttons, enclosure – $80
- Manufacturing & testing – $150
→ **Total BOM per unit**: $1,255
→ **Suggested retail**: $1,799

**Physical Specifications**

- Dimensions: 12 cm × 10 cm × 3.5 cm
- Weight: 180 g
- Power: 12 V @ 3 A (30 W typical, 36 W peak)
- Connectors: USB-C 3.1 Gen 2 (10 Gbps), DC barrel jack (12 V)
- Indicators: Power (green), Activity (blue), Status (RGB)
- Cooling: Active, temperature-controlled

**Key Characteristics**

- Models supported: LLaMA 2 7B (primary), 13B (supported)
- Context window: 4k tokens
- Compute precision: SFixed 12.20 (during matrix operations)
- Weight storage: I8E format on eMMC/DRAM with on-core dequantization
- KV cache storage: FixedPoint
- Interface: USB-C (data), barrel jack (power)
- Performance target: ~5.2 tok/s (7B model)

**Development Path**

- Phase 1: Block-level validation on Kria KV260 (individual blocks only — KV260 cannot fit full 7B decoder) → 3 months
- Phase 2: Custom ZU9EG board prototype (5 units) → 6 months, $120k NRE
- Phase 3: First production batch (100 units) → target retail $1,799, 12 months total timeline

**Hardware implementation – parallel64 & resource estimates**

- parallel64: 64-wide matrix multiply → 64 DSPs per instance
- Required instances: ~8 active (QKV proj ×3, Attn out ×1, FFN ×3, vocab proj ×1)
- DSP usage: ~512 / 4272 (12%)
- LUT & BRAM usage comfortable for ZU9EG (274K LUTs, 912 BRAMs available)

**Memory Layout & Budget**

- Weights streamed via AXI row fetch; no full-model on-chip mirroring
- KV cache (4k context): FixedPoint K/V banks in DRAM
- Intermediates + buffers: budgeted within platform DDR envelope
- DDR bandwidth is managed via burst reads and staged fetch/compute overlap

**Final Checklist**

- 7B/13B only
- Context 4k
- AXI bursts ≤256 beats per row (verified for both models)
- 32-bit address sufficient
- Timing closure @400 MHz feasible
- DDR bandwidth & prefetch overlap planned
- eMMC + first-boot conversion strategy viable

**Notes & Recommendations**

- Keep KV write path enabled for decode operation
- Use compile-time WordsPerRow assertions
- Add CI checks for KV cache budget vs. DDR size

---