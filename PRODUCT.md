
# Product

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

---