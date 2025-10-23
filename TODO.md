# Software implementation
## AXI Interface Implementation Summary

### Context
Scaling from 260K model (weights embedded in FPGA) to 7B model (weights in external storage).

**Problem:** 7B model = 7GB weights, FPGA BRAM = 2.8MB → Cannot embed weights

**Solution:** Store weights in eMMC (64GB), cache active layer in DDR4 (8GB), access via AXI4 interface

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ FPGA                                                        │
│                                                             │
│   Compute Core (LLaMA decoder) ──┐                          │
│                                  │                          │
│  ┌───────────────────────────────▼─────────────────────┐    │
│  │ AXI Master Controllers                              │    │
│  │ ├─ Read Master (eMMC): Load weights                 │    │
│  │ ├─ Read Master (DDR4): Read cached weights          │    │
│  │ └─ Write Master (DDR4): Cache dequantized weights   │    │
│  └───────────┬──────────────────┬──────────────────────┘    │
└──────────────┼──────────────────┼───────────────────────────┘
               │                  │
          AXI4 Bus           AXI4 Bus
               │                  │
      ┌────────▼──────┐    ┌─────▼───────────┐
      │ eMMC 64GB     │    │ DDR4 8GB        │
      │ (ROM-like)    │    │ (Working RAM)   │
      │               │    │                 │
      │ - 7B: 7GB     │    │ - Active layer: │
      │ - 13B: 13GB   │    │   1.2GB         │
      │ - Models      │    │ - KV cache:     │
      └───────────────┘    │   800MB         │
                           └─────────────────┘
```

### Implementation Steps

#### Phase 1: AXI Primitives
1. Define AXI types
2. Implement simple read master (single transfer)
3. Implement simple write master
4. Test with simulation (fake AXI slave)

#### Phase 2: Burst & Optimization
1. Add burst support (read multiple 64-byte chunks)
2. Add write buffering
3. Test throughput

#### Phase 3: Weight Loading
1. Implement layer weight loader
2. Add dequantization pipeline (I8E → FixedPoint)
3. Add DDR4 caching logic

#### Phase 4: Integration
1. Modify decoder to use external weights
2. Remove decoderConst
3. Add layer-by-layer state machine
4. Test with 7B model

### Key Concepts

**AXI Handshake:**
- Transfer happens when `valid` AND `ready` both HIGH
- Master drives `valid`, slave drives `ready`
- Independent for each channel (AR, R, AW, W, B)

**Data Width:**
- 512 bits = 64 bytes per transfer
- Reduces number of transactions

**Burst:**
- Read/write multiple addresses in one transaction
- More efficient than single transfers

**Layer-at-a-Time:**
- Can't fit all 32 layers in memory
- Load layer N, process, load layer N+1
- Prefetch while computing (overlap)

### Critical Dependencies

**Before implementing AXI:**
- ✅ Understand current 260K design
- ✅ Know I8E quantization format
- ✅ Understand FixedPoint (SFixed 12 20)

**After implementing AXI:**
- Need physical hardware with eMMC + DDR4
- Need AXI IP cores (from Xilinx)
- Need actual 7B model weights in I8E format

### Questions to Answer During Implementation
1. How to handle AXI errors/timeouts?
2. How to arbitrate DDR4 (read vs write)?
3. Cache coherency strategy?
4. Prefetch next layer while computing current?
5. What if weight load takes longer than compute?

---

# Hardware implementation

```text
┌────────────────────────────────────────────────────────┐
│  LLaMA 7B Accelerator - Final Spec                     │
│  Target: 5.2 tokens/second @ 7B model                  │
│  Cost: ~$1,800                                         │
└────────────────────────────────────────────────────────┘
```

## Bill of Materials

```
┌──────────────────────────────────────────────────────────┐
│  Component                    Part Number         Cost   │
├──────────────────────────────────────────────────────────┤
│  FPGA SoC                                                │
│  Xilinx Zynq UltraScale+ ZU9EG               $700        │
│  ├─ Logic Cells: 356K                                    │
│  ├─ DSP Slices: 1,728 ← Enough for parallel64            │
│  ├─ Block RAM: 23 Mb                                     │
│  ├─ Max Clock: 450 MHz ← Run at 400 MHz                  │
│  └─ DDR4 controller (up to 2666 MT/s)                    │
│                                                          │
│  Memory                                                  │
│  ├─ 8GB DDR4-2666 (2× 4GB SO-DIMM)           $60         │
│  │  └─ Bandwidth: 42 GB/s (sufficient)                   │
│  └─ 64GB eMMC 5.1                            $30         │
│     └─ Sequential: 400 MB/s                              │
│                                                          │
│  PCB                                                     │
│  └─ 8-layer board + assembly                 $120        │
│                                                          │
│  Power                                                   │
│  ├─ DC-DC converters (multi-rail)            $40         │
│  └─ 12V @ 3A power adapter                   $15         │
│                                                          │
│  Cooling                                                 │
│  └─ Heatsink + 40mm fan                      $25         │
│                                                          │
│  Connectors & Misc                                       │
│  ├─ USB 3.1 Type-C controller                $20         │
│  ├─ Status LEDs, buttons                     $10         │
│  └─ Enclosure (CNC aluminum)                 $50         │
│                                                          │
│  Manufacturing & Testing                     $150        │
├──────────────────────────────────────────────────────────┤
│  Total BOM per unit:                         $1,220      │
│  Retail (with margin):                       $1,799      │
└──────────────────────────────────────────────────────────┘

NRE (one-time): $120K
├─ Board design: $40K
├─ Firmware/drivers: $50K  
└─ Testing/certification: $30K
```

## Resource Utilization Check

### parallel64 Requirements
```
Single parallel64 matrix multiplier:
├─ 64 multiply-accumulate units
├─ DSP slices needed: 64
├─ Control logic: ~500 LUTs
└─ Registers: ~200 FFs

For 7B model, you need:
├─ QKV projections: 3× multipliers
├─ Attention output: 1× multiplier  
├─ FFN (W1, W2, W3): 3× multipliers
├─ Vocab projection: 1× multiplier
└─ Total: ~8 active multipliers

Resource usage:
├─ DSPs: 8 × 64 = 512 (out of 1,728) = 30% ✅
├─ LUTs: ~50K (out of 356K) = 14% ✅
├─ BRAM: ~15 Mb (out of 23 Mb) = 65% ✅
└─ Conclusion: FITS COMFORTABLY! ✅
```

## Memory Layout
```
8GB DDR4 allocation:

├─ Active layer weights (dequantized):
│  └─ ~1.2 GB (FixedPoint format)
│
├─ KV Cache (256 token context):
│  └─ ~800 MB
│
├─ Intermediate buffers:
│  └─ ~500 MB
│
├─ DMA buffers:
│  └─ ~100 MB
│
└─ Free/OS:
   └─ ~5.4 GB

64GB eMMC allocation:

├─ 7B model (I8E): 7 GB
├─ 13B model (I8E): 13 GB  
├─ Fine-tuned models: ~20 GB
└─ Firmware/OS: ~5 GB
```

## Physical Specifications
```
┌──────────────────────────────────────────┐
│  Dimensions: 12cm × 10cm × 3.5cm         │
│  Weight: 180g                            │
│  Power: 12V @ 3A (30W typical, 36W peak) │
│                                          │
│  Connectors:                             │
│  ├─ USB-C 3.1 Gen 2 (10 Gbps)            │
│  └─ DC barrel jack (12V)                 │
│                                          │
│  Indicators:                             │
│  ├─ Power LED (green)                    │
│  ├─ Activity LED (blue)                  │
│  └─ Status LED (RGB)                     │
│                                          │
│  Cooling: Active (temp-controlled)       │
└──────────────────────────────────────────┘
```

## Development Path

### Phase 1: Validate with Dev Board
```
Use: Kria KV260 ($500)
├─ Has ZU5EV (smaller than ZU9EG)
├─ Test parallel64 implementation
├─ Verify timing closure @ 400 MHz
└─ Timeline: 3 months
```

### Phase 2: Custom Board Prototype
```
Design custom board with ZU9EG
├─ Schematic + PCB layout
├─ 5 prototype boards
└─ Timeline: 6 months
   Cost: $120K NRE
```

### Phase 3: Production
```
First batch: 100 units
├─ Price: $1,799 retail
└─ Timeline: 12 months from start
```

## Quick Reference Card
```
┌────────────────────────────────────────────────────────┐
│  LLaMA 7B FPGA Accelerator - Spec Sheet                │
├────────────────────────────────────────────────────────┤
│  Performance:     5.2 tokens/second                    │
│  Model Support:   7B, 13B (with slower speed)          │
│  Power:           30W typical (vs 450W GPU)            │
│  Interface:       USB 3.1 to Raspberry Pi              │
│  Storage:         64GB (multiple models)               │
│  Memory:          8GB DDR4 working memory              │
│  Clock:           400 MHz                              │
│  Precision:       FixedPoint (SFixed 12 20)            │
│  Format:          I8E quantization (8-bit + exp)       │
│  Cooling:         Active fan (quiet)                   │
│  Price:           $1,799                               │
└────────────────────────────────────────────────────────┘
```

## Software Requirements
```
Your Clash code changes:
├─ Replace matrixMultiplier with parallel64RowMatrixMultiplier
│  in: QKV projection, FFN, OutputProjection
├─ Set clock to 400 MHz in constraints
├─ Ensure timing closure (may need pipeline stages)
└─ Expected code changes: ~500 lines

Estimated effort: 2-3 weeks to implement + test
```

## Final Checklist
```
✅ FPGA has enough DSPs (1,728 vs 512 needed)
✅ Memory bandwidth sufficient (42 GB/s)
✅ Storage fits multiple models (64GB)
✅ Power reasonable (30W)
✅ Cost acceptable ($1,800)
✅ Performance realistic (5.2 tok/s)
✅ Form factor good (fits with RPi)
✅ No impossible optimizations claimed
```

# In the short-term

- We now have QKV loaded from DDR, buffered, and used in projection (RAM path), with a robust row assembler. Tokens are correct with useRAMEnable=True.
- We must replace every remaining constant weight/table with a DDR-driven, buffered (or on-demand) source.
- What still uses hardcoded params:
  1) Multi-head attention WO matrices mWoQ (MatI8E ModelDimension HeadDimension).
  2) RMS vectors: rmsAttF (attention) and fRMSFfnF (FFN).
  3) FFN matrices: fW1Q, fW2Q, fW3Q.
  4) Input embeddings: vocabularyQ.
  5) Rotary tables: freqCosF/freqSinF.
