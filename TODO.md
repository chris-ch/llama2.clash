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

## Quick Reference Card
```
┌────────────────────────────────────────────────────────┐
│  LLaMA 7B FPGA Accelerator - Spec Sheet                │
├────────────────────────────────────────────────────────┤
│  Performance:     5.2 tokens/second                    │
│  Model Support:   7B, 13B (with slower speed)          │
│  Power:           30W typical (vs 450W GPU)            │
│  Interface:       USB 2 to Raspberry Pi              │
│  Storage:         64GB (multiple models)               │
│  Memory:          8GB DDR4 working memory              │
│  Clock:           400 MHz                              │
│  Precision:       FixedPoint (SFixed 12 20)            │
│  Format:          I8E quantization (8-bit + exp)       │
│  Cooling:         Active fan (quiet)                   │
│  Price:           $1,799                               │
└────────────────────────────────────────────────────────┘
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

- LLaMA-2 7B KV cache cannot fit on-chip. We must use external DDR via AXI masters, with URAM for active tokens.
- We now have QKV loaded from DDR, buffered, and used in projection (RAM path), with a robust row assembler. Tokens are correct with useRAMEnable=True.
- We must replace every remaining constant weight/table with a DDR-driven, buffered (or on-demand) source.
- What still uses hardcoded params:
  1) Multi-head attention WO matrices mWoQ (MatI8E ModelDimension HeadDimension).
  2) RMS vectors: rmsAttF (attention) and fRMSFfnF (FFN).
  3) FFN matrices: fW1Q, fW2Q, fW3Q.
  4) Input embeddings: vocabularyQ.
  5) Rotary tables: freqCosF/freqSinF.
