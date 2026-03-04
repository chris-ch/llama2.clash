# Design choices and constraints
    Toolchain: Clash 1.8.2, GHC 9.6.7.
    Models: LLaMA 2 7B (primary), 13B (supported). 70B is out of scope.
    Context window: < 16k tokens.
    AXI policy: streaming reads for inference; writes disabled in synthesized compute core. A boot-time write path may be enabled to populate/copy/cached data in DDR.
    Memory access: no register mirrors; strictly streaming reads with burst support (≤256-beat bursts).
    Boot: offline external binary loaded at boot; optional on-device FP32→I8E conversion on first boot, then persist I8E to eMMC to avoid re-conversion.

# Software implementation
## AXI Interface Implementation Summary
### Context

    Scaling from 260K to 7B/13B (weights external, but temporarily embedded in FPGA as scaffolding during early development) ).
    **Problem:** 7B ≈ 7 GB, 13B ≈ 13 GB of weights → cannot embed in BRAM.
    **Solution:** Store weights on eMMC (64 GB). Cache the active layer in DDR4 (8 GB). Compute core fetches rows via AXI4 read bursts.

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
        Read Master (eMMC): used by PS/DMA for staging layers or by a boot-time converter.
        Read Master (DDR4): used by the compute core to fetch cached I8E rows.
        Write Master (DDR4): disabled in synthesized decoder; enabled for boot-time conversion/caching or tests.

### Implementation Steps
#### Phase 1: AXI Primitives

    Define AXI types (512-bit data, 32-bit address).
    Implement simple read master (single-beat).
    Implement read burst master (≤256 beats).
    Optional: minimal write path (for boot-only writes/tests). Synthesize OFF by default.
    Test with simulation (fake AXI slave / DRAMBackedAxiSlave).

#### Phase 2: Throughput & Robustness

    Burst support: single-burst-per-row for 7B/13B (arsize=6, arburst=INCR, arlen=beats-1 ≤ 255).
    Optional: burst chunking utility (not required for 7B/13B; keep for future).
    Optional write buffering (boot path only).
    Measure effective throughput and stalls.

#### Phase 3: Weight Loading and Format

    Boot-time loader:
        If I8E artifact exists on eMMC: copy layer N to DDR.
        Else: read FP32 from eMMC, convert row-wise to I8E, write I8E artifact back to eMMC (preferred), and copy active layer to DDR.
    Dequantization pipeline in compute core: I8E → SFixed (SFixed 12 20).
    DDR4 caching logic: layer-at-a-time; overlap prefetch with compute.

#### Phase 4: Integration

    Modify decoder to use external I8E weights via AXI row fetchers.
    Remove decoderConst.
    Add layer-by-layer control/state machine.
    Validate on 7B; then 13B.

### Key Concepts

    **AXI handshake:** valid ∧ ready on each channel (AR, R, AW, W, B).
    **Data width**: 512 bits (64 bytes/beat).
    **Burst rules**: arlen ≤ 255 (≤256 beats). 7B/13B rows fit a single burst.
    **Layer-at-a-Time**: fetch layer N, compute, prefetch N+1.

### Critical Dependencies

**Before implementing AXI**:

    ✅ Current 260K design understood.
    ✅ I8E format agreed (row-wise mantissas + 1 exponent).
    ✅ FixedPoint target (SFixed 12 20) established.

**After implementing AXI**:

    eMMC + DDR4 hardware available.
    Xilinx AXI IPs or PS DMA available.
    7B/13B model weights available as FP32 and/or I8E.
    For 13B: verify timing closure and burst throughput.

### Questions to Answer During Implementation

    How to handle AXI errors/timeouts? (OKAY-only now; add SLVERR path later.)
    DDR arbitration (read-vs-write) during boot-only writes?
    Cache coherency (only PS writes during boot; compute is read-only).
    Prefetch overlap policy (double-buffer layer regions)?
    Compute vs load balance: what if load stalls compute?

---

# Hardware implementation

## LLaMA 7B/13B Accelerator – Final Spec (revised)

    Target: ~5.2 tokens/s @ 7B (13B proportionally slower).
    Cost: ~$1,800.

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

**Synthesis notes for 13B**:
- Burst lengths per row:
            ModelDimension=5120 → WordsPerRow = ceil((5120+62)/63) = 82 beats (single burst).
            HiddenDimension=13824 → WordsPerRow = ceil((13824+62)/63) = 220 beats (single burst).
- Both satisfy AXI arlen ≤ 255.

## Memory Layout (revised with <16k context)

    Active layer weights (dequantized/SFixed) ≈ 1.2 GB (7B-class).
    KV cache budget depends on precision and context:
        Per-token KV size (bytes) = 2 · NumLayers · NumKVHeads · HeadDim · bytesPerElem. For 7B: NumLayers=32, NumKVHeads=32, HeadDim=128 → 2 · 32 · 32 · 128 = 262,144 elements/token.
            8-bit KV: ≈ 256 KB/token → 16k tokens ≈ 4.0 GB.
            16-bit KV: ≈ 512 KB/token → 16k tokens ≈ 8.0 GB.
        To keep total under 8 GB with an active layer, prefer 8-bit KV for long contexts, or cap context well below 16k if KV is 16-bit.
    Intermediate + DMA buffers: ~0.5–0.7 GB.
    The remainder for OS/free.

eMMC allocation

    7B I8E: ~7 GB.
    13B I8E: ~13 GB.
    Optional: FP32 originals retained for reference or first-boot conversion (delete after conversion to reclaim space).
    Firmware/OS: ~5 GB.

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

    Performance: ~5.2 tok/s (7B).
    Model Support: 7B, 13B (70B excluded).
    Precision: FixedPoint (SFixed 12 20) in compute; I8E weights on DDR.
    Context: < 16k tokens (recommend 8-bit KV for long contexts).
    Interface/Power/Clock: unchanged.

## Software Requirements

    Replace matrixMultiplier with parallel64RowMatrixMultiplier in QKV, FFN, OutputProjection.
    400 MHz clock; insert pipeline stages as required for timing closure.
    AXI row fetchers:
        512-bit, single-burst-per-row for 7B/13B (arsize=6, arburst=INCR, arlen=beats-1).
        Keep burst-chunking utility in software/sim; not required for 7B/13B.
    AXI writes:
        Synthesized compute core: writes disabled.
        Boot path/tests: enable write master to populate DDR (or to persist I8E to eMMC).
    Expected code changes: ~500 lines; 2–3 weeks including tests.

## DRAM Weight Migration Status

### Completed
    ✅ Q matrices  — DRAM-backed via AXI; 8 heads, 8 masters in arbiter
    ✅ K matrices  — DRAM-backed via AXI; 4 KV heads, 4 masters in arbiter
    ✅ V matrices  — DRAM-backed via AXI; 4 KV heads, 4 masters in arbiter
    ✅ WO matrices — DRAM-backed via 2-level arbiter; 8 Q-heads, 8 masters
    All 24 AXI masters share a single DRAM slave via a 2-level arbiter tree:
      MHA top arbiter (2 masters): QKV sub-arbiter (16) + WO sub-arbiter (8).
    HC parallel path runs alongside DRAM for regression assertions during migration.
    OutputChecker stale-data bug found and fixed (rising-edge capture pattern).
    All 101 tests pass on branch `dram`.

### Next: FFN Weight Matrices (W1, W2, W3)

FFN has three matrices per layer, computed sequentially: W1 (gate) → W3 (up) → W2 (down).

Matrix shapes (260K model: ModelDimension=64, HiddenDimension=172):
  W1: `MatI8E HiddenDimension ModelDimension` — 172 rows × 64 cols (input: Vec 64)
  W3: `MatI8E HiddenDimension ModelDimension` — 172 rows × 64 cols (input: Vec 64)
  W2: `MatI8E ModelDimension HiddenDimension` — 64 rows × 172 cols (input: Vec 172)
Beats per row: W1/W3 → WordsPerRow ModelDimension (same as Q/K/V); W2 → WordsPerRow HiddenDimension.
`W1Matrix`, `W2Matrix`, `W3Matrix` already in `WeightsLayout.MatrixType`.
No per-head indexing; `headIdxInt = 0` always (one projector per matrix per layer).

Key technical challenges:

1. **Sequential pipeline, not parallel**: unlike Q/K/V/WO (all heads in parallel),
   FFN runs Gate→Up→Down in strict order. Only one DRAM master is active at a time.
   Design choice: 3 separate weight loaders with a 3-master sub-arbiter inside
   a new `FFNProjector` module (cleaner; only the active phase issues AR requests).

2. **Two input widths**: W1/W3 take `Vec ModelDimension FixedPoint` (64 elements);
   W2 takes `Vec HiddenDimension FixedPoint` (172 elements = `gateUpLatched`).
   The generic RowComputeUnit already supports arbitrary column widths after WO migration.

3. **Arbiter extension at transformer layer level**:
   Currently `transformerLayer` passes `dramSlaveIn` directly to MHA.
   FFN needs its own DRAM slave. Add a 2-master arbiter inside `transformerLayer`:
     Master 0 → MHA's existing arbiter tree
     Master 1 → FFN's 3-master sub-arbiter (W1, W3, W2 — sequential, so at most 1 active)

4. **FFN is much larger** at 7B scale: HiddenDimension=11008 → 220-beat bursts.
   The generic loader already handles multi-beat rows; 260K simulation uses short rows.

Migration steps:
    Step 1: Add `w1WeightLoader`, `w3WeightLoader`, `w2WeightLoader` to `WeightLoader.hs`.
            Use `Layout.W1Matrix`/`W3Matrix`/`W2Matrix`; HC from `PARAM.fW1Q`/`fW3Q`/`fW2Q`.
    Step 2: Build `FFNProjector` module replacing `feedForwardCore`'s 3× `parallelRowMatrixMultiplier`.
            Each phase (FFNGate/FFNUp/FFNDown) uses its weight loader + RowScheduler + RowComputeUnit + OutputAccumulator.
            3-master internal sub-arbiter; only active phase drives AR requests.
    Step 3: Expose `ffnAxiMasterOut` from `FFNProjector`; wire into `TransformerLayer.hs`.
            Add 2-master top arbiter: MHA master + FFN master (sequential, no contention in practice).
    Step 4: Retain HC cross-check (row-level and output-level assertions) until tests pass.
    Step 5: Validate: all 101 tests pass with DRAM FFN + HC cross-check active.
    Step 6: (Optional cleanup) Remove HC path and `FeedForwardNetworkComponentQ` param.

### Remaining small parameters (low priority — not synthesis blockers at 260K scale)
    ⬜ rmsAttF  — Vec 64 FixedPoint per layer (attention RMS norm weights)
    ⬜ fRMSFfnF — Vec 64 FixedPoint per layer (FFN RMS norm weights)
    ⬜ rmsFinalWeightF — Vec 64 FixedPoint (final output norm weights)
    ⬜ rotaryEncoding — Vec SequenceLength (Vec 16 FixedPoint) cos/sin tables
    These are tiny; can live in small BRAMs or stay as ROM constants.

## Final Checklist

    ✅ Models: 7B/13B only; 70B excluded.
    ✅ Context window < 16k; DDR budget met with 8-bit KV for long contexts.
    ✅ AXI bursts per row ≤ 256 beats (7B/13B OK).
    ✅ 32-bit AXI address OK with per-layer windows < 4 GiB.
    ✅ FPGA resources sufficient; timing at 400 MHz achievable with pipelining.
    ✅ DDR bandwidth adequate; overlap prefetch with compute.
    ✅ eMMC capacity sufficient; first-boot FP32→I8E conversion feasible, then reuse I8E.

## Notes and tips

    Keep the DRAMBackedAxiSlave’s write path behind a flag (tests/boot only). Inference builds should be read-only for simplicity.
    Use the compile-time WordsPerRow dim to size buffers and to assert arlen ≤ 255 in 7B/13B builds.
    Add a CI test that computes KV budget from model constants and fails if the requested context×precision would exceed the DDR allocation.
