# Plan: Eliminate Large Vec Registers via BRAM-backed Accumulators

## Problem

Vivado fails to elaborate the design for large models (110m, 7b) because several
components hold entire activation vectors in Clash `register` state — which maps
directly to flip-flop arrays.  The offending pattern is `OutputAccumulator`:

```haskell
-- OutputAccumulator.hs (today)
qOut = register (repeat 0) $
  mux rowDone (replace rowIndex rowResult qOut) qOut
```

`qOut :: Vec N FixedPoint` becomes a `reg [32*N-1:0]` in the generated Verilog.

### Affected sites and register sizes

| Site | Vec size | 260k bits | 110m bits | 7b bits |
|---|---|---|---|---|
| `FFNProjector` w1 accum (`gateRaw`) | `HiddenDimension` | 5 504 | **65 536** | **352 256** |
| `FFNProjector` w3 accum (`gateUpLatched`) | `HiddenDimension` | 5 504 | **65 536** | **352 256** |
| `FFNProjector` w2 accum (`outputResult`) | `ModelDimension` | 2 048 | 24 576 | 131 072 |
| `WOHeadProjector` per head | `ModelDimension` | 2 048 | 24 576 | 131 072 |
| `QueryHeadProjector` per head | `ModelDimension` | 2 048 | 24 576 | 131 072 |
| `KeyValueHeadProjector` per head | `HeadDimension` | 256 | 2 048 | 4 096 |

For the 7b model the two FFN hidden-dimension accumulators alone add up to
**~88 KB of flip-flop state** — plus the intermediate `gateRaw` and
`gateUpLatched` `Vec HiddenDimension` registers that hold results between FFN
phases, pushing the total over **170 KB**.  Vivado cannot elaborate this.

The 260k simulation model is fine (all sizes are small).  The 110m model is
borderline; the 7b model definitively fails.

## Root cause

Every `OutputAccumulator` instantiation introduces a `Vec numRows FixedPoint`
register.  For large models `numRows` is `HiddenDimension` (up to 11 008 for 7b)
or `ModelDimension` (up to 4 096 for 7b), producing registers measured in tens
of kilobits.

Beyond the accumulators, `FFNProjector` also holds two full
`Vec HiddenDimension` snapshots (`gateRaw`, `gateUpLatched`) between phases.
These must be eliminated at the same time.

## Approach

Replace every large `Vec` register with a BRAM-backed streaming output.
Results are written element-by-element into a dedicated activation BRAM slot as
they are produced by `RowComputeUnit`, and are read back element-by-element by
the downstream phase.

### Key invariant

`RowComputeUnit` produces one `(rowIndex, rowResult)` pair per row in strictly
increasing index order.  This makes the BRAM write sequentially addressed,
which is the ideal access pattern.

### Activation BRAM slot allocation

The activation BRAM already has named slots (slot 0–3).  The FFN intermediate
results (`gateRaw`, `gateUpLatched`) can reuse existing free slots or be
allocated new ones.  The WO and QKV projector outputs already flow through
slot 2 / slot 3 post-residual; their intermediate accumulators are the only
new consumers.

### Interface change per component

Instead of:
```haskell
oaOutput :: Signal dom (Vec numRows FixedPoint)   -- large Vec register
```

The accumulator writes to BRAM and exposes:
```haskell
oaBramWrite :: Signal dom (Maybe (BramAddr, FixedPoint))  -- element-by-element write
oaDone      :: Signal dom Bool                            -- all rows written
```

Downstream reads back via the shared BRAM read port, one element per cycle,
instead of indexing a Vec.

## Prioritisation (completed)

All originally-identified Vec register sites have been eliminated.  The order
of implementation was:

1. **`FFNProjector` w1 / w3 accumulators + `gateRaw` / `gateUpLatched`** ✓
2. **`FFNProjector` w2 accumulator** ✓
3. **`WOHeadProjector` per-head accumulator** ✓  (serial-heads, activation BRAM)
4. **`KVCacheBankController` QDot timing bug** ✓  (bug fix, not a Vec refactor)
5. **`QueryHeadProjector`** ✓  (pair-wise RoPE inline, BRAM write output)
6. **`KeyValueHeadProjector`** ✓  (K: inline RoPE; V: direct; KVBC word BRAMs)

## FFN refactor (implemented)

The three FFN phases are BRAM-mediated as planned:

```
FPGate:  RowComputeUnit writes w1 results → FFN BRAM slot A[0..HiddenDim-1]
FPUp:    RowComputeUnit writes w3 results → FFN BRAM slot B[0..HiddenDim-1]
         post-FPUp pass: SiLU(A)⊙B → FFN BRAM slot C[0..HiddenDim-1]
FPDown:  RowComputeUnit reads slot C as column, writes w2 results
         → FFN BRAM slot C offset [2×HiddenDim .. 2×HiddenDim+ModelDim-1]
```

No `Vec HiddenDimension` register appears anywhere in this design.

`DecoderSpec` `maxCycles` was increased from 20 000 → 60 000 to account for
the serial FPDown column read latency.

## Simulation impact (resolved)

All 113 tests pass with the full BRAM-backed design.  The cycle budget increase
in `DecoderSpec` was the only simulation accommodation needed.

## Remaining cleanup

- `KeyValueHeadProjector/OutputAccumulator.hs` — dead code, can be deleted.
- `QueryHeadProjector/OutputAccumulator.hs` — dead code, can be deleted.

## Next steps

- Run Vivado elaboration against the 110m or 7b model config to confirm the
  flip-flop count is within bounds and synthesis succeeds.

## Status

- [x] FFNProjector w1/w3 accumulators + intermediate Vec registers
      — `gateRaw`, `gateUpLatched`, and the w1/w3 `OutputAccumulator` Vec
        registers replaced by a 2-slot FFN-internal BRAM
        (depth = 2 × HiddenDimension).
      — FPDown column now read serially from BRAM (1 element/cycle).
        DecoderSpec `maxCycles` increased from 20 000 → 60 000 to account for
        the serial FPDown latency (acceptable per plan).
      — ffnOut0 reference values match Phase 1 baseline within tolerance.
- [x] FFNProjector w2 accumulator (outputResult Vec ModelDimension)
      — w2 results written element-by-element to FFN BRAM slot C
        (depth extended to 2×HiddenDim + ModelDim).
      — FeedForwardNetwork residual FSM reads slot C via exposed BRAM port;
        projectorReadyIn gated by resIdle + not coreValidOutRise to prevent
        the 1-cycle gap where FPDone→FPIdle could fire before resIdle latches.
- [x] WOHeadProjector per-head accumulator
      — OutputAccumulator (Vec ModelDimension FixedPoint per head × 12 heads)
        eliminated. Heads now run serially gated by headActive counter.
      — Head 0 reads residual from slot 0; heads 1..N-1 accumulate into slot 2.
        Each head writes slot2[i] = rdBase[i] + projectedRow[i] element-by-element.
      — woHeadsCapture (Vec ModelDimension snapshot) and the separate residual FSM
        both removed from MultiHeadAttention.
      — Per-head attention outputs latched individually at perHeadDoneRise to
        remain stable throughout the serial WO phase.
      — writeDone fires one cycle after the last head's outputValid pulse.
      — ffnOut0 reference values and all model-nano decoder tests still pass.
- [x] KVCacheBankController QDot counter timing bug
      — `qDotDone` fired spuriously on the first cycle of every QDot phase
        because `qDotCounter` still held `maxBound` from the previous phase
        (the reset to 0 takes effect the *next* cycle).  This caused QDot1 to
        accumulate zero contributions (dotAcc1=0) and QDot0 row 2+ to also
        fire immediately, producing wrong attention scores for seqPos≥1.
      — Fix: guard `qDotDone` with `.&&. (not <$> (enterQDot0 .||. enterQDot1))`.
      — Verified: model-nano multi-token test (seqPos=1) and model-260k
        ffnOut0 reference test both pass after the fix.
- [x] QueryHeadProjector
      — OutputAccumulator (Vec HeadDimension FixedPoint per head) eliminated.
        QueryHeadProjector.hs now writes directly to a per-head Q BRAM via
        pair-wise RoPE encoding on-the-fly (`Maybe (Index HeadDimension, FixedPoint)`).
      — QueryHeadProjector/OutputAccumulator.hs still exists but is no longer
        imported or used; can be deleted.
- [x] KeyValueHeadProjector
      — `Vec HeadDimension FixedPoint` OutputAccumulators (K and V) eliminated.
      — K path: pair-wise RoPE applied inline as rows complete (same pattern as
        QueryHeadProjector); emits `Maybe (Index HeadDimension, FixedPoint)`.
      — V path: each row result emitted directly as `Maybe (Index HeadDimension, FixedPoint)`.
      — KVCacheBankController: `latchedKWords`/`latchedVWords` (Vec FFs) replaced
        by K and V AXI-word BRAMs (depth = WordsPerFPVec HeadDimension).
        Elements are packed into 512-bit words on arrival via `insertFP` helper
        (matching fixedPointVecPackerVec's little-endian layout); completed words
        are written to BRAM.  Write master reads from BRAM using `nextWriteBeat`
        as the read address (1-cycle ahead) so data is available when each beat fires.
      — QKVProjection and MultiHeadAttention wiring updated accordingly.
      — Dead code: KeyValueHeadProjector/OutputAccumulator.hs and
        QueryHeadProjector/OutputAccumulator.hs are no longer imported by anyone.
      — All 113 tests pass after the refactor.
