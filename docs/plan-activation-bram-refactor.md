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

## Prioritisation

1. **`FFNProjector` w1 / w3 accumulators + `gateRaw` / `gateUpLatched`**
   Largest registers; primary cause of Vivado failure for 110m and 7b.
   Proof-of-concept already exists from the WO head projector prototype (see
   git history for the reverted Task 1 attempt — the streaming approach itself
   was correct; it was reverted only because of simulation speed and a Phase C
   residual-write bug).

2. **`WOHeadProjector` per-head accumulator**
   Already prototyped.  The serial-heads approach (one head at a time using the
   shared activation BRAM) correctly eliminates the `Vec ModelDimension`
   accumulator.  The simulation slowdown (8× due to serialising NumQueryHeads)
   is acceptable for synthesis; tests just need a larger `maxCycles` budget.
   The Phase C bug (spurious extra write) must be fixed:
   ```haskell
   -- WRONG (causes extra write on pipeline drain):
   inResWritePhase = (prevResActive .||. resDrain) .&&. prevResPhase .==. pure 1
   -- CORRECT:
   inResWritePhase = resActive .&&. prevResPhase .==. pure 1
   ```

3. **`QueryHeadProjector` / `KeyValueHeadProjector`**
   Smaller registers; lower urgency.  Tackle after FFN and WO are done.

## FFN refactor sketch

The three FFN phases become BRAM-mediated:

```
FPGate:  RowComputeUnit writes w1 results → BRAM slot A[0..HiddenDim-1]
FPUp:    RowComputeUnit writes w3 results → BRAM slot B[0..HiddenDim-1]
         (simultaneously reads slot A to apply SiLU element-by-element
          and writes SiLU(A)⊙B → BRAM slot C[0..HiddenDim-1])
FPDown:  RowComputeUnit reads slot C as column input, writes w2 results
         → BRAM slot D[0..ModelDim-1]
```

The element-wise `SiLU(gate) * up` product can be computed during a short
post-FPUp pass that reads slots A and B and writes slot C — or folded into the
FPDown column fetch if timing permits.

No `Vec HiddenDimension` register appears anywhere in this design.

## Simulation impact

The BRAM-backed approach serialises work that was previously done in a single
`replace` operation.  For large models the DRAM fetch is already the bottleneck
so the extra BRAM read cycles are hidden.  For the 260k simulation model the
cycle budget in `DecoderSpec` may need to increase; this is acceptable.

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
- [ ] QueryHeadProjector
- [ ] KeyValueHeadProjector
