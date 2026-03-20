# Outstanding Work: Activation BRAM Refactor

The `LayerData` wide bus has been replaced with an internal activation BRAM.
The pipeline is functional and the end-to-end simulation produces correct output.
Three tasks remain.

---

## Context: BRAM Layout

Four static slots, each `ModelDimension` words × 32 bits, flat-addressed as
`ActivationBramAddr = Index (4 * ModelDimension)`:

| Slot | Base address | Content |
|---|---|---|
| 0 | `0` | `inputVector` — written by init FSM, read by QKV and MHA residual |
| 1 | `ModelDimension` | *(unused)* |
| 2 | `2 * ModelDimension` | `attentionOutput` — written by MHA residual FSM, read by FFN |
| 3 | `3 * ModelDimension` | `feedForwardOutput` — written by FFN residual FSM, copied to slot 0 at layer boundary |

The BRAM is instantiated inside `TransformerLayer.hs` and exposed via a
`trueDualPortBlockRam` with port A (read) and port B (write).
**Read latency is 1 cycle**: address issued at cycle T, data available at T+1.

---

## Task 1 — Eliminate `OutputAccumulator` Vec Register

### What exists today

`WOHeadProjector.hs` wraps `OutputAccumulator.hs`, which holds a
`Vec ModelDimension FixedPoint` register (`qOut`).  It is built row-by-row:
each time `rowDone` fires with `(rowIndex, rowResult)`, element `rowIndex`
is replaced in `qOut`.

`MultiHeadAttention.hs` then:
1. Collects `woVecs :: Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint))`
2. Gates each by its `woValid`, sums across heads → `woHeads :: Signal dom (Vec ModelDimension FixedPoint)`
3. Snapshots `woHeads` at `attentionDone` via `regEn`
4. Runs a sequential residual FSM: reads slot 0 element-by-element, adds `woHeads[i]`, writes slot 2

This means there are `NumQueryHeads` wide Vec registers in flight simultaneously
before the residual FSM starts.

### Goal

Eliminate the `Vec ModelDimension FixedPoint` registers in `OutputAccumulator`
by writing WO rows directly into BRAM slot 2 as they complete, accumulating
in place. The residual add (slot0 + WO result) then reads slot 0 and slot 2
in a final pass instead of holding `woHeadsCapture`.

### Proposed approach

**Phase A — Zero-initialise slot 2 before WO projection starts**

Add a short FSM that writes `0` to all `ModelDimension` addresses of slot 2
before the WO heads begin. This runs in `ModelDimension` cycles and costs nothing
relative to the DRAM fetch latency of the WO weights.

**Phase B — Accumulate directly into BRAM slot 2**

Replace `OutputAccumulator` with a BRAM-write interface. When `rowDone` fires
for head `h` at row `i` with result `r`:
1. Read `slot2[i]` (one BRAM read cycle latency)
2. Write `slot2[i] += r`

Since multiple heads may complete different rows concurrently, accesses must be
serialised. Two strategies:

- **Serial heads** (simplest): run the `NumQueryHeads` WO projectors one at a time.
  Total WO latency multiplies by `NumQueryHeads`, but DRAM fetch is the bottleneck
  anyway, so the impact on overall throughput is likely negligible.

- **Row-granularity arbitration** (more complex): allow concurrent heads but
  queue write-back requests to slot 2; a single-port write arbitrator drains the queue.
  Saves cycles but adds logic.

Start with serial heads as the simpler option and measure the cycle count impact.

**Phase C — Replace residual FSM in `MultiHeadAttention.hs`**

After all WO heads complete, slot 2 holds `sum_h(WO_h[i])` for each `i`.
Replace the `woHeadsCapture`/`resLoadCounter` FSM with a new sequential pass:
- Read slot 0 address `i` and slot 2 address `i`
- Write slot 2 address `i` = slot0_data + slot2_data

This requires reading two BRAM addresses per element. Options:
- Use port A for slot 0 and port B for slot 2 (if dual-port read is available)
- Interleave reads over 2 cycles per element (simpler, doubles this FSM's duration)

### Relevant files

- `project/llama2/LLaMa2/Layer/Attention/QueryHeadProjector/OutputAccumulator.hs` — to be replaced
- `project/llama2/LLaMa2/Layer/Attention/WOHeadProjector.hs` — drives `OutputAccumulator`, exposes `woOut :: Signal dom (Vec ModelDimension FixedPoint)`
- `project/llama2/LLaMa2/Layer/Attention/MultiHeadAttention.hs` — collects `woVecs`, sums to `woHeads`, runs residual FSM
- `project/llama2/LLaMa2/Layer/TransformerLayer.hs` — owns the BRAM; may need slot 2 zero-init routed through its port mux

### Validation

The `ffnOut0` reference values below serve as the regression check after this change.

---

## Task 2 — Extend `ffnOut0` Reference Test

### What exists today

`test/LLaMa2/Decoder/DecoderSpec.hs` has the test:

```
"ffnOut0 matches Phase 1 baseline: token 0 layers 0-1"
```

It runs 20 000 cycles with prompt `[1]` (BOS token only) and checks `ffnOut0`
(i.e. `ffnOutput[0]`) at the first two `layerDone` events.

### Goal

Extend the test to cover all 5 layers of token 0. The reference values from
the Phase 1 baseline are:

| `layerDone` event | Layer | `ffnOutput[0]` |
|---|---|---|
| 0 | 0 | 0.34451 |
| 1 | 1 | 0.74667 |
| 2 | 2 | −0.38245 |
| 3 | 3 | −0.12680 |
| 4 | 4 | −0.40527 |

Tolerance: 0.001 (current test uses this successfully for layers 0–1).

The current 20 000-cycle budget covers 2 layers; covering all 5 layers of token 0
requires roughly 50 000 cycles (each layer is ~9 000–10 000 cycles at model-260k).

---

## Task 3 — Verify Multi-Token Autoregressive Loop

### Goal

Run the model-nano configuration (`--flag model-nano`) through at least 3 tokens
and confirm that the autoregressive state (KV cache, seqPos counter, slot 0 copy)
is consistent across token boundaries.

The model-nano config has fewer layers and smaller dimensions, making it practical
to run in a test (~100 000 cycles for 3 tokens).

Suggested test approach:
- Drive prompt `[1]` (BOS) with `--steps` sufficient to emit 3 tokens
- Capture the generated token IDs and compare against a known-good reference
  from the Phase 1 baseline (`cabal run llama2 --flag model-nano`)
- Alternatively, check that `seqPos` increments correctly and that `ffnOut0`
  at each layer boundary is numerically close to the Phase 1 value for
  the same token/layer combination

### Relevant files

- `test/LLaMa2/Decoder/DecoderSpec.hs` — add a new `describe "multi-token"` block
- `project/llama2/LLaMa2/Decoder/Decoder.hs` — `ffnStreamOut` accumulator, `lastLayerComplete`
- `project/llama2/LLaMa2/Layer/TransformerLayer.hs` — copy FSM (slot 3 → slot 0)
