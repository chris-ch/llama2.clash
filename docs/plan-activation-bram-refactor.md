# Architectural Plan: Activation BRAM Refactor

## Problem

`LayerData` is a Haskell record carrying full activation vectors between pipeline stages.
Clash maps it to a flat combinational bus:

- 110M model: **451,212-bit** bus (55 KB/wire)
- 7B model: **2,359,328-bit** bus (294 KB/wire)

This causes Vivado to OOM during both elaboration and synthesis for the 7B model,
and makes the full decoder unsynthesisable at scale.

Root location: `layerDataReg = register initialLayerData nextLayerData` in `Decoder.hs`
and the wide mux tree feeding it. Also `activeLayerProcessor` in `LayerRunner.hs`.

---

## Goal

Replace `LayerData` (wide bus) with `LayerDataAddr` (8-bit constant) plus an
on-chip **Activation BRAM**. Activation vectors live in BRAM slots; pipeline stages
receive a slot address and a narrow read/write port — never a full `Vec`.

Inter-stage bus shrinks from **451K bits to 8 bits** (110M) and from **2.36M bits to
8 bits** (7B).

---

## BRAM Layout

Four static slots, each `ModelDimension` words × 32 bits:

| Slot | Content | Written by | Read by |
|---|---|---|---|
| 0 | `inputVector` | `layerInputStage` | QKV projection, MHA residual |
| 1 | `queryVectors` | QKV projection | KV attend phase |
| 2 | `attentionOutput` | MHA residual adder | FFN stage |
| 3 | `feedForwardOutput` | FFN residual adder | Layer boundary copy |

`keyVectors` / `valueVectors` already live in external DRAM KV cache — no change.

**110M:** 4 × 768 × 32 bits = 98,304 bits → fits in 3 × 36Kb BRAMs
**7B:** 4 × 4096 × 32 bits = 524,288 bits → ~15 × 36Kb BRAMs

Since the pipeline is strictly sequential (QKV → Attn+WO → FFN), port A (reads)
and port B (writes) are never contended. A small mux on `processingPhase` routes them.

---

## New Types (`Types/LayerData.hs`)

```haskell
type NumActivationSlots  = 4
type ActivationBramDepth = NumActivationSlots * ModelDimension
type ActivationBramAddr  = Index ActivationBramDepth

data LayerDataAddr = LayerDataAddr
  { inputVecSlot   :: Index NumActivationSlots  -- always 0 (compile-time constant)
  , queryVecSlot   :: Index NumActivationSlots  -- always 1
  , attnOutputSlot :: Index NumActivationSlots  -- always 2
  , ffnOutputSlot  :: Index NumActivationSlots  -- always 3
  } deriving (Generic, NFDataX, BitPack)

initialLayerDataAddr :: LayerDataAddr
initialLayerDataAddr = LayerDataAddr 0 1 2 3
-- All fields are compile-time constants → Clash folds to literals.
-- The 451K-bit layerDataReg in Decoder.hs disappears entirely.
```

`LayerData` stays in the file marked `{- Simulation only -}`.

---

## Phased Implementation

### Phase 1 — Foundation (no test breakage)

#### 1. `LLaMa2/Types/LayerData.hs`
- Add `LayerDataAddr`, slot types, `ActivationBramAddr`, `initialLayerDataAddr`.
- Keep `LayerData` with comment `{- Simulation only: not used in synthesised modules -}`.

#### 2. `LLaMa2/Memory/ActivationBRAM.hs` *(new file)*
Thin wrapper over `trueDualPortRam` from `DualPortRAM.hs`:

```haskell
data ActivationBramReadPort dom = ActivationBramReadPort
  { rdAddr :: Signal dom ActivationBramAddr }

data ActivationBramWritePort dom = ActivationBramWritePort
  { wrAddr :: Signal dom ActivationBramAddr
  , wrData :: Signal dom (Maybe FixedPoint) }

activationBram
  :: HiddenClockResetEnable dom
  => ActivationBramReadPort dom
  -> ActivationBramWritePort dom
  -> Signal dom FixedPoint   -- read data, 1-cycle latency
```

#### 3. `LLaMa2/Numeric/RmsNormSeq.hs`
- Expose internal `counter :: Signal dom (Index n)` as an additional return value.
- Callers use this to drive the BRAM read address in sync with element consumption.

```haskell
-- Before
rmsNormSeq :: ... -> (Signal dom Bool, Signal dom (Vec n FixedPoint))

-- After
rmsNormSeq :: ... -> (Signal dom Bool, Signal dom (Vec n FixedPoint), Signal dom (Index n))
--                                                                     ^^^^^^^^^^^^^^^^^^^^
--                                                                     exposed counter
```

---

### Phase 2 — Stage Interfaces

#### 4. `LLaMa2/Layer/FeedForward/FeedForwardNetwork.hs`
- Replace `Signal dom (Vec ModelDimension FixedPoint)` input with:
  `attnOutputSlot :: Index NumActivationSlots` + `ActivationBramReadPort` + `ActivationBramWritePort`
- Use exposed `rmsNormSeq` counter to drive BRAM read address.
- Sequential residual add: read slot 2 element `i`, add FFN core output `i`, write slot 3 address `i`.
- Return `ffnWriteDone :: Signal dom Bool` instead of a wide Vec.

#### 5. `LLaMa2/Layer/Attention/QKVProjection.hs`
- Replace full-Vec input with `inputVecSlot` + `ActivationBramReadPort`.
- Use exposed `rmsNormSeq` counter to drive BRAM read address (pre-issued 1 cycle early).
- Q/K/V output still goes to KV cache DRAM directly as before.

#### 6. `LLaMa2/Layer/Attention/MultiHeadAttention.hs`
- Remove `Signal dom LayerData` input entirely.
- New inputs: `inputVecSlot`, `attnOutputSlot`, `ActivationBramReadPort`, `ActivationBramWritePort`.
- `residualAdder` becomes a sequential FSM: read slot 0 addr `i`, add WO output `i`, write slot 2 addr `i`.
- `OutputAccumulator` writes directly to BRAM slot 2 as each row completes
  (eliminates its internal `Vec ModelDimension FixedPoint` accumulator register too).
- Return `attnWriteDone :: Signal dom Bool` instead of a wide Vec.

#### 7. `LLaMa2/Layer/Attention/KVCache.hs`
- Remove `Signal dom LayerData` input.
- Accept `keyVec`, `valVec`, `queries` directly (the `LayerData` projection glue was the only use).
- Minimal change.

---

### Phase 3 — Orchestration

#### 8. `LLaMa2/Layer/TransformerLayer.hs`
- Instantiate `ActivationBRAM` internally.
- Route BRAM ports to each stage via `processingPhase` mux (combinational, 12-bit address).
- `transformerLayer` return type loses all `Signal dom (Vec ...)` output fields.
- Only completion pulses (`qkvDone`, `attnDone`, `ffnDone`) remain in outputs.

#### 9. `LLaMa2/Decoder/LayerRunner.hs`
- Remove `qkvData`, `attnData`, `ffnData` wide record-update signals.
- `layerInputStage` → `layerBramInit`: a copy FSM running `ModelDimension` cycles.
  - Layer 0: write embedding Vec into slot 0 (one element/cycle).
  - Layer N>0: BRAM copy slot 3 → slot 0 (read addr `i`, write addr `i+1`-cycle pipeline).
- `LayerOutputs` loses all `Signal dom LayerData` / `Signal dom (Vec ...)` fields.
- Update `layerRunnerTop` `{-# ANN #-}` synthesis annotation (remove wide port names).

#### 10. `LLaMa2/Decoder/Decoder.hs`
- Delete `layerDataReg` and the `nextLayerData` wide mux tree entirely.
- `initialLayerData` → `initialLayerDataAddr` (8-bit constant, zero cost).
- Add BRAM-drain FSM after last layer: reads slot 3 one element/cycle into a local
  buffer (or directly streams) to feed `logitsProjector`.
  *(Or refactor `logitsProjector` to accept a BRAM port directly — can be a follow-on.)*
- Update `decoder` and `topEntity` type signatures.

---

### Phase 4 — Simulation Compatibility

#### 11. `LLaMa2/Simulation/LayerDataBramAdapter.hs` *(new, simulation-only)*

```haskell
-- Convert a LayerData into BRAM write commands for test setup
layerDataToWrites
  :: LayerDataAddr
  -> LayerData
  -> [(ActivationBramAddr, FixedPoint)]

-- Reconstruct a LayerData from a BRAM snapshot for reference comparison
layerDataFromBram
  :: LayerDataAddr
  -> Vec ActivationBramDepth FixedPoint
  -> LayerData

-- Drive BRAM write signals for ModelDimension cycles (simulation stimulus helper)
initBramFromLayerData
  :: LayerDataAddr -> LayerData
  -> [(Signal dom Bool, Signal dom ActivationBramAddr, Signal dom FixedPoint)]
```

#### 12. Test files
- `TransformerLayerSpec.hs`, `QKVProjectionSpec.hs`, `DecoderSpec.hs`
- Replace direct `LayerData { inputVector = repeat 1.0, ... }` with BRAM
  pre-population via `initBramFromLayerData` for the first `ModelDimension` cycles.
- Post-simulation comparison uses `layerDataFromBram` to extract and compare Vecs.

---

## Key Gotchas

### BRAM Read Latency (Most Critical)
`blockRamU` / `trueDualPortRam` are **synchronous-read**: data arrives one cycle
*after* the address is presented. Every FSM must pre-issue the read address one cycle
early. An off-by-one here causes silent wrong results.

Pattern: FSM counter `i` drives read address → data for element `i` arrives on cycle
`i+1` when the consumer processes it. The FSM should be in a "wait for first data"
state on cycle 1, then process element 0 on cycle 1, element 1 on cycle 2, etc.

**Validate with a simulation checker** comparing BRAM element reads against expected
`Vec` values before touching synthesis.

### `rmsNormSeq` Two-Pass
It reads the input vector twice (accumulate pass, then normalise pass). Both passes
drive the BRAM read address via the exposed counter. The counter must reset and re-run
for the second pass — no contention since the passes are sequential, but the
`validIn` triggering logic must account for the full 2-pass latency.

### Write-then-Read Ordering
`trueDualPortRam` (write-before-read mode) may return the written value on the same
cycle if the same address is simultaneously read and written. At layer boundaries,
port A reads slot 3 addr `i` and port B writes slot 0 addr `i` — different physical
addresses (slot 3 base ≠ slot 0 base), so no hazard. Verify this with explicit
address range assertions.

### Layer Boundary Copy Latency
Copying slot 3 → slot 0 adds `ModelDimension` cycles at each layer boundary:
- 260K model: 64 cycles × NumLayers = negligible
- 110M model: 768 cycles × NumLayers
- 7B model: 4096 cycles × NumLayers

Still negligible relative to DRAM weight-fetch latency, but test timing budgets
(`maxCycles` in specs) need adjusting.

### `OutputAccumulator` Secondary Wide Register
`OutputAccumulator` in `WOHeadProjector` holds a `Vec ModelDimension FixedPoint`
accumulator register internally. Address this in Phase 2 (MHA step): it should write
directly to BRAM slot 2 as each row completes rather than accumulating in a Vec.

### `layerRunnerTop` Synthesis Annotation
Currently has `PortName` entries for `"qkv_output"`, `"attn_output"`, `"ffn_output"`
mapped to `Signal System LayerData`. These port names must be updated in Phase 3.
Any Vivado IP-XACT or constraint files referencing these names will need updating too.

---

## What Disappears

| Item | Location | Why removed |
|---|---|---|
| `layerDataReg` (451K-bit FF bank) | `Decoder.hs` | Replaced by BRAM |
| `nextLayerData` wide mux tree | `Decoder.hs` | Address is a constant |
| `qkvData`/`attnData`/`ffnData` signals | `LayerRunner.hs` | No wide updates needed |
| `residualAdder` combinational function | `MultiHeadAttention.hs` | Replaced by sequential FSM |
| `OutputAccumulator` Vec register | `WOHeadProjector.hs` | Writes directly to BRAM |
| FFN residual add wide signal | `FeedForwardNetwork.hs` | Replaced by sequential FSM |
| Wide input Vec in QKV/FFN | `QKVProjection.hs`, `FeedForwardNetwork.hs` | BRAM read port replaces it |

---

## Files Changed Summary

| File | Change |
|---|---|
| `Types/LayerData.hs` | Add `LayerDataAddr` types; keep `LayerData` simulation-only |
| `Memory/ActivationBRAM.hs` | **New**: BRAM wrapper with read/write port records |
| `Numeric/RmsNormSeq.hs` | Expose `counter` in return type |
| `Layer/FeedForward/FeedForwardNetwork.hs` | BRAM ports replace Vec input/output |
| `Layer/Attention/QKVProjection.hs` | BRAM read port replaces Vec input |
| `Layer/Attention/MultiHeadAttention.hs` | Remove `LayerData`; sequential residual FSM |
| `Layer/Attention/KVCache.hs` | Remove `LayerData` import; take vecs directly |
| `Layer/TransformerLayer.hs` | Instantiate BRAM; route ports by phase |
| `Decoder/LayerRunner.hs` | Remove wide signals; `layerBramInit` copy FSM |
| `Decoder/Decoder.hs` | Delete `layerDataReg` + mux tree; add drain FSM |
| `Simulation/LayerDataBramAdapter.hs` | **New**: simulation-only adapter helpers |
| `*Spec.hs` test files | BRAM pre-population via adapter |
