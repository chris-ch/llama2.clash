# TODO

## Vivado elaboration OOM (110m / 7b models)

Vivado runs out of memory elaborating the generated Verilog for large models.
Two distinct root causes identified from inspecting the generated Verilog for
the 110m model (`temp/clash-verilog-model-110m/`).

---

### 1. Remaining `Vec ModelDimension FixedPoint` registers

`FPVecLoader` and `FFNProjector` still hold full `Vec ModelDimension FixedPoint`
values in flip-flop registers.  For 110m (ModelDimension=768): **24 576 bits
each**, and there are three of them.

| Register | Bits | File | Description |
|---|---|---|---|
| `outputVec_1` | 24 576 | `FPVecLoader` @ `QKVProjection` | rmsAtt weight vector, assembled from DRAM burst before use |
| `outputVec_2` | 24 576 | `FPVecLoader` @ `FeedForwardNetwork` | rmsFFN weight vector, same pattern |
| `wordBuffer_1` | 24 576 | `WeightsLayout` (DRAM burst buffer) | 48 × 512-bit AXI words assembled to build a ModelDimension Vec |
| `xHatLatched` | 24 576 | `FFNProjector` | Normed activation latched for the full w1/w3 row computation |

Root cause: `RowComputeUnit` accepts `rcColumn :: Signal dom (Vec ModelDimension
FixedPoint)`.  This forces every caller to hold the full activation vector in a
register for the duration of row computation.  The DRAM burst buffers accumulate
the same-sized Vec from AXI bursts before it can be passed in.

Fix: redesign `RowComputeUnit` to read its column input from a BRAM (one element
per cycle, via an address counter), eliminating the Vec column port entirely.
Callers write the activation into a small activation BRAM once, then RowComputeUnit
streams it.  The `FPVecLoader` burst buffers are replaced by the same BRAM.
Both rmsAtt and rmsFFN weight vectors, plus xHat, would be stored in per-use
BRAMs instead of Vec registers.

---

### 2. Large wires from flat multi-layer Clash elaboration

Clash inlines all `NumLayers` layer instances into a single flat Verilog module.
The per-layer output buses are concatenated into enormous wires that Vivado must
hold in memory during elaboration — even though they contain no flip-flops.

| Wire | Bits | Description |
|---|---|---|
| `kvBankResultsVec` | 303 300 | All KV bank controller outputs for all layers |
| `c$latchedHeadOutputs_app_arg_0` | 294 912 | 12 layers × `Vec NumQueryHeads (Vec HeadDimension FixedPoint)` |

Fix: add `{-# NOINLINE #-}` to the per-layer processing function (e.g.
`activeLayerProcessor` or `decoder`) so Clash emits it as a separate Verilog
module that is instantiated `NumLayers` times, rather than inlining everything
into one flat module.  This both reduces elaboration memory and produces cleaner
hierarchy for timing analysis.

See: https://clash-lang.org/docs/user-guide/synthesis-pragmas/#noinline
