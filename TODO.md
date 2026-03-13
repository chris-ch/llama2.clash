# To do

**Pre-synthesis tasks**

1. **Merge KV DRAM to single bank per head** — current interface is `Vec NumLayers (Vec NumKeyValueHeads (Slave.AxiSlaveIn dom))` (e.g. 32×8 = 256 AXI ports), which is infeasible for synthesis. Since `KVCacheLayout` already encodes `layerIdx` as an address offset, collapse to `Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)` — one bank per KV head covering all layers via addressing. Update `LayerStack`, `MultiHeadAttention`, `KVCache`, `KVCacheBankController`, and `Decoder` interfaces accordingly.

2. **Strip all trace scaffolding from `llama2/LLaMa2/`** — remove all `traceEdgeC`, `traceChangeC`, `traceWhenC` calls and the `TraceUtils` import throughout the design. These were implementation aids only and must be absent from the synthesis-ready codebase.

3. **Remove vestigial `loadTrigger` / `layerChanged` from Decoder** — these signals appear to be remnants of the old weight-prefetch scheme. Verify they serve no purpose with on-demand AXI weight loading and delete if dead.

4. ~~**Annotate `Decoder` as the top-level synthesis entity**~~ ✓ — `topEntity = exposeClockResetEnable decoderTop` with full `Synthesize` port-name annotation added to `Decoder.hs`.

5. **End-to-end simulation test** — full decoder simulation exercising multiple token generation steps through all `NumLayers` passes of the single layer instance, verifying correct token output.

6. **Trial synthesis + resource estimation** — run Clash + FPGA toolchain to confirm the design synthesises cleanly and check LUT/FF/BRAM utilisation.
