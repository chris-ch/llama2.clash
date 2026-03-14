# To do — Verilog Generation

## Blocker: elaboration OOM

Clash's HDL elaboration of `topEntity` runs out of memory due to
`Vec VocabularySize FixedPoint` (512 elements even on NANO) forcing
normalisation of a large comparison tree in RAM.

**Findings (2026-03-14):**
- NANO model: 21 GB before OOM on a 30 GB machine
- 260K model: did not finish in 10+ hours
- All 50+ modules compile through GHC+Clash in ~17 s — design is type-correct

**Mitigation options (pick one):**

1. **Split sampler** — annotate `tokenSampler` / `outputProjection` as a
   separate `topEntity` in its own module; synthesise independently.
   Low risk, minimal code change.

2. **BRAM argmax** — replace the combinatorial `Vec VocabularySize` argmax
   with a sequential BRAM-backed scan, reducing the normalised comparison
   tree to O(1) per cycle. Correct long-term solution; larger change.

3. **Larger machine** — run synthesis on a host with ≥ 64 GB RAM, no code
   changes needed.

---

## Hierarchical Verilog output

Apply `{-# NOINLINE #-}` to major sub-components so Clash emits a
separate `.v` file per block (instantiated in the parent) rather than
one flat module. Benefits: smaller per-file elaboration, faster
incremental re-runs, cleaner output for downstream EDA tools.

Candidate boundaries:
- `LayerRunner.activeLayerProcessor`
- `InputEmbedding.inputEmbedding`
- `OutputProjection.logitsProjector`
- `Sampler.tokenSampler`

Prerequisite: OOM blocker resolved first (elaboration must complete).

---

## Downstream EDA

Once Verilog is generated:
- Run Vivado / Quartus for LUT / FF / BRAM resource counts
- Check timing closure target (400 MHz per PRODUCT.md)
- Verify DSP usage (~512 / 1728 estimated in PRODUCT.md)
