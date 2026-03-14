# To do

**Pre-synthesis tasks**

1. **Trial synthesis + resource estimation** — run Clash + FPGA toolchain to confirm the design synthesises cleanly and check LUT/FF/BRAM utilisation.

   **Findings (2026-03-14):**
   - ✓ All 50+ modules compile through GHC+Clash in ~17 s — design is type-correct and synthesis-annotated.
   - ✓ `clash-ghc` added as `build-tool-depends` in `llama2.cabal`; `synth.sh` uses `cabal exec -- clash` — no global install required.
   - ✗ HDL elaboration of `topEntity` OOM'd on this machine (30 GB, NANO took 21 GB before kill; 260K ran 10+ h).
   - Root cause: `Vec VocabularySize FixedPoint` (512 elements even on NANO) forces Clash to normalise a large comparison tree in RAM. Blocked until one of the mitigations below is applied.
   - FPGA resource counts (LUT/FF/BRAM) require Vivado or Quartus on the generated Verilog — not installed here.

   **Mitigation options for elaboration OOM (choose one):**
   - **Split sampler**: annotate `tokenSampler` / `outputProjection` as a separate `topEntity` in its own module; synthesise independently from the rest of the decoder.
   - **BRAM argmax**: replace the combinatorial `Vec VocabularySize` argmax with a sequential BRAM-backed scan, reducing the normalised comparison tree to O(1) per cycle.
   - **Larger machine**: run synthesis on a host with ≥ 64 GB RAM without code changes.
