#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# synth.sh — Run Clash Verilog synthesis for the decoder top entity.
#
# Prerequisites:
#   - cabal (clash-ghc is declared as a build-tool-depends in llama2.cabal
#     and is fetched/built automatically by cabal — no global install needed)
#   - At least 64 GB RAM recommended for MODEL_260K elaboration
#
# Usage:
#   ./synth.sh [model-nano|model-260k]   (default: model-260k)
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL=${1:-model-260k}
OUT=/tmp/clash-verilog-$MODEL

case $MODEL in
  model-nano)   CPP_FLAG=-DMODEL_NANO   ;;
  model-260k)   CPP_FLAG=-DMODEL_260K   ;;
  *)            echo "Unknown model: $MODEL"; exit 1 ;;
esac

echo "=== Clash Verilog synthesis: $MODEL ==="
echo "    Output dir : $OUT"

mkdir -p "$OUT"

cabal exec -- clash --verilog "$CPP_FLAG" \
  -iproject/llama2 \
  -outputdir "$OUT" \
  project/llama2/LLaMa2/Decoder/Decoder.hs \
  2>&1 | tee "$OUT/clash.log"

echo ""
echo "=== Verilog files generated ==="
find "$OUT" -name "*.v" | sort
echo ""
echo "=== Clash log summary ==="
grep -E "took:|error:|warning:|Compiling" "$OUT/clash.log" | tail -20
