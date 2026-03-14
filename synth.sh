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

# Ensure the library is compiled so Clash can load precompiled .hi files
cabal build lib:llama2 2>&1

HI_DIR=$(find dist-newstyle -path "*/llama2-0.1.0.0/build" -type d | head -1)
echo "    .hi dir    : $HI_DIR"

mkdir -p "$OUT"

# Default extensions from llama2.cabal — Clash doesn't read the cabal file
EXTENSIONS=(
  -XAllowAmbiguousTypes
  -XBangPatterns
  -XDataKinds
  -XDeriveAnyClass
  -XDeriveGeneric
  -XDerivingVia
  -XFlexibleContexts
  -XFlexibleInstances
  -XGeneralizedNewtypeDeriving
  -XImplicitParams
  -XInstanceSigs
  -XKindSignatures
  -XLambdaCase
  -XMonoLocalBinds
  -XNamedFieldPuns
  -XNoImplicitPrelude
  -XNumericUnderscores
  -XOverloadedStrings
  -XRecordWildCards
  -XScopedTypeVariables
  -XTemplateHaskell
  -XTupleSections
  -XTypeApplications
  -XTypeFamilies
  -XTypeOperators
  -XUndecidableInstances
)

cabal exec -- clash --verilog "$CPP_FLAG" \
  -iproject/llama2 \
  -hidir "$HI_DIR" \
  -odir  "$HI_DIR" \
  "${EXTENSIONS[@]}" \
  -outputdir "$OUT" \
  project/llama2/LLaMa2/Decoder/Decoder.hs \
  2>&1 | tee "$OUT/clash.log"

echo ""
echo "=== Verilog files generated ==="
find "$OUT" -name "*.v" | sort
echo ""
echo "=== Clash log summary ==="
grep -E "took:|error:|warning:|Compiling" "$OUT/clash.log" | tail -20
