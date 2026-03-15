#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# synth.sh — Run Clash Verilog synthesis for the LLaMA-2 decoder.
#
# Prerequisites:
#   - cabal (clash-ghc is declared as a build-tool-depends in llama2.cabal
#     and is fetched/built automatically by cabal — no global install needed)
#   - At least 64 GB RAM recommended for MODEL_260K full elaboration
#
# Usage:
#   ./synth.sh [model-nano|model-260k|model-15m|model-42m|model-110m|model-7b|model-13b|model-70b] [full|hierarchical]
#
#   full (default): synthesise the complete decoder in one Clash invocation.
#
#   hierarchical: synthesise each major block independently, bottom-up.
#     Leaf blocks are elaborated first in separate Clash processes; the top
#     entity is attempted last.  Stops at the first OOM/failure so you can
#     identify the blocking block without waiting for the full run.
#
#     Order:
#       1. token_sampler      (Sampler.hs)
#       2. input_embedding    (InputEmbedding.hs)
#       3. logits_projector   (OutputProjection.hs)
#       4. layer_runner       (LayerRunner.hs)
#       5. decoder/topEntity  (Decoder.hs)
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL=${1:-model-260k}
MODE=${2:-full}
OUT=/tmp/clash-verilog-$MODEL

case $MODEL in
  model-nano)   CPP_FLAG=-DMODEL_NANO   ;;
  model-260k)   CPP_FLAG=-DMODEL_260K   ;;
  model-15m)    CPP_FLAG=-DMODEL_15M    ;;
  model-42m)    CPP_FLAG=-DMODEL_42M    ;;
  model-110m)   CPP_FLAG=-DMODEL_110M   ;;
  model-7b)     CPP_FLAG=-DMODEL_7B     ;;
  model-13b)    CPP_FLAG=-DMODEL_13B    ;;
  model-70b)    CPP_FLAG=-DMODEL_70B    ;;
  *)            echo "Unknown model: $MODEL"; exit 1 ;;
esac

case $MODE in
  full|hierarchical) ;;
  *) echo "Unknown mode: $MODE (use 'full' or 'hierarchical')"; exit 1 ;;
esac

echo "=== Clash Verilog synthesis: $MODEL ($MODE) ==="
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

clash_run() {
  local label=$1
  local src=$2
  local main_is=$3
  local log="$OUT/clash-${label}.log"
  echo ""
  echo "--- [$label] $src (main-is: $main_is) ---"
  cabal exec -- clash --verilog "$CPP_FLAG" \
    -iproject/llama2 \
    -hidir "$HI_DIR" \
    -odir  "$HI_DIR" \
    "${EXTENSIONS[@]}" \
    -main-is "$main_is" \
    -outputdir "$OUT/$label" \
    "$src" \
    2>&1 | tee "$log"
  echo "--- [$label] done ---"
}

case $MODE in
  full)
    mkdir -p "$OUT/full"
    cabal exec -- clash --verilog "$CPP_FLAG" \
      -iproject/llama2 \
      -hidir "$HI_DIR" \
      -odir  "$HI_DIR" \
      "${EXTENSIONS[@]}" \
      -outputdir "$OUT/full" \
      project/llama2/LLaMa2/Decoder/Decoder.hs \
      2>&1 | tee "$OUT/clash.log"
    ;;

  hierarchical)
    # Bottom-up: leaves first, top entity last.
    # Each step runs in its own Clash process — a failure here means that
    # block is the OOM bottleneck; blocks above it are not attempted.
    clash_run "1-sampler"      project/llama2/LLaMa2/Sampling/Sampler.hs           samplerTop
    clash_run "2-input-embed"  project/llama2/LLaMa2/Embedding/InputEmbedding.hs   inputEmbeddingTop
    clash_run "3-output-proj"  project/llama2/LLaMa2/Embedding/OutputProjection.hs logitsProjectorTop
    clash_run "4-layer-runner" project/llama2/LLaMa2/Decoder/LayerRunner.hs        layerRunnerTop
    clash_run "5-decoder"      project/llama2/LLaMa2/Decoder/Decoder.hs            topEntity
    ;;
esac

echo ""
echo "=== Verilog files generated ==="
find "$OUT" -name "*.v" | sort
echo ""
echo "=== Clash log summary ==="
grep -rE "took:|error:|warning:|Compiling|Killed" "$OUT"/*.log "$OUT"/**/*.log 2>/dev/null | tail -30
