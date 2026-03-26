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

# ---------------------------------------------------------------------------
# Post-synthesis design complexity report
#
# Vivado RTL elaboration can crash (OOM / silent kill) when individual wires
# exceed ~50 K bits, because it builds per-bit internal data structures.
# This report surfaces the widest wires per module so you can spot the
# bottleneck before running Vivado.
# ---------------------------------------------------------------------------
echo ""
echo "=== Design complexity report ==="
echo "    (Widest declared wires per .v file — crash risk grows above ~50 K bits)"
echo ""
printf "  %-10s  %-10s  %-8s  %s\n" "MaxBits" "Lines" "Size" "File"
printf "  %-10s  %-10s  %-8s  %s\n" "-------" "-----" "----" "----"

while IFS= read -r vfile; do
    max_bits=$(grep -oP '(?<=\[)[0-9]+(?=:0\])' "$vfile" 2>/dev/null \
               | awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m+0}')
    lines=$(wc -l < "$vfile")
    size=$(du -sh "$vfile" | cut -f1)
    rel="${vfile#$OUT/}"
    # Highlight files where the widest wire exceeds 50 K bits
    if [ "$max_bits" -gt 50000 ] 2>/dev/null; then
        flag="  <-- WARNING: wide bus"
    else
        flag=""
    fi
    printf "  %-10s  %-10s  %-8s  %s%s\n" "$max_bits" "$lines" "$size" "$rel" "$flag"
done < <(find "$OUT" -name "*.v" | sort)

echo ""
echo "=== System memory ==="
free -h | head -2

echo ""
echo "--- Vivado elab memory estimate ---"
# Heuristic: Vivado needs roughly 30–100 bytes per bit for the widest wire
# in its internal representation.  The estimate below uses the 100-byte
# worst-case factor so you know the safe lower bound on available RAM.
all_max=$(find "$OUT" -name "*.v" -exec grep -oP '(?<=\[)[0-9]+(?=:0\])' {} \; 2>/dev/null \
          | awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m+0}')
if [ "${all_max:-0}" -gt 0 ]; then
    # Heuristic: Vivado's elaboration is dominated by expression-tree nodes,
    # not raw storage.  Empirically, wires wider than ~50 K bits cause
    # multi-GB memory spikes; at 300 K bits expect 50–200 GB peak usage.
    echo "    Widest wire across design : ${all_max} bits"
    avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
    avail_gb=$(awk "BEGIN{printf \"%.1f\", $avail_kb / (1024*1024)}")
    echo "    Available RAM now         : ${avail_gb} GB"
    if [ "$all_max" -gt 50000 ]; then
        echo ""
        echo "    ACTION: wire(s) wider than 50 K bits detected — Vivado elab of the"
        echo "    full topEntity is likely to OOM (empirically: >50 K bits -> multi-GB"
        echo "    spikes; >300 K bits -> 50-200 GB peak).  Consider running elab on the"
        echo "    sub-entities (InputEmbedding, rowComputeUnit, etc.) individually"
        echo "    to isolate which module carries the oversized bus."
    fi
fi
