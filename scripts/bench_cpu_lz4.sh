#!/usr/bin/env bash
# 3) CPU LZ4 - BYTE data, upstream v1.9.4 + Kunpeng build in ONE run.
#
#   bench_cpu_lz4.sh [preset=vec12] [iters=10] [warmup=3] [cpu_core=2]
#
# The two libraries are built by scripts/build_ref_lz4_cpu.sh (auto-run if the
# binaries are missing).  Both use the block API with 64 KiB blocks, single
# thread, pinned to one core -- the same shape as the NPU kernels.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source scripts/lz4_bench_common.sh

PRESET="${1:-vec12}"
ITERS="${2:-10}"
WARMUP="${3:-3}"
CORE="${4:-2}"

if [ "$PRESET" = "-h" ] || [ "$PRESET" = "--help" ]; then
    sed -n '2,10p' "$0"
    echo; echo "dataset presets:"; lz4_print_presets
    exit 0
fi

PAT=$(lz4_pattern "$PRESET") || exit 2
lz4_require_files "$PAT" || exit 3
lz4_verify_dataset "$PAT" || exit 2
lz4_banner "CPU LZ4 (upstream v1.9.4 + Kunpeng)" "$PAT" "$ITERS" "$WARMUP" "core=$CORE"

BENCH_PATTERN="$PAT" \
BENCH_CSV="logs/bench_cpu_lz4_${PRESET}.csv" \
BENCH_LOG="logs/bench_cpu_lz4_${PRESET}.log" \
BENCH_NO_NPU=1 \
bash scripts/run_bench_lz4_cpu.sh "$ITERS" "$WARMUP" "$CORE"
