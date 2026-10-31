#!/usr/bin/env bash
# 6) CPU vs NPU LZ4 - BYTE data, same files for all three implementations.
#
#   bench_cpu_vs_lz4.sh [preset=vec12] [iters=10] [warmup=3] [cpu_core=2]
#
#   lz4cpu_up  upstream LZ4 v1.9.4   (single thread, 64 KiB blocks)
#   lz4cpu_kp  kunpengcompute/lz4    (single thread, 64 KiB blocks)
#   lz4npu     this project's Ascend C single-buffer kernel
#
# The dataset is hash-verified first (for data/fp and data/large); every codec
# reads the exact same bytes.  Fairness: same 64 KiB blocks, same iters/warmup,
# CPU pinned to one core, NPU kernel-only (E2E is a separate CSV column).
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
    sed -n '2,13p' "$0"
    echo; echo "dataset presets:"; lz4_print_presets
    echo; echo "examples:"
    echo "  bash scripts/bench_cpu_vs_lz4.sh vec12 10 3"
    echo "  bash scripts/bench_cpu_vs_lz4.sh large1g 3 1"
    echo "  bash scripts/bench_cpu_vs_lz4.sh large5g 3 1"
    exit 0
fi

PAT=$(lz4_pattern "$PRESET") || exit 2
lz4_require_files "$PAT" || exit 3
lz4_verify_dataset "$PAT" || exit 2
lz4_banner "CPU vs NPU LZ4 (byte)" "$PAT" "$ITERS" "$WARMUP" "cpu_core=$CORE"

BENCH_PATTERN="$PAT" \
BENCH_CSV="logs/bench_cpu_vs_lz4_${PRESET}.csv" \
BENCH_LOG="logs/bench_cpu_vs_lz4_${PRESET}.log" \
bash scripts/run_bench_lz4_cpu.sh "$ITERS" "$WARMUP" "$CORE"
