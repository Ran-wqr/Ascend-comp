#!/usr/bin/env bash
# CPU vs NPU LZ4 on the ~1 GB test set (data/large).
#
#   run_bench_lz4_cpu_large.sh [iters=3] [warmup=1] [cpu_core=2]
#
# Defaults are smaller than the 12-vector run because each file is 1e9 bytes:
# 3 iterations still amortise everything, and a full matrix takes a few minutes.
# Uses exactly the same harness as run_bench_lz4_cpu.sh (same binaries, same
# 64 KiB blocks, same statistics), just a different file glob and output paths.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ITERS="${1:-3}"
WARMUP="${2:-1}"
CORE="${3:-2}"

mkdir -p logs data/large
if [ ! -f data/large/manifest.json ] || [ ! -f data/large/enwik9.bin ]; then
    echo "[large] generating data/large (downloads enwik9 if missing) ..."
    python3 scripts/gen_large_testdata.py || exit 1
fi
echo "[large] verifying hashes ..."
python3 scripts/gen_large_testdata.py --verify || { echo "[large] TEST SET MISMATCH - aborting"; exit 2; }

BENCH_PATTERN="data/large/*.bin" \
BENCH_CSV="logs/bench_lz4_cpu_large.csv" \
BENCH_LOG="logs/bench_lz4_cpu_large.log" \
bash scripts/run_bench_lz4_cpu.sh "$ITERS" "$WARMUP" "$CORE"
