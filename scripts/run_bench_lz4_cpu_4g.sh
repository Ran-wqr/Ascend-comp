#!/usr/bin/env bash
# CPU vs NPU LZ4 on ONE 4 GB file (data/large/large_4G.bin).
#
#   run_bench_lz4_cpu_4g.sh [iters=3] [warmup=1] [cpu_core=2]
#
# Why 4,000,000,000 bytes and not 4 GiB (4,294,967,296): the NPU kernels use
# 32-bit offsets, and `total+31` alignment would wrap at exactly 4 GiB. 4e9
# keeps a safety margin while still being "4 GB" in the same decimal convention
# as the 1 GB files.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ITERS="${1:-3}"
WARMUP="${2:-1}"
CORE="${3:-2}"
TARGET="data/large/large_4G.bin"

mkdir -p logs data/large
if [ ! -f "$TARGET" ] || [ "$(stat -c%s "$TARGET" 2>/dev/null)" != "4000000000" ]; then
    echo "[4g] generating $TARGET (uses data/large/enwik9.bin) ..."
    python3 scripts/gen_large_testdata.py --only text4g --size 4000000000 || exit 1
fi
echo "[4g] verifying hashes ..."
python3 scripts/gen_large_testdata.py --verify || { echo "[4g] TEST SET MISMATCH - aborting"; exit 2; }

BENCH_PATTERN="$TARGET" \
BENCH_CSV="logs/bench_lz4_cpu_4g.csv" \
BENCH_LOG="logs/bench_lz4_cpu_4g.log" \
bash scripts/run_bench_lz4_cpu.sh "$ITERS" "$WARMUP" "$CORE"
