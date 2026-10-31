#!/usr/bin/env bash
# CPU vs NPU LZ4 on THREE single 5 GB files (data/large/large_5G_*.bin).
#
#   run_bench_lz4_cpu_5g.sh [iters=3] [warmup=1] [cpu_core=2]
#
# 5 GB = 5,000,000,000 B (decimal, same convention as the 1 GB / 4 GB files).
# The NPU harness splits each file into 1 GiB windows, so a single >4 GiB file
# is safe (each window's kernel spans stay below 2^32).
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ITERS="${1:-3}"
WARMUP="${2:-1}"
CORE="${3:-2}"

mkdir -p logs data/large
need=0
for f in text struct random; do
    p="data/large/large_5G_${f}.bin"
    [ -f "$p" ] && [ "$(stat -c%s "$p" 2>/dev/null)" = "5000000000" ] || need=1
done
if [ "$need" = 1 ]; then
    echo "[5g] generating three 5 GB files (uses data/large/enwik9.bin) ..."
    python3 scripts/gen_large_testdata.py --size 5000000000 \
        --only text5g --only struct5g --only random5g || exit 1
fi
echo "[5g] verifying hashes ..."
python3 scripts/gen_large_testdata.py --verify || { echo "[5g] TEST SET MISMATCH - aborting"; exit 2; }

BENCH_PATTERN="data/large/large_5G_*.bin" \
BENCH_CSV="logs/bench_lz4_cpu_5g.csv" \
BENCH_LOG="logs/bench_lz4_cpu_5g.log" \
bash scripts/run_bench_lz4_cpu.sh "$ITERS" "$WARMUP" "$CORE"
