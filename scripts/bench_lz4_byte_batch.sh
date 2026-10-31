#!/usr/bin/env bash
# 2) NPU LZ4 - batch (pointer-array) mode, BYTE data.
#
#   bench_lz4_byte_batch.sh [iters=10] [warmup=3] [dtype=all] [MB=128] [item=uniform|mixed]
#
# Batch items are generated in memory (like the nvCOMP batch tests): MB controls
# the total payload, item=uniform makes every item 64 KiB, item=mixed uses
# variable lengths (1..65536 B).  dtype=all runs 1, 2 and 4.
#
# There is no file-based batch driver on purpose: batch items are independent
# variable-length buffers, not files (see README_BENCH_LZ4.md).
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source scripts/lz4_bench_common.sh

ITERS="${1:-10}"
WARMUP="${2:-3}"
DTYPE="${3:-all}"
MB="${4:-128}"
ITEM="${5:-mixed}"

if [ "$ITERS" = "-h" ] || [ "$ITERS" = "--help" ]; then
    sed -n '2,12p' "$0"
    echo; echo "examples:"
    echo "  bash scripts/bench_lz4_byte_batch.sh 10 3 all 128 mixed"
    echo "  bash scripts/bench_lz4_byte_batch.sh 10 3 1 64 uniform"
    exit 0
fi

case "$ITEM" in uniform|mixed) ;; *) echo "[err] item must be uniform|mixed"; exit 2 ;; esac
case "$DTYPE" in all) DS="1 2 4" ;; 1|2|4) DS="$DTYPE" ;; *) echo "[err] dtype must be all|1|2|4"; exit 2 ;; esac

mkdir -p logs
LOG=logs/bench_lz4_byte_batch.log
: > "$LOG"
lz4_banner "LZ4 batch (byte, synthetic items)" "in-memory ${MB} MiB, item=$ITEM" "$ITERS" "$WARMUP" "dtype=$DTYPE"

rc=0
for d in $DS; do
    ./build/bench_lz4 "$MB" "$ITERS" "$WARMUP" "$d" --mode batch --item "$ITEM" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
done
echo "CSV: logs/bench_lz4_batch_d{1,2,4}.csv"
lz4_result "$rc" "byte batch, ${MB} MiB, item=$ITEM"
