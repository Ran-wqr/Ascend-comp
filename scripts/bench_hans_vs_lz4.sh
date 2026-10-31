#!/usr/bin/env bash
# 7) HANS vs NPU LZ4 - FLOATING-POINT data, same files for all three codecs.
#
#   bench_hans_vs_lz4.sh [preset=fp8m] [iters=10] [warmup=3]
#
#   lz4raw  no preprocessing       (64 KiB blocks)
#   lz4     byte-plane separation  (32 KiB blocks)
#   hans    official CANN op
#
# All three run through build/bench_fp on the same bytes, same warmup/iters,
# same ACL-event timing, same statistics, same bit-exact verification.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source scripts/lz4_bench_common.sh

PRESET="${1:-fp8m}"
ITERS="${2:-10}"
WARMUP="${3:-3}"

if [ "$PRESET" = "-h" ] || [ "$PRESET" = "--help" ]; then
    sed -n '2,12p' "$0"
    echo; echo "dataset presets:"; lz4_print_presets
    echo; echo "examples:"
    echo "  bash scripts/bench_hans_vs_lz4.sh fp8m 10 3"
    echo "  bash scripts/bench_hans_vs_lz4.sh fp16_8m 10 3"
    exit 0
fi

PAT=$(lz4_pattern "$PRESET") || exit 2
lz4_require_files "$PAT" || exit 3
case "$PAT" in
    *data/fp/*) ;;
    *) echo "[err] this script needs an FP dataset (data/fp/<dtype>/...)" >&2; exit 2 ;;
esac
lz4_verify_dataset "$PAT" || exit 2
lz4_banner "HANS vs NPU LZ4 (FP)" "$PAT" "$ITERS" "$WARMUP" "lz4raw / lz4 / hans"

mkdir -p logs
LOG="logs/bench_hans_vs_lz4_${PRESET}.log"
CSV="logs/bench_hans_vs_lz4_${PRESET}.csv"
: > "$LOG"
echo "dataset,codec,dtype,elems,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,verify" > "$CSV"

rc=0
for f in $PAT; do
    dt=$(lz4_dtype_of "$f")
    [ -n "$dt" ] || { echo "[skip] cannot infer dtype: $f" | tee -a "$LOG"; continue; }
    for c in lz4raw lz4 hans; do
        row=$(./build/bench_fp "$c" "$dt" "$f" "$ITERS" "$WARMUP" csv 2>>"$LOG")
        if [ $? -ne 0 ] || [ -z "$row" ]; then
            echo "[FAIL] $f codec=$c" | tee -a "$LOG"; rc=1; continue
        fi
        echo "$row" >> "$CSV"
    done
done

python3 scripts/summarize_fp.py "$CSV" --size 0 | tee -a "$LOG"
echo "CSV: $CSV"
lz4_result "$rc" "HANS vs LZ4, $PRESET"
