#!/usr/bin/env bash
# 4) NPU LZ4 on FLOATING-POINT data - single-buffer, both variants.
#
#   bench_lz4_fp.sh [preset=fp8m] [iters=10] [warmup=3] [codecs=both|raw|plane]
#
#   raw   = lz4raw : no preprocessing, 64 KiB blocks
#   plane = lz4    : byte-plane separation (2/4 planes), 32 KiB blocks
#
# dtype (fp16/bf16/fp32) is taken from the path data/fp/<dtype>/...
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source scripts/lz4_bench_common.sh

PRESET="${1:-fp8m}"
ITERS="${2:-10}"
WARMUP="${3:-3}"
CODECS="${4:-both}"

if [ "$PRESET" = "-h" ] || [ "$PRESET" = "--help" ]; then
    sed -n '2,11p' "$0"
    echo; echo "dataset presets:"; lz4_print_presets
    exit 0
fi

case "$CODECS" in
    both)  CLIST="lz4raw lz4" ;;
    raw)   CLIST="lz4raw" ;;
    plane) CLIST="lz4" ;;
    *) echo "[err] codecs must be both|raw|plane"; exit 2 ;;
esac

PAT=$(lz4_pattern "$PRESET") || exit 2
lz4_require_files "$PAT" || exit 3
case "$PAT" in
    *data/fp/*) ;;
    *) echo "[err] this script needs an FP dataset (data/fp/<dtype>/...)" >&2; exit 2 ;;
esac
lz4_verify_dataset "$PAT" || exit 2
lz4_banner "NPU LZ4 on FP data" "$PAT" "$ITERS" "$WARMUP" "codecs=$CODECS"

mkdir -p logs
LOG="logs/bench_lz4_fp_${PRESET}.log"
CSV="logs/bench_lz4_fp_${PRESET}.csv"
: > "$LOG"
echo "dataset,codec,dtype,elems,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,verify" > "$CSV"

rc=0
for f in $PAT; do
    dt=$(lz4_dtype_of "$f")
    [ -n "$dt" ] || { echo "[skip] cannot infer dtype: $f" | tee -a "$LOG"; continue; }
    for c in $CLIST; do
        row=$(./build/bench_fp "$c" "$dt" "$f" "$ITERS" "$WARMUP" csv 2>>"$LOG")
        if [ $? -ne 0 ] || [ -z "$row" ]; then
            echo "[FAIL] $f codec=$c" | tee -a "$LOG"; rc=1; continue
        fi
        echo "$row" >> "$CSV"
    done
done

python3 scripts/summarize_perf.py "$CSV" --size 0 | tee -a "$LOG"
echo "CSV: $CSV"
lz4_result "$rc" "FP LZ4 ($CODECS), $PRESET"
