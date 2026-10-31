#!/usr/bin/env bash
# 5) HANS on FLOATING-POINT data (official CANN aclnn op).
#
#   bench_hans_fp.sh [preset=fp8m] [iters=10] [warmup=3]
#
# Uses the same binary / timing / verification as the LZ4 FP script
# (build/bench_fp, codec=hans), so the numbers are directly comparable.
# HANS requires elems % 64 == 0 and elems >= 32768; all data/fp files satisfy it.
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
    sed -n '2,9p' "$0"
    echo; echo "dataset presets:"; lz4_print_presets
    exit 0
fi

PAT=$(lz4_pattern "$PRESET") || exit 2
lz4_require_files "$PAT" || exit 3
case "$PAT" in
    *data/fp/*) ;;
    *) echo "[err] this script needs an FP dataset (data/fp/<dtype>/...)" >&2; exit 2 ;;
esac
lz4_verify_dataset "$PAT" || exit 2
lz4_banner "HANS on FP data" "$PAT" "$ITERS" "$WARMUP" "statistic=true reshuff=false"

mkdir -p logs
LOG="logs/bench_hans_fp_${PRESET}.log"
CSV="logs/bench_hans_fp_${PRESET}.csv"
: > "$LOG"
echo "dataset,codec,dtype,elems,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,verify" > "$CSV"

rc=0
for f in $PAT; do
    dt=$(lz4_dtype_of "$f")
    [ -n "$dt" ] || { echo "[skip] cannot infer dtype: $f" | tee -a "$LOG"; continue; }
    row=$(./build/bench_fp hans "$dt" "$f" "$ITERS" "$WARMUP" csv 2>>"$LOG")
    if [ $? -ne 0 ] || [ -z "$row" ]; then
        echo "[FAIL] $f codec=hans" | tee -a "$LOG"; rc=1; continue
    fi
    echo "$row" >> "$CSV"
done

python3 scripts/summarize_perf.py "$CSV" --size 0 | tee -a "$LOG"
echo "CSV: $CSV"
lz4_result "$rc" "HANS, $PRESET"
