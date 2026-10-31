#!/usr/bin/env bash
# 1) NPU LZ4 - single-buffer mode, BYTE data.
#
#   bench_lz4_byte_single.sh [preset=synthetic] [iters=10] [warmup=3] [dtype=all] [MB=128]
#
# preset=synthetic : generated in memory, MB MiB per dtype (fast, no files)
# preset=vec12|large1g|large4g|large5g|largeall : iterate real files
#                   (single-buffer kernel, dtype forced to 1)
#
# Change the dataset: pass another preset, or PATTERN="my/*.bin", or add a
# preset in scripts/lz4_bench_common.sh.  List presets with --help.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source scripts/lz4_bench_common.sh

PRESET="${1:-synthetic}"
ITERS="${2:-10}"
WARMUP="${3:-3}"
DTYPE="${4:-all}"
MB="${5:-128}"

if [ "$PRESET" = "-h" ] || [ "$PRESET" = "--help" ]; then
    sed -n '2,16p' "$0"
    echo; echo "dataset presets:"; lz4_print_presets
    echo; echo "synthetic example: bash scripts/bench_lz4_byte_single.sh synthetic 10 3 all 128"
    echo "file example     : bash scripts/bench_lz4_byte_single.sh vec12 10 3"
    echo "custom glob      : PATTERN='data/t_big_*.bin' bash scripts/bench_lz4_byte_single.sh vec12"
    exit 0
fi

mkdir -p logs
LOG=logs/bench_lz4_byte_single.log
: > "$LOG"

if [ "$PRESET" = "synthetic" ]; then
    case "$DTYPE" in
        all) DS="1 2 4" ;;
        1|2|4) DS="$DTYPE" ;;
        *) echo "[err] dtype must be all|1|2|4"; exit 2 ;;
    esac
    lz4_banner "LZ4 single (byte, synthetic)" "in-memory ${MB} MiB/dtype" "$ITERS" "$WARMUP" "dtype=$DTYPE"
    rc=0
    for d in $DS; do
        ./build/bench_lz4 "$MB" "$ITERS" "$WARMUP" "$d" --mode single 2>&1 | tee -a "$LOG"
        [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    done
    echo "CSV: logs/bench_lz4_single_d{1,2,4}.csv"
    lz4_result "$rc" "byte single, synthetic, ${MB} MiB"
    exit $?
fi

PAT=$(lz4_pattern "$PRESET") || exit 2
lz4_require_files "$PAT" || exit 3
lz4_verify_dataset "$PAT" || exit 2
lz4_banner "LZ4 single (byte, files)" "$PAT" "$ITERS" "$WARMUP" "dtype=1 (forced)"

CSV="logs/bench_lz4_byte_single_${PRESET}.csv"
echo "dataset,codec,dtype,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,e2e_ms,verify" > "$CSV"
rc=0
for f in $PAT; do
    row=$(./build/bench_lz4_file "$f" 1 "$ITERS" "$WARMUP" csv 2>>"$LOG")
    if [ $? -ne 0 ] || [ -z "$row" ]; then
        echo "[FAIL] $f" | tee -a "$LOG"; rc=1; continue
    fi
    echo "$row" >> "$CSV"
done
python3 scripts/summarize_perf.py "$CSV" --size 0 | tee -a "$LOG"
echo "CSV: $CSV"
lz4_result "$rc" "byte single, $PRESET"
