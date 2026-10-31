#!/usr/bin/env bash
# HANS baseline benchmark over the fixed FP16/BF16/FP32 test set.
#
#   run_bench_hans.sh [iters] [warmup]
#
# Guarantees:
#   * the test set is verified against data/fp/manifest.json (sha256) BEFORE running
#   * one binary, one measurement path, one warmup/iters setting for all dtypes
#   * results: logs/bench_hans.log (table) + logs/bench_hans.csv (structured)
#   * each file is executed exactly once (table is formatted from the CSV row)
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh

ITERS="${1:-10}"
WARMUP="${2:-3}"
LOGDIR=logs
mkdir -p "$LOGDIR"

[ -f data/fp/manifest.json ] || { echo "[hans] missing data/fp (run scripts/gen_fp_data.py)"; exit 2; }
echo "[hans] verifying fixed test set ..."
python3 scripts/gen_fp_data.py --verify || { echo "[hans] TEST SET MISMATCH - aborting"; exit 2; }

[ -x hans/bench_hans ] || bash scripts/build_hans.sh || exit 1

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}

LOG="$LOGDIR/bench_hans.log"
CSV="$LOGDIR/bench_hans.csv"
: > "$LOG"; : > "$CSV"
echo "dataset,dtype,elems,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,verify" > "$CSV"

echo "=== HANS baseline (CANN aclnn, statistic=true, reshuff=false) ===" | tee -a "$LOG"
printf "%-30s %-4s %9s %10s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %s\n" \
    dataset dtype elems in_bytes ratio c_med c_min c_max c_sd d_mean d_med d_min d_max verify | tee -a "$LOG"

fail=0
for dtype in fp16 bf16 fp32; do
    for f in data/fp/$dtype/*.bin; do
        [ -e "$f" ] || continue
        row=$(./hans/bench_hans "$dtype" "$f" "$ITERS" "$WARMUP" csv 2>>"$LOG")
        rc=$?
        if [ "$rc" -ne 0 ] || [ -z "$row" ]; then
            echo "[FAIL] $f rc=$rc" | tee -a "$LOG"
            fail=1
            continue
        fi
        echo "$row" >> "$CSV"
        echo "$row" | awk -F, '{printf "%-28s %-4s %9s %10s %8.4f %8.2f %8.3f %8.3f %8.3f %8.2f %8.3f %8.3f %8.3f %s\n", $1,$2,$3,$4,$6,$8,$9,$10,$11,$12,$13,$14,$15,$17}' | tee -a "$LOG"
    done
done

echo "-------------------------------------------------------------------------------" | tee -a "$LOG"
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS  (iters=$ITERS warmup=$WARMUP)" | tee -a "$LOG"
else
    echo "RESULT: FAIL" | tee -a "$LOG"
    exit 1
fi
