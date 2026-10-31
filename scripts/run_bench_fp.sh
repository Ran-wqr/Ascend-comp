#!/usr/bin/env bash
# Unified FP16/BF16/FP32 benchmark: LZ4 (byte-plane separation) vs HANS (CANN).
#
#   run_bench_fp.sh [iters] [warmup]
#
# Both codecs run through the SAME binary (build/bench_fp), the same input
# bytes, the same warmup/iters, the same ACL-event timing, the same statistics
# and the same bit-exact verification.  The test set is hash-verified first.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh

ITERS="${1:-10}"
WARMUP="${2:-3}"
LOGDIR=logs
mkdir -p "$LOGDIR"

[ -f data/fp/manifest.json ] || { echo "[fp] missing data/fp (run scripts/gen_fp_data.py)"; exit 2; }
echo "[fp] verifying fixed test set ..."
python3 scripts/gen_fp_data.py --verify || { echo "[fp] TEST SET MISMATCH - aborting"; exit 2; }

[ -x build/bench_fp ] || { echo "[fp] missing build/bench_fp (run scripts/build.sh)"; exit 1; }

CSV="$LOGDIR/bench_fp.csv"
LOG="$LOGDIR/bench_fp.log"
: > "$CSV"; : > "$LOG"
echo "dataset,codec,dtype,elems,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,verify" > "$CSV"

{
echo "=== FP16/BF16/FP32: LZ4(raw) vs LZ4(plane) vs HANS(CANN aclnn) ==="
echo "manifest_sha256: $(sha256sum data/fp/manifest.json | cut -d' ' -f1)"
echo "iters=$ITERS warmup=$WARMUP  |  soc=$(npu-smi info 2>/dev/null | grep -m1 -o 'Ascend[0-9A-Za-z]*')"
} | tee -a "$LOG"

fail=0
for dtype in fp16 bf16 fp32; do
    for f in data/fp/$dtype/*.bin; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        # 1 MiB is a small-size stress case only: HANS per-core fixed tail (8448 B)
        # inflates its ratio there, so it must not be used for the headline number.
        ok=1
        for codec in lz4raw lz4 hans; do
            row=$(./build/bench_fp "$codec" "$dtype" "$f" "$ITERS" "$WARMUP" csv 2>>"$LOG")
            rc=$?
            if [ "$rc" -ne 0 ] || [ -z "$row" ]; then
                echo "[FAIL] $base codec=$codec rc=$rc" | tee -a "$LOG"
                ok=0; fail=1; break
            fi
            echo "$row" >> "$CSV"
        done
        [ "$ok" -eq 1 ] || continue
    done
done

python3 scripts/summarize_fp.py "$CSV" --size 0 | tee -a "$LOG"

echo "-------------------------------------------------------------------------------" | tee -a "$LOG"
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS  (iters=$ITERS warmup=$WARMUP)" | tee -a "$LOG"
else
    echo "RESULT: FAIL" | tee -a "$LOG"
    exit 1
fi
