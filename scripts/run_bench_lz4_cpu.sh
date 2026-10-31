#!/usr/bin/env bash
# CPU vs NPU LZ4 benchmark on the byte-type vectors (data/t_*.bin).
#
#   run_bench_lz4_cpu.sh [iters=10] [warmup=3] [cpu_core=2]
#
# Codecs compared (same files, same 64 KiB block size, same iters/warmup):
#   lz4cpu_up : upstream LZ4 v1.9.4, single thread      (official CPU reference)
#   lz4cpu_kp : kunpengcompute/lz4 (KZL/NEON)           (extra, ratio may differ)
#   lz4npu    : this project's Ascend C kernels         (kernel-only + E2E)
#
# Fairness notes:
#   * the CPU side is pinned to one physical core with taskset (640-core host)
#   * CPU timing is wall-clock around the block loop; there is no H2D/D2H.
#     The NPU side reports kernel-only (events) as the primary number and
#     e2e (H2D+kernel+D2H) as a separate column.
#   * CPU and NPU both use 64 KiB blocks and the block API, no container header.
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
# the NPU binary needs the CANN runtime (libascendcl.so) and the device libs
source /usr/local/Ascend/cann-9.0.0/set_env.sh

ITERS="${1:-10}"
WARMUP="${2:-3}"
CORE="${3:-2}"
PATTERN="${BENCH_PATTERN:-data/t_*.bin}"
CSV="${BENCH_CSV:-logs/bench_lz4_cpu.csv}"
LOG="${BENCH_LOG:-logs/bench_lz4_cpu.log}"
mkdir -p logs

[ -x build/bench_lz4_cpu_up ] && [ -x build/bench_lz4_cpu_kp ] || bash scripts/build_ref_lz4_cpu.sh || exit 1
[ -x build/bench_lz4_file ] || { echo "[cpu] missing build/bench_lz4_file (run scripts/build.sh)"; exit 1; }

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}

: > "$CSV"; : > "$LOG"
echo "dataset,codec,dtype,in_bytes,comp_bytes,ratio,comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_sd,dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_sd,e2e_ms,verify" > "$CSV"

{
echo "=== CPU vs NPU LZ4 (block=64 KiB, single thread on CPU) ==="
echo "iters=$ITERS warmup=$WARMUP cpu_core=$CORE"
echo "cpu=$(lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs || echo unknown)  cores=$(nproc)"
echo "up=$(git -C ref/lz4_up_src rev-parse --short HEAD 2>/dev/null)  kp=$(git -C ref/lz4_kp_src rev-parse --short HEAD 2>/dev/null)"
} | tee -a "$LOG"

fail=0
for f in $PATTERN; do
    [ -e "$f" ] || continue
    ok=1
    for spec in "up:build/bench_lz4_cpu_up" "kp:build/bench_lz4_cpu_kp"; do
        label="${spec%%:*}"; bin="${spec#*:}"
        row=$(taskset -c "$CORE" "$bin" "$f" "lz4cpu_$label" "$ITERS" "$WARMUP" 64 csv 2>>"$LOG")
        rc=$?
        if [ "$rc" -ne 0 ] || [ -z "$row" ]; then
            echo "[FAIL] $f codec=lz4cpu_$label rc=$rc" | tee -a "$LOG"; ok=0; fail=1; break
        fi
        echo "$row" >> "$CSV"
    done
    [ "$ok" -eq 1 ] || continue
    if [ "${BENCH_NO_NPU:-0}" = "1" ]; then continue; fi
    row=$(./build/bench_lz4_file "$f" 1 "$ITERS" "$WARMUP" csv 2>>"$LOG")
    rc=$?
    if [ "$rc" -ne 0 ] || [ -z "$row" ]; then
        echo "[FAIL] $f codec=lz4npu rc=$rc" | tee -a "$LOG"; fail=1; continue
    fi
    echo "$row" >> "$CSV"
done

if [ "${BENCH_NO_NPU:-0}" = "1" ]; then
    python3 scripts/summarize_perf.py "$CSV" --size 0 | tee -a "$LOG"
else
    python3 scripts/summarize_lz4_cpu.py "$CSV" | tee -a "$LOG"
fi

echo "-------------------------------------------------------------------------------" | tee -a "$LOG"
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS  (iters=$ITERS warmup=$WARMUP)" | tee -a "$LOG"
else
    echo "RESULT: FAIL" | tee -a "$LOG"
    exit 1
fi
