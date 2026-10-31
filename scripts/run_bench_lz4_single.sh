#!/usr/bin/env bash
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p logs

MB="${1:-128}"; ITERS="${2:-10}"; WARMUP="${3:-3}"
rc=0
run() {
    local log="$1"; shift
    echo ">>> $*"
    "$@" 2>&1 | tee "$log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
}

run logs/bench_lz4_single_d1.log ./build/bench_lz4 "$MB" "$ITERS" "$WARMUP" 1 --mode single
run logs/bench_lz4_single_d2.log ./build/bench_lz4 "$MB" "$ITERS" "$WARMUP" 2 --mode single
run logs/bench_lz4_single_d4.log ./build/bench_lz4 "$MB" "$ITERS" "$WARMUP" 4 --mode single

echo "------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then echo "BENCH(lz4/single): PASS"; else echo "BENCH(lz4/single): FAIL"; fi
exit "$rc"
