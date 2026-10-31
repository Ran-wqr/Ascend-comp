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

run logs/bench_snappy_batch.log ./build/bench_snappy "$MB" "$ITERS" "$WARMUP" --mode batch

echo "------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then echo "BENCH(snappy/batch): PASS"; else echo "BENCH(snappy/batch): FAIL"; fi
exit "$rc"
