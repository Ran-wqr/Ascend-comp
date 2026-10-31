#!/usr/bin/env bash
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p logs

rc=0
bash scripts/run_bench_lz4.sh    "$@" || rc=1
bash scripts/run_bench_snappy.sh "$@" || rc=1
echo "============================================================"
if [ "$rc" -eq 0 ]; then echo "BENCH(ALL): PASS"; else echo "BENCH(ALL): FAIL"; fi
exit "$rc"
