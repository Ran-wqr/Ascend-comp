#!/usr/bin/env bash
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p logs

rc=0
bash scripts/run_tests_lz4_single.sh || rc=1
bash scripts/run_tests_lz4_batch.sh  || rc=1
echo "============================================================"
if [ "$rc" -eq 0 ]; then echo "TESTS(lz4): PASS"; else echo "TESTS(lz4): FAIL"; fi
exit "$rc"
