#!/usr/bin/env bash
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p logs

log="logs/test_lz4_batch.log"
echo ">>> test_lz4 --mode batch"
./build/test_lz4 --mode batch 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
echo "------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then echo "TESTS(lz4/batch): PASS"; else echo "TESTS(lz4/batch): FAIL"; fi
exit "$rc"
