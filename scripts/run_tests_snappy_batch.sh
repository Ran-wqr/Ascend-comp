#!/usr/bin/env bash
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p logs

log="logs/test_snappy_batch.log"
echo ">>> test_snappy --mode batch"
./build/test_snappy --mode batch 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
echo "------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then echo "TESTS(snappy/batch): PASS"; else echo "TESTS(snappy/batch): FAIL"; fi
exit "$rc"
