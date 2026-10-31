#!/usr/bin/env bash
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
mkdir -p logs

log="logs/test_snappy_single.log"
echo ">>> test_snappy --mode single"
./build/test_snappy --mode single 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
echo "------------------------------------------------------------"
if [ "$rc" -eq 0 ]; then echo "TESTS(snappy/single): PASS"; else echo "TESTS(snappy/single): FAIL"; fi
exit "$rc"
