#!/usr/bin/env bash
# Full regression: build -> unit tests -> interop (strict) -> optional bench.
#   usage: regress.sh [--with-bench] [--bench-args "16 3 1"]
set +u
set -o pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

WITH_BENCH=0
BENCH_ARGS="16 3 1"
while [ $# -gt 0 ]; do
    case "$1" in
        --with-bench) WITH_BENCH=1; shift ;;
        --bench-args) BENCH_ARGS="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
done

rc=0
echo "== [1/4] build =="
bash scripts/build.sh || exit 1

echo "== [2/4] unit tests (single + batch, both algos) =="
bash scripts/run_tests.sh || rc=1

echo "== [3/4] interop (strict: SKIP counts as failure) =="
bash scripts/run_all.sh --strict || rc=1

if [ "$WITH_BENCH" = 1 ]; then
    echo "== [4/4] benchmarks =="
    bash scripts/run_bench.sh $BENCH_ARGS || rc=1
else
    echo "== [4/4] benchmarks skipped (use --with-bench) =="
fi

echo "============================================================"
if [ "$rc" -eq 0 ]; then echo "REGRESS: PASS"; else echo "REGRESS: FAIL"; fi
exit "$rc"
