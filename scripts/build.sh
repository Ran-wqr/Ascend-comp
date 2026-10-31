#!/usr/bin/env bash
# Build all Ascend C targets. Fails loudly on cmake/make errors (pipefail).
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u
cd "$HERE"
rm -rf build
mkdir -p build logs
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee ../logs/cmake.log
make -j"${JOBS:-4}" 2>&1 | tee ../logs/make.log
echo "[build] OK -> $HERE/build/{lz4_demo,snappy_demo,test_lz4,test_snappy,bench_lz4,bench_snappy,bench_fp,bench_lz4_file}"
