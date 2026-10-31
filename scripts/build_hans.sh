#!/usr/bin/env bash
# Build the host-side HANS benchmark (official CANN aclnn op).
# Kept separate from scripts/build.sh because it links the aclnn/opapi host
# libraries instead of being compiled by bisheng as an Ascend C kernel.
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u

INC=/usr/local/Ascend/cann-9.0.0/aarch64-linux/include
LIB=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64
LIB2=/usr/local/Ascend/cann-9.0.0/lib64

mkdir -p "$HERE/logs"
g++ -O2 -std=c++17 -I"$INC" -I"$INC/aclnnop" \
    -o "$HERE/hans/bench_hans" "$HERE/hans/bench_hans.cpp" \
    -L"$LIB" -L"$LIB2" -lopapi_math -lnnopbase -lascendcl -lpthread \
    2>&1 | tee "$HERE/logs/build_hans.log"
echo "[build_hans] OK -> $HERE/hans/bench_hans"
