#!/usr/bin/env bash
# Build the CPU reference libraries and benchmarks:
#   ref/lz4_up_src  : upstream LZ4 v1.9.4 (official reference, no Kunpeng patch)
#   ref/lz4_kp_src  : kunpengcompute/lz4 dev-1.9.4 (Kunpeng/KZL optimizations)
#
# Both are built with the same flags and linked into two separate binaries
# (they export identical LZ4_* symbols, so they cannot share one binary).
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

KP_REPO="https://gitcode.com/boostkit/lz4.git"
KP_COMMIT="${KP_COMMIT:-9b09896f}"      # dev-1.9.4 HEAD at the time of writing
UP_COMMIT="${UP_COMMIT:-5ff83968}"      # LZ4 v1.9.4 (parent of the first KZL commit)
CFLAGS_CPU="${CFLAGS_CPU:--O3 -march=native -flto -fno-plt}"

mkdir -p ref build logs

if [ ! -d ref/lz4_kp_src/.git ]; then
    echo "[build_ref] cloning $KP_REPO"
    git clone --quiet -b dev-1.9.4 "$KP_REPO" ref/lz4_kp_src
fi
git -C ref/lz4_kp_src checkout --quiet "$KP_COMMIT"

if [ ! -d ref/lz4_up_src/.git ]; then
    echo "[build_ref] creating upstream worktree copy"
    cp -a ref/lz4_kp_src ref/lz4_up_src
fi
git -C ref/lz4_up_src checkout --quiet "$UP_COMMIT"

echo "[build_ref] upstream : $(git -C ref/lz4_up_src rev-parse --short HEAD) $(grep -m1 'define LZ4_VERSION_RELEASE' ref/lz4_up_src/lib/lz4.h | awk '{print $3}')"
echo "[build_ref] kunpeng  : $(git -C ref/lz4_kp_src rev-parse --short HEAD)"

for kind in up kp; do
    ( cd "ref/lz4_${kind}_src/lib" && make -s clean >/dev/null 2>&1 || true
      make -s liblz4.a CFLAGS="$CFLAGS_CPU" ) >> "logs/build_ref_lz4_cpu.log" 2>&1
done

gcc -O3 $CFLAGS_CPU -I ref/lz4_up_src/lib \
    benchmarks/bench_lz4_cpu.c ref/lz4_up_src/lib/liblz4.a \
    -o build/bench_lz4_cpu_up -lm
gcc -O3 $CFLAGS_CPU -I ref/lz4_kp_src/lib \
    benchmarks/bench_lz4_cpu.c ref/lz4_kp_src/lib/liblz4.a \
    -o build/bench_lz4_cpu_kp -lm

{
  echo "date=$(date -Iseconds)"
  echo "cpu=$(lscpu | awk -F: '/^Vendor ID/{v=$2} /^Model:/{m=$2} /^CPU max MHz/{f=$2} END{printf "%s model%s @%s MHz", v, m, f}' | xargs)"
  echo "cores=$(nproc)"
  echo "gcc=$(gcc -dumpfullversion)"
  echo "cflags=$CFLAGS_CPU"
  echo "up_commit=$(git -C ref/lz4_up_src rev-parse HEAD)"
  echo "kp_commit=$(git -C ref/lz4_kp_src rev-parse HEAD)"
} > logs/bench_lz4_cpu.env

echo "[build_ref] OK -> build/bench_lz4_cpu_{up,kp}"
