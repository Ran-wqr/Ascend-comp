#!/usr/bin/env bash
# Build the official LZ4 / google-snappy reference codecs (interop checks).
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/ref"
[ -d lz4 ]    || git clone --depth 1 https://github.com/lz4/lz4.git lz4
[ -d snappy ] || git clone --depth 1 https://github.com/google/snappy.git snappy

if [ ! -f snappy/snappy-stubs-public.h ]; then
cat > snappy/snappy-stubs-public.h <<'EOS'
#ifndef THIRD_PARTY_SNAPPY_OPENSOURCE_SNAPPY_STUBS_PUBLIC_H_
#define THIRD_PARTY_SNAPPY_OPENSOURCE_SNAPPY_STUBS_PUBLIC_H_
#include <cstddef>
#include <sys/uio.h>
#define SNAPPY_MAJOR 1
#define SNAPPY_MINOR 2
#define SNAPPY_PATCHLEVEL 1
#define SNAPPY_VERSION ((SNAPPY_MAJOR << 16) | (SNAPPY_MINOR << 8) | SNAPPY_PATCHLEVEL)
namespace snappy {
}
#endif
EOS
fi

g++ -O2 -I lz4/lib -o tools/ref_lz4 tools/ref_lz4.cpp lz4/lib/lz4.c
g++ -O2 -I snappy  -o tools/ref_snappy tools/ref_snappy.cpp \
    snappy/snappy.cc snappy/snappy-sinksource.cc snappy/snappy-stubs-internal.cc snappy/snappy-c.cc
echo "[build_ref] OK -> ref/tools/{ref_lz4,ref_snappy}"
