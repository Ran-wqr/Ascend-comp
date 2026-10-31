#!/usr/bin/env python3
"""Verify NPU-computed per-chunk decompressed sizes against the expected chunking."""
import sys, os, struct

sizes_file, orig_file, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
total = os.path.getsize(orig_file)
blob = open(sizes_file, "rb").read()
n = len(blob) // 4
got = [struct.unpack_from("<I", blob, i * 4)[0] for i in range(n)]

num_chunks = max(1, (total + chunk - 1) // chunk)
exp = [min(chunk, total - c * chunk) for c in range(num_chunks)]

if got != exp:
    print("size mismatch: got=%s exp=%s" % (got[:6], exp[:6]))
    sys.exit(1)
print("sizes ok (%d chunks)" % n)
