#!/usr/bin/env python3
"""Compare a reference-decoded batch output against the concatenation of the
non-empty batch items (avoids 0-length blocks, which the reference tools reject)."""
import sys

list_file, decoded = sys.argv[1], sys.argv[2]
names = [l.strip() for l in open(list_file) if l.strip()]
blob = open(decoded, "rb").read()

expected = b"".join(open(n, "rb").read() for n in names)
if blob != expected:
    print("batch mismatch: decoded=%d expected=%d" % (len(blob), len(expected)))
    sys.exit(1)
print("batch ok: %d bytes over %d items" % (len(blob), len(names)))
