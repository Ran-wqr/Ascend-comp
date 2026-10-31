#!/usr/bin/env python3
"""Generate a variable-sized, non-contiguous batch (independent items + list file)."""
import os, random, sys

random.seed(7)
d = "data/batch"
os.makedirs(d, exist_ok=True)
for f in os.listdir(d):
    p = os.path.join(d, f)
    if os.path.isfile(p):
        os.remove(p)

sizes = [0, 1, 2, 3, 4, 5, 31, 32, 33, 100, 255, 256, 257, 1000, 4095, 4096, 4097,
         65535, 65536] + [random.randint(0, 65536) for _ in range(21)]

names = []
for idx, s in enumerate(sizes):
    kind = idx % 4
    if kind == 0:
        b = bytes(random.getrandbits(8) for _ in range(s))
    elif kind == 1:
        b = (b"pattern-%04d;" % idx) * (s // 12 + 1)
        b = b[:s]
    elif kind == 2:
        b = bytes([(i * 7 + idx) & 0xff for i in range(s)])
    else:
        b = b"\x00" * s
    fn = "%s/item_%03d_%d.bin" % (d, idx, s)
    open(fn, "wb").write(b)
    names.append(fn)

open(d + "/list.txt", "w").write("\n".join(names) + "\n")
print("batch items:", len(names))
