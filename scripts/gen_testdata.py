#!/usr/bin/env python3
"""Generate a set of test vectors covering compressible / incompressible / edge cases."""
import os, random, struct

random.seed(1234)
os.makedirs("data", exist_ok=True)

def w(name, b):
    open("data/" + name, "wb").write(b)

w("t_empty.bin", b"")                                   # 0 bytes
w("t_1byte.bin", b"\x42")                               # 1 byte
w("t_tiny.bin", b"hello lz4 snappy ascend")             # 23 bytes
w("t_small.bin", bytes([(i * 37 + 11) & 0xff for i in range(1000)]))  # period 256, ~3.7 periods
w("t_repeat.bin", b"ABCDABCD" * 8192)                   # 64 KiB, highly compressible
w("t_zeros.bin", b"\x00" * 65536)                       # exactly one chunk of zeros
w("t_chunkboundary.bin", bytes([(i * 7) & 0xff for i in range(65536 + 33)]))  # crosses a chunk
w("t_text.bin", b"the quick brown fox jumps over the lazy dog. " * 5000)
buf = bytearray()
for i in range(20000):
    buf += struct.pack("<i", i % 97) + b"payload-%d;" % i
w("t_struct.bin", bytes(buf))                          # 20000 records: int32 + "payload-N;"
w("t_random.bin", bytes(random.getrandbits(8) for _ in range(300000)))

# larger multi-block inputs (exercise the block-strided chunk loop on the NPU)
big = bytearray()
for i in range(2 * 1024 * 1024):
    big.append((i * 31 + (i >> 8)) & 0xff)
w("t_big_struct.bin", bytes(big))                      # 2 MiB, 32 chunks, slow-changing pattern
w("t_big_random.bin", bytes(random.getrandbits(8) for _ in range(2 * 1024 * 1024)))  # incompressible control

print("generated:", sorted((f, os.path.getsize("data/" + f)) for f in os.listdir("data")))
