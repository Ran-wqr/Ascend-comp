#!/usr/bin/env python3
"""Fixed FP16 / BF16 / FP32 test-set generator for the LZ4-vs-HANS comparison.

Determinism contract
--------------------
* Same generator + same seed => byte-identical files.
* `manifest.json` records dtype/elems/bytes/sha256/source for every file.
* `--verify` regenerates into a temp dir and compares sha256; any mismatch is a
  hard failure.  Every run script must call `--verify` first, so two benchmark
  sessions can never silently use different data.

HANS constraints (CANN 9.0.0, npu_hans_encode)
----------------------------------------------
    elems % 64 == 0  and  elems >= 32768
Both 1 MiB and 8 MiB per dtype satisfy this: 1 MiB = 524288 elems (fp16/bf16)
or 262144 elems (fp32).

Usage
-----
    python3 scripts/gen_fp_data.py                 # generate (idempotent)
    python3 scripts/gen_fp_data.py --verify        # hash check only
    python3 scripts/gen_fp_data.py --sizes 1 8 64  # add 64 MiB (on demand)
    python3 scripts/gen_fp_data.py --no-weights    # synthetic only
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import tempfile

DTYPES = {"fp16": 2, "bf16": 2, "fp32": 4}
CLASSES = ["weights", "gauss", "uniform", "smooth", "zeros", "repeat", "sparse", "wideexp"]
# fp32-only: emulates a natively-trained fp32 tensor (random low mantissa bits).
# Without it, fp32 "weights" would be a bf16 cast with 16 zero mantissa bits and
# would overstate the compressibility of real fp32 data.
FP32_ONLY_CLASSES = ["f32nat"]


# --------------------------------------------------------------------- rng --
def rng_for(seed, dtype, cls, mib):
    # stable integer derivation (never uses Python's randomized hash())
    h = 1469598103934665603
    for ch in ("%s|%s|%s|%d" % (dtype, cls, mib, seed)).encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return random.Random(h)


# ------------------------------------------------------------------- casts --
def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def pack_f32(v):
    return struct.pack("<f", v)


def pack_f16(v):
    """Round-to-nearest-even fp16 with saturation (records nothing; callers count)."""
    if v != v:                       # NaN
        return struct.pack("<H", 0x7E00)
    if v > 65504.0:
        return struct.pack("<H", 0x7C00)
    if v < -65504.0:
        return struct.pack("<H", 0xFC00)
    return struct.pack("<e", v)


def pack_bf16(v):
    """Truncating f32 -> bf16 (matches the usual hardware cast direction)."""
    b = struct.pack("<f", v)
    return b[2:4]


PACK = {"fp16": pack_f16, "bf16": pack_bf16, "fp32": pack_f32}


def pack_values(dtype, vals):
    p = PACK[dtype]
    out = bytearray()
    if dtype == "fp32":
        for v in vals:
            out += p(v)
    else:
        for v in vals:
            out += p(v)
    return bytes(out)


# ---------------------------------------------------------------- synthetic --
def gen_values(cls, n, rng):
    if cls == "gauss":
        return [rng.gauss(0.0, 0.02) for _ in range(n)]
    if cls == "smooth":
        import math
        out = []
        for i in range(n):
            t = i / 1024.0
            out.append(0.25 * math.sin(t) + 0.1 * math.cos(t * 0.25) + (i / float(n)) * 0.01)
        return out
    if cls == "zeros":
        return [0.0] * n
    if cls == "repeat":
        row = [rng.gauss(0.0, 0.02) for _ in range(512)]
        return (row * (n // 512 + 1))[:n]
    if cls == "sparse":
        out = []
        for _ in range(n):
            out.append(rng.gauss(0.0, 0.02) if rng.random() < 0.05 else 0.0)
        return out
    if cls == "wideexp":
        out = []
        for i in range(n):
            e = rng.randint(-40, 40)
            out.append(rng.choice((-1.0, 1.0)) * (2.0 ** e) * (1.0 + rng.random() * 0.5))
        return out
    raise ValueError(cls)


def gen_synthetic(dtype, cls, nbytes, rng):
    if cls == "uniform":
        return rng.randbytes(nbytes)
    es = DTYPES[dtype]
    n = nbytes // es
    return pack_values(dtype, gen_values(cls, n, rng))


# ------------------------------------------------------------------ weights --
def read_safetensors_slice(path, tensor, nbytes):
    """Return raw BF16 bytes of the first `nbytes` of `tensor` (header-only decode)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        if tensor not in hdr:
            raise KeyError("tensor %r not found in %s" % (tensor, path))
        info = hdr[tensor]
        start, end = info["data_offsets"]
        avail = end - start
        if avail < nbytes:
            raise ValueError("tensor %s has %d bytes, need %d" % (tensor, avail, nbytes))
        f.seek(8 + hlen + start)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise IOError("short read from %s" % path)
    if info.get("dtype") != "BF16":
        raise ValueError("expected BF16 source tensor, got %s" % info.get("dtype"))
    return raw


def cast_bf16_slice(raw, dtype):
    n = len(raw) // 2
    if dtype == "bf16":
        return raw
    if dtype == "fp32":
        # exact: bf16 -> f32 is a 16-bit left shift
        out = bytearray(n * 4)
        for i in range(n):
            u16 = raw[2 * i] | (raw[2 * i + 1] << 8)     # little-endian u16
            struct.pack_into("<I", out, i * 4, u16 << 16)
        return bytes(out)
    # fp16: go through float with saturation
    clipped = 0
    out = bytearray(n * 2)
    for i in range(n):
        u16 = raw[2 * i] | (raw[2 * i + 1] << 8)         # little-endian u16
        v = struct.unpack("<f", struct.pack("<I", u16 << 16))[0]
        if v == v and (v > 65504.0 or v < -65504.0):
            clipped += 1
        out[2 * i:2 * i + 2] = pack_f16(v)
    return bytes(out), clipped


# ----------------------------------------------------------------- manifest --
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(outdir, sizes, seed, weights_path, tensor):
    os.makedirs(outdir, exist_ok=True)
    entries = []
    for dtype in ("fp16", "bf16", "fp32"):
        es = DTYPES[dtype]
        d = os.path.join(outdir, dtype)
        os.makedirs(d, exist_ok=True)
        for mib in sizes:
            nbytes = mib * 1024 * 1024
            n = nbytes // es
            assert n % 64 == 0 and n >= 32768, (dtype, mib, n)
            for cls in CLASSES + FP32_ONLY_CLASSES:
                if cls == "weights":
                    if not weights_path or not os.path.exists(weights_path):
                        continue
                    src_bytes = (nbytes // es) * 2      # BF16 source: 2 B per value
                    raw = read_safetensors_slice(weights_path, tensor, src_bytes)
                    res = cast_bf16_slice(raw, dtype)
                    note = ""
                    if isinstance(res, tuple):
                        data, clipped = res
                        note = "cast bf16->fp16, %d values clipped" % clipped
                    else:
                        data = res
                        note = "cast bf16->%s" % dtype
                    source = "%s:%s[0:%d BF16 bytes]" % (os.path.basename(weights_path), tensor, src_bytes)
                elif cls == "f32nat":
                    if dtype != "fp32":
                        continue
                    if not weights_path or not os.path.exists(weights_path):
                        continue
                    src_bytes = (nbytes // 4) * 2      # BF16 source: 2 B per value
                    raw = read_safetensors_slice(weights_path, tensor, src_bytes)
                    rng = rng_for(seed, "f32nat", cls, mib)
                    n = nbytes // 4
                    out = bytearray(nbytes)
                    for i in range(n):
                        u16 = raw[2 * i] | (raw[2 * i + 1] << 8)
                        low = rng.getrandbits(16)   # plausible native mantissa noise
                        struct.pack_into("<I", out, i * 4, (u16 << 16) | low)
                    data = bytes(out)
                    note = "bf16 high 16 bits + deterministic random low 16 bits (native fp32 emulation)"
                    source = "%s:%s[0:%d BF16 bytes]" % (os.path.basename(weights_path), tensor, src_bytes)
                else:
                    rng = rng_for(seed, dtype, cls, mib)
                    data = gen_synthetic(dtype, cls, nbytes, rng)
                    note = "synthetic"
                    source = "generator seed=%d" % seed
                assert len(data) == nbytes, (dtype, cls, mib, len(data), nbytes)
                name = "%s_%s_%dMiB.bin" % (dtype, cls, mib)
                p = os.path.join(d, name)
                with open(p, "wb") as f:
                    f.write(data)
                entries.append({
                    "name": "%s/%s" % (dtype, name),
                    "dtype": dtype,
                    "class": cls,
                    "elems": n,
                    "bytes": nbytes,
                    "sha256": sha256_file(p),
                    "source": source,
                    "note": note,
                })
    manifest = {
        "version": 1,
        "generator": "scripts/gen_fp_data.py",
        "seed": seed,
        "sizes_mib": sizes,
        "hans_constraints": {"elems_multiple_of": 64, "elems_min": 32768},
        "weights": {"path": weights_path, "tensor": tensor} if weights_path else None,
        "files": sorted(entries, key=lambda e: e["name"]),
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def verify(outdir, sizes, seed, weights_path, tensor):
    mp = os.path.join(outdir, "manifest.json")
    if not os.path.exists(mp):
        print("[verify] missing %s" % mp)
        return 1
    old = json.load(open(mp, encoding="utf-8"))
    if old.get("seed") != seed:
        print("[verify] seed mismatch: manifest=%s arg=%s" % (old.get("seed"), seed))
        return 1
    tmp = tempfile.mkdtemp(prefix="fpdata_verify_")
    try:
        new = generate(tmp, sizes, seed, weights_path, tensor)
        a = {e["name"]: e for e in old["files"]}
        b = {e["name"]: e for e in new["files"]}
        if set(a) != set(b):
            print("[verify] file set differs: missing=%s extra=%s"
                  % (sorted(set(a) - set(b))[:4], sorted(set(b) - set(a))[:4]))
            return 1
        bad = [k for k in a if a[k]["sha256"] != b[k]["sha256"]]
        if bad:
            print("[verify] SHA-256 MISMATCH for %d files, e.g. %s" % (len(bad), bad[:3]))
            return 1
        print("[verify] OK: %d files, sha256 identical (seed=%s, sizes=%s)"
              % (len(a), seed, sizes))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fp")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 8],
                    help="MiB per file (default: 1 8)")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--weights", default="/workspace/ql/wqr/ENEC_910C/models/BF16/Qwen3-32B/model-00001-of-00017.safetensors")
    ap.add_argument("--tensor", default="model.layers.0.mlp.down_proj.weight")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    wp = None if a.no_weights else a.weights
    if wp and not os.path.exists(wp):
        print("[warn] weights file not found (%s); generating synthetic only" % wp)
        wp = None

    if a.verify:
        sys.exit(verify(a.out, a.sizes, a.seed, wp, a.tensor))

    m = generate(a.out, a.sizes, a.seed, wp, a.tensor)
    total = sum(e["bytes"] for e in m["files"])
    print("[gen] %d files, %.1f MiB, seed=%d -> %s"
          % (len(m["files"]), total / 1048576.0, a.seed, a.out))
    for e in m["files"]:
        print("  %-34s %-5s %9d B  %s" % (e["name"], e["dtype"], e["bytes"], e["sha256"][:12]))


if __name__ == "__main__":
    main()
