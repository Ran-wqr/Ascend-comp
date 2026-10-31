#!/usr/bin/env python3
"""Large (~5 GB total) byte-stream test set for CPU-vs-NPU LZ4.

Five 1,000,000,000-byte files (decimal 1 GB, same size as enwik9):

  enwik9.bin           real corpus (en.wikipedia XML dump), highly compressible text
  large_struct_1G.bin  per 64 KiB chunk: 48 KiB of repeated records + 16 KiB fresh noise
                       (~75% patterned / 25% incompressible -> mid ratio)
  large_random_1G.bin  deterministic random bytes  (incompressible control, ratio ~1.0)
  large_binary_1G.bin  4 KiB pages, 2 KiB template + 2 KiB noise (binary/DB-like)
  large_mixed_1G.bin   250 MB text + 250 MB records + 250 MB random + 250 MB zeros

Why these three: they span the compressibility range, and 1 GB removes the two
effects that dominate the small vectors -- CPU L1/L2 residency and the NPU's
~1 ms per-launch overhead.

Reproducibility
  * enwik9: downloaded from mattmahoney.net (322,592,222 B zip), size checked to
    be exactly 1e9 bytes; sha256 recorded in manifest.json.
  * synthetic: fixed seed, deterministic loop order.
  * `--verify` re-hashes every file against manifest.json (no regeneration).

Usage
  python3 scripts/gen_large_testdata.py                 # generate missing files
  python3 scripts/gen_large_testdata.py --verify        # hash check only
  python3 scripts/gen_large_testdata.py --only struct   # subset
  python3 scripts/gen_large_testdata.py --size 2000 --out data/large2G
"""
import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import time
import zipfile

BYTES_1G = 1000 * 1000 * 1000
CHUNK = 64 * 1024
STRUCT_TEMPLATE = 48 * 1024          # patterned part of each chunk
BINARY_PAGE = 4 * 1024               # 50% template + 50% noise per page
MIXED_SEG = 250 * 1000 * 1000        # each of the 4 mixed segments
ENWIK9_URL = "https://mattmahoney.net/dc/enwik9.zip"


def sha256_file(p, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def ensure_enwik9(outdir, total, keep_zip):
    dst = os.path.join(outdir, "enwik9.bin")
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        return dst
    zpath = os.path.join(outdir, "enwik9.zip")
    if not os.path.exists(zpath):
        print("[gen] downloading enwik9.zip ...")
        subprocess.check_call(["curl", "-sL", "--retry", "3", "-o", zpath, ENWIK9_URL])
    print("[gen] extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extract("enwik9", outdir)
    src = os.path.join(outdir, "enwik9")
    if os.path.getsize(src) != total:
        raise SystemExit("[gen] enwik9 size %d != expected %d" % (os.path.getsize(src), total))
    os.replace(src, dst)
    if not keep_zip:
        os.remove(zpath)
    return dst


def build_template():
    b = bytearray()
    i = 0
    while len(b) < STRUCT_TEMPLATE:
        b += struct.pack("<I", i & 0xffff)
        b += ("payload-%08d;" % (i % 1000)).encode()
        i += 1
    return bytes(b[:STRUCT_TEMPLATE])


def gen_struct(dst, total, seed):
    tpl = build_template()
    rng = random.Random(seed)
    noise_len = CHUNK - len(tpl)
    with open(dst, "wb") as f:
        left = total
        while left > 0:
            n = min(CHUNK, left)
            buf = bytearray(tpl)
            if n > len(tpl):
                buf += rng.randbytes(n - len(tpl))
            else:
                buf = buf[:n]
            f.write(buf)
            left -= n


def gen_random(dst, total, seed):
    rng = random.Random(seed)
    block = 8 << 20
    with open(dst, "wb") as f:
        left = total
        while left > 0:
            n = min(block, left)
            f.write(rng.randbytes(n))
            left -= n


def gen_binary(dst, total, seed):
    rng = random.Random(seed)
    tpl = (bytes(range(256)) * 16)[:BINARY_PAGE]
    half = BINARY_PAGE // 2
    with open(dst, "wb") as f:
        left = total
        while left > 0:
            page = bytearray(tpl[:half])
            page += rng.randbytes(half)
            n = min(len(page), left)
            f.write(page[:n])
            left -= n


def gen_text4g(dst, total, srcdir):
    """Stream the real enwik9 text repeatedly until `total` bytes."""
    src = os.path.join(srcdir, "enwik9.bin")
    if not os.path.exists(src):
        raise SystemExit("[gen] text4g needs %s (run without --only first)" % src)
    with open(src, "rb") as g, open(dst, "wb") as f:
        left = total
        while left > 0:
            g.seek(0)
            n = min(left, os.path.getsize(src))
            chunk = g.read(n)
            f.write(chunk)
            left -= len(chunk)


def gen_mixed(dst, total, seed, srcdir):
    seg = total // 4
    rng = random.Random(seed)
    enw = os.path.join(srcdir, "enwik9.bin")
    with open(dst, "wb") as f:
        # 1) text (real corpus slice, synthetic fallback)
        if os.path.exists(enw):
            with open(enw, "rb") as g:
                f.write(g.read(seg))
        else:
            tpl = (b"the quick brown fox jumps over the lazy dog. " * 2000)[:65536]
            left = seg
            while left > 0:
                n = min(len(tpl), left)
                f.write(tpl[:n]); left -= n
        # 2) record-like (48 KiB template + noise per 64 KiB)
        tpl2 = build_template()
        left = seg
        while left > 0:
            n = min(CHUNK, left)
            buf = bytearray(tpl2)
            if n > len(tpl2):
                buf += rng.randbytes(n - len(tpl2))
            f.write(buf[:n]); left -= n
        # 3) incompressible
        left = seg
        while left > 0:
            n = min(8 << 20, left)
            f.write(rng.randbytes(n)); left -= n
        # 4) zeros
        z = b"\x00" * (1 << 20)
        left = total - 3 * seg
        while left > 0:
            n = min(len(z), left)
            f.write(z[:n]); left -= n


SPECS = {
    "enwik9": {"file": "enwik9.bin", "kind": "real corpus (Wikipedia XML)", "seed": None},
    "struct": {"file": "large_struct_1G.bin", "kind": "48 KiB records + 16 KiB noise per 64 KiB chunk", "seed": 0x5EED0001},
    "random": {"file": "large_random_1G.bin", "kind": "deterministic random bytes", "seed": 0x5EED0002},
    "binary": {"file": "large_binary_1G.bin", "kind": "4 KiB pages: 2 KiB template + 2 KiB noise", "seed": 0x5EED0003},
    "mixed": {"file": "large_mixed_1G.bin", "kind": "250 MB text + records + random + zeros", "seed": 0x5EED0004},
    # single-file variants (processed in 1 GiB windows on the NPU side)
    "text4g": {"file": "large_4G.bin", "kind": "enwik9 repeated to 4 GB", "seed": None},
    "text5g": {"file": "large_5G_text.bin", "kind": "enwik9 repeated to 5 GB", "seed": None},
    "struct5g": {"file": "large_5G_struct.bin", "kind": "48 KiB records + 16 KiB noise per 64 KiB chunk", "seed": 0x5EED0005},
    "random5g": {"file": "large_5G_random.bin", "kind": "deterministic random bytes", "seed": 0x5EED0006},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/large")
    ap.add_argument("--size", type=int, default=BYTES_1G, help="bytes per file (default 1e9)")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--only",
                    choices=["enwik9", "struct", "random", "binary", "mixed",
                             "text4g", "text5g", "struct5g", "random5g"],
                    action="append")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    manifest_path = os.path.join(a.out, "manifest.json")
    wanted = a.only or ["enwik9", "struct", "random", "binary", "mixed"]

    if a.verify:
        if not os.path.exists(manifest_path):
            raise SystemExit("[verify] missing %s" % manifest_path)
        m = json.load(open(manifest_path, encoding="utf-8"))
        bad = 0
        for e in m["files"]:
            p = os.path.join(a.out, e["file"])
            if not os.path.exists(p):
                print("[verify] MISSING %s" % p); bad += 1; continue
            sz = os.path.getsize(p)
            h = sha256_file(p)
            if sz != e["bytes"] or h != e["sha256"]:
                print("[verify] MISMATCH %s (size %d/%d)" % (e["file"], sz, e["bytes"])); bad += 1
            else:
                print("[verify] OK %-24s %d B %s" % (e["file"], sz, h[:16]))
        print("[verify] %s" % ("FAIL" if bad else "OK: all files match manifest.json"))
        sys.exit(1 if bad else 0)

    files = []
    for key in wanted:
        spec = SPECS[key]
        dst = os.path.join(a.out, spec["file"])
        if key == "enwik9":
            ensure_enwik9(a.out, a.size, a.keep_zip)
        elif os.path.exists(dst) and os.path.getsize(dst) == a.size:
            print("[gen] exists, skip: %s" % dst)
        else:
            t0 = time.time()
            if key == "struct":
                gen_struct(dst, a.size, a.seed ^ spec["seed"])
            elif key == "binary":
                gen_binary(dst, a.size, a.seed ^ spec["seed"])
            elif key == "mixed":
                gen_mixed(dst, a.size, a.seed ^ spec["seed"], a.out)
            elif key in ("text4g", "text5g"):
                gen_text4g(dst, a.size, a.out)
            elif key == "struct5g":
                gen_struct(dst, a.size, a.seed ^ spec["seed"])
            elif key == "random5g":
                gen_random(dst, a.size, a.seed ^ spec["seed"])
            else:
                gen_random(dst, a.size, a.seed ^ spec["seed"])
            print("[gen] %-24s %d B in %.1fs" % (spec["file"], a.size, time.time() - t0))
        files.append({
            "file": spec["file"],
            "label": key,
            "bytes": os.path.getsize(dst),
            "sha256": sha256_file(dst),
            "kind": spec["kind"],
            "seed": a.seed if spec["seed"] is not None else None,
            "source": ENWIK9_URL if key == "enwik9" else "generator",
        })

    # merge with an existing manifest so `--only` does not drop other entries
    merged = {}
    if os.path.exists(manifest_path):
        for e in json.load(open(manifest_path, encoding="utf-8")).get("files", []):
            merged[e["file"]] = e
    for e in files:
        merged[e["file"]] = e
    sizes = sorted({e["bytes"] for e in merged.values()})
    out = {
        "version": 1,
        "generator": "scripts/gen_large_testdata.py",
        "sizes_bytes": sizes,
        "files": sorted(merged.values(), key=lambda e: e["file"]),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print("\n[gen] manifest -> %s" % manifest_path)
    for e in out["files"]:
        print("  %-24s %12d B  %s  %s" % (e["file"], e["bytes"], e["sha256"][:16], e["kind"]))


if __name__ == "__main__":
    main()
