#!/usr/bin/env python3
"""Side-by-side FP comparison: LZ4 raw / LZ4 plane-split / HANS.

Reads logs/bench_fp.csv (produced by scripts/run_bench_fp.sh) and prints one
row per dataset.  All three codecs were run by the same binary, on the same
input bytes, with the same warmup/iters and the same verification.
"""
import argparse
import csv
import sys


def load(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["dataset"], r["codec"])] = r
    return rows


def f(r, k, d=float("nan")):
    if not r:
        return d
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="logs/bench_fp.csv")
    ap.add_argument("--size", type=int, default=8, help="MiB filter (0 = all)")
    a = ap.parse_args()

    rows = load(a.csv)
    datasets = sorted({k[0] for k in rows})
    if a.size:
        datasets = [d for d in datasets if d.endswith("_%dMiB.bin" % a.size)]

    print("%-26s %-5s %9s %8s %8s %8s %-6s %9s %9s %9s %s" % (
        "dataset", "dtype", "in(B)", "raw_r", "plane_r", "hans_r",
        "best", "raw_c", "plane_c", "hans_c", "verify"))
    print("-" * 128)

    wins = {"lz4raw": 0, "lz4": 0, "hans": 0}
    n = 0
    for d in datasets:
        raw, pl, ha = rows.get((d, "lz4raw")), rows.get((d, "lz4")), rows.get((d, "hans"))
        if not (raw and pl and ha):
            print("%-26s (missing codec)" % d)
            continue
        rr, pr, hr = f(raw, "ratio"), f(pl, "ratio"), f(ha, "ratio")
        best = min((rr, "lz4raw"), (pr, "lz4"), (hr, "hans"))[1]
        wins[best] += 1
        n += 1
        ok = all(r["verify"] == "OK" for r in (raw, pl, ha))
        print("%-26s %-5s %9s %8.4f %8.4f %8.4f %-6s %9.2f %9.2f %9.2f %s" % (
            d, raw["dtype"], raw["in_bytes"], rr, pr, hr, best,
            f(raw, "comp_ms_median"), f(pl, "comp_ms_median"), f(ha, "comp_ms_median"),
            "OK" if ok else "FAIL"))

    print("-" * 128)
    print("best ratio count (n=%d): lz4raw=%d  lz4(plane)=%d  hans=%d"
          % (n, wins["lz4raw"], wins["lz4"], wins["hans"]))
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
