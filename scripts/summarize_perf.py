#!/usr/bin/env python3
"""Generic result table for any benchmark CSV produced by this project.

Works with the three 18-column schemas (bench_fp / bench_lz4_file / cpu runner)
and with the 17-column standalone HANS CSV.  Columns are read by name, so a
schema difference only shows up as a missing codec entry, not a parse error.

  summarize_perf.py <csv> [--size MiB|0] [--codecs a,b,c]
"""
import argparse
import csv
import sys


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def gbs(nbytes, ms):
    return nbytes / (ms * 1e6) if ms and ms > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--size", type=int, default=0,
                    help="keep only *_<N>MiB.bin datasets (0 = all)")
    ap.add_argument("--codecs", default="",
                    help="comma-separated codec order (default: as found)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8")))
    if not rows:
        print("(empty CSV)")
        return 1
    for r in rows:                      # standalone HANS CSV has no codec column
        if not r.get("codec"):
            r["codec"] = "hans"

    datasets = []
    for r in rows:
        d = r["dataset"]
        if a.size and not d.endswith("_%dMiB.bin" % a.size):
            continue
        if d not in datasets:
            datasets.append(d)
    datasets.sort()

    found = []
    for r in rows:
        if r["codec"] not in found:
            found.append(r["codec"])
    codecs = a.codecs.split(",") if a.codecs else found

    head = "%-26s %9s" % ("dataset", "in(B)")
    for c in codecs:
        head += " | %-9s %7s %7s" % (c + "_r", "cGB/s", "dGB/s")
    head += " | best"
    print(head)
    print("-" * len(head))

    by = {}
    for r in rows:
        by.setdefault(r["dataset"], {})[r["codec"]] = r

    bad = 0
    for d in datasets:
        base = by.get(d, {})
        if not base:
            continue
        nb = fnum(next(iter(base.values())), "in_bytes")
        line = "%-26s %9.0f" % (d, nb)
        best, best_r = "-", float("inf")
        for c in codecs:
            r = base.get(c)
            if not r:
                line += " | %-9s %7s %7s" % ("-", "-", "-")
                continue
            ratio = fnum(r, "ratio")
            line += " | %-9.4f %7.2f %7.2f" % (
                ratio,
                gbs(nb, fnum(r, "comp_ms_median")),
                gbs(nb, fnum(r, "dec_ms_median")))
            if ratio == ratio and ratio < best_r:
                best, best_r = c, ratio
            if r.get("verify") != "OK":
                bad += 1
        print(line + " | %s" % best)

    print("-" * len(head))
    print("rows=%d  verify_fail=%d  (ratio lower is better)" % (len(rows), bad))
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
