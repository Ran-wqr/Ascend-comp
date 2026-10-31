#!/usr/bin/env python3
"""CPU vs NPU LZ4 summary from logs/bench_lz4_cpu.csv.

Columns produced by both benchmarks are identical:
  dataset, codec, dtype, in_bytes, comp_bytes, ratio,
  comp_ms_{mean,median,min,max,sd}, dec_ms_{...}, e2e_ms, verify

Throughput is derived from the median: GB/s = in_bytes / (ms * 1e6).
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


def num(r, k):
    if not r:
        return float("nan")
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def gbs(nbytes, ms):
    return (nbytes / (ms * 1e6)) if ms and ms > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="logs/bench_lz4_cpu.csv")
    a = ap.parse_args()

    rows = load(a.csv)
    datasets = sorted({k[0] for k in rows})

    print("%-24s %9s %7s %7s %7s | %8s %8s %8s | %8s %8s %8s | %s" % (
        "dataset", "in(B)", "up_r", "kp_r", "npu_r",
        "up_c", "kp_c", "npu_c", "up_d", "kp_d", "npu_d", "note"))
    print("(c/d = GB/s from median; ratios lower is better)  " + "-" * 34)

    n = 0
    wins = {"cpu": 0, "npu": 0, "tie": 0}
    for d in datasets:
        up, kp, npu = rows.get((d, "lz4cpu_up")), rows.get((d, "lz4cpu_kp")), rows.get((d, "lz4npu"))
        if not (up and kp and npu):
            print("%-24s (missing codec)" % d)
            continue
        nb = int(float(up["in_bytes"]))
        u, k, q = (num(x, "comp_ms_median") for x in (up, kp, npu))
        if q and q > 0:
            if q < min(u, k):
                wins["npu"] += 1
            elif min(u, k) < q:
                wins["cpu"] += 1
            else:
                wins["tie"] += 1
        n += 1
        ok = all(r["verify"] == "OK" for r in (up, kp, npu))
        note = "" if ok else "VERIFY-FAIL"
        if nb < 4096:
            note = (note + " ").strip()
            note = (note + " tiny(cache-resident CPU)").strip()
        print("%-24s %9d %7.4f %7.4f %7.4f | %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f | %s" % (
            d, nb,
            num(up, "ratio"), num(kp, "ratio"), num(npu, "ratio"),
            gbs(nb, u), gbs(nb, k), gbs(nb, q),
            gbs(nb, num(up, "dec_ms_median")),
            gbs(nb, num(kp, "dec_ms_median")),
            gbs(nb, num(npu, "dec_ms_median")),
            note))

    print("-" * 118)
    print("compression winner (median, n=%d): cpu=%d  npu=%d  tie=%d"
          % (n, wins["cpu"], wins["npu"], wins["tie"]))
    print("up=upstream LZ4 v1.9.4, kp=kunpengcompute/lz4, npu=Ascend C kernels")
    print("npu comp/dec = kernel-only (ACL events); npu e2e = H2D+kernel+D2H (see CSV)")
    print("CPU is single-thread, pinned with taskset; both sides use 64 KiB blocks.")
    print("Small inputs are L1/L2-resident on the CPU and dominated by the NPU launch")
    print("overhead (~1 ms). For throughput conclusions use data/large:")
    print("  bash scripts/run_bench_lz4_cpu_large.sh 3 1   # 5 x 1 GB")
    print("  bash scripts/run_bench_lz4_cpu_4g.sh 3 1      # single 4 GB")
    print("  bash scripts/run_bench_lz4_cpu_5g.sh 3 1      # three 5 GB files")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
