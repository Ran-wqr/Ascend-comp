/*
 * bench_common.h -- timing + data generation + reporting shared by the
 * benchmarks.  Mirrors nvCOMP benchmarks/benchmark_common.h:
 *   - device-side timing with events (CUDA events -> aclrtEvent)
 *   - data shapes: random / runs / text / structured / zeros
 *   - report: input size, compressed size, ratio, comp ms, comp GB/s,
 *             decomp ms, decomp GB/s, verify result
 */
#ifndef NVCOMP_ASCEND_BENCH_COMMON_H
#define NVCOMP_ASCEND_BENCH_COMMON_H

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include "acl/acl.h"

static inline size_t b_round32(size_t x) { return (x + 31u) & ~(size_t)31u; }

/* GB/s from bytes and milliseconds (decimal GB, same convention as nvCOMP) */
static inline double b_gbs(size_t bytes, double ms)
{
    return (ms > 0.0) ? ((double)bytes / (ms * 1e6)) : 0.0;
}

struct BenchEvents {
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    int init()
    {
        if (aclrtCreateEvent(&start) != ACL_SUCCESS) return 0;
        if (aclrtCreateEvent(&end)   != ACL_SUCCESS) return 0;
        return 1;
    }
    void fini()
    {
        if (start) aclrtDestroyEvent(start);
        if (end)   aclrtDestroyEvent(end);
    }
    float elapsed()
    {
        float ms = -1.0f;
        aclError e = aclrtEventElapsedTime(&ms, start, end);
        if (e != ACL_SUCCESS) {
            fprintf(stderr, "[ACL ERROR] %s:%d aclrtEventElapsedTime -> %d\n",
                    __FILE__, __LINE__, (int)e);
            return -1.0f;
        }
        return ms;
    }
};

/* ---------------- data generators ---------------- */

static std::vector<uint8_t> b_runs(size_t n, size_t run)
{
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)((i / (run ? run : 1)) & 0xff);
    return v;
}

static std::vector<uint8_t> b_random(size_t n, uint32_t seed)
{
    std::mt19937 eng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)d(eng);
    return v;
}

static std::vector<uint8_t> b_zeros(size_t n) { return std::vector<uint8_t>(n, 0); }

static std::vector<uint8_t> b_repeat(size_t n, const char* pat)
{
    size_t L = std::strlen(pat);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)pat[i % L];
    return v;
}

static std::vector<uint8_t> b_text(size_t n)
{
    return b_repeat(n, "the quick brown fox jumps over the lazy dog. ");
}

static std::vector<uint8_t> b_structured(size_t n)
{
    std::vector<uint8_t> v(n);
    size_t i = 0;
    uint32_t rec = 0;
    while (i < n) {
        int32_t id = (int32_t)(rec % 97);
        uint8_t tmp[4] = {(uint8_t)(id & 0xff), (uint8_t)((id >> 8) & 0xff),
                          (uint8_t)((id >> 16) & 0xff), (uint8_t)((id >> 24) & 0xff)};
        for (int k = 0; k < 4 && i < n; k++) v[i++] = tmp[k];
        char tag[32];
        int tn = std::snprintf(tag, sizeof(tag), "payload-%u;", rec);
        for (int k = 0; k < tn && i < n; k++) v[i++] = (uint8_t)tag[k];
        rec++;
    }
    return v;
}

/* ---------------- reporting ---------------- */

static void bench_header(const char* algo, uint32_t chunk, uint32_t dtype,
                         const char* unitLabel)
{
    std::fprintf(stdout,
        "\n=== %s benchmark (chunk=%uB, dtype=%u) ===\n", algo, chunk, dtype);
    std::fprintf(stdout,
        "%-16s %7s %10s %10s %8s %10s %9s %10s %9s %6s\n",
        "dataset", unitLabel, "in(MB)", "comp(MB)", "ratio",
        "comp(ms)", "compGB/s", "dec(ms)", "decGB/s", "verify");
}

static void bench_row(const char* name, uint32_t items, size_t inBytes, size_t compBytes,
                      double compMs, double decMs, bool ok)
{
    const double inMB   = (double)inBytes / (1024.0 * 1024.0);
    const double compMB = (double)compBytes / (1024.0 * 1024.0);
    const double ratio  = inBytes ? (double)compBytes / (double)inBytes : 0.0;
    std::fprintf(stdout,
        "%-16s %7u %10.2f %10.2f %8.4f %10.3f %9.2f %10.3f %9.2f %6s\n",
        name, items, inMB, compMB, ratio, compMs, b_gbs(inBytes, compMs),
        decMs, b_gbs(inBytes, decMs), ok ? "OK" : "FAIL");
}


/* ---------------- statistics / end-to-end / environment ---------------- */
#include <algorithm>
#include <chrono>

static std::vector<float> g_compS, g_decS;
static double g_h2dMs = 0.0, g_kernMs = 0.0, g_d2hMs = 0.0;

static void bench_reset_samples() { g_compS.clear(); g_decS.clear(); }
static inline void bench_add_comp(float ms) { g_compS.push_back(ms); }
static inline void bench_add_dec(float ms)  { g_decS.push_back(ms); }

struct Stat5 { double mean, med, mn, mx, sd; };

static Stat5 stat5(std::vector<float> v)
{
    Stat5 r{0,0,0,0,0};
    if (v.empty()) return r;
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (float x : v) sum += x;
    r.mean = sum / (double)v.size();
    r.mn = v.front();
    r.mx = v.back();
    r.med = (v.size() % 2) ? v[v.size()/2]
                           : 0.5 * (v[v.size()/2 - 1] + v[v.size()/2]);
    double var = 0.0;
    for (float x : v) var += (x - r.mean) * (x - r.mean);
    double sdv = (v.size() > 1) ? (var / (double)(v.size() - 1)) : 0.0;
    r.sd = std::sqrt(sdv);
    return r;
}

static void bench_print_stats(const char* name)
{
    Stat5 c = stat5(g_compS), d = stat5(g_decS);
    std::fprintf(stdout,
        "stats/%-10s comp_ms mean=%.3f med=%.3f min=%.3f max=%.3f sd=%.3f | "
        "dec_ms mean=%.3f med=%.3f min=%.3f max=%.3f sd=%.3f\n",
        name, c.mean, c.med, c.mn, c.mx, c.sd, d.mean, d.med, d.mn, d.mx, d.sd);
}

static void bench_print_e2e(const char* name, size_t inBytes)
{
    const double total = g_h2dMs + g_kernMs + g_d2hMs;
    std::fprintf(stdout,
        "e2e/%-12s h2d=%.3fms kern=%.3fms d2h=%.3fms total=%.3fms GB/s=%.2f\n",
        name, g_h2dMs, g_kernMs, g_d2hMs, total,
        total > 0.0 ? ((double)inBytes / (total * 1e6)) : 0.0);
}

static std::string read_version(const char* path)
{
    std::ifstream f(path);
    if (!f.is_open()) return std::string("unavailable");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("Version=", 0) == 0) return line.substr(8);
    }
    return std::string("unavailable");
}

static std::string soc_name()
{
    FILE* fp = popen("npu-smi info 2>/dev/null | grep -m1 -o 'Ascend[0-9A-Za-z]*'", "r");
    if (!fp) return std::string("unknown");
    char buf[128] = {0};
    if (!fgets(buf, sizeof(buf), fp)) { pclose(fp); return std::string("unknown"); }
    pclose(fp);
    std::string v(buf);
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
    return v.empty() ? std::string("unknown") : v;
}

static void bench_env(const char* algo, const char* mode, const char* item,
                      uint32_t chunk, uint32_t maxb, uint32_t dtype, uint32_t seed)
{
    std::string soc = soc_name();
    std::string cann = read_version("/usr/local/Ascend/cann-9.0.0/compiler/version.info");
    std::string drv  = read_version("/usr/local/Ascend/driver/version.info");
    std::string stamp = "unknown";
    {
        FILE* d = popen("date -Iseconds 2>/dev/null", "r");
        if (d) {
            char tb[64] = {0};
            if (fgets(tb, sizeof(tb), d)) { stamp = tb; while (!stamp.empty() && (stamp.back()=='\n'||stamp.back()=='\r')) stamp.pop_back(); }
            pclose(d);
        }
    }
    std::fprintf(stdout,
        "[env] time=%s algo=%s soc=%s cann=%s driver=%s chunk=%u maxb=%u dtype=%u seed=%u mode=%s item=%s\n",
        stamp.c_str(), algo, soc.c_str(), cann.c_str(), drv.c_str(), chunk, maxb, dtype, seed, mode, item);
}

static void bench_csv_begin(FILE* fp)
{
    std::fprintf(fp,
        "dataset,items,in_bytes,comp_bytes,ratio,"
        "comp_ms_mean,comp_ms_median,comp_ms_min,comp_ms_max,comp_ms_std,"
        "dec_ms_mean,dec_ms_median,dec_ms_min,dec_ms_max,dec_ms_std,"
        "e2e_h2d_ms,e2e_kern_ms,e2e_d2h_ms,e2e_total_ms,e2e_gbs,verify\n");
}

static void bench_csv_row(FILE* fp, const char* name, uint32_t items,
                          size_t inBytes, size_t compBytes, bool ok)
{
    Stat5 c = stat5(g_compS), d = stat5(g_decS);
    const double total = g_h2dMs + g_kernMs + g_d2hMs;
    const double ratio = inBytes ? (double)compBytes / (double)inBytes : 0.0;
    const double gbs = total > 0.0 ? ((double)inBytes / (total * 1e6)) : 0.0;
    std::fprintf(fp,
        "%s,%u,%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
        "%.4f,%.4f,%.4f,%.4f,%.3f,%s\n",
        name, items, inBytes, compBytes, ratio,
        c.mean, c.med, c.mn, c.mx, c.sd, d.mean, d.med, d.mn, d.mx, d.sd,
        g_h2dMs, g_kernMs, g_d2hMs, total, gbs, ok ? "OK" : "FAIL");
}

#endif // NVCOMP_ASCEND_BENCH_COMMON_H
