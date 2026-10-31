/*
 * test_common.h -- minimal test harness + data generators + detailed
 * per-case reporting (dataset, units, ratio, throughput).
 */
#ifndef NVCOMP_ASCEND_TEST_COMMON_H
#define NVCOMP_ASCEND_TEST_COMMON_H

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include "acl/acl.h"

static int g_pass = 0;
static int g_fail = 0;
static const char* g_group = "";

static uint64_t g_sum_in = 0;
static uint64_t g_sum_comp = 0;
static double   g_sum_comp_ms = 0.0;
static double   g_sum_dec_ms  = 0.0;

#define TCHECK(cond)                                                            \
    do {                                                                        \
        if (cond) { g_pass++; }                                                 \
        else {                                                                  \
            g_fail++;                                                           \
            std::fprintf(stdout, "[FAIL] %-14s %s:%d  %s\n",                    \
                         g_group, __FILE__, __LINE__, #cond);                   \
        }                                                                       \
    } while (0)

static inline size_t t_round32(size_t x) { return (x + 31u) & ~(size_t)31u; }

/* device-event timer (reused across cases) */
struct TEvents {
    aclrtEvent a = nullptr;
    aclrtEvent b = nullptr;
    int init()
    {
        if (aclrtCreateEvent(&a) != ACL_SUCCESS) return 0;
        if (aclrtCreateEvent(&b) != ACL_SUCCESS) return 0;
        return 1;
    }
    void fini()
    {
        if (a) { aclrtDestroyEvent(a); a = nullptr; }
        if (b) { aclrtDestroyEvent(b); b = nullptr; }
    }
    float elapsed()
    {
        float ms = -1.0f;
        aclrtEventElapsedTime(&ms, a, b);
        return ms;
    }
};

static TEvents g_ev;

static int g_verbose = 1;

struct TestStat {
    uint32_t n = 0;
    uint32_t chunks = 0;
    uint32_t items = 0;
    uint32_t minItem = 0;
    uint32_t maxItem = 0;
    uint32_t launches = 0;
    uint64_t comp = 0;
};

/* one detailed line per case */
static void t_report(const char* group, const char* desc,
                     uint64_t inBytes, uint32_t units, const char* unitKey,
                     uint32_t launches, uint64_t compBytes,
                     double compMs, double decMs, bool hasDec, bool ok,
                     bool accumulate = true)
{
    char ratio[24], cgb[24], dgb[24];
    if (inBytes > 0) std::snprintf(ratio, sizeof(ratio), "%.4f", (double)compBytes / (double)inBytes);
    else             std::snprintf(ratio, sizeof(ratio), "--");
    if (inBytes > 0 && compMs > 0.0)
        std::snprintf(cgb, sizeof(cgb), "%.2f", (double)inBytes / (compMs * 1e6));
    else
        std::snprintf(cgb, sizeof(cgb), "--");
    if (hasDec && inBytes > 0 && decMs > 0.0)
        std::snprintf(dgb, sizeof(dgb), "%.2f", (double)inBytes / (decMs * 1e6));
    else
        std::snprintf(dgb, sizeof(dgb), "--");

    if (!ok || g_verbose) {
        if (hasDec) {
            std::fprintf(stdout,
                "[%s] grp=%-16s %s=%-6u launch=%-2u n=%-8llu comp=%-9llu ratio=%-7s "
                "comp_ms=%8.3f comp_GBs=%6s dec_ms=%8.3f dec_GBs=%6s  data=%s\n",
                ok ? " OK " : "FAIL", group, unitKey, units, launches,
                (unsigned long long)inBytes, (unsigned long long)compBytes, ratio,
                compMs, cgb, decMs, dgb, desc);
        } else {
            std::fprintf(stdout,
                "[%s] grp=%-16s %s=%-6u launch=%-2u n=%-8llu comp=%-9llu ratio=%-7s "
                "comp_ms=%8.3f comp_GBs=%6s dec_ms=      -- dec_GBs=    --  data=%s\n",
                ok ? " OK " : "FAIL", group, unitKey, units, launches,
                (unsigned long long)inBytes, (unsigned long long)compBytes, ratio,
                compMs, cgb, desc);
        }
    }

    if (accumulate) {
        g_sum_in += inBytes;
        g_sum_comp += compBytes;
        if (compMs > 0.0) g_sum_comp_ms += compMs;
        if (hasDec && decMs > 0.0) g_sum_dec_ms += decMs;
    }
}


/* ---- data generators ---- */

static std::vector<uint8_t> gen_runs(size_t n, size_t run)
{
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)((i / (run ? run : 1)) & 0xff);
    return v;
}

static std::vector<uint8_t> gen_random(size_t n, uint32_t seed, int maxByte = 255)
{
    std::mt19937 eng(seed);
    std::uniform_int_distribution<int> d(0, maxByte);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)d(eng);
    return v;
}

static std::vector<uint8_t> gen_zeros(size_t n) { return std::vector<uint8_t>(n, 0); }

static std::vector<uint8_t> gen_repeat(size_t n, const char* pat)
{
    size_t L = std::strlen(pat);
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)pat[i % L];
    return v;
}

static std::vector<uint8_t> gen_text(size_t n)
{
    return gen_repeat(n, "the quick brown fox jumps over the lazy dog. ");
}

static std::vector<uint8_t> gen_structured(size_t n)
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

static int test_summary(const char* suite)
{
    std::fprintf(stdout, "\n== %s: %d passed, %d failed ==\n", suite, g_pass, g_fail);

    char ratio[24];
    if (g_sum_in > 0) std::snprintf(ratio, sizeof(ratio), "%.4f", (double)g_sum_comp / (double)g_sum_in);
    else              std::snprintf(ratio, sizeof(ratio), "--");

    std::fprintf(stdout, "   aggregate: n=%llu comp=%llu ratio=%s",
                 (unsigned long long)g_sum_in, (unsigned long long)g_sum_comp, ratio);
    if (g_sum_comp_ms > 0.0) {
        char cgb[24];
        std::snprintf(cgb, sizeof(cgb), "%.2f", (double)g_sum_in / (g_sum_comp_ms * 1e6));
        std::fprintf(stdout, " comp_ms=%.3f comp_GBs=%s", g_sum_comp_ms, cgb);
    }
    if (g_sum_dec_ms > 0.0) {
        char dgb[24];
        std::snprintf(dgb, sizeof(dgb), "%.2f", (double)g_sum_in / (g_sum_dec_ms * 1e6));
        std::fprintf(stdout, " dec_ms=%.3f dec_GBs=%s", g_sum_dec_ms, dgb);
    }
    std::fprintf(stdout, "\n");

    if (g_fail == 0) std::fprintf(stdout, "RESULT: PASS\n");
    else             std::fprintf(stdout, "RESULT: FAIL\n");
    return (g_fail == 0) ? 0 : 1;
}


#endif
