// HANS baseline benchmark (official CANN aclnn op).
//
//   bench_hans <fp16|bf16|fp32> <file.bin> [iters=10] [warmup=3]
//
// Same measurement contract as benchmarks/bench_lz4.asc:
//   * ACL events around the op, warmup first, per-iteration samples
//   * mean / median / min / max / sd
//   * bit-exact verification of the round trip (outside the timed region)
//   * compressed size = mantissa + fixed_used + var_used + pdf (from the op header)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_hans_encode.h"
#include "aclnnop/aclnn_hans_decode.h"

#define CK(x) do { aclError __s = (x); if (__s != ACL_SUCCESS) { \
    std::fprintf(stderr, "[ACL ERROR] %s -> %d\n", #x, (int)__s); return 1; } } while (0)
#define OP(x) do { aclnnStatus __s = (x); if (__s != 0) { \
    std::fprintf(stderr, "[OP ERROR] %s -> %d\n", #x, (int)__s); return 1; } } while (0)

static bool read_file(const char* p, std::vector<uint8_t>& out)
{
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "[IO] open failed: %s\n", p); return false; }
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
    long n = std::ftell(f);
    if (n < 0) { std::fclose(f); return false; }
    std::rewind(f);
    out.resize((size_t)n);
    if (n && std::fread(out.data(), 1, (size_t)n, f) != (size_t)n) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

struct Stat5 { double mean, med, mn, mx, sd; };
static Stat5 stat5(std::vector<double> v)
{
    Stat5 r{0,0,0,0,0};
    if (v.empty()) return r;
    std::sort(v.begin(), v.end());
    double s = 0; for (double x : v) s += x;
    r.mean = s / (double)v.size();
    r.mn = v.front(); r.mx = v.back();
    r.med = (v.size() % 2) ? v[v.size()/2] : 0.5 * (v[v.size()/2-1] + v[v.size()/2]);
    double var = 0; for (double x : v) var += (x - r.mean) * (x - r.mean);
    r.sd = v.size() > 1 ? std::sqrt(var / (double)(v.size()-1)) : 0.0;
    return r;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <fp16|bf16|fp32> <file.bin> [iters] [warmup]\n", argv[0]);
        return 2;
    }
    const std::string dts = argv[1];
    const uint32_t iters  = (argc > 3) ? (uint32_t)std::atoi(argv[3]) : 10u;
    const uint32_t warmup = (argc > 4) ? (uint32_t)std::atoi(argv[4]) : 3u;

    aclDataType dt;
    int elemBytes;
    if      (dts == "fp16") { dt = ACL_FLOAT16; elemBytes = 2; }
    else if (dts == "bf16") { dt = ACL_BF16;    elemBytes = 2; }
    else if (dts == "fp32") { dt = ACL_FLOAT;   elemBytes = 4; }
    else { std::fprintf(stderr, "bad dtype\n"); return 2; }

    std::vector<uint8_t> hIn;
    if (!read_file(argv[2], hIn)) return 3;
    if (hIn.empty()) { std::fprintf(stderr, "[skip] empty input\n"); return 0; }
    if (hIn.size() % (size_t)elemBytes != 0) { std::fprintf(stderr, "[err] size not a multiple of element size\n"); return 3; }
    const uint32_t elems = (uint32_t)(hIn.size() / (size_t)elemBytes);
    if (elems % 64u != 0u || elems < 32768u) {
        std::fprintf(stderr, "[skip] HANS requires elems%%64==0 && elems>=32768 (got %u)\n", elems);
        return 0;
    }

    const size_t inBytes    = hIn.size();
    const size_t mantBytes  = (size_t)elems * (size_t)(elemBytes - 1);
    const size_t mantElems  = mantBytes / (size_t)elemBytes;
    const size_t vecCores   = 64;
    const size_t bound      = inBytes + inBytes / 64u + 8448u * vecCores;
    const size_t fixedBytes = (bound + 31u) & ~(size_t)31u;
    const size_t varBytes   = (bound + 31u) & ~(size_t)31u;

    CK(aclInit(nullptr));
    CK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CK(aclrtCreateStream(&stream));

    void *dIn=nullptr, *dPdf=nullptr, *dMant=nullptr, *dFixed=nullptr, *dVar=nullptr, *dOut=nullptr;
    CK(aclrtMalloc(&dIn,   inBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dPdf,  256u * 4u,  ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dMant, mantBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dFixed, fixedBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dVar,  varBytes,   ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dOut,  inBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(dIn, inBytes, hIn.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemset(dPdf, 256u * 4u, 0, 256u * 4u));
    CK(aclrtMemset(dFixed, fixedBytes, 0, fixedBytes));
    CK(aclrtMemset(dVar, varBytes, 0, varBytes));

    int64_t inDims[1]    = { (int64_t)elems };
    int64_t pdfDims[2]   = { 1, 256 };
    int64_t mantDims[1]  = { (int64_t)mantElems };
    int64_t fixedDims[1] = { (int64_t)(fixedBytes / (size_t)elemBytes) };
    int64_t varDims[1]   = { (int64_t)(varBytes / (size_t)elemBytes) };

    aclTensor* tIn    = aclCreateTensor(inDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, inDims, 1, dIn);
    aclTensor* tPdf   = aclCreateTensor(pdfDims, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, pdfDims, 2, dPdf);
    aclTensor* tMant  = aclCreateTensor(mantDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, mantDims, 1, dMant);
    aclTensor* tFixed = aclCreateTensor(fixedDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, fixedDims, 1, dFixed);
    aclTensor* tVar   = aclCreateTensor(varDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, varDims, 1, dVar);
    aclTensor* tOut   = aclCreateTensor(inDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, inDims, 1, dOut);

    // encoder / decoder executors are created once (host setup, not timed)
    uint64_t wsE = 0, wsD = 0;
    aclOpExecutor* execE = nullptr;
    aclOpExecutor* execD = nullptr;
    OP(aclnnHansEncodeGetWorkspaceSize(tIn, tPdf, true, false, tMant, tFixed, tVar, &wsE, &execE));
    OP(aclnnHansDecodeGetWorkspaceSize(tMant, tFixed, tVar, tPdf, false, tOut, &wsD, &execD));
    void* wsEbuf = nullptr; void* wsDbuf = nullptr;
    if (wsE) CK(aclrtMalloc(&wsEbuf, wsE, ACL_MEM_MALLOC_HUGE_FIRST));
    if (wsD) CK(aclrtMalloc(&wsDbuf, wsD, ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtEvent ev0 = nullptr, ev1 = nullptr;
    CK(aclrtCreateEvent(&ev0));
    CK(aclrtCreateEvent(&ev1));

    for (uint32_t w = 0; w < warmup; w++) {
        OP(aclnnHansEncode(wsEbuf, wsE, execE, stream));
        OP(aclnnHansDecode(wsDbuf, wsD, execD, stream));
    }
    CK(aclrtSynchronizeStream(stream));

    std::vector<double> compS, decS;
    compS.reserve(iters); decS.reserve(iters);
    for (uint32_t i = 0; i < iters; i++) {
        CK(aclrtRecordEvent(ev0, stream));
        OP(aclnnHansEncode(wsEbuf, wsE, execE, stream));
        CK(aclrtRecordEvent(ev1, stream));
        CK(aclrtSynchronizeEvent(ev1));
        float ms = -1.0f;
        if (aclrtEventElapsedTime(&ms, ev0, ev1) != ACL_SUCCESS || ms < 0.0f) {
            std::fprintf(stderr, "[err] event timing failed\n"); return 4;
        }
        compS.push_back(ms);
    }
    for (uint32_t i = 0; i < iters; i++) {
        CK(aclrtRecordEvent(ev0, stream));
        OP(aclnnHansDecode(wsDbuf, wsD, execD, stream));
        CK(aclrtRecordEvent(ev1, stream));
        CK(aclrtSynchronizeEvent(ev1));
        float ms = -1.0f;
        if (aclrtEventElapsedTime(&ms, ev0, ev1) != ACL_SUCCESS || ms < 0.0f) {
            std::fprintf(stderr, "[err] event timing failed\n"); return 4;
        }
        decS.push_back(ms);
    }

    // compressed size from the op header (first 512 B of `fixed`)
    int32_t hdr[128];
    CK(aclrtMemcpy(hdr, sizeof(hdr), dFixed, sizeof(hdr), ACL_MEMCPY_DEVICE_TO_HOST));
    uint32_t coreDim = (uint32_t)hdr[1];
    if (coreDim == 0u || coreDim > 64u) coreDim = 64u;
    uint64_t fixedUsed = 0, varUsed = 0;
    for (uint32_t c = 0; c < coreDim; c++) {
        if (hdr[8 + c]  > 0) fixedUsed += (uint32_t)hdr[8 + c];
        if (hdr[64 + c] > 0) varUsed   += (uint32_t)hdr[64 + c];
    }
    const uint64_t total = (uint64_t)mantBytes + fixedUsed + varUsed + 256u * 4u;

    // verification (outside the timed region)
    std::vector<uint8_t> hOut(inBytes);
    CK(aclrtMemcpy(hOut.data(), inBytes, dOut, inBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    size_t bad = 0, firstBad = 0;
    for (size_t i = 0; i < inBytes; i++) {
        if (hOut[i] != hIn[i]) { if (!bad) firstBad = i; bad++; }
    }

    Stat5 c = stat5(compS), d = stat5(decS);
    const char* base = std::strrchr(argv[2], '/');
    base = base ? base + 1 : argv[2];
    const bool csv = (argc > 5) && (std::strcmp(argv[5], "csv") == 0);
    if (csv) {
        std::printf("%s,%s,%u,%zu,%llu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%s\n",
                    base, dts.c_str(), elems, inBytes, (unsigned long long)total,
                    (double)total / (double)inBytes,
                    c.mean, c.med, c.mn, c.mx, c.sd,
                    d.mean, d.med, d.mn, d.mx, bad ? "FAIL" : "OK");
    } else {
        std::printf("%-28s %-4s %9u %10zu %8.4f %8.2f %8.2f %8.3f %8.3f %8.3f %8.2f %8.3f %8.3f %8.3f %s\n",
                    base, dts.c_str(), elems, inBytes, (double)total / (double)inBytes,
                    c.mean, c.med, c.mn, c.mx, c.sd,
                    d.mean, d.med, d.mn, d.mx,
                    bad ? "FAIL" : "OK");
    }
    std::fprintf(stderr, "[hans] %s: comp %.2f GB/s (med), dec %.2f GB/s (med)%s\n",
                 base, (double)inBytes / (c.med * 1e6), (double)inBytes / (d.med * 1e6),
                 bad ? " MISMATCH" : "");

    aclrtDestroyEvent(ev0); aclrtDestroyEvent(ev1);
    if (wsEbuf) aclrtFree(wsEbuf);
    if (wsDbuf) aclrtFree(wsDbuf);
    aclrtFree(dIn); aclrtFree(dPdf); aclrtFree(dMant); aclrtFree(dFixed); aclrtFree(dVar); aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return bad ? 1 : 0;
}
