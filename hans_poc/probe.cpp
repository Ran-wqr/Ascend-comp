// HANS aclnn probe: round-trip + compressed-size accounting.
//   usage: probe <elemBytes:2|4> <elems> [reshuff:0|1]
// elemBytes=2 -> ACL_FLOAT16, 4 -> ACL_FLOAT. (BF16 selected with elemBytes=2 + arg3 "bf16")
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_hans_encode.h"
#include "aclnnop/aclnn_hans_decode.h"

#define CK(x) do { aclError __s = (x); if (__s != ACL_SUCCESS) { \
    std::printf("[FAIL] %s -> %d\n", #x, (int)__s); return 1; } } while (0)
#define OP(x) do { aclnnStatus __s = (x); if (__s != 0) { \
    std::printf("[FAIL] %s -> %d\n", #x, (int)__s); return 1; } } while (0)

static uint64_t g_st = 12345u;
static uint32_t rnd() {
    g_st = g_st * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(g_st >> 33);
}

int main(int argc, char** argv)
{
    const int elemBytes = (argc > 1) ? std::atoi(argv[1]) : 2;
    const uint32_t elems  = (argc > 2) ? (uint32_t)std::atoi(argv[2]) : 32768u;
    const bool reshuff    = (argc > 3) ? (std::atoi(argv[3]) != 0) : false;
    const bool bf16       = (argc > 4) ? (std::strcmp(argv[4], "bf16") == 0) : false;

    if (elemBytes != 2 && elemBytes != 4) { std::printf("elemBytes must be 2 or 4\n"); return 2; }
    if (elems % 64u != 0u || elems < 32768u) {
        std::printf("HANS requires elems %% 64 == 0 and elems >= 32768\n"); return 2;
    }

    const size_t inBytes    = (size_t)elems * (size_t)elemBytes;
    const size_t mantBytes  = (size_t)elems * (size_t)(elemBytes - 1);
    const uint32_t mantElems = (uint32_t)(mantBytes / elemBytes);

    // upper bound from the op doc: input + input/64 + 8448 * vector cores
    const size_t vecCores = 64;
    const size_t bound    = inBytes + inBytes / 64u + 8448u * vecCores;
    const size_t fixedBytes = (bound + 31u) & ~(size_t)31u;
    const size_t varBytes   = reshuff ? 32u : ((bound + 31u) & ~(size_t)31u);

    const aclDataType dt = (elemBytes == 4) ? ACL_FLOAT : (bf16 ? ACL_BF16 : ACL_FLOAT16);

    std::vector<uint8_t> hIn(inBytes);
    for (uint32_t i = 0; i < elems; i++) {
        if (elemBytes == 4) {
            uint32_t v = 0x3F000000u + (rnd() % 0x02000000u);
            std::memcpy(hIn.data() + (size_t)i * 4, &v, 4);
        } else {
            uint16_t v = (uint16_t)(0x3800u + (rnd() % 0x0400u));
            std::memcpy(hIn.data() + (size_t)i * 2, &v, 2);
        }
    }

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
    CK(aclrtMemset(dMant, mantBytes, 0, mantBytes));
    CK(aclrtMemset(dFixed, fixedBytes, 0xAA, fixedBytes));
    CK(aclrtMemset(dVar, varBytes, 0xAA, varBytes));
    CK(aclrtMemset(dOut, inBytes, 0, inBytes));

    int64_t inDims[1]   = { (int64_t)elems };
    int64_t pdfDims[2]  = { 1, 256 };
    int64_t mantDims[1] = { (int64_t)mantElems };
    int64_t fixedDims[1]= { (int64_t)(fixedBytes / (size_t)elemBytes) };
    int64_t varDims[1]  = { (int64_t)(varBytes / (size_t)elemBytes) };

    aclTensor* tIn    = aclCreateTensor(inDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, inDims, 1, dIn);
    aclTensor* tPdf   = aclCreateTensor(pdfDims, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, pdfDims, 2, dPdf);
    aclTensor* tMant  = aclCreateTensor(mantDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, mantDims, 1, dMant);
    aclTensor* tFixed = aclCreateTensor(fixedDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, fixedDims, 1, dFixed);
    aclTensor* tVar   = aclCreateTensor(varDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, varDims, 1, dVar);
    aclTensor* tOut   = aclCreateTensor(inDims, 1, dt, nullptr, 0, ACL_FORMAT_ND, inDims, 1, dOut);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    OP(aclnnHansEncodeGetWorkspaceSize(tIn, tPdf, /*statistic=*/true, reshuff,
                                       tMant, tFixed, tVar, &wsSize, &exec));
    void* ws = nullptr;
    if (wsSize) CK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    OP(aclnnHansEncode(ws, wsSize, exec, stream));
    CK(aclrtSynchronizeStream(stream));

    int32_t hdr[128];
    CK(aclrtMemcpy(hdr, sizeof(hdr), dFixed, sizeof(hdr), ACL_MEMCPY_DEVICE_TO_HOST));
    uint32_t coreDim = (uint32_t)hdr[1];
    uint64_t fixedUsed = 0, varUsed = 0;
    if (coreDim == 0u || coreDim > 64u) {
        std::printf("[warn] suspicious coreDim=%u (header magic=%d)\n", coreDim, hdr[0]);
        coreDim = 64;
    }
    for (uint32_t c = 0; c < coreDim; c++) {
        if (hdr[8 + c]  > 0) fixedUsed += (uint32_t)hdr[8 + c];
        if (hdr[64 + c] > 0) varUsed   += (uint32_t)hdr[64 + c];
    }

    aclOpExecutor* execD = nullptr;
    uint64_t wsD = 0;
    OP(aclnnHansDecodeGetWorkspaceSize(tMant, tFixed, tVar, tPdf, reshuff, tOut, &wsD, &execD));
    void* wsDec = nullptr;
    if (wsD) CK(aclrtMalloc(&wsDec, wsD, ACL_MEM_MALLOC_HUGE_FIRST));
    OP(aclnnHansDecode(wsDec, wsD, execD, stream));
    CK(aclrtSynchronizeStream(stream));

    std::vector<uint8_t> hOut(inBytes);
    CK(aclrtMemcpy(hOut.data(), inBytes, dOut, inBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    int bad = 0, firstBad = -1;
    for (size_t i = 0; i < inBytes; i++) {
        if (hOut[i] != hIn[i]) { if (!bad) firstBad = (int)i; bad++; }
    }

    const uint64_t total = (uint64_t)mantBytes + fixedUsed + varUsed + 256u * 4u;
    std::printf("[hans] dtype=%s elems=%u in=%zu B | mantissa=%zu fixed_used=%llu var_used=%llu pdf=1024\n",
                (elemBytes == 4 ? "fp32" : (bf16 ? "bf16" : "fp16")), elems, inBytes,
                mantBytes, (unsigned long long)fixedUsed, (unsigned long long)varUsed);
    std::printf("[hans] total=%llu ratio=%.4f coreDim=%u reshuff=%d | roundtrip=%s (bad=%d first=%d)\n",
                (unsigned long long)total, (double)total / (double)inBytes, coreDim, (int)reshuff,
                bad ? "MISMATCH" : "bit-exact", bad, firstBad);

    if (ws)     aclrtFree(ws);
    if (wsDec)  aclrtFree(wsDec);
    aclrtFree(dIn); aclrtFree(dPdf); aclrtFree(dMant); aclrtFree(dFixed); aclrtFree(dVar); aclrtFree(dOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return bad ? 1 : 0;
}
