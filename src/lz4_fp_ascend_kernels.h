/*
 * lz4_fp_ascend_kernels.h
 *
 * FP16 / BF16 / FP32 variant of LZ4: byte-plane separation (S-planes) followed
 * by a standard LZ4 block.  Lossless: the transform is a pure byte permutation.
 *
 *   fp16/bf16 (2 B/elem):  plane0 = low bytes, plane1 = high bytes
 *   fp32      (4 B/elem):  plane0..3 = byte 0..3 of each element
 *
 * The exponent/sign plane is highly repetitive for real tensors while the
 * mantissa planes are close to noise, which is exactly the split HANS exploits.
 *
 * Why 32 KiB blocks: the transform needs a second UB buffer, so the FP path
 * uses FP_BLOCK = 32 KiB (input + transformed + output + hash table fits in
 * the 184 KiB usable UB).  The byte format of each block is still a standard
 * LZ4 block; the FP container (v2) records elemBytes/transform so a decoder
 * knows to invert the permutation.
 */
#ifndef LZ4_FP_ASCEND_KERNELS_H
#define LZ4_FP_ASCEND_KERNELS_H

#include "kernel_operator.h"
#include "lz4_ascend_kernels.h"

namespace lz4fp {

constexpr uint32_t FP_BLOCK      = 32u * 1024u;                  // original bytes / block
constexpr uint32_t FP_MAXCOMP    = FP_BLOCK + FP_BLOCK / 255u + 64u;
constexpr uint32_t FP_SLOT       = (FP_MAXCOMP + 31u) & ~31u;
constexpr uint32_t FP_MAX_LAUNCH = 32u;

__global__ __vector__ void lz4fp_compress_kernel(GM_ADDR in, GM_ADDR out, GM_ADDR outSizes,
                                                 uint32_t totalBytes, uint32_t numBlocks,
                                                 uint32_t elemBytes)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qIn;
    TQue<TPosition::VECOUT, 1> qOut, qSz;
    TBuf<TPosition::VECCALC>   htBuf, xfBuf;
    pipe.InitBuffer(qIn,  1, FP_BLOCK);
    pipe.InitBuffer(qOut, 1, FP_SLOT);
    pipe.InitBuffer(qSz,  1, 32);
    pipe.InitBuffer(htBuf, lz4a::HASH_SIZE * sizeof(int32_t));
    pipe.InitBuffer(xfBuf, FP_BLOCK);

    GlobalTensor<uint8_t> inG, outG;
    inG.SetGlobalBuffer((__gm__ uint8_t*)in, (uint32_t)((totalBytes + 31u) & ~31u));
    outG.SetGlobalBuffer((__gm__ uint8_t*)out, numBlocks * FP_SLOT);
    GlobalTensor<int32_t> szG;
    szG.SetGlobalBuffer((__gm__ int32_t*)outSizes, numBlocks * 8);

    LocalTensor<int32_t> ht = htBuf.Get<int32_t>();
    LocalTensor<uint8_t> xf = xfBuf.Get<uint8_t>();
    const uint32_t nb = GetBlockNum();

    for (uint32_t b = GetBlockIdx(); b < numBlocks; b += nb) {
        uint32_t start = b * FP_BLOCK;
        uint32_t len   = (start + FP_BLOCK <= totalBytes) ? FP_BLOCK : (totalBytes - start);
        if (len == 0u) continue;
        uint32_t lenR  = (len + 31u) & ~31u;

        LocalTensor<uint8_t> src = qIn.AllocTensor<uint8_t>();
        DataCopy(src, inG[start], lenR);
        qIn.EnQue(src);
        LocalTensor<uint8_t> s = qIn.DeQue<uint8_t>();

        const uint32_t es = (elemBytes == 4u) ? 4u : 2u;
        const uint32_t n  = len / es;
        for (uint32_t p = 0; p < es; p++) {
            uint32_t base = p * n;
            for (uint32_t i = 0; i < n; i++) xf.SetValue(base + i, s.GetValue(i * es + p));
        }

        LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
        uint32_t clen = lz4a::lz4CompressT<1u>(xf, len, dst, ht);
        qOut.EnQue(dst);
        qIn.FreeTensor(s);

        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        szl.SetValue(0, (int32_t)clen);
        qSz.EnQue(szl);

        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        DataCopy(szG[b * 8], szr, 8);
        qSz.FreeTensor(szr);

        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        DataCopy(outG[b * FP_SLOT], d, FP_SLOT);
        qOut.FreeTensor(d);
    }
}

__global__ __vector__ void lz4fp_decompress_kernel(GM_ADDR comp, GM_ADDR inSizes, GM_ADDR out,
                                                   uint32_t totalBytes, uint32_t numBlocks,
                                                   uint32_t elemBytes)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qIn, qSz;
    TQue<TPosition::VECOUT, 1> qOut;
    TBuf<TPosition::VECCALC>   xfBuf;
    pipe.InitBuffer(qIn,  1, FP_SLOT);
    pipe.InitBuffer(qSz,  1, 32);
    pipe.InitBuffer(qOut, 1, FP_BLOCK);
    pipe.InitBuffer(xfBuf, FP_BLOCK);

    GlobalTensor<uint8_t> compG, outG;
    compG.SetGlobalBuffer((__gm__ uint8_t*)comp, numBlocks * FP_SLOT);
    outG.SetGlobalBuffer((__gm__ uint8_t*)out, (uint32_t)((totalBytes + 31u) & ~31u));
    GlobalTensor<int32_t> szG;
    szG.SetGlobalBuffer((__gm__ int32_t*)inSizes, numBlocks * 8);

    LocalTensor<uint8_t> xf = xfBuf.Get<uint8_t>();
    const uint32_t nb = GetBlockNum();

    for (uint32_t b = GetBlockIdx(); b < numBlocks; b += nb) {
        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        DataCopy(szl, szG[b * 8], 8);
        qSz.EnQue(szl);
        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        uint32_t clen = (uint32_t)szr.GetValue(0);
        qSz.FreeTensor(szr);
        if (clen == 0u || clen > FP_SLOT) continue;

        uint32_t start = b * FP_BLOCK;
        uint32_t len   = (start + FP_BLOCK <= totalBytes) ? FP_BLOCK : (totalBytes - start);
        uint32_t clenR = (clen + 31u) & ~31u;

        LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
        DataCopy(compL, compG[b * FP_SLOT], clenR);
        qIn.EnQue(compL);
        LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

        bool ok = false;
        uint32_t op = lz4a::lz4Decompress(cl, clen, xf, FP_BLOCK, ok);
        qIn.FreeTensor(cl);
        if (!ok || op != len) continue;          /* host-side verify will fail */

        const uint32_t es = (elemBytes == 4u) ? 4u : 2u;
        const uint32_t n  = len / es;

        LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
        for (uint32_t p = 0; p < es; p++) {
            uint32_t base = p * n;
            for (uint32_t i = 0; i < n; i++) dst.SetValue(i * es + p, xf.GetValue(base + i));
        }
        qOut.EnQue(dst);

        uint32_t outR = (len + 31u) & ~31u;
        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        DataCopy(outG[start], d, outR);
        qOut.FreeTensor(d);
    }
}

} // namespace lz4fp

#endif // LZ4_FP_ASCEND_KERNELS_H
