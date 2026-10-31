#ifndef SNAPPY_ASCEND_KERNELS_H
#define SNAPPY_ASCEND_KERNELS_H

#include "kernel_operator.h"

using namespace AscendC;

namespace snappya {

constexpr uint32_t CHUNK      = 64 * 1024;
constexpr uint32_t HASH_BITS  = 12;
constexpr uint32_t HASH_SIZE  = 1u << HASH_BITS;
constexpr uint32_t MAXCOMP    = CHUNK + CHUNK / 6 + 64;   // >= snappy MaxCompressedLength()
constexpr uint32_t SLOT       = (MAXCOMP + 31) & ~31u;
constexpr uint32_t MAX_COPY_LENGTH   = 64;
constexpr uint32_t MAX_COPY_DISTANCE = 32768;
constexpr uint32_t MAX_LAUNCH = 32;

/* ---------------------------------------------------------------- device ---- */

__aicore__ inline uint32_t rd32(const LocalTensor<uint8_t>& t, uint32_t o)
{
    return (uint32_t)t.GetValue(o) | ((uint32_t)t.GetValue(o + 1) << 8) |
           ((uint32_t)t.GetValue(o + 2) << 16) | ((uint32_t)t.GetValue(o + 3) << 24);
}

// standard snappy 4-byte hash (kHashMul32 = 0x1e35a7bd), 12 bits
__aicore__ inline uint32_t snapHash(uint32_t v)
{
    return (uint32_t)((v * 0x1e35a7bdu) >> (32 - HASH_BITS));
}

__aicore__ inline void emitLiteral(const LocalTensor<uint8_t>& dst, uint32_t& op,
                                   const LocalTensor<uint8_t>& src, uint32_t srcOff, uint32_t l)
{
    uint32_t n = l - 1u;
    if (n < 60u) {
        dst.SetValue(op++, (uint8_t)(n << 2));
    } else if (n < 256u) {
        dst.SetValue(op++, (uint8_t)(60 << 2));
        dst.SetValue(op++, (uint8_t)n);
    } else if (n < 65536u) {
        dst.SetValue(op++, (uint8_t)(61 << 2));
        dst.SetValue(op++, (uint8_t)n);
        dst.SetValue(op++, (uint8_t)(n >> 8));
    } else if (n < (1u << 24)) {
        dst.SetValue(op++, (uint8_t)(62 << 2));
        dst.SetValue(op++, (uint8_t)n);
        dst.SetValue(op++, (uint8_t)(n >> 8));
        dst.SetValue(op++, (uint8_t)(n >> 16));
    } else {
        dst.SetValue(op++, (uint8_t)(63 << 2));
        dst.SetValue(op++, (uint8_t)n);
        dst.SetValue(op++, (uint8_t)(n >> 8));
        dst.SetValue(op++, (uint8_t)(n >> 16));
        dst.SetValue(op++, (uint8_t)(n >> 24));
    }
    for (uint32_t k = 0; k < l; k++) dst.SetValue(op + k, src.GetValue(srcOff + k));
    op += l;
}

// emit a copy element; long matches are split into <=64-byte copies at the same distance
__aicore__ inline void emitCopy(const LocalTensor<uint8_t>& dst, uint32_t& op,
                                uint32_t dist, uint32_t len)
{
    while (len > 0u) {
        if (len >= 4u && len <= 11u && dist < 2048u) {
            dst.SetValue(op++, (uint8_t)(((dist & 0x700u) >> 3) | ((len - 4u) << 2) | 0x01u));
            dst.SetValue(op++, (uint8_t)(dist & 0xffu));
            len = 0u;
        } else {
            uint32_t c = (len < MAX_COPY_LENGTH) ? len : MAX_COPY_LENGTH;
            dst.SetValue(op++, (uint8_t)(((c - 1u) << 2) | 0x02u));
            dst.SetValue(op++, (uint8_t)(dist & 0xffu));
            dst.SetValue(op++, (uint8_t)((dist >> 8) & 0xffu));
            len -= c;
        }
    }
}

__aicore__ inline uint32_t snapCompress(const LocalTensor<uint8_t>& src, uint32_t len,
                                        const LocalTensor<uint8_t>& dst,
                                        const LocalTensor<int32_t>& ht)
{
    uint32_t op = 0u;
    {
        uint32_t x = len;
        while (x > 0x7fu) { dst.SetValue(op++, (uint8_t)(x | 0x80u)); x >>= 7; }
        dst.SetValue(op++, (uint8_t)x);
    }

    Duplicate(ht, (int32_t)-1, (int32_t)HASH_SIZE);

    uint32_t nextEmit = 0u, ip = 0u;
    if (len >= 4u) {
        const uint32_t limit = len - 4u;
        while (ip <= limit) {
            uint32_t key = rd32(src, ip);
            uint32_t h   = snapHash(key);
            int32_t cand = ht.GetValue(h);
            ht.SetValue(h, (int32_t)ip);

            if (cand >= 0 && (ip - (uint32_t)cand) <= MAX_COPY_DISTANCE &&
                rd32(src, (uint32_t)cand) == key) {
                uint32_t base = (uint32_t)cand;
                if (ip > nextEmit) emitLiteral(dst, op, src, nextEmit, ip - nextEmit);
                uint32_t m = 4u;
                uint32_t maxm = len - ip;
                while (m < maxm && src.GetValue(base + m) == src.GetValue(ip + m)) m++;
                emitCopy(dst, op, ip - base, m);
                ip += m;
                nextEmit = ip;
            } else {
                ip++;
            }
        }
    }
    if (nextEmit < len) emitLiteral(dst, op, src, nextEmit, len - nextEmit);
    return op;
}

// nvcompBatchedSnappyGetDecompressSizeAsync equivalent (the varint preamble).
/* Bounded size-only walker: validates the whole element chain without writing
 * output.  Returns 0 and ok=false for any malformed / inconsistent stream. */
__aicore__ inline uint32_t snapSizeOnly(const LocalTensor<uint8_t>& comp, uint32_t clen, bool& ok)
{
    ok = false;
    if (clen == 0u) { ok = true; return 0u; }

    uint32_t ip = 0u;
    uint64_t ulen = 0u;
    {
        uint32_t shift = 0u;
        for (;;) {
            if (ip >= clen) return 0u;
            uint8_t b = comp.GetValue(ip++);
            ulen |= (uint64_t)(b & 0x7fu) << shift;
            shift += 7u;
            if ((b & 0x80u) == 0u) break;
            if (shift > 35u) return 0u;
        }
    }
    if (ulen > 0xFFFFFFFFull) return 0u;

    uint32_t op = 0u;
    while (ip < clen) {
        uint8_t tag = comp.GetValue(ip++);
        uint32_t t = (uint32_t)(tag & 3u);
        if (t == 0u) {
            uint32_t l = (uint32_t)(tag >> 2);
            if (l < 60u) {
                l += 1u;
            } else {
                uint32_t nb = l - 59u;
                if (clen - ip < nb) return 0u;
                l = 0u;
                for (uint32_t k = 0; k < nb; k++) l |= ((uint32_t)comp.GetValue(ip++) << (8u * k));
                l += 1u;
            }
            if (l > clen - ip) return 0u;
            ip += l; op += l;
        } else {
            uint32_t l, off;
            if (t == 1u) {
                if (clen - ip < 1u) return 0u;
                l   = (uint32_t)((tag >> 2) & 0x7u) + 4u;
                off = ((uint32_t)(tag & 0xe0u) << 3) | (uint32_t)comp.GetValue(ip);
                ip += 1u;
            } else if (t == 2u) {
                if (clen - ip < 2u) return 0u;
                l   = (uint32_t)(tag >> 2) + 1u;
                off = (uint32_t)comp.GetValue(ip) | ((uint32_t)comp.GetValue(ip + 1) << 8);
                ip += 2u;
            } else {
                if (clen - ip < 4u) return 0u;
                l   = (uint32_t)(tag >> 2) + 1u;
                off = (uint32_t)comp.GetValue(ip) | ((uint32_t)comp.GetValue(ip + 1) << 8) |
                      ((uint32_t)comp.GetValue(ip + 2) << 16) | ((uint32_t)comp.GetValue(ip + 3) << 24);
                ip += 4u;
            }
            if (off == 0u || off > op) return 0u;
            op += l;
        }
        if (op > ulen) return 0u;
    }
    if (op != ulen) return 0u;
    ok = true;
    return op;
}

/* Bounded Snappy block decoder.
 * Every read of `comp` is checked against `clen` and every write of `out`
 * against `outCap`; the declared length in the varint preamble must match the
 * number of bytes actually produced.  `ok` is set to false on any inconsistency,
 * so callers can report a hard error instead of trusting a bogus length. */
__aicore__ inline uint32_t snapDecompress(const LocalTensor<uint8_t>& comp, uint32_t clen,
                                          const LocalTensor<uint8_t>& out, uint32_t outCap,
                                          bool& ok)
{
    ok = false;
    if (clen == 0u) return 0u;                       /* "no data" item */

    uint32_t ip = 0u, op = 0u;
    uint64_t ulen = 0u;
    { // varint preamble: <= 5 bytes for a 32-bit length
        uint32_t shift = 0u;
        for (;;) {
            if (ip >= clen) return 0u;               /* truncated varint */
            uint8_t b = comp.GetValue(ip++);
            ulen |= (uint64_t)(b & 0x7fu) << shift;
            shift += 7u;
            if ((b & 0x80u) == 0u) break;
            if (shift > 35u) return 0u;              /* varint too long */
        }
    }
    if (ulen > (uint64_t)outCap) return 0u;           /* declared size > capacity */

    while (ip < clen) {
        uint8_t tag = comp.GetValue(ip++);
        uint32_t t = (uint32_t)(tag & 3u);
        if (t == 0u) {
            uint32_t l = (uint32_t)(tag >> 2);
            if (l < 60u) {
                l += 1u;
            } else {
                uint32_t nb = l - 59u;               /* 1..4 extra little-endian bytes */
                if (clen - ip < nb) return 0u;       /* truncated length */
                l = 0u;
                for (uint32_t k = 0; k < nb; k++) l |= ((uint32_t)comp.GetValue(ip++) << (8u * k));
                l += 1u;
            }
            if (l > clen - ip) return 0u;            /* literal overruns input */
            if (l > outCap - op) return 0u;          /* literal overruns output */
            for (uint32_t k = 0; k < l; k++) out.SetValue(op + k, comp.GetValue(ip + k));
            ip += l; op += l;
        } else {
            uint32_t l, off;
            if (t == 1u) {
                if (clen - ip < 1u) return 0u;
                l   = (uint32_t)((tag >> 2) & 0x7u) + 4u;
                off = ((uint32_t)(tag & 0xe0u) << 3) | (uint32_t)comp.GetValue(ip);
                ip += 1u;
            } else if (t == 2u) {
                if (clen - ip < 2u) return 0u;
                l   = (uint32_t)(tag >> 2) + 1u;
                off = (uint32_t)comp.GetValue(ip) | ((uint32_t)comp.GetValue(ip + 1) << 8);
                ip += 2u;
            } else {
                if (clen - ip < 4u) return 0u;
                l   = (uint32_t)(tag >> 2) + 1u;
                off = (uint32_t)comp.GetValue(ip) | ((uint32_t)comp.GetValue(ip + 1) << 8) |
                      ((uint32_t)comp.GetValue(ip + 2) << 16) | ((uint32_t)comp.GetValue(ip + 3) << 24);
                ip += 4u;
            }
            if (off == 0u || off > op) return 0u;     /* invalid back-reference */
            if (l > outCap - op) return 0u;           /* match overruns output */
            for (uint32_t k = 0; k < l; k++) out.SetValue(op + k, out.GetValue(op + k - off));
            op += l;
        }
        if (op > ulen) return 0u;                     /* produced > declared */
    }
    if (op != ulen) return 0u;                        /* declared != produced */
    ok = true;
    return op;
}

/* ---------------------------------------------------------------- kernels --- */

__global__ __vector__ void snappy_compress_kernel(GM_ADDR in, GM_ADDR out, GM_ADDR outSizes,
                                                  uint32_t totalBytes, uint32_t numChunks)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qIn;
    TQue<TPosition::VECOUT, 1> qOut, qSz;
    TBuf<TPosition::VECCALC>   htBuf;
    pipe.InitBuffer(qIn,  1, CHUNK);
    pipe.InitBuffer(qOut, 1, SLOT);
    pipe.InitBuffer(qSz,  1, 32);
    pipe.InitBuffer(htBuf, HASH_SIZE * sizeof(int32_t));

    GlobalTensor<uint8_t> inG, outG;
    inG.SetGlobalBuffer((__gm__ uint8_t*)in, (uint32_t)((totalBytes + 31u) & ~31u));
    outG.SetGlobalBuffer((__gm__ uint8_t*)out, numChunks * SLOT);

    LocalTensor<int32_t> ht = htBuf.Get<int32_t>();

    const uint32_t nb = GetBlockNum();
    for (uint32_t c = GetBlockIdx(); c < numChunks; c += nb) {
        uint32_t start = c * CHUNK;
        uint32_t len   = (start + CHUNK <= totalBytes) ? CHUNK : (totalBytes - start);
        uint32_t lenR  = (len + 31u) & ~31u;

        LocalTensor<uint8_t> src = qIn.AllocTensor<uint8_t>();
        if (lenR) DataCopy(src, inG[start], lenR);
        qIn.EnQue(src);
        LocalTensor<uint8_t> s = qIn.DeQue<uint8_t>();

        LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
        uint32_t clen = snapCompress(s, len, dst, ht);
        qOut.EnQue(dst);

        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        szl.SetValue(0, (int32_t)clen);
        qSz.EnQue(szl);

        qIn.FreeTensor(s);

        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        GlobalTensor<int32_t> szG;
        szG.SetGlobalBuffer((__gm__ int32_t*)outSizes, numChunks * 8);
        DataCopy(szG[c * 8], szr, 8);
        qSz.FreeTensor(szr);

        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        if (clen > SLOT) {                      /* cannot happen unless format broke */
            LocalTensor<int32_t> e = qSz.AllocTensor<int32_t>();
            e.SetValue(0, -1);
            qSz.EnQue(e);
            LocalTensor<int32_t> er = qSz.DeQue<int32_t>();
            GlobalTensor<int32_t> eg;
            eg.SetGlobalBuffer((__gm__ int32_t*)outSizes, numChunks * 8);
            DataCopy(eg[c * 8], er, 8);
            qSz.FreeTensor(er);
            qOut.FreeTensor(d);
            continue;
        }
        DataCopy(outG[c * SLOT], d, SLOT);
        qOut.FreeTensor(d);
    }
}

__global__ __vector__ void snappy_decompress_kernel(GM_ADDR comp, GM_ADDR out,
                                                    GM_ADDR inSizes, GM_ADDR outProduced,
                                                    uint32_t numChunks)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qIn, qSz, qPs;
    TQue<TPosition::VECOUT, 1> qOut;
    pipe.InitBuffer(qIn,  1, SLOT);
    pipe.InitBuffer(qSz,  1, 32);
    pipe.InitBuffer(qPs,  1, 32);
    pipe.InitBuffer(qOut, 1, CHUNK);

    GlobalTensor<uint8_t> compG, outG;
    compG.SetGlobalBuffer((__gm__ uint8_t*)comp, numChunks * SLOT);
    outG.SetGlobalBuffer((__gm__ uint8_t*)out, numChunks * CHUNK);

    GlobalTensor<int32_t> szG, psG;
    szG.SetGlobalBuffer((__gm__ int32_t*)inSizes, numChunks * 8);
    psG.SetGlobalBuffer((__gm__ int32_t*)outProduced, numChunks * 8);

    const uint32_t nb = GetBlockNum();
    for (uint32_t c = GetBlockIdx(); c < numChunks; c += nb) {
        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        DataCopy(szl, szG[c * 8], 8);
        qSz.EnQue(szl);
        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        uint32_t clen = (uint32_t)szr.GetValue(0);
        qSz.FreeTensor(szr);
        if (clen == 0u) continue;

        if (clen > SLOT) {                       /* corrupt size table */
            LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
            psl.SetValue(0, -1);
            qPs.EnQue(psl);
            LocalTensor<int32_t> psr = qPs.DeQue<int32_t>();
            DataCopy(psG[c * 8], psr, 8);
            qPs.FreeTensor(psr);
            continue;
        }
        uint32_t clenR = (clen + 31u) & ~31u;

        LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
        DataCopy(compL, compG[c * SLOT], clenR);
        qIn.EnQue(compL);
        LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

        LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
        bool decOk = false;
        uint32_t op = snapDecompress(cl, clen, dst, CHUNK, decOk);
        qOut.EnQue(dst);

        LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
        psl.SetValue(0, decOk ? (int32_t)op : -1);
        qPs.EnQue(psl);

        qIn.FreeTensor(cl);

        LocalTensor<int32_t> psr = qPs.DeQue<int32_t>();
        DataCopy(psG[c * 8], psr, 8);
        qPs.FreeTensor(psr);

        uint32_t opR = (op + 31u) & ~31u;
        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        DataCopy(outG[c * CHUNK], d, opR);
        qOut.FreeTensor(d);
    }
}

__global__ __vector__ void snappy_size_kernel(GM_ADDR comp, GM_ADDR inSizes,
                                              GM_ADDR outSizes, uint32_t numChunks)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qIn, qSz;
    TQue<TPosition::VECOUT, 1> qPs;
    pipe.InitBuffer(qIn, 1, SLOT);
    pipe.InitBuffer(qSz, 1, 32);
    pipe.InitBuffer(qPs, 1, 32);

    GlobalTensor<uint8_t> compG;
    compG.SetGlobalBuffer((__gm__ uint8_t*)comp, numChunks * SLOT);
    GlobalTensor<int32_t> szG, psG;
    szG.SetGlobalBuffer((__gm__ int32_t*)inSizes, numChunks * 8);
    psG.SetGlobalBuffer((__gm__ int32_t*)outSizes, numChunks * 8);

    const uint32_t nb = GetBlockNum();
    for (uint32_t c = GetBlockIdx(); c < numChunks; c += nb) {
        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        DataCopy(szl, szG[c * 8], 8);
        qSz.EnQue(szl);
        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        uint32_t clen = (uint32_t)szr.GetValue(0);
        qSz.FreeTensor(szr);
        if (clen == 0u) continue;

        if (clen > SLOT) {
            LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
            psl.SetValue(0, -1);
            qPs.EnQue(psl);
            LocalTensor<int32_t> psr = qPs.DeQue<int32_t>();
            DataCopy(psG[c * 8], psr, 8);
            qPs.FreeTensor(psr);
            continue;
        }
        uint32_t clenR = (clen + 31u) & ~31u;

        LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
        DataCopy(compL, compG[c * SLOT], clenR);
        qIn.EnQue(compL);
        LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

        bool szOk = false;
        uint32_t op = snapSizeOnly(cl, clen, szOk);
        qIn.FreeTensor(cl);

        LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
        psl.SetValue(0, szOk ? (int32_t)op : -1);
        qPs.EnQue(psl);
        LocalTensor<int32_t> psr = qPs.DeQue<int32_t>();
        DataCopy(psG[c * 8], psr, 8);
        qPs.FreeTensor(psr);
    }
}

/* ============================ batch API =================================== */

constexpr uint32_t MAXB = 256;
constexpr uint32_t MAX_ITEM_BYTES = 16u * 1024u * 1024u;                 // per-item cap
constexpr uint32_t MAX_BLOCKS = (MAX_ITEM_BYTES + CHUNK - 1u) / CHUNK;   // 64 KiB blocks per item

__global__ __vector__ void snappy_batch_compress_kernel(
    GM_ADDR uncompPtrs, GM_ADDR uncompBytes,
    GM_ADDR compPtrs,   GM_ADDR compBytes,
    GM_ADDR blockSizes, uint32_t batchSize)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qpIn, qpOut, qszIn, qIn;
    TQue<TPosition::VECOUT, 1> qOut, qszOut, qBs;
    TBuf<TPosition::VECCALC>   htBuf;
    pipe.InitBuffer(qpIn,   1, MAXB * 8);
    pipe.InitBuffer(qpOut,  1, MAXB * 8);
    pipe.InitBuffer(qszIn,  1, MAXB * 4);
    pipe.InitBuffer(qIn,    1, CHUNK);
    pipe.InitBuffer(qOut,   1, SLOT);
    pipe.InitBuffer(qszOut, 1, 32);
    pipe.InitBuffer(qBs,    1, MAX_BLOCKS * sizeof(int32_t));
    pipe.InitBuffer(htBuf, HASH_SIZE * sizeof(int32_t));

    GlobalTensor<uint64_t> ipG, opG;
    ipG.SetGlobalBuffer((__gm__ uint64_t*)uncompPtrs, batchSize);
    opG.SetGlobalBuffer((__gm__ uint64_t*)compPtrs, batchSize);
    GlobalTensor<uint32_t> iszG;
    iszG.SetGlobalBuffer((__gm__ uint32_t*)uncompBytes, batchSize);
    GlobalTensor<int32_t> oszG, bsG;
    oszG.SetGlobalBuffer((__gm__ int32_t*)compBytes, batchSize * 8);
    bsG.SetGlobalBuffer((__gm__ int32_t*)blockSizes, batchSize * MAX_BLOCKS);

    const uint32_t ptrCount = ((batchSize + 3u) / 4u) * 4u;
    const uint32_t szCount  = ((batchSize + 7u) / 8u) * 8u;

    LocalTensor<uint64_t> ipL = qpIn.AllocTensor<uint64_t>();
    DataCopy(ipL, ipG, ptrCount);
    qpIn.EnQue(ipL);
    LocalTensor<uint64_t> ip = qpIn.DeQue<uint64_t>();

    LocalTensor<uint64_t> opL = qpOut.AllocTensor<uint64_t>();
    DataCopy(opL, opG, ptrCount);
    qpOut.EnQue(opL);
    LocalTensor<uint64_t> op = qpOut.DeQue<uint64_t>();

    LocalTensor<uint32_t> iszL = qszIn.AllocTensor<uint32_t>();
    DataCopy(iszL, iszG, szCount);
    qszIn.EnQue(iszL);
    LocalTensor<uint32_t> isz = qszIn.DeQue<uint32_t>();

    LocalTensor<int32_t> ht = htBuf.Get<int32_t>();
    const uint32_t nb = GetBlockNum();

    for (uint32_t i = GetBlockIdx(); i < batchSize; i += nb) {
        uint32_t ilen = isz.GetValue(i);
        uint32_t blocks = (ilen + CHUNK - 1u) / CHUNK;
        uint64_t inAddr  = ip.GetValue(i);
        uint64_t outAddr = op.GetValue(i);

        LocalTensor<int32_t> bsl = qBs.AllocTensor<int32_t>();
        uint32_t compOff = 0u, inOff = 0u, compTotal = 0u;
        int compErr = 0;

        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t l = ilen - inOff;
            if (l > CHUNK) l = CHUNK;
            uint32_t lR = (l + 31u) & ~31u;

            GlobalTensor<uint8_t> ig;
            ig.SetGlobalBuffer((__gm__ uint8_t*)(inAddr + inOff), lR);

            LocalTensor<uint8_t> src = qIn.AllocTensor<uint8_t>();
            DataCopy(src, ig, lR);
            qIn.EnQue(src);
            LocalTensor<uint8_t> s = qIn.DeQue<uint8_t>();

            LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
            uint32_t clen = snapCompress(s, l, dst, ht);
            qOut.EnQue(dst);
            qIn.FreeTensor(s);

            uint32_t clenW = (clen + 31u) & ~31u;

            LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
            if (clen > SLOT) {                        /* output cannot fit: hard error */
                qOut.FreeTensor(d);
                compErr = 1;
                break;
            }
            if (clenW) {
                GlobalTensor<uint8_t> og;
                og.SetGlobalBuffer((__gm__ uint8_t*)(outAddr + compOff), clenW);
                DataCopy(og, d, clenW);
            }
            qOut.FreeTensor(d);

            bsl.SetValue(b, (int32_t)clen);
            compOff   += clenW;
            compTotal += clen;
            inOff     += l;
        }
        if (blocks == 0u) bsl.SetValue(0, 0);
        uint32_t nwrite = blocks ? blocks : 1u;
        uint32_t bCnt = ((nwrite + 7u) / 8u) * 8u;   /* 8 int32 per 32B */
        for (uint32_t b = nwrite; b < bCnt; b++) bsl.SetValue(b, 0);
        qBs.EnQue(bsl);

        LocalTensor<int32_t> szl = qszOut.AllocTensor<int32_t>();
        szl.SetValue(0, compErr ? -1 : (int32_t)compTotal);
        qszOut.EnQue(szl);

        LocalTensor<int32_t> bsr = qBs.DeQue<int32_t>();
        DataCopy(bsG[i * MAX_BLOCKS], bsr, bCnt);
        qBs.FreeTensor(bsr);

        LocalTensor<int32_t> szr = qszOut.DeQue<int32_t>();
        DataCopy(oszG[i * 8], szr, 8);
        qszOut.FreeTensor(szr);
    }
    qpIn.FreeTensor(ip);
    qpOut.FreeTensor(op);
    qszIn.FreeTensor(isz);
}

__global__ __vector__ void snappy_batch_decompress_kernel(
    GM_ADDR compPtrs, GM_ADDR compBytes,
    GM_ADDR outPtrs,  GM_ADDR outCapBytes, GM_ADDR outActualBytes,
    GM_ADDR blockSizes, uint32_t batchSize)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qcp, qop, qszIn, qcapIn, qbsIn, qIn;
    TQue<TPosition::VECOUT, 1> qOut, qpsOut;
    pipe.InitBuffer(qcp,    1, MAXB * 8);
    pipe.InitBuffer(qop,    1, MAXB * 8);
    pipe.InitBuffer(qszIn,  1, MAXB * 32);
    pipe.InitBuffer(qcapIn, 1, MAXB * 4);
    pipe.InitBuffer(qbsIn,  1, MAX_BLOCKS * sizeof(int32_t));
    pipe.InitBuffer(qIn,    1, SLOT);
    pipe.InitBuffer(qOut,   1, CHUNK);
    pipe.InitBuffer(qpsOut, 1, 32);

    GlobalTensor<uint64_t> cpG, opG;
    cpG.SetGlobalBuffer((__gm__ uint64_t*)compPtrs, batchSize);
    opG.SetGlobalBuffer((__gm__ uint64_t*)outPtrs, batchSize);
    GlobalTensor<int32_t>  cszG,  aszG, bsG;
    cszG.SetGlobalBuffer((__gm__ int32_t*)compBytes, batchSize * 8);
    aszG.SetGlobalBuffer((__gm__ int32_t*)outActualBytes, batchSize * 8);
    bsG.SetGlobalBuffer((__gm__ int32_t*)blockSizes, batchSize * MAX_BLOCKS);
    GlobalTensor<uint32_t> capG;
    capG.SetGlobalBuffer((__gm__ uint32_t*)outCapBytes, batchSize);

    const uint32_t ptrCount = ((batchSize + 3u) / 4u) * 4u;
    const uint32_t szCount  = batchSize * 8;
    const uint32_t capCount = ((batchSize + 7u) / 8u) * 8u;

    LocalTensor<uint64_t> cpL = qcp.AllocTensor<uint64_t>();
    DataCopy(cpL, cpG, ptrCount);
    qcp.EnQue(cpL);
    LocalTensor<uint64_t> cp = qcp.DeQue<uint64_t>();

    LocalTensor<uint64_t> opL = qop.AllocTensor<uint64_t>();
    DataCopy(opL, opG, ptrCount);
    qop.EnQue(opL);
    LocalTensor<uint64_t> op = qop.DeQue<uint64_t>();

    LocalTensor<int32_t> cszL = qszIn.AllocTensor<int32_t>();
    DataCopy(cszL, cszG, szCount);
    qszIn.EnQue(cszL);
    LocalTensor<int32_t> csz = qszIn.DeQue<int32_t>();

    LocalTensor<uint32_t> capL = qcapIn.AllocTensor<uint32_t>();
    DataCopy(capL, capG, capCount);
    qcapIn.EnQue(capL);
    LocalTensor<uint32_t> cap = qcapIn.DeQue<uint32_t>();

    const uint32_t nb = GetBlockNum();
    for (uint32_t i = GetBlockIdx(); i < batchSize; i += nb) {
        uint32_t cbytes = (uint32_t)csz.GetValue(i * 8);
        uint32_t capacity = cap.GetValue(i);
        LocalTensor<int32_t> psl = qpsOut.AllocTensor<int32_t>();

        uint32_t blocks = (capacity + CHUNK - 1u) / CHUNK;
        if (cbytes == 0u || blocks == 0u) {
            psl.SetValue(0, 0);
            qpsOut.EnQue(psl);
            LocalTensor<int32_t> psr = qpsOut.DeQue<int32_t>();
            DataCopy(aszG[i * 8], psr, 8);
            qpsOut.FreeTensor(psr);
            continue;
        }

        uint64_t cAddr = cp.GetValue(i);
        uint64_t oAddr = op.GetValue(i);

        uint32_t bCnt = ((blocks * 4u + 31u) / 32u) * 8u;
        LocalTensor<int32_t> bsl = qbsIn.AllocTensor<int32_t>();
        DataCopy(bsl, bsG[i * MAX_BLOCKS], bCnt);
        qbsIn.EnQue(bsl);
        LocalTensor<int32_t> bs = qbsIn.DeQue<int32_t>();

        uint32_t compOff = 0u, outOff = 0u, produced = 0u;
        int decErr = 0;
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t clen = (uint32_t)bs.GetValue(b);
            if (clen == 0u) break;
            if (clen > SLOT) { decErr = 1; break; }   /* corrupt block table */
            uint32_t clenR = (clen + 31u) & ~31u;

            GlobalTensor<uint8_t> cg;
            cg.SetGlobalBuffer((__gm__ uint8_t*)(cAddr + compOff), clenR);

            LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
            DataCopy(compL, cg, clenR);
            qIn.EnQue(compL);
            LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

            uint32_t room = (capacity > outOff) ? (capacity - outOff) : 0u;
            uint32_t blockCap = (room > CHUNK) ? CHUNK : room;
            LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
            bool decOk = false;
            uint32_t opn = snapDecompress(cl, clen, dst, blockCap, decOk);
            qOut.EnQue(dst);
            qIn.FreeTensor(cl);
            if (!decOk) { decErr = 1; break; }

            uint32_t opnR = (opn + 31u) & ~31u;
            LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
            if (opnR) {
                GlobalTensor<uint8_t> og;
                og.SetGlobalBuffer((__gm__ uint8_t*)(oAddr + outOff), opnR);
                DataCopy(og, d, opnR);
            }
            qOut.FreeTensor(d);

            compOff  += clenR;
            outOff   += opn;
            produced += opn;
        }
        qbsIn.FreeTensor(bs);

        psl.SetValue(0, decErr ? -1 : (int32_t)produced);
        qpsOut.EnQue(psl);
        LocalTensor<int32_t> psr = qpsOut.DeQue<int32_t>();
        DataCopy(aszG[i * 8], psr, 8);
        qpsOut.FreeTensor(psr);
    }
    qcp.FreeTensor(cp);
    qop.FreeTensor(op);
    qszIn.FreeTensor(csz);
    qcapIn.FreeTensor(cap);
}


// nvcompBatchedSnappyGetDecompressSizeAsync equivalent for the pointer-array API.
__global__ __vector__ void snappy_batch_size_kernel(
    GM_ADDR compPtrs, GM_ADDR compBytes, GM_ADDR blockSizes,
    GM_ADDR outSizes, uint32_t batchSize)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qcp, qszIn, qbsIn, qIn;
    TQue<TPosition::VECOUT, 1> qpsOut;
    pipe.InitBuffer(qcp,    1, MAXB * 8);
    pipe.InitBuffer(qszIn,  1, MAXB * 32);
    pipe.InitBuffer(qbsIn,  1, MAX_BLOCKS * sizeof(int32_t));
    pipe.InitBuffer(qIn,    1, SLOT);
    pipe.InitBuffer(qpsOut, 1, 32);

    GlobalTensor<uint64_t> cpG;
    cpG.SetGlobalBuffer((__gm__ uint64_t*)compPtrs, batchSize);
    GlobalTensor<int32_t> cszG, aszG, bsG;
    cszG.SetGlobalBuffer((__gm__ int32_t*)compBytes, batchSize * 8);
    aszG.SetGlobalBuffer((__gm__ int32_t*)outSizes, batchSize * 8);
    bsG.SetGlobalBuffer((__gm__ int32_t*)blockSizes, batchSize * MAX_BLOCKS);

    const uint32_t ptrCount = ((batchSize + 3u) / 4u) * 4u;
    const uint32_t szCount  = batchSize * 8;

    LocalTensor<uint64_t> cpL = qcp.AllocTensor<uint64_t>();
    DataCopy(cpL, cpG, ptrCount);
    qcp.EnQue(cpL);
    LocalTensor<uint64_t> cp = qcp.DeQue<uint64_t>();

    LocalTensor<int32_t> cszL = qszIn.AllocTensor<int32_t>();
    DataCopy(cszL, cszG, szCount);
    qszIn.EnQue(cszL);
    LocalTensor<int32_t> csz = qszIn.DeQue<int32_t>();

    const uint32_t nb = GetBlockNum();
    for (uint32_t i = GetBlockIdx(); i < batchSize; i += nb) {
        uint32_t cbytes = (uint32_t)csz.GetValue(i * 8);
        LocalTensor<int32_t> psl = qpsOut.AllocTensor<int32_t>();
        if (cbytes == 0u) {
            psl.SetValue(0, 0);
            qpsOut.EnQue(psl);
            LocalTensor<int32_t> psr = qpsOut.DeQue<int32_t>();
            DataCopy(aszG[i * 8], psr, 8);
            qpsOut.FreeTensor(psr);
            continue;
        }

        LocalTensor<int32_t> bsl = qbsIn.AllocTensor<int32_t>();
        DataCopy(bsl, bsG[i * MAX_BLOCKS], MAX_BLOCKS);
        qbsIn.EnQue(bsl);
        LocalTensor<int32_t> bs = qbsIn.DeQue<int32_t>();

        uint64_t cAddr = cp.GetValue(i);
        uint32_t compOff = 0u, produced = 0u;
        int szErr = 0;
        for (uint32_t b = 0; b < MAX_BLOCKS; b++) {
            uint32_t clen = (uint32_t)bs.GetValue(b);
            if (clen == 0u) break;
            if (clen > SLOT) { szErr = 1; break; }
            uint32_t clenR = (clen + 31u) & ~31u;

            GlobalTensor<uint8_t> cg;
            cg.SetGlobalBuffer((__gm__ uint8_t*)(cAddr + compOff), clenR);

            LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
            DataCopy(compL, cg, clenR);
            qIn.EnQue(compL);
            LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

            bool blkOk = false;
            uint32_t blkSz = snapSizeOnly(cl, clen, blkOk);
            qIn.FreeTensor(cl);
            if (!blkOk) { szErr = 1; break; }
            produced += blkSz;
            compOff += clenR;
        }
        qbsIn.FreeTensor(bs);

        psl.SetValue(0, szErr ? -1 : (int32_t)produced);
        qpsOut.EnQue(psl);
        LocalTensor<int32_t> psr = qpsOut.DeQue<int32_t>();
        DataCopy(aszG[i * 8], psr, 8);
        qpsOut.FreeTensor(psr);
    }
    qcp.FreeTensor(cp);
    qszIn.FreeTensor(csz);
}

} // namespace snappya

#endif // SNAPPY_ASCEND_KERNELS_H
