#ifndef LZ4_ASCEND_KERNELS_H
#define LZ4_ASCEND_KERNELS_H

#include "kernel_operator.h"

using namespace AscendC;

namespace lz4a {
constexpr uint32_t CHUNK      = 64 * 1024;
//哈希表一共有4096(2^12)个位置，4096*4=16384字节=16KiB
constexpr uint32_t HASH_BITS  = 12;
constexpr uint32_t HASH_SIZE  = 1u << HASH_BITS;
//为压缩结果预留的最大容量
constexpr uint32_t MAXCOMP    = CHUNK + CHUNK / 6 + 64;
//按照 32 字节对齐后，每个压缩块占用的固定存储空间
constexpr uint32_t SLOT       = (MAXCOMP + 31) & ~31u;
//控制 Host 一次最多启动 32 个用于处理数据块的执行 Block
constexpr uint32_t MAX_LAUNCH = 32;

/* ---------------------------------------------------------------- device ---- */

__aicore__ inline uint32_t brev32(uint32_t v)
{
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
    v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
    v = (v >> 16) | (v << 16);
    return v;
}

__aicore__ inline uint32_t lz4hash(uint32_t key)
{
    return (brev32(key) + (key ^ 0xc375u)) & (HASH_SIZE - 1u);
}

__aicore__ inline uint32_t rd32(const LocalTensor<uint8_t>& t, uint32_t o)
{
    return (uint32_t)t.GetValue(o) | ((uint32_t)t.GetValue(o + 1) << 8) |
           ((uint32_t)t.GetValue(o + 2) << 16) | ((uint32_t)t.GetValue(o + 3) << 24);
}

/* dst 压缩结果写入的位置
op 当前压缩输出的写入下标
src 原始输入数据
litStart Literal 起始位置
litLen Literal 长度
matchOff Match 距离
matchLen Match 长度 */
__aicore__ inline void emitSeq(const LocalTensor<uint8_t>& dst, uint32_t& op,
                               const LocalTensor<uint8_t>& src,
                               uint32_t litStart, uint32_t litLen,
                               uint32_t matchOff, uint32_t matchLen)
{
    //Literal 长度和 Match长度都被限制在15以内，如果超过 15，则在后续字节中使用额外的字节来表示实际长度。
    uint32_t tokLit = (litLen < 15u) ? litLen : 15u;
    uint32_t tokMat = 0u;
    if (matchLen > 0u) {
        // Token中的Match值=MatchLen−4
        uint32_t m = matchLen - 4u;
        tokMat = (m < 15u) ? m : 15u;
    }
    // 生成token
    dst.SetValue(op++, (uint8_t)((tokLit << 4) | tokMat));
    // 处理额外长度的litLen
    if (litLen >= 15u) {
        uint32_t r = litLen - 15u;
        while (r >= 255u) { dst.SetValue(op++, (uint8_t)255); r -= 255u; }
        dst.SetValue(op++, (uint8_t)r);
    }
    //将Literal数据从src复制到dst，循环逐字节复制
    for (uint32_t k = 0; k < litLen; k++) dst.SetValue(op + k, src.GetValue(litStart + k));
    op += litLen;
    // 写入Offset(两个字节)
    if (matchLen > 0u) {
        dst.SetValue(op++, (uint8_t)(matchOff & 0xffu));//低8位
        dst.SetValue(op++, (uint8_t)((matchOff >> 8) & 0xffu));//高8位
        // 处理额外长度的matchlen
        if (matchLen - 4u >= 15u) {
            uint32_t r = (matchLen - 4u) - 15u;
            while (r >= 255u) { dst.SetValue(op++, (uint8_t)255); r -= 255u; }
            dst.SetValue(op++, (uint8_t)r);
        }
    }
}

// Compare TS bytes; TS == sizeof(input element) == {1,2,4}.
template <uint32_t TS>
__aicore__ inline bool eqRange(const LocalTensor<uint8_t>& t, uint32_t a, uint32_t b)
{
    for (uint32_t k = 0; k < TS; k++) {
        if (t.GetValue(a + k) != t.GetValue(b + k)) return false;
    }
    return true;
}

// Typed greedy LZ4 block compressor.  Match candidates are restricted to TS-byte
// aligned positions (nvCOMP NVCOMP_TYPE_CHAR/SHORT/INT semantics); matches are
// extended in TS-byte units.  The emitted stream is still a standard byte-level
// LZ4 block (match lengths / offsets are byte counts).
template <uint32_t TS>
__aicore__ inline uint32_t lz4CompressT(const LocalTensor<uint8_t>& src, uint32_t len,
                                        const LocalTensor<uint8_t>& dst,
                                        const LocalTensor<int32_t>& ht)
{
    //初始化哈希表，-1
    if (len == 0u) {
        /* standard LZ4 empty block: a single zero token */
        dst.SetValue(0u, (uint8_t)0u);
        return 1u;
    }

    Duplicate(ht, (int32_t)-1, (int32_t)HASH_SIZE);
    // pos：当前扫描到哪里 anchor：尚未编码的数据从哪里开始 op：压缩结果写到哪里
    uint32_t op = 0u, anchor = 0u, pos = 0u;
    if (len > 12u) {
        const uint32_t limit = len - 12u;              // byte rule: last match start <= len-12
        while (pos <= limit) {
            // 从指定位置连续读取 4 个字节，把它们组合成一个 32 位整数。
            uint32_t key = rd32(src, pos);
            // 计算哈希值，使用 lz4hash 函数对 key 进行哈希计算，得到一个哈希值 h。
            uint32_t h   = lz4hash(key);
            // 读取哈希表中对应位置的值，作为候选匹配位置 cand。
            int32_t cand = ht.GetValue(h);
            // 记录当前位置，h->pos
            ht.SetValue(h, (int32_t)pos);
            // 检查匹配条件：候选位置有效、距离不超过 65535 字节，并且在候选位置和当前扫描位置的 4 个字节相等。
            if (cand >= 0 && (pos - (uint32_t)cand) <= 65535u && rd32(src, (uint32_t)cand) == key) {//？？？
                uint32_t c    = (uint32_t)cand;
                // 计算最大匹配长度 maxm，确保至少保留 5 个字节的字面量数据（byte rule）。
                uint32_t maxm = len - 5u - pos;        // keep >= 5 literal bytes (byte rule)
                //m已经匹配的字节数，初始为4
                uint32_t m    = 4u;                    // 4 % TS == 0 for TS in {1,2,4}
                //eqRange<TS>() 虽然一次检查 TS 个字节，但实现中仍然是通过 GetValue() 循环逐字节比较，不是直接调用某个专用的向量比较指令
                while (m + TS <= maxm && eqRange<TS>(src, c + m, pos + m)) m += TS;
                emitSeq(dst, op, src, anchor, pos - anchor, pos - c, m);
                pos += m;                              // m is a multiple of TS
                anchor = pos;
            } else {
                pos += TS;
            }
        }
    }
    if (anchor < len) emitSeq(dst, op, src, anchor, len - anchor, 0u, 0u);
    return op;
}

// nvcompBatchedLZ4GetDecompressSizeAsync equivalent: parse the stream without
// producing output and report the decompressed size.
__aicore__ inline uint32_t lz4SizeOnly(const LocalTensor<uint8_t>& comp, uint32_t clen, bool& ok)
{
    uint32_t ip = 0u, op = 0u;
    ok = true;
    while (ip < clen) {
        uint8_t tok = comp.GetValue(ip++);
        uint32_t lit = (uint32_t)(tok >> 4);
        if (lit == 15u) {
            uint8_t b;
            do {
                if (ip >= clen) { ok = false; return 0u; }
                b = comp.GetValue(ip++);
                if (lit > 0xFFFFFFFFu - b) { ok = false; return 0u; }
                lit += b;
            } while (b == 255u);
        }
        if (lit > clen - ip) { ok = false; return 0u; }      /* input bound */
        ip += lit;
        if (op > 0xFFFFFFFFu - lit) { ok = false; return 0u; }
        op += lit;
        if (ip >= clen) break;
        if (clen - ip < 2u) { ok = false; return 0u; }       /* offset */
        uint32_t off = (uint32_t)comp.GetValue(ip) | ((uint32_t)comp.GetValue(ip + 1) << 8);
        ip += 2u;
        if (off == 0u || off > op) { ok = false; return 0u; }
        uint32_t m = (uint32_t)(tok & 15);
        if (m == 15u) {
            uint8_t b;
            do {
                if (ip >= clen) { ok = false; return 0u; }
                b = comp.GetValue(ip++);
                if (m > 0xFFFFFFFFu - b) { ok = false; return 0u; }
                m += b;
            } while (b == 255u);
        }
        if (m > 0xFFFFFFFFu - 4u) { ok = false; return 0u; }
        m += 4u;
        if (op > 0xFFFFFFFFu - m) { ok = false; return 0u; }
        op += m;
    }
    return op;
}

/* 读取 Token（标记字节）
复制 Literal（原始字节）
读取 Offset 和 Match Length
从已解压数据中复制 Match */
__aicore__ inline uint32_t lz4Decompress(const LocalTensor<uint8_t>& comp, uint32_t clen,
                                         const LocalTensor<uint8_t>& out, uint32_t outCap, bool& ok)
{
    uint32_t ip = 0u, op = 0u;
    ok = true;
    while (ip < clen) {
        uint8_t tok = comp.GetValue(ip++);
        uint32_t litLen = (uint32_t)(tok >> 4);
        if (litLen == 15u) {
            uint8_t b;
            do {
                if (ip >= clen) { ok = false; return op; }             /* truncated */
                b = comp.GetValue(ip++);
                if (litLen > 0xFFFFFFFFu - b) { ok = false; return op; }
                litLen += b;
            } while (b == 255u);
        }
        if (litLen > clen - ip) { ok = false; return op; }             /* input bound */
        if (litLen > outCap - op) { ok = false; return op; }           /* output bound */
        for (uint32_t k = 0; k < litLen; k++) out.SetValue(op + k, comp.GetValue(ip + k));
        ip += litLen;
        op += litLen;

        if (ip >= clen) break;                                         /* last sequence */

        if (clen - ip < 2u) { ok = false; return op; }                 /* offset truncated */
        uint32_t off = (uint32_t)comp.GetValue(ip) | ((uint32_t)comp.GetValue(ip + 1) << 8);
        ip += 2u;
        if (off == 0u || off > op) { ok = false; return op; }          /* invalid offset */

        uint32_t mlen = (uint32_t)(tok & 15);
        if (mlen == 15u) {
            uint8_t b;
            do {
                if (ip >= clen) { ok = false; return op; }
                b = comp.GetValue(ip++);
                if (mlen > 0xFFFFFFFFu - b) { ok = false; return op; }
                mlen += b;
            } while (b == 255u);
        }
        if (mlen > 0xFFFFFFFFu - 4u) { ok = false; return op; }
        mlen += 4u;
        if (mlen > outCap - op) { ok = false; return op; }             /* output bound */
        for (uint32_t k = 0; k < mlen; k++) out.SetValue(op + k, out.GetValue(op + k - off));
        op += mlen;
    }
    return op;
}

/* ---------------------------------------------------------------- kernels --- */
//GM_ADDR 表示设备端 Global Memory（全局内存）中的地址。
__global__ __vector__ void lz4_compress_kernel(GM_ADDR in, GM_ADDR out, GM_ADDR outSizes,
                                               uint32_t totalBytes, uint32_t numChunks,
                                               uint32_t dtype)
{
    TPipe pipe;
    // TPosition::VECIN：表示向量计算输入侧使用的缓冲区UB位置。
    // 1：表示队列的缓冲区数量为 1。
    // qIn：这个队列的变量名。
    TQue<TPosition::VECIN, 1>  qIn;
    TQue<TPosition::VECOUT, 1> qOut, qSz;
    // 这里用于保存 LZ4 的哈希表。
    TBuf<TPosition::VECCALC>   htBuf;
    // pipe.InitBuffer(队列, 缓冲区数量, 每块缓冲区字节数);
    pipe.InitBuffer(qIn,  1, CHUNK);
    pipe.InitBuffer(qOut, 1, SLOT);
    //压缩长度
    pipe.InitBuffer(qSz,  1, 32);
    // pipe.InitBuffer(缓冲区, 字节数);
    pipe.InitBuffer(htBuf, HASH_SIZE * sizeof(int32_t));

    //创建了一个用于访问 NPU 全局内存中字节数据的对象。
    GlobalTensor<uint8_t> inG, outG;
    //告诉编译器，把传进来的 in 按照指向 GM 中 uint8_t 数据的指针来使用。
    //SetGlobalBuffer() 则让 inG 关联到这个地址和相应的数据范围。
    inG.SetGlobalBuffer((__gm__ uint8_t*)in, (uint32_t)((totalBytes + 31u) & ~31u));
    outG.SetGlobalBuffer((__gm__ uint8_t*)out, numChunks * SLOT);
    // 取得能够访问这块工作空间的 LocalTensor<int32_t> 对象。每个 int32_t 占 4 字节，所以它能保存 4096 个元素。
    LocalTensor<int32_t> ht = htBuf.Get<int32_t>();

    const uint32_t nb = GetBlockNum();
    for (uint32_t c = GetBlockIdx(); c < numChunks; c += nb) {
        // 第c个数据块的起始位置
        uint32_t start = c * CHUNK;
        // 计算当前数据块的长度，如果剩余数据不足一个完整的 CHUNK，则只取剩余部分。
        uint32_t len   = (start + CHUNK <= totalBytes) ? CHUNK : (totalBytes - start);
        // 将 len 向上对齐到 32 字节的倍数，确保数据块长度满足硬件要求。
        uint32_t lenR  = (len + 31u) & ~31u;

        // 将当前数据块从全局内存复制到本地缓冲区 src 中，以便进行压缩操作。
        LocalTensor<uint8_t> src = qIn.AllocTensor<uint8_t>();
        if (lenR) DataCopy(src, inG[start], lenR);//DataCopy(目标, 来源, 元素数量);
        qIn.EnQue(src);
        //DeQue：从队列取得数据
        LocalTensor<uint8_t> s = qIn.DeQue<uint8_t>();
        //AllocTensor：分配一个新的本地张量，用于存放压缩结果。
        LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
        uint32_t clen;
        // 调用 LZ4 压缩函数，把输入数据 s 压缩到 dst 中，并把实际压缩结果长度返回给 clen。
        // s	待压缩的输入数据
        // len	输入数据长度（字节）
        // dst	存放压缩结果的缓冲区
        // ht	进行重复匹配时使用的哈希表
        // clen	最终压缩数据的实际长度
        if (dtype == 2u)      clen = lz4CompressT<2u>(s, len, dst, ht);
        else if (dtype == 4u) clen = lz4CompressT<4u>(s, len, dst, ht);
        else                  clen = lz4CompressT<1u>(s, len, dst, ht);
        // 把压缩结果对应的 Tensor 入队，交给后续输出阶段。
        qOut.EnQue(dst);
        //记录压缩结果的长度，存入 szl 中，并将其入队到 qSz 队列中。
        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        szl.SetValue(0, (int32_t)clen);
        qSz.EnQue(szl);

        qIn.FreeTensor(s);

        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        GlobalTensor<int32_t> szG;
        szG.SetGlobalBuffer((__gm__ int32_t*)outSizes, numChunks * 8);
        // 将压缩结果长度写入全局内存中的 outSizes 数组中，表示每个数据块的压缩长度。
        DataCopy(szG[c * 8], szr, 8);
        qSz.FreeTensor(szr);
        //！！！但 lz4CompressT() 实际只写入了前 clen 个有效字节。剩余部分未在这段代码中初始化，因此不应该把整个 SLOT 都当作有效压缩数据。
        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        DataCopy(outG[c * SLOT], d, SLOT);
        qOut.FreeTensor(d);
    }
}

__global__ __vector__ void lz4_decompress_kernel(GM_ADDR comp, GM_ADDR out,
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

        if (clen > SLOT) {   /* corrupt: block longer than the slot */
            LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
            psl.SetValue(0, (int32_t)-1);
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
        uint32_t op = lz4Decompress(cl, clen, dst, CHUNK, decOk);
        qOut.EnQue(dst);

        LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
        psl.SetValue(0, decOk ? (int32_t)op : (int32_t)-1);
        qPs.EnQue(psl);

        qIn.FreeTensor(cl);

        LocalTensor<int32_t> psr = qPs.DeQue<int32_t>();
        DataCopy(psG[c * 8], psr, 8);
        qPs.FreeTensor(psr);

        uint32_t opR = decOk ? ((op + 31u) & ~31u) : 0u;
        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        if (opR) DataCopy(outG[c * CHUNK], d, opR);
        qOut.FreeTensor(d);
    }
}

// Decompressed-size query: no output buffer is written (mirrors nvCOMP's
// output_decompressed == false path).
__global__ __vector__ void lz4_size_kernel(GM_ADDR comp, GM_ADDR inSizes,
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
            psl.SetValue(0, (int32_t)-1);
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
        uint32_t op = lz4SizeOnly(cl, clen, szOk);
        if (!szOk) op = 0xFFFFFFFFu;
        qIn.FreeTensor(cl);

        LocalTensor<int32_t> psl = qPs.AllocTensor<int32_t>();
        psl.SetValue(0, (int32_t)op);
        qPs.EnQue(psl);
        LocalTensor<int32_t> psr = qPs.DeQue<int32_t>();
        DataCopy(psG[c * 8], psr, 8);
        qPs.FreeTensor(psr);
    }
}

/* ============================ batch API =================================== */

constexpr uint32_t MAXB = 256;   // max items per batch launch (host loops beyond)
constexpr uint32_t MAX_ITEM_BYTES = 16u * 1024u * 1024u;                 // per-item cap (nvCOMP LZ4 max)
constexpr uint32_t MAX_BLOCKS = (MAX_ITEM_BYTES + CHUNK - 1u) / CHUNK;   // 64 KiB blocks per item

// nvcompBatchedLZ4CompressAsync equivalent: pointer arrays + per-item sizes.
__global__ __vector__ void lz4_batch_compress_kernel(
    GM_ADDR uncompPtrs, GM_ADDR uncompBytes,
    GM_ADDR compPtrs,   GM_ADDR compBytes,
    GM_ADDR blockSizes, uint32_t batchSize, uint32_t dtype)
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
            uint32_t clen;
            if (dtype == 2u)      clen = lz4CompressT<2u>(s, l, dst, ht);
            else if (dtype == 4u) clen = lz4CompressT<4u>(s, l, dst, ht);
            else                  clen = lz4CompressT<1u>(s, l, dst, ht);
            qOut.EnQue(dst);
            qIn.FreeTensor(s);

            uint32_t clenW = (clen + 31u) & ~31u;
            if (clenW > SLOT) clenW = SLOT;

            LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
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
        szl.SetValue(0, (int32_t)compTotal);
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

// nvcompBatchedLZ4DecompressAsync equivalent.
__global__ __vector__ void lz4_batch_decompress_kernel(
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
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t clen = (uint32_t)bs.GetValue(b);
            if (clen == 0u) break;
            if (clen > SLOT) { produced = 0xFFFFFFFFu; break; }
            uint32_t clenR = (clen + 31u) & ~31u;

            GlobalTensor<uint8_t> cg;
            cg.SetGlobalBuffer((__gm__ uint8_t*)(cAddr + compOff), clenR);

            LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
            DataCopy(compL, cg, clenR);
            qIn.EnQue(compL);
            LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

            LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
            uint32_t capLeft = (capacity - outOff > CHUNK) ? CHUNK : (capacity - outOff);
            bool decOk = false;
            uint32_t opn = lz4Decompress(cl, clen, dst, capLeft, decOk);
            qOut.EnQue(dst);
            qIn.FreeTensor(cl);

            if (!decOk) {
                LocalTensor<uint8_t> dd = qOut.DeQue<uint8_t>();
                qOut.FreeTensor(dd);
                produced = 0xFFFFFFFFu;
                break;
            }
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

        psl.SetValue(0, (int32_t)produced);
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


// nvcompBatchedLZ4GetDecompressSizeAsync equivalent for the pointer-array API.
__global__ __vector__ void lz4_batch_size_kernel(
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
        for (uint32_t b = 0; b < MAX_BLOCKS; b++) {
            uint32_t clen = (uint32_t)bs.GetValue(b);
            if (clen == 0u) break;
            if (clen > SLOT) { produced = 0xFFFFFFFFu; break; }
            uint32_t clenR = (clen + 31u) & ~31u;

            GlobalTensor<uint8_t> cg;
            cg.SetGlobalBuffer((__gm__ uint8_t*)(cAddr + compOff), clenR);

            LocalTensor<uint8_t> compL = qIn.AllocTensor<uint8_t>();
            DataCopy(compL, cg, clenR);
            qIn.EnQue(compL);
            LocalTensor<uint8_t> cl = qIn.DeQue<uint8_t>();

            bool szOk = false;
            uint32_t sz = lz4SizeOnly(cl, clen, szOk);
            qIn.FreeTensor(cl);
            if (!szOk) { produced = 0xFFFFFFFFu; break; }
            produced += sz;
            compOff += clenR;
        }
        qbsIn.FreeTensor(bs);

        psl.SetValue(0, (int32_t)produced);
        qpsOut.EnQue(psl);
        LocalTensor<int32_t> psr = qpsOut.DeQue<int32_t>();
        DataCopy(aszG[i * 8], psr, 8);
        qpsOut.FreeTensor(psr);
    }
    qcp.FreeTensor(cp);
    qszIn.FreeTensor(csz);
}

} // namespace lz4a

#endif // LZ4_ASCEND_KERNELS_H
