# 设计说明：从 nvCOMP（CUDA SIMT）到昇腾 910C（Ascend C SIMD）

本文说明将 nvCOMP `branch-2.2` 的 LZ4 / Snappy 压缩与解压迁移到昇腾 910C 的
**算法语义迁移**方法与目标硬件上的工程取舍。

---

## 1. 参考实现（nvCOMP branch-2.2）的并行结构

### 1.1 LZ4（`src/LZ4Kernels.cuh`, `src/lowlevel/LZ4CompressionKernels.cu`）

- 每个 chunk 由 **1 个 warp（32 线程）** 协作完成压缩与解压（`LZ4_*_THREADS_PER_CHUNK = 32`）。
- 依赖 warp 级原语：
  - `__match_any_sync` / 软件回退（`warpMatchAny`）—— 找哈希碰撞的 lane；
  - `__ballot_sync`（`warpBallot`）—— 在 warp 内定位“第一个匹配位置”；
  - `__shfl_sync`（`shuffleLiterals`）—— 字面量按对齐方式跨 lane 拼接；
  - `__syncwarp` / `__syncthreads` —— 协作拷贝之间的同步。
- 哈希：`hash(key) = (brev(key) + (key ^ 0xc375)) & (size-1)`，键为 4 字节。
- 格式约束：min-match 4，offset ≤ 65535，**末尾 5 字节必为字面量**，**末次匹配起点不晚于 `len-12`**。

### 1.2 Snappy（`src/SnappyKernels.cuh`, `src/lowlevel/SnappyBatchKernels.cu`）

- 每个 chunk 由 **多个 warp** 流水线协作：
  - **WARP0** 编码字面量 / 拷贝（`StoreLiterals` / `StoreCopy`）；
  - **WARP1** 用 12-bit 哈希找 4 字节匹配（`FindFourByteMatch`，用 `BALLOT` / `SHFL`）；
  - 解压为 **3 warp 流水线**：解码符号 → 预取字节流 → 还原 LZ77。
- 依赖 `__ballot_sync`、`__shfl_sync`、`__match_any_sync`、`SYNCWARP`、`NANOSLEEP` 等。
- 格式：`varint(uncompressed_len)` + 元素（literal / copy-1B-offset / copy-2B-offset / copy-4B-offset），
  `MAX_COPY_LENGTH=64`、`MAX_COPY_DISTANCE=32768`。

**结论：这些实现是深度 warp 协作的 SIMT 代码，无法也不应逐行翻译到 910C。**

---

## 2. 目标硬件：910C / 910B（`dav-2201`）AIV 分析

通过 `npu-smi`、CANN 头文件与实测得出：

| 事实 | 依据 |
| --- | --- |
| 提供 **AIV 向量核** 作为可编程计算单元；`<<<blocks, nullptr, stream>>>` 直调，`GetBlockIdx()/GetBlockNum()` 可用 | 实测 hello_world 8/8 blocks |
| **UB = 192 KB**（向量可用区约 184 KB） | `kernel_utils_constants.h`：`__NPU_ARCH__==2201 ⇒ TOTAL_UB_SIZE=192*1024`；平台 `ub_size=196608` |
| `GlobalTensor` / `LocalTensor` / `DataCopy`（GM↔UB、UB↔UB）工作正常 | 实测 |
| **`DataCopy` 长度必须 32B 对齐**（否则尾部被丢弃） | 实测：4112B 拷贝只落地 4096B |
| **AIV 标量单元可任意 `GetValue/SetValue` 访问 UB**（任意下标、任意顺序） | 实测：字节反转 + FNV 哈希一致 |
| 逐元素算术 `vmax/vmin/vsub` 在 `dav_c220` 走 **half/float 通路**，**不支持 uint8/int8** | 编译报错 `maybe need a type '__ubuf__ half *'` |
| 逐元素 `Compare/vcmp` 同样 **只支持 half/float** | 编译报错 |
| `Duplicate` 支持 int32（用于哈希表清零） | 实测 |
| TQue 的 `EnQue/DeQue` 正确插入 MTE↔Scalar/Vector 同步 | 实测（缺失时读到全 0） |

**关键判断：** 910C 的向量算术语义偏向半精度/浮点数据通路，而 LZ4/Snappy 是**字节级、
强串行依赖**的算法。因此正确的工程做法是：

> **把批量数据搬运（MTE/向量数据通道）与哈希表清零（向量 `Duplicate`）交给向量侧，
> 把 LZ77 的解析（读 4 字节 → 哈希 → 查表 → 比较 → 发射 token）放到 AIV 的设备端
> 标量单元。** 这正是任务所允许的“设备端 Scalar 控制逻辑”，且核心压缩/解压
> **完全在 NPU 核内执行**，主机仅负责 `aclrtMemcpy`。

同时，nvCOMP 中以 **warp 协作**实现的三件事在 910C 上被重新组织：

| nvCOMP（SIMT） | 本移植（AIV） |
| --- | --- |
| 32 lane 并行“扫描 + ballot”找匹配 | 单核内串行贪心扫描（哈希表在 UB），块间并行 |
| `shuffleLiterals` 跨 lane 拼字面量 | 标量顺序写 + `DataCopy` 批量搬运 |
| `__ballot_sync` 找第一个匹配 lane | 标量循环直接定位 |
| 多 warp 流水线（Snappy 编/解码） | 每块单核处理一个 chunk；块间并行，核数=grid |

并行度来源由 **warp 内 SIMT** 变为 **chunk 级 block 并行 + AIV 向量数据通道**。

---

## 3. 数据/内存布局

- 主机把输入按 **64 KiB** 分块（最后一块可不足）。
- 设备端：`nb = min(numChunks, 32)` 个 block；每个 block 以 `for (c = blockIdx; c < numChunks; c += nb)` 处理多个 chunk。
- GM 侧：
  - 输入：连续 `totalBytes`；
  - 压缩输出：每 chunk 一个固定槽 `SLOT = ceil((CHUNK + CHUNK/6 + 64)/32)*32` 字节；
  - 每 chunk 压缩长度：32B（一个 int32）；
  - 解压输出：每 chunk `CHUNK`。
- UB 侧（每 block）：
  - LZ4 压缩：`src(CHUNK) + dst(SLOT) + hash(4096*4B)` ≈ **154 KB**
  - LZ4 解压：`comp(SLOT) + out(CHUNK)` ≈ **139 KB**
  - Snappy 压缩：`src + dst + hash` ≈ **146 KB**
  - Snappy 解压：`comp + out` ≈ **139 KB**
  - 均在 184 KB 可用 UB 之内。

设备核结构：

```
__global__ __vector__ void lz4_compress_kernel(in, out, outSizes, totalBytes, numChunks)
    TPipe pipe; TQue VECIN/VECOUT; TBuf<hash>;
    for each chunk (strided):
        DataCopy(src ← GM)  → EnQue/DeQue  (MTE2→scalar 同步)
        lz4Compress(src,len,dst,ht)         (设备端标量 + 向量 Duplicate)
        DataCopy(out ← dst)                 (scalar→MTE3 同步)
```

---

## 4. 算法（保留 nvCOMP 语义 / 数据格式）

### 4.1 LZ4（标准 raw block）

- 哈希族沿用 nvCOMP：`(brev32(key) + (key ^ 0xc375)) & (HASH_SIZE-1)`，`HASH_SIZE=4096`。
- 贪心匹配，min-match 4，offset ≤ 65535；匹配扩展上限 `len-5-i`（末尾 5 字节字面量）；
  主循环 `i ≤ len-12`（末次匹配起点限制）。
- 序列编码：`token` → 可选字面量长度扩展 → 字面量 → 2 字节 offset(LE) → 可选匹配长度扩展。
- **类型化（对齐）变体**：对应 nvCOMP 的 `nvcompBatchedLZ4Opts_t.data_type`
  （`NVCOMP_TYPE_CHAR/UCHAR/BITS`=1 字节、`SHORT/USHORT`=2、`INT/UINT`=4）。
  匹配候选位置限制在 `sizeof(T)` 对齐处、匹配按 T 单位扩展；**输出仍是标准字节级 LZ4 块**
  （长度/偏移都是字节数），已被官方 LZ4 解码器验证可解。实现为模板 `lz4CompressT<TS>`，
  `TS ∈ {1,2,4}`，`dtype` 运行时分派。
- **解压尺寸查询**：`lz4SizeOnly` 对应 nvCOMP `output_decompressed == false` 路径
  （只解析、不写输出），对应公开接口 `nvcompBatchedLZ4GetDecompressSizeAsync`。

### 4.2 Snappy（标准 block）

- 哈希：`(v * 0x1e35a7bd) >> (32-12)`（snappy 标准 `kHashMul32`）。
- `varint(uncompressed_len)` 前缀 + 元素流；`MAX_COPY_LENGTH=64`、`MAX_COPY_DISTANCE=32768`；
  长匹配拆成多个 ≤64 字节的同距离拷贝。
- 解码支持全部四种元素（literal / copy-1B / copy-2B / copy-4B），可解官方编码器输出。
- **解压尺寸查询**：`snapSizeOnly` 读取 varint 前缀，对应 nvCOMP 的
  `get_uncompressed_sizes_kernel` / `nvcompBatchedSnappyGetDecompressSizeAsync`。

### 4.3 批量 API（指针数组语义）

nvCOMP 的批量接口不是“一个大 buffer 切成等长块”，而是：

```
nvcompBatchedLZ4CompressAsync(
    const void* const* device_uncompressed_ptrs,   // 每项输入指针（可互不连续）
    const size_t*      device_uncompressed_bytes,  // 每项长度（可变）
    ..., void* const* device_compressed_ptrs,      // 每项输出指针
    size_t*            device_compressed_bytes,    // 每项压缩后长度（输出）
    ..., cudaStream_t stream);
```

在 Ascend 上迁移的关键点与做法：

1. **设备端读取设备指针数组**：把 `device_*_ptrs`（`__gm__ uint64_t[]`）通过一次
   `DataCopy` 搬进 UB，用 `LocalTensor<uint64_t>::GetValue(i)` 取出地址，再
   `SetGlobalBuffer((__gm__ uint8_t*)addr, ...)` 驱动 GM↔UB 搬运（已在目标机验证）。
2. **块并行**：`blockIdx` 以 `i = GetBlockIdx(); i < batchSize; i += GetBlockNum()`
   方式分摊 batch；每个 item 用独立的 `(in ptr, out ptr, size)`。
3. **每项输出大小**：以 32B 槽写入 `device_compressed_bytes`（避免多块写同一数组的竞争）。
4. **temp / scratch**：nvCOMP 把哈希表放在 GM temp 空间；本移植把哈希表放在每个 block 的
   **UB**（更贴合 Ascend 内存层级），因此 `temp` 指针可忽略（见限制说明）。
5. **解压**：额外接受每项缓冲**容量**数组，并输出每项**实际**解压大小
   （对应 `device_uncompressed_bytes` / `device_actual_uncompressed_bytes`）。

约束（因 UB 容量）：单次 launch 的 item 数 ≤ 256，单 item 长度 ≤ 64 KiB（与 nvCOMP
建议的 64 KiB chunk 一致）；更大批量由主机分批循环。

---

## 5. 正确性证据

`scripts/run_all.sh` 对 12 个向量 × {LZ4 d1, LZ4 d2, LZ4 d4, Snappy} 做四向验证，
再加上 3 个**批量 API**（指针数组、40 个变长 item）用例，**51/51 通过**：

- **A** NPU 压缩 → NPU 解压，逐字节比对（含空文件、1 字节、跨 64 KiB 边界、2 MiB 多块、
  纯随机不可压、高重复、文本、结构化数据）。
- **B** NPU 压缩产物交给 **官方 lz4 / google-snappy** 解码并还原（格式兼容）。
- **C** 官方编码器产物交给 **NPU 解压** 并还原（解码器兼容）。
- **D** NPU 解压尺寸查询结果 == 期望的分块大小。
- **batch** 指针数组批量 API：40 个互不连续、变长（0/1/奇数/32 对齐/65535/65536/随机）的
  item，逐项校验（A）并做官方参考双向互操作（B/C）。

参考实现来自 `github.com/lz4/lz4` 与 `github.com/google/snappy`，用 `gcc/g++` 独立编译。

**验证方法说明**：A 是自洽往返（NPU 编 → NPU 解 → 逐字节比对原文件）；B/C 是**与官方实现
的双向互操作**——B 证明我们产出的字节流是合法的标准格式，C 证明我们的解码器不依赖自家
编码器；D 用一个**独立算法**（尺寸解析）交叉验证数据。三者结合可排除“自说自话的自定义
格式”。局限：B/C 仅对非空文件执行；尚未做随机模糊（fuzz）与损坏流测试。

---

## 5.1 自动化测试与性能基准

- `tests/`（对应 nvCOMP `tests/`）：`test_lz4` / `test_snappy`。覆盖 0~64 的每个尺寸、
  边界尺寸、5 种数据形态、LZ4 `dtype=1/2/4`、多 chunk、解压尺寸查询、批量 API；
  输出缓冲预填 `0xAA` 后逐字节比对。当前 **LZ4 122 项 / Snappy 105 项全部通过**。
- `benchmarks/`（对应 nvCOMP `benchmarks/`）：`bench_lz4` / `bench_snappy`。
  用 ACL 设备事件（等价 CUDA events）在 warmup 后计时，报告压缩率与压缩/解压吞吐（GB/s），
  并对每个数据集做往返校验。
- 设备端算法与核函数已抽到 `src/*_kernels.h`，demo / tests / benchmarks 共享同一份实现，
  避免算法代码复制。

## 6. 覆盖范围与已知限制

**已迁移并验证的算法功能**（nvCOMP 的 LZ4/Snappy 算法本体、数据格式与批处理语义）：

- LZ4 压缩 / 解压 / 解压尺寸查询 / 类型化（1、2、4 字节对齐）输入
- Snappy 压缩 / 解压 / 解压尺寸查询
- **批量（指针数组）压缩 / 解压**：变长、互不连续的独立 item + 每项输出大小 + 解压容量/实际大小

**未迁移（容器层 / 接口层，非算法本体）**：

- **HLIF 容器格式**：`src/highlevel/{LZ4,Snappy}HlifKernels.cu` +
  `nvcomp_common_deps/hlif_shared_types.hpp` 的 `CommonHeader`（magic、version、
  `FormatType`、`comp_data_size`、`decomp_data_size`、`num_chunks`、
  `include_chunk_starts`、整体/逐 chunk 校验和、`uncomp_chunk_size`、`comp_data_offset`）
  以及 chunk 起始偏移表。`LZ4Manager` / `SnappyManager`（C++ 高层 API）走的就是这条路径。
  注意它**复用同一份 cipher**（`compressStream` / `do_snap`），因此压缩算法本身已覆盖。
- **C API 与 C++ Manager 接口外壳**：`nvcompBatchedLZ4*` / `nvcompBatchedSnappy*` 的
  **字面函数签名**（批量语义本身已迁移，见第 4.3 节）与 `LZ4Manager` / `SnappyManager` 类；
  `*GetTempSize` / `*GetMaxOutputChunkSize` 辅助计算。
- **GM temp 空间**：nvCOMP 把哈希表放在调用方传入的 GM temp 缓冲；本移植把哈希表放在
  每 block 的 UB（更符合 Ascend 内存层级），因此不消费 `device_temp_ptr`。
- **逐 chunk 状态与错误上报**、**损坏流越界检查**（`OOB_CHECKING`）、**可选校验和**。

其他工程性限制：

- **功能优先，未做性能优化**：解析在 AIV 标量单元串行执行；可通过块内向量化批量拷贝
  （32B 对齐区段）、更细的分块流水、或 `GatherMask` 等进一步加速。
- UB 中的哈希表固定 4096 项（12-bit）；nvCOMP 最高 14-bit（只影响压缩率，不影响格式）。
- 交叉验证（B/C）在文件非空时进行；尚未做 fuzz 与损坏流失配测试。


---

## 12. 可靠性与错误处理（整改后）

### 12.1 Batch 元数据

Kernel 用 `DataCopy` 搬运元数据时必须 32 B 对齐，因此按元素取整读取：

- 指针数组：`ceil(batchSize / 4) * 4` 个 `uint64`（32 B 粒度）；
- 大小 / 容量 / 输出长度：`ceil(batchSize / 8) * 8` 个 `int32`。

主机侧使用**相同取整后的长度**分配并零初始化（`std::vector(ptrCount, 0)`），
再整体 H2D。主机分批切片时 `base` 是 256 的倍数（同时是 4 和 8 的倍数），
所以最后一批的取整读取仍落在总分配范围内。

### 12.2 LZ4 解压校验

`lz4Decompress(comp, clen, out, outCap, ok)` 在每次访问前检查：

| 检查 | 处理 |
| --- | --- |
| token 之后的 literal 扩展字节 | `ip >= clen` → 失败 |
| literal 长度 | `litLen > clen - ip`（输入）或 `> outCap - op`（输出）→ 失败 |
| offset 两字节 | `clen - ip < 2` → 失败 |
| offset 合法性 | `off == 0 || off > op` → 失败 |
| match 扩展字节 / 长度 | 同上，并防 `0xFFFFFFFF` 溢出 |
| match 输出 | `mlen > outCap - op` → 失败 |

所有长度比较都用 `x > limit - y` 形式，避免无符号加法溢出。

`lz4SizeOnly(comp, clen, ok)` 做同样的结构校验（包括 offset），
因此尺寸查询对损坏流也会返回失败而不是错误长度。

### 12.2b Snappy 解压校验

Snappy 解码器原先**没有任何边界检查**（属于任务清单之外的同类缺陷，已一并修复）：

| 检查 | 处理 |
| --- | --- |
| varint 前导（uncompressed length） | 最多 5 字节；`ip >= clen` 或 `shift > 35` → 失败 |
| 声明长度 vs 容量 | `ulen > outCap` → 失败 |
| literal 长度扩展字节 | `clen - ip < nb` → 失败 |
| literal 数据 | `l > clen - ip` 或 `l > outCap - op` → 失败 |
| copy 的 offset 字节 | 1/2/4 字节按类型检查剩余输入 |
| offset 合法性 | `off == 0 || off > op` → 失败 |
| copy 输出 | `l > outCap - op` → 失败 |
| 一致性 | `op > ulen`（中途）或 `op != ulen`（结尾）→ 失败 |

`snapSizeOnly` 用同一套规则做“只走不写”的校验，因此尺寸查询对损坏流返回 `-1`。

### 12.3 错误状态

- 单 chunk 解压核 / 尺寸核（LZ4 与 Snappy 相同）：失败时把 `produced` 写成 `-1`（`0xFFFFFFFF`）；
- 批量解压核 / 批量尺寸核（LZ4 与 Snappy 相同）：某 item 失败时该 item 的 `actualSize` 写成 `-1`；
- 批量压缩核：输出放不下（`clen > SLOT`）时该 item 的 `compressedSize` 写成 `-1`；
- 主机侧（demo / tests / benchmark）识别 `-1` 并判 FAIL，绝不把 `-1` 当长度使用；
- 批量解压按 item 的 `capacity` 逐块限制输出（`min(CHUNK, capacity - outOff)`）。

### 12.4 空输入

| 场景 | 表示 | 说明 |
| --- | --- | --- |
| LZ4 单缓冲区 / 文件 | 1 字节 `0x00` | 标准空块 |
| Snappy 单缓冲区 / 文件 | 1 字节 `0x00` | varint(0) |
| Batch 中的空 item | 0 个块，`clen = 0` | “无数据 item”，与标准空块区分 |

官方 LZ4 / Snappy 解码器均可解出 0 字节；NPU 也能解官方产生的空块。

### 12.5 批大小

`MAXB = 256` 是单次 Kernel Launch 上限。主机在 demo / tests / benchmark 中
按 256 切片循环发射，因此 API 层支持任意项数（受内存限制）。
测试覆盖 1 / 3 / 31 / 32 / 33 / 255 / 256 / 257。
