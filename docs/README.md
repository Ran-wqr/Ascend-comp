# nvcomp_ascend 完全指南（初学者版）

> 这份文档假设你**几乎不懂** CUDA / Ascend / 压缩算法。
> 我们从"压缩是什么"讲起，用比喻 + 逐行代码解释，带你彻底看懂这个项目。
>
> 配套文档：功能覆盖见 [`../README.md`](../README.md)，设计取舍见 [`design.md`](design.md)。

---

## 第 0 章 先建立直觉

### 0.1 什么是"压缩"

想象你要搬一箱书。如果每本书都单独装一个盒子，很占地方；
如果发现"第 1 本书和第 100 本书一模一样"，你就可以只带一本，
再留张纸条写"第 100 本 = 第 1 本"。**压缩就是找重复、用更短的描述代替重复。**

- **无损压缩**：还原后和原来**一个字节都不差**（本项目的目标）。
- **有损压缩**：还原后允许有差异（比如 JPEG 图片、MP3 音乐），本项目**不用**。

### 0.2 LZ4 和 Snappy 是两种"打包方法"

它们都基于同一个思路（LZ77）：**"往前找，看这串字节之前出现过没有"**。
区别只在"打包格式"（写进文件的字节长什么样）：

| | LZ4 | Snappy |
| --- | --- | --- |
| 是谁的 | NVIDIA nvCOMP 里的 LZ4 实现 | NVIDIA nvCOMP 里的 Snappy 实现 |
| 输出格式 | 标准 **LZ4 raw block** | 标准 **Snappy block** |
| 开头 | 直接是第一个"序列" | 先写一段"原始长度"（varint） |
| 最短匹配 | 4 字节 | 4 字节 |
| 最远回溯 | 65535 字节 | 32768 字节 |
| 特殊规则 | **末尾 5 字节必须原样保留**、末次匹配不能太靠后 | 没有这条限制 |

**关键点**：本项目**没有发明新格式**。我们产出的字节，用官方的 LZ4 / Snappy
解码器也能解开（文档后面会讲怎么验证）。

### 0.3 为什么要放到 NPU 上

CPU 做压缩是"一个人搬书"，一次搬一本。
NPU（昇腾 910）里有几十个**并行计算单元**，可以"几十个人同时搬不同的箱子"。
本项目把数据切成很多块（chunk），**每块交给一个 block 并行处理**。

### 0.4 一个程序，两个世界

本项目的每个 `.asc` 文件里其实有**两段代码**：

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  主机（CPU）                 │        │  设备（NPU）                  │
│  - 读文件                    │  启动   │  - 真正做压缩/解压的核函数      │
│  - 申请设备内存              │ ─────► │    （__global__ __vector__）   │
│  - 把数据拷到设备            │        │  - 在 AIV 核上并行执行          │
│  - 把结果拷回主机、比对      │ ◄───── │                              │
└─────────────────────────────┘  结果   └──────────────────────────────┘
```

- **主机代码**：普通的 C++，用 ACL（Ascend 的计算库）管理内存和启动。
- **设备代码**：用 Ascend C 写，运行在 NPU 上，**看不到你的硬盘、也不能随便 printf 调试**。

---

## 第 1 章 先认识这些名词（大白话 + 比喻）

| 名词 | 一句话解释 | 生活比喻 | 在本项目哪里出现 |
| --- | --- | --- | --- |
| **主机 / Host** | 你登录的那台服务器的 CPU | 你的办公室 | 所有 `run_*` 函数、`main` |
| **设备 / Device** | 昇腾 NPU 芯片 | 外面的仓库 | 核函数运行的地方 |
| **GM** | Global Memory，设备上的大内存 | 仓库的大货架 | `GlobalTensor`、`dIn/dComp/dOut` |
| **UB** | Unified Buffer，每个计算单元旁的小高速缓存（约 184KB） | 你手边的工作台 | `LocalTensor`、`TQue`、`TBuf` |
| **核函数 / Kernel** | 在设备上运行的函数 | 仓库里的工人 | `__global__ __vector__ void xxx_kernel` |
| **block** | 一次启动里的一个并行单元 | 一个工人 | `GetBlockIdx()`、`GetBlockNum()` |
| **Ascend C** | 华为的设备端编程语言（C++ 方言） | 仓库工人的操作手册 | 所有 `.asc` 设备代码 |
| **DataCopy** | 在 GM 和 UB 之间搬数据 | 用小推车运货 | `DataCopy(...)` |
| **TQue** | 带"排队+同步"的缓冲区 | 传送带 | `qIn.AllocTensor/EnQue/DeQue` |
| **TBuf** | 普通缓冲区（不管同步） | 一张空桌子 | `htBuf`（哈希表） |
| **chunk** | 一个独立压缩单元（本项目 ≤64KB） | 一个箱子 | `CHUNK` |
| **哈希表** | 用"指纹"快速找到"之前在哪见过" | 字典的目录 | `ht` |
| **token** | LZ4 里描述"这段怎么还原"的控制字节 | 快递单 | `emitSeq` |

### 1.1 你只需要懂这几样 C 语言

**（1）整数类型**

| 写法 | 含义 | 范围 |
| --- | --- | --- |
| `uint8_t` | 无符号 8 位整数 | 0 ~ 255（正好一个字节） |
| `uint32_t` | 无符号 32 位整数 | 0 ~ 约 42 亿 |
| `int32_t` | 有符号 32 位整数 | 约 -21 亿 ~ 21 亿 |

> 压缩处理的是"字节"，所以到处是 `uint8_t`；位置/长度用 `uint32_t`。

**（2）位运算**（压缩格式里大量使用）

```c
a & b    // 按位与
a | b    // 按位或
a ^ b    // 按位异或（不同为1）
a << 3   // 左移 3 位（相当于乘 8）
a >> 3   // 右移 3 位（相当于除 8）
```

例：`(0x0A << 4) | 0x03` = `0xA3`。**这就是 LZ4 把"字面量长度"和"匹配长度"塞进同一个字节的办法。**

**（3）指针与数组**

```c
uint8_t* p;        // p 指向某个字节
p[0]               // p 指向的第 0 个字节
```

**（4）`__aicore__` / `__global__ __vector__`**

- `__aicore__`：这个函数在设备上跑。
- `__global__ __vector__`：这是一个**核函数**（由主机启动、在设备上并行执行）。

---

## 第 2 章 目录导览：每个文件夹/文件是干什么的

```
nvcomp_ascend/
├── CMakeLists.txt          ← 编译说明书
├── common_ascend.h         ← 主机侧小工具（读文件/写文件/错误检查）
├── README.md               ← 项目总览（功能覆盖表）
├── docs/
│   ├── README.md           ← 你正在读的这份（初学者详解）
│   └── design.md           ← 设计取舍（为什么这么迁移）
├── src/
│   ├── lz4_ascend.asc      ← ★ LZ4 全部代码（设备算法 + 核函数 + 主机驱动）
│   └── snappy_ascend.asc   ← ★ Snappy 全部代码
├── scripts/                ← 编译/生成数据/跑测试的脚本
├── ref/                    ← 官方 LZ4/Snappy 源码 + 交叉验证小工具
├── data/                   ← 测试数据（可删除重建）
├── logs/                   ← 运行日志（可删除）
└── build/                  ← 编译产物（可删除重建）
```

### 2.1 每个文件的"它是什么 / 为什么需要 / 删了会怎样"

| 文件/目录 | 它是什么 | 为什么需要 | 删了会怎样 |
| --- | --- | --- | --- |
| `CMakeLists.txt` | 告诉 CMake 怎么用 `bisheng` 编译 `.asc` | 没有它无法生成可执行文件 | 编译失败 |
| `common_ascend.h` | 三个主机工具函数 | 避免每个文件重复写读文件代码 | 编译报"找不到 read_file" |
| `src/lz4_ascend.asc` | LZ4 的**全部**逻辑 | 核心产物 | 没有 LZ4 功能 |
| `src/snappy_ascend.asc` | Snappy 的**全部**逻辑 | 核心产物 | 没有 Snappy 功能 |
| `scripts/build.sh` | 一键编译 | 少打字、少出错 | 可手动 cmake/make |
| `scripts/build_ref.sh` | 编译官方参考编解码器 | 用于"外部验证" | B/C 验证列变 NA |
| `scripts/gen_testdata.py` | 造 12 个测试文件 | 覆盖各种边界情况 | 测试数据缺失 |
| `scripts/gen_batch.py` | 造 40 个变长 item | 测批量 API | 批量测试缺失 |
| `scripts/check_sizes.py` | 校验尺寸查询结果 | 验证 D 列 | D 列失败 |
| `scripts/check_batch.py` | 校验批量参考解码结果 | 验证批量 B/C | 批量 B/C 失败 |
| `scripts/run_all.sh` | 一键跑全部验证 | 复现结果 | 需手动逐条跑 |
| `ref/lz4/`、`ref/snappy/` | 官方源码（第三方） | **证明我们的格式是标准的** | 无法交叉验证 |
| `ref/tools/ref_lz4.cpp` | 官方 LZ4 编/解码小工具 | 喂/吃我们的数据 | 同上 |
| `data/` | 测试输入与中间产物 | 测试用 | 脚本会重建 |
| `logs/` | 日志 | 排查问题 | 可删 |
| `build/` | 编译目录 | 放可执行文件 | 重新编译即可 |

> **记住一句话**：真正的"源代码"只有 `src/` + `scripts/` + `common_ascend.h` + `CMakeLists.txt` + `docs/`。
> 其他都能重新生成。

---

## 第 3 章 一个程序是怎么跑起来的（6 步）

用"把家里的书搬到图书馆仓库、整理、再搬回来"来理解：

```
① 读文件           你从书架取出书                read_file()
② 申请设备内存      在仓库租一块地方              aclrtMalloc()
③ 拷到设备          把书运到仓库                  aclrtMemcpy(HOST→DEVICE)
④ 启动核函数        让仓库工人整理                kernel<<<blocks, nullptr, stream>>>(...)
⑤ 拷回主机          把整理结果运回                aclrtMemcpy(DEVICE→HOST)
⑥ 比对              检查有没有丢页                memcmp / 逐字节比较
```

对应到代码（以 `lz4_demo data/t_text.bin` 为例）：

```cpp
read_file(inPath, hIn);                          // ①
aclrtMalloc((void**)&dIn,   inBytes,   ...);     // ②
aclrtMalloc((void**)&dComp, compBytes, ...);
aclrtMemcpy(dIn, inBytes, hIn.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);   // ③
lz4_compress_kernel<<<nb, nullptr, stream>>>(dIn, dComp, dSizes, totalBytes, numChunks, dtype); // ④
aclrtSynchronizeStream(stream);                  // 等设备干完
aclrtMemcpy(hSizes.data(), sizeBytes, dSizes, sizeBytes, ACL_MEMCPY_DEVICE_TO_HOST); // ⑤
// ⑥ 逐字节比较 recon 和 hIn
```

**为什么需要 `aclrtSynchronizeStream`？**
`<<<>>>` 是**异步**的——主机说完"开始"就往下走了。必须等设备真正算完，
才能把结果拷回来，否则会拷到一半数据。同步 = "等工人干完再取货"。

**为什么内存要 32 字节对齐？**
设备搬数据的最小单位是 32 字节（像一个托盘）。你要求搬 100 字节，它会搬 128 字节。
本项目统一用 `(len + 31) & ~31` 把长度向上取整到 32 的倍数。

---

## 第 4 章 压缩原理：LZ4 是怎么"变魔术"的

### 4.1 核心思想：找"之前出现过的相同片段"

```
原始数据:  A B C D A B C D A B C D A B C D ...
位置:      0 1 2 3 4 5 6 7 8 ...
                    ↑
              位置 4 的 "ABCD" 和位置 0 的 "ABCD" 一模一样！
```

于是我们不用再写一遍 "ABCD"，而是写一条指令：

> "从这里开始，**复制**前面第 **4** 个字节往前的 **4** 个字节。"

- 这个"4"叫 **offset（回溯距离）**。
- 复制的长度叫 **match length（匹配长度）**。

### 4.2 手算一个例子

输入：`"ABCD"` 重复 100 次 = 400 字节。

压缩器的思路：

1. 位置 0~3 是 `ABCD`，之前没见过 → 记为**字面量（literal）**。
2. 位置 4 读到 `ABCD`，查哈希表发现位置 0 也是 `ABCD` → 找到匹配，offset = 4。
3. 往后一直比，发现后面全都一样（因为每 4 字节重复一次），
   于是匹配长度能延伸到 **391 字节**。
4. 但**规则要求末尾至少留 5 字节原样**，所以只能匹配到 391（400 - 5 - 4 = 391）。
5. 剩下最后 5 字节作为字面量。

压缩后大致是：

```
[token][ABCD][offset=4][长度扩展...][token][最后5字节]
 1B     4B      2B        2B          1B      5B      ≈ 15 字节
```

**400 字节 → 约 15 字节**。这就是压缩。

### 4.3 压缩后的字节长什么样（LZ4 的"序列"格式）

LZ4 的每一段叫一个 **sequence（序列）**，格式固定：

```
┌────────┬──────────────┬──────────┬──────────┬──────────────┐
│ token  │ 字面量长度扩展 │ 字面量    │ offset   │ 匹配长度扩展   │
│ 1 字节  │ 0~N 字节      │ L 字节    │ 2 字节    │ 0~N 字节      │
└────────┴──────────────┴──────────┴──────────┴──────────────┘
```

**token 一个字节拆成两半：**

```
 bit: 7 6 5 4 | 3 2 1 0
      └──┬──┘   └──┬──┘
     字面量长度   匹配长度-4
     (0~15)      (0~15)
```

- 如果字面量长度 ≥ 15，token 里就写 15，然后在后面用"扩展字节"继续写。
- 扩展字节的规则：每个字节写 255 表示"还要继续"，最后一个字节 < 255 表示结束。
  （所以长度 300 = 255 + 45，写两个字节 `FF 2D`。）
- **最后一个 sequence 只有字面量，没有 offset/匹配部分。**

### 4.4 哈希表：为什么要它

如果每到一个位置都"往回翻 65535 字节"去找重复，太慢了。
于是用**哈希表**（一个 4096 项的数组）当"目录"：

```
hash(4个字节) ──► 表里的下标 ──► 记录"上次在这个指纹出现在哪个位置"
```

- 每次到一个新位置，先算 4 字节的"指纹"（哈希值）；
- 查表：如果表里有记录，就得到"候选位置 cand"；
- 再**真的比较一下**那 4 个字节（防止哈希碰撞误判）；
- 然后把当前位置写回表里（更新"最近一次出现"）。

**哈希表初始值是 -1**，表示"这个指纹还没出现过"。

### 4.5 LZ4 必须遵守的三条规则

| 规则 | 原因 |
| --- | --- |
| 匹配最短 4 字节 | 太短的匹配不划算（描述它比原样写还长） |
| offset ≤ 65535 | offset 只用 2 字节存 |
| **末尾 5 字节必须是字面量** | 解码器需要保证读匹配时不会越界 |
| **最后一个匹配的起点 ≤ len-12** | 同上，给解码器留安全余量 |

代码里就体现为两个数字：

```cpp
const uint32_t limit = len - 12u;   // 主循环最多走到这里
uint32_t maxm = len - 5u - pos;     // 匹配最多延伸到"末尾留5字节"
```

---

## 第 5 章 逐行读懂 `lz4CompressT`（压缩核心）

先看变量（非常重要，建议抄下来）：

| 变量 | 含义 | 生活比喻 |
| --- | --- | --- |
| `src` | 输入数据（在 UB 里） | 待整理的书 |
| `dst` | 输出数据（在 UB 里） | 打包好的箱子 |
| `len` | 输入有多少字节 | 一共多少本书 |
| `ht` | 哈希表 | 目录本 |
| `pos` | 当前扫描到哪 | 你正看到第几本 |
| `anchor` | 上一段"还没被写出去"的起点 | 上一个打包口 |
| `op` | 输出写到第几字节 | 箱子装到哪了 |
| `cand` | 哈希表给的候选位置 | 目录说"第几本和它一样" |

代码 + 逐行注释：

```cpp
template <uint32_t TS>                       // TS = 对齐字节数(1/2/4)
__aicore__ inline uint32_t lz4CompressT(
    const LocalTensor<uint8_t>& src, uint32_t len,
    const LocalTensor<uint8_t>& dst,
    const LocalTensor<int32_t>& ht)
{
    // ① 把哈希表全部填成 -1（用向量指令一次填 4096 个）
    Duplicate(ht, (int32_t)-1, (int32_t)HASH_SIZE);

    uint32_t op = 0u, anchor = 0u, pos = 0u;

    if (len > 12u) {                          // 太短就没必要找匹配
        const uint32_t limit = len - 12u;     // 规则：末次匹配起点 <= len-12

        while (pos <= limit) {                // ② 从前往后扫描
            uint32_t key = rd32(src, pos);    // ③ 读 4 个字节当"指纹"
            uint32_t h   = lz4hash(key);      // ④ 算哈希，得到表下标
            int32_t cand = ht.GetValue(h);    // ⑤ 查表：上次在哪见过
            ht.SetValue(h, (int32_t)pos);     // ⑥ 把当前位置写回表

            // ⑦ 判断：有候选 且 距离合法 且 4 字节真的相等
            if (cand >= 0 &&
                (pos - (uint32_t)cand) <= 65535u &&
                rd32(src, (uint32_t)cand) == key) {

                uint32_t c    = (uint32_t)cand;
                uint32_t maxm = len - 5u - pos;   // 末尾留 5 字节
                uint32_t m    = 4u;               // 已知 4 字节相同

                // ⑧ 继续往后比，能延多长延多长（按 TS 字节一步）
                while (m + TS <= maxm && eqRange<TS>(src, c + m, pos + m))
                    m += TS;

                // ⑨ 写一个序列：字面量 [anchor,pos) + 匹配(offset=pos-c, 长度 m)
                emitSeq(dst, op, src, anchor, pos - anchor, pos - c, m);

                pos += m;                         // ⑩ 跳过整段匹配
                anchor = pos;                     // 下一个"未写出的起点"
            } else {
                pos += TS;                        // 没匹配：往后挪
            }
        }
    }

    // ⑪ 收尾：把剩下的字节全当字面量写出去（没有匹配部分）
    if (anchor < len) emitSeq(dst, op, src, anchor, len - anchor, 0u, 0u);

    return op;   // 返回压缩后总字节数
}
```

### 5.1 用例子走一遍

输入 `"ABCDABCDABCD..."`（400 字节），`TS=1`：

| 步骤 | pos | 读到的 key | cand | 动作 |
| --- | --- | --- | --- | --- |
| 1 | 0 | `ABCD` | -1 | 没匹配，pos→1 |
| 2 | 1 | `BCDA` | -1 | pos→2 |
| 3 | 2 | `CDAB` | -1 | pos→3 |
| 4 | 3 | `DABC` | -1 | pos→4 |
| 5 | 4 | `ABCD` | **0** | 匹配！延伸到 m=391，写出序列 |
| 6 | 395 | — | — | 395 > limit(388)，退出循环 |
| 7 | — | — | — | 收尾：写 5 字节字面量 |

最终：`op` 约 15 字节。

### 5.2 `TS` 是干什么的

`TS` = 1 / 2 / 4，对应 nvCOMP 的 `NVCOMP_TYPE_CHAR` / `SHORT` / `INT`。

- `TS=1`：每个字节位置都可以当匹配起点（默认）。
- `TS=2`：只从**偶数**位置开始找（2 字节对齐）。
- `TS=4`：只从**4 的倍数**位置开始找。

**输出格式完全一样**（都是字节级 LZ4），只是"找匹配的粒度"不同。
代码里就是 `pos += TS` 和 `m += TS`。

---

## 第 6 章 逐行读懂 `emitSeq`（怎么写一个序列）

```cpp
__aicore__ inline void emitSeq(
    const LocalTensor<uint8_t>& dst, uint32_t& op,     // 输出 + 输出位置(会更新)
    const LocalTensor<uint8_t>& src,
    uint32_t litStart, uint32_t litLen,               // 字面量从哪来、多长
    uint32_t matchOff, uint32_t matchLen)             // 匹配距离、匹配长度
{
    // ① 算 token 的两个半字节
    uint32_t tokLit = (litLen < 15u) ? litLen : 15u;   // 字面量长度，最多写15
    uint32_t tokMat = 0u;
    if (matchLen > 0u) {
        uint32_t m = matchLen - 4u;                    // 匹配长度从4开始算，所以减4
        tokMat = (m < 15u) ? m : 15u;
    }
    // ② 写 token：高4位|低4位
    dst.SetValue(op++, (uint8_t)((tokLit << 4) | tokMat));

    // ③ 如果字面量 >= 15，写扩展字节
    if (litLen >= 15u) {
        uint32_t r = litLen - 15u;                     // 超出的部分
        while (r >= 255u) { dst.SetValue(op++, 255); r -= 255u; }
        dst.SetValue(op++, (uint8_t)r);                // 最后一个 <255
    }

    // ④ 把字面量原样拷贝过去
    for (uint32_t k = 0; k < litLen; k++)
        dst.SetValue(op + k, src.GetValue(litStart + k));
    op += litLen;

    // ⑤ 如果有匹配，写 2 字节 offset + 匹配长度扩展
    if (matchLen > 0u) {
        dst.SetValue(op++, (uint8_t)(matchOff & 0xffu));         // 低字节
        dst.SetValue(op++, (uint8_t)((matchOff >> 8) & 0xffu));  // 高字节
        if (matchLen - 4u >= 15u) {
            uint32_t r = (matchLen - 4u) - 15u;
            while (r >= 255u) { dst.SetValue(op++, 255); r -= 255u; }
            dst.SetValue(op++, (uint8_t)r);
        }
    }
}
```

**为什么长度扩展要用 255 循环？**
因为 4 位只能表示 0~15。超过 15 的部分用一串字节表示，
每个 255 表示"再加 255，还没完"，最后一个 <255 表示"结束"。

例：长度 300 → `300 - 15 = 285` → `285 = 255 + 30` → 写 `FF 1E`。

---

## 第 7 章 逐行读懂 `lz4Decompress`（解压）

解压就是**照着 token 把数据重新拼出来**，完全反过来：

```cpp
__aicore__ inline uint32_t lz4Decompress(
    const LocalTensor<uint8_t>& comp, uint32_t clen,   // 压缩数据 + 长度
    const LocalTensor<uint8_t>& out)                   // 输出缓冲
{
    uint32_t ip = 0u;   // 读压缩数据的位置
    uint32_t op = 0u;   // 写输出数据的位置

    while (ip < clen) {
        // ① 读 token
        uint8_t tok = comp.GetValue(ip++);
        uint32_t litLen = (uint32_t)(tok >> 4);        // 高4位 = 字面量长度

        // ② 如果等于15，读扩展字节
        if (litLen == 15u) {
            uint8_t b;
            do { b = comp.GetValue(ip++); litLen += b; } while (b == 255u);
        }

        // ③ 把字面量原样复制到输出
        for (uint32_t k = 0; k < litLen; k++)
            out.SetValue(op + k, comp.GetValue(ip + k));
        ip += litLen;
        op += litLen;

        // ④ 如果已经读完，说明这是最后一个"只有字面量"的序列
        if (ip >= clen) break;

        // ⑤ 读 2 字节 offset（小端：低字节在前）
        uint32_t off = (uint32_t)comp.GetValue(ip)
                     | ((uint32_t)comp.GetValue(ip + 1) << 8);
        ip += 2u;

        // ⑥ 读匹配长度（低4位 + 扩展），再加 4
        uint32_t mlen = (uint32_t)(tok & 15);
        if (mlen == 15u) {
            uint8_t b;
            do { b = comp.GetValue(ip++); mlen += b; } while (b == 255u);
        }
        mlen += 4u;

        // ⑦ 从"当前位置往前 off 字节"复制 mlen 字节
        for (uint32_t k = 0; k < mlen; k++)
            out.SetValue(op + k, out.GetValue(op + k - off));
        op += mlen;
    }
    return op;   // 解压后总字节数
}
```

### 7.1 关键点：为什么"重叠复制"也能对

假设 `off = 2`，要复制 6 字节，输出里已有 `AB`：

```
输出:  ... A B
要复制:      A B A B A B
             ↑ 从 op-off 开始，边复制边产生新的字节
```

因为循环是 `k` 从 0 递增，`out[op+k-off]` 可能正好是**刚才刚写进去的字节**，
于是自然实现了"ABABAB"这种周期重复。这正是 LZ4 能高效压缩重复内容的原因。

> 前提：`off >= 1` 且 `off <= op`（合法压缩流一定满足）。

---

## 第 8 章 Snappy 有什么不同

Snappy 和 LZ4 的**思路完全一样**（找重复），只是**打包格式**不同。

### 8.1 开头多一段"原始长度"

Snappy 的数据流最前面是 `varint(原始长度)`：

```cpp
// snapCompress 开头
uint32_t x = len;
while (x > 0x7fu) { dst.SetValue(op++, (uint8_t)(x | 0x80u)); x >>= 7; }
dst.SetValue(op++, (uint8_t)x);
```

`varint` = "每字节低 7 位存数据，最高位表示还有没有下一字节"。
比如 300 = `0xAC 0x02`。

**为什么需要它？** 因为 Snappy 解码时要知道"应该解出多少字节"，用来校验。

### 8.2 四种"元素"（对应 LZ4 的 sequence）

Snappy 没有 token，而是用**每个元素第一个字节的低 2 位**区分类型：

| 低2位 | 类型 | 长度 | 距离 |
| --- | --- | --- | --- |
| `00` | 字面量 | `(b>>2)+1`；≥60 时用额外 1~4 字节 | — |
| `01` | 短拷贝 | `((b>>2)&7)+4`（4~11） | 11 位：`((b&0xE0)<<3)|b1` |
| `10` | 中拷贝 | `(b>>2)+1`（1~64） | 16 位：`b1|(b2<<8)` |
| `11` | 长拷贝 | `(b>>2)+1` | 32 位：4 字节 |

**本项目压缩时只产生 `00` / `01` / `10`**（和 nvCOMP 一样），
但**解压时四种都支持**，所以能解官方编码器的输出。

### 8.3 `emitLiteral` 图解

```cpp
uint32_t n = l - 1u;                 // 长度先减 1（格式规定）
if (n < 60u) {
    dst[op++] = (n << 2);            // 低2位=00，高6位=长度-1
} else if (n < 256u) {
    dst[op++] = (60 << 2);           // 标记"后面还有1字节长度"
    dst[op++] = n;
}
// ... 61/62/63 分别对应 2/3/4 字节长度
for (k...) dst[op+k] = src[srcOff+k]; // 原样拷贝
```

### 8.4 `emitCopy` 图解

```cpp
while (len > 0u) {
    if (len >= 4 && len <= 11 && dist < 2048) {
        // 短拷贝：2 字节
        dst[op++] = ((dist & 0x700) >> 3) | ((len - 4) << 2) | 0x01;
        dst[op++] = (dist & 0xff);
        len = 0;
    } else {
        // 中拷贝：3 字节，每次最多 64
        uint32_t c = (len < 64) ? len : 64;
        dst[op++] = ((c - 1) << 2) | 0x02;
        dst[op++] = (dist & 0xff);
        dst[op++] = ((dist >> 8) & 0xff);
        len -= c;                     // 超过 64 就拆成多次
    }
}
```

**为什么长匹配要拆成多个 64？** 因为中拷贝的长度字段只有 6 位（最大 64）。
拆开后每次都用同一个 `dist`，解码时依然正确（因为数据是周期重复的）。

### 8.5 和 LZ4 的对比总结

| | LZ4 | Snappy |
| --- | --- | --- |
| 开头 | 无 | `varint(原始长度)` |
| 控制字节 | 1 个 token（高4/低4） | 每个元素低2位 |
| 最短匹配 | 4 | 4 |
| 最大距离 | 65535 | 32768 |
| 单次最大匹配 | 无上限（用扩展字节） | 64（更长的拆分） |
| 末尾 5 字节规则 | **有** | **无** |
| 长度扩展 | 255 循环 | literal 用 60~63 标记 |

---

## 第 9 章 核函数外壳：数据怎么进出 NPU

### 9.1 三个"容器"

| 名字 | 是什么 | 比喻 |
| --- | --- | --- |
| `TPipe` | 管理 UB 的"总管" | 工作台管理员 |
| `TQue` | 带同步的队列缓冲 | 传送带（有"进货口/出货口"） |
| `TBuf` | 普通缓冲 | 一张空桌子 |

### 9.2 固定套路（背下来）

```cpp
// 1) 申请
LocalTensor<uint8_t> t = q.AllocTensor<uint8_t>();

// 2) 从 GM 搬到 UB（MTE2）
DataCopy(t, globalTensor, 长度);

// 3) 入队（告诉框架：数据准备好了，可以插同步）
q.EnQue(t);

// 4) 出队（等到数据真的可用）
LocalTensor<uint8_t> r = q.DeQue<uint8_t>();

// 5) 用 r 做计算/标量读写
...
// 6) 用完后释放
q.FreeTensor(r);
```

**为什么必须 `EnQue/DeQue`？**
`DataCopy` 是**搬运引擎（MTE）**在干，计算是**标量/向量单元**在干，两者是异步的。
`EnQue/DeQue` 会在中间插一个"等待"，保证"数据搬完了才开始算"。
**如果漏了，你会读到全 0 或旧数据**——本项目开发时就踩过这个坑。

### 9.3 逐行读 `lz4_compress_kernel`

```cpp
__global__ __vector__ void lz4_compress_kernel(
    GM_ADDR in, GM_ADDR out, GM_ADDR outSizes,   // 三个 GM 地址
    uint32_t totalBytes, uint32_t numChunks,     // 总字节数、块数
    uint32_t dtype)                              // 1/2/4
{
    // ① 建工作台 + 4 个缓冲
    TPipe pipe;
    TQue<TPosition::VECIN, 1>  qIn;      // 输入：GM→UB
    TQue<TPosition::VECOUT, 1> qOut, qSz;// 输出：UB→GM
    TBuf<TPosition::VECCALC>   htBuf;    // 哈希表
    pipe.InitBuffer(qIn,  1, CHUNK);     // 64KB
    pipe.InitBuffer(qOut, 1, SLOT);      // 76544B
    pipe.InitBuffer(qSz,  1, 32);
    pipe.InitBuffer(htBuf, HASH_SIZE * sizeof(int32_t));

    // ② 把 GM 地址包装成 GlobalTensor（可被 DataCopy 使用）
    GlobalTensor<uint8_t> inG, outG;
    inG.SetGlobalBuffer((__gm__ uint8_t*)in, (totalBytes + 31u) & ~31u);
    outG.SetGlobalBuffer((__gm__ uint8_t*)out, numChunks * SLOT);

    LocalTensor<int32_t> ht = htBuf.Get<int32_t>();

    // ③ 并行分摊：每个 block 处理 c = bid, bid+nb, bid+2nb, ...
    const uint32_t nb = GetBlockNum();
    for (uint32_t c = GetBlockIdx(); c < numChunks; c += nb) {

        uint32_t start = c * CHUNK;                       // 这一块在输入里的起点
        uint32_t len   = (start + CHUNK <= totalBytes)
                         ? CHUNK : (totalBytes - start);   // 最后一块可能不满
        if (len == 0u) continue;
        uint32_t lenR  = (len + 31u) & ~31u;               // 32B 对齐

        // ④ 搬入
        LocalTensor<uint8_t> src = qIn.AllocTensor<uint8_t>();
        DataCopy(src, inG[start], lenR);
        qIn.EnQue(src);
        LocalTensor<uint8_t> s = qIn.DeQue<uint8_t>();

        // ⑤ 压缩（核心！）
        LocalTensor<uint8_t> dst = qOut.AllocTensor<uint8_t>();
        uint32_t clen;
        if (dtype == 2u)      clen = lz4CompressT<2u>(s, len, dst, ht);
        else if (dtype == 4u) clen = lz4CompressT<4u>(s, len, dst, ht);
        else                  clen = lz4CompressT<1u>(s, len, dst, ht);
        qOut.EnQue(dst);

        // ⑥ 记录压缩后长度（写进 32B 的槽）
        LocalTensor<int32_t> szl = qSz.AllocTensor<int32_t>();
        szl.SetValue(0, (int32_t)clen);
        qSz.EnQue(szl);
        qIn.FreeTensor(s);

        LocalTensor<int32_t> szr = qSz.DeQue<int32_t>();
        GlobalTensor<int32_t> szG;
        szG.SetGlobalBuffer((__gm__ int32_t*)outSizes, numChunks * 8);
        DataCopy(szG[c * 8], szr, 8);
        qSz.FreeTensor(szr);

        // ⑦ 搬出
        LocalTensor<uint8_t> d = qOut.DeQue<uint8_t>();
        DataCopy(outG[c * SLOT], d, SLOT);
        qOut.FreeTensor(d);
    }
}
```

### 9.4 为什么输出槽是 `SLOT = 76544`

- 压缩最坏情况会比原数据**稍微变大**（要加 token/offset）。
- 本项目取 `MAXCOMP = CHUNK + CHUNK/6 + 64 = 76522`，再向上取整到 32 的倍数 = **76544**。
- 每块固定占 76544 字节，**位置 = 块号 × 76544**，这样主机和核函数都容易算。

---

## 第 10 章 单文件模式的完整数据流

命令：`./build/lz4_demo data/t_text.bin out.lz4blocks 1`

### 10.1 主机侧：准备 5 块设备内存

```
文件名: t_text.bin (225000 字节)
numChunks = ceil(225000 / 65536) = 4

设备内存:
  dIn      ┌───────────────────────────────┐ 225000 字节（向上取整到32）
           │ 原始数据                       │
           └───────────────────────────────┘

  dComp    ┌────────┬────────┬────────┬────────┐  每块 76544B
           │ chunk0 │ chunk1 │ chunk2 │ chunk3 │
           └────────┴────────┴────────┴────────┘

  dSizes   ┌────┬────┬────┬────┐  每块 32B（前4字节放压缩长度）
           │ c0 │ c1 │ c2 │ c3 │
           └────┴────┴────┴────┘

  dOut     ┌────────┬────────┬────────┬────────┐  每块 65536B
           │ chunk0 │ chunk1 │ chunk2 │ chunk3 │
           └────────┴────────┴────────┴────────┘

  dProd    ┌────┬────┬────┬────┐  每块 32B（前4字节放实际解压长度）
           │ c0 │ c1 │ c2 │ c3 │
           └────┴────┴────┴────┘
```

### 10.2 完整流程（带编号）

```
① read_file("t_text.bin") ─► hIn (主机内存)

② aclrtMalloc ×5  ─► dIn/dComp/dSizes/dOut/dProd

③ aclrtMemcpy hIn ─► dIn

④ nb = min(numChunks, 32) = 4
   lz4_compress_kernel<<<4, nullptr, stream>>>(...)
        block0: chunk0   block1: chunk1
        block2: chunk2   block3: chunk3      （并行！）

⑤ aclrtSynchronizeStream

⑥ 拷回 dSizes ─► hSizes；拷回 dComp ─► hComp
   （hSizes[c*8] = 每块压缩后长度；hComp 里是压缩数据）

⑦ lz4_decompress_kernel<<<4, nullptr, stream>>>(...)
        block0: chunk0   block1: chunk1
        block2: chunk2   block3: chunk3

⑧ aclrtSynchronizeStream

⑨ 拷回 dOut ─► hOut；拷回 dProd ─► hProd
   recon = hOut[chunk0 的前 hProd[0] 字节]
         + hOut[chunk1 的前 hProd[1] 字节]
         + ...

⑩ 逐字节比较 recon 和 hIn
   → 全相同：打印 "ROUNDTRIP OK"
   → 有不同：打印 "ROUNDTRIP FAIL first_bad=..."

⑪ 写 out.lz4blocks：每块 = [4字节长度][压缩数据]
```

### 10.3 为什么"拼接"而不是直接用 hOut

因为 `dOut` 每块固定占 `CHUNK` 字节，但最后一块可能只解出很少字节。
所以要按 `dProd`（每块实际产出长度）来截取，再拼起来。

---

## 第 11 章 批量模式：为什么需要、怎么实现

### 11.1 为什么需要"批量"

真实使用中，你往往有**很多个独立的小文件/小数据块**要压缩，比如：

```
a.bin (100 字节)
b.bin (5000 字节)
c.bin (0 字节)
d.bin (65536 字节)
...
```

nvCOMP 的批量接口允许你**一次调用**处理它们，而且：

- 每个 item 的**长度可以不同**；
- 每个 item 的**内存在哪里都行**（不用连续）；
- 每个 item 有**自己的输出位置**；
- 返回**每个 item 压缩后的大小**。

### 11.2 怎么表示"一堆内存地址"

用**指针数组**：

```c
uint64_t inPtrs[3]  = { 地址A, 地址B, 地址C };   // 每项输入在哪
uint32_t inBytes[3] = { 100, 5000, 65536 };      // 每项多长
uint64_t outPtrs[3] = { 地址X, 地址Y, 地址Z };   // 每项输出到哪
```

这些数组本身也要放到**设备内存**里（因为核函数只能读设备内存）。

### 11.3 难点：设备端怎么"读出指针"

设备代码拿到的是 `GM_ADDR uncompPtrs`（一个地址），要取出第 i 个指针：

```cpp
// ① 把整个指针数组搬进 UB（uint64 就是 8 字节的地址）
LocalTensor<uint64_t> ipL = qpIn.AllocTensor<uint64_t>();
DataCopy(ipL, ipG, ptrCount);
qpIn.EnQue(ipL);
LocalTensor<uint64_t> ip = qpIn.DeQue<uint64_t>();

// ② 取出第 i 项的地址
uint64_t inAddr = ip.GetValue(i);

// ③ 把这个地址包装成 GlobalTensor，就能正常 DataCopy 了
GlobalTensor<uint8_t> ig;
ig.SetGlobalBuffer((__gm__ uint8_t*)inAddr, ilenR);
DataCopy(src, ig, ilenR);          // 从"那一项"读数据
```

> 这是本项目在目标机上**单独验证过**的关键技巧。

### 11.4 批量核函数的结构

```cpp
__global__ __vector__ void lz4_batch_compress_kernel(
    GM_ADDR uncompPtrs, GM_ADDR uncompBytes,   // 输入：指针数组 + 长度数组
    GM_ADDR compPtrs,   GM_ADDR compBytes,     // 输出：指针数组 + 压缩长度数组
    uint32_t batchSize, uint32_t dtype)
{
    // 1) 把指针/长度数组搬进 UB
    // 2) 每个 block 分摊 item
    for (uint32_t i = GetBlockIdx(); i < batchSize; i += GetBlockNum()) {
        uint32_t ilen = isz.GetValue(i);        // 这一项多长
        uint64_t inAddr  = ip.GetValue(i);      // 这一项从哪读
        uint64_t outAddr = op.GetValue(i);      // 这一项写到哪

        // 3) 搬入 → 压缩 → 搬出（和单文件完全一样）
        ...
        // 4) 把这一项的压缩长度写回 compBytes[i]
        DataCopy(oszG[i * 8], szr, 8);
    }
}
```

### 11.5 为什么"每项 32 字节"记录长度

多个 block 会**同时**写长度数组。如果它们写同一个 32 字节块，会互相覆盖。
所以让**每一项独占一个 32 字节槽**（`i*8` 个 int32），各写各的，互不干扰。

### 11.6 批量模式的验证

`run_all.sh` 用 40 个 item（长度 0/1/2/3/4/5/31/32/33/…/65535/65536/随机）测：

- **A**：批量压缩 → 批量解压 → 40 项逐字节比对；
- **B**：批量压缩产物 → 官方解码器；
- **C**：官方编码器产物 → NPU 批量解压。

---


### 11.7 单个 item 超过 64 KiB 怎么办

UB 每块只有约 184 KiB，不能把整个大 item 一次读进来。做法是把 item 按 64 KiB 切成若干 **block**，逐块压缩：

`
item (≤ 16 MiB)
  ├─ block0 (64 KiB) → 压缩块 → 记录 blockSizes[0]
  ├─ block1 (64 KiB) → 压缩块 → 记录 blockSizes[1]
  └─ ...
`

- 每块仍是**标准** LZ4 / Snappy 块，可被官方解码器单独解；
- 每项额外有一个 `blockSizes` 表（int32），解压时据此定位各块；
- item ≤ 64 KiB 时只有 1 块，行为与旧版完全一致；
- 上限 **16 MiB**（`MAX_ITEM_BYTES`），与 nvCOMP LZ4 同口径。

## 第 12 章 主机 `main` 的四种模式

| 命令 | 做什么 | 用到哪些核函数 |
| --- | --- | --- |
| `lz4_demo <输入> [块流] [dtype]` | 压缩→解压自检，导出块流 | `lz4_compress_kernel` + `lz4_decompress_kernel` |
| `lz4_demo --decode <块流> <输出>` | 只解压外部块流 | `lz4_decompress_kernel` |
| `lz4_demo --size <块流> <大小文件>` | 只查询每块解压大小 | `lz4_size_kernel` |
| `lz4_demo --batch <列表文件> [dtype]` | 批量压缩→解压→逐项比对 | `lz4_batch_compress_kernel` + `lz4_batch_decompress_kernel` |

Snappy 完全对称，只是没有 `dtype` 参数。

---

## 第 13 章 怎么运行 & 怎么看输出

### 13.1 第一次运行

```bash
# ① 必须先进环境（否则找不到 .so、cmake 找不到 ASC）
source /usr/local/Ascend/cann-9.0.0/set_env.sh

cd /workspace/ql/wqr/nvcomp_ascend

# ② 编译
bash scripts/build.sh
#   看到 "[build] OK -> .../build/{lz4_demo,snappy_demo}" 就成功了

# ③ 编译官方参考工具（用于交叉验证）
bash scripts/build_ref.sh
#   看到 "[build_ref] OK -> ref/tools/{ref_lz4,ref_snappy}"

# ④ 跑全量验证（约 10~15 分钟）
bash scripts/run_all.sh
```

### 13.2 单文件运行

```bash
./build/lz4_demo data/t_text.bin /tmp/t.lz4blocks 1
```

输出逐行解释：

```
[lz4] input=data/t_text.bin bytes=225000 chunks=4 chunk=65536 dtype=1
      ↑ 输入文件          ↑ 字节数   ↑ 切成4块   ↑ 每块64KB  ↑ 对齐方式
[lz4] compressed=1118 ratio=0.005  roundtrip_mismatch=0
      ↑ 压缩后1118字节  ↑ 压缩率0.5%  ↑ 不一致字节数=0
[lz4] ROUNDTRIP OK  (all 225000 bytes identical)
      ↑ 成功：225000 字节完全一致
```

- `ratio` 越小说明压得越好（`0.005` = 压到原来的 0.5%）。
- `roundtrip_mismatch=0` 是核心指标，必须为 0。

### 13.3 全量验证输出

```
lz4/d1     t_text.bin             A=OK   B=OK   C=OK   D=OK
snappy     t_zeros.bin            A=OK   B=OK   C=OK   D=OK
batch/lz4d1 list(40 items)         A=OK   B=OK   C=OK   D=NA
...
TOTAL pass=51 fail=0
```

| 列 | 含义 | 怎么算成功 |
| --- | --- | --- |
| **A** | NPU 压缩 → NPU 解压 → 和原文件比 | 逐字节相同 |
| **B** | NPU 压缩产物 → **官方解码器** | 能解出原文件 |
| **C** | **官方编码器**产物 → NPU 解压 | 能解出原文件 |
| **D** | NPU 查询解压大小 | 等于期望块大小 |
| `NA` | 空文件时 B/C 跳过 | 正常 |

### 13.4 日志文件

| 文件 | 内容 |
| --- | --- |
| `logs/cmake.log` | CMake 配置输出 |
| `logs/make.log` | 编译输出（有错误先看这里） |
| `logs/gen_testdata.log` | 生成了哪些测试文件 |
| `logs/gen_batch.log` | 批量测试集信息 |
| `logs/run_all.log` | 每个用例一行结果 |
| `logs/summary.log` | 汇总表（带表头） |

---

## 第 14 章 验证方法为什么可信

初学者常问："你自己压、自己解，万一两边都错但刚好对上呢？"

所以我们做了 **4 层验证**：

```
        ┌──────────────┐   A: 往返比对   ┌──────────────┐
        │  NPU 压缩     │ ──────────────► │  NPU 解压     │
        └──────┬───────┘                 └──────────────┘
               │ B: 交给官方解码
               ▼
        ┌──────────────┐
        │ 官方 LZ4 解码 │ ──► 和原文件比
        └──────────────┘

        ┌──────────────┐
        │ 官方 LZ4 编码 │ ──► C: 交给 NPU 解压 ──► 和原文件比
        └──────────────┘

        D: NPU 尺寸查询 ──► 和"应该多大"比（独立算法交叉验证）
```

- **A** 证明"能自洽往返"。
- **B** 证明"我们产出的确实是标准 LZ4/Snappy"（否则官方解码器会报错）。
- **C** 证明"我们的解码器不依赖自家编码器"。
- **D** 用另一条独立逻辑（只解析不输出）再算一遍大小。

> 官方实现来自 `github.com/lz4/lz4` 与 `github.com/google/snappy`，
> 用 `g++` 独立编译，和我们的代码**没有任何共享**。

---

## 第 15 章 常见报错与解决

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `error while loading shared libraries: libascendcl.so` | 没进环境 | `source /usr/local/Ascend/cann-9.0.0/set_env.sh` |
| `find_package(ASC)` 失败 | 同上 | 同上，再 `rm -rf build && bash scripts/build.sh` |
| 编译报 `maybe need a type '__ubuf__ half *'` | 对 `uint8_t` 用了 `Add/Max/Min/Compare` 等向量指令 | dav-c220 这些指令只支持 half/float；字节运算改用标量 |
| 结果全 0 / 数据不对 | 忘了 `EnQue/DeQue`，搬运算术和计算竞争 | 按第 9.2 节固定套路补齐 |
| 最后一段数据丢失 | `DataCopy` 长度不是 32 的倍数 | 用 `(len+31)&~31` 向上取整 |
| 编译报 UB 超限 | 缓冲总大小 > 184KB | 减小 `CHUNK`/`MAXB`，或按第 3 章预算表重算 |
| `batch too big (max 256)` | 批量 item 超过 `MAXB` | 拆成多次，或调大 `MAXB`（注意 UB） |
| `item too large (max 16777216)` | 单个 item > 16 MiB | 拆小（≤16 MiB），或改更大 CHUNK 后重编译 |
| `ACL ERROR ... -> 507035` | 核函数里队列不平衡（如对空队列 DeQue） | 检查每个分支的 Alloc/EnQue/DeQue/Free 是否成对 |

---

## 第 16 章 建议的代码阅读顺序

不要一上来读 900 行。按这个顺序：

1. **本文第 4 章**（LZ4 原理）——先懂"要干什么"。
2. **`lz4CompressT`**（第 5 章）——最短的核心逻辑，约 30 行。
3. **`emitSeq`**（第 6 章）——搞懂输出格式。
4. **`lz4Decompress`**（第 7 章）——反过来验证理解。
5. **`lz4_compress_kernel`**（第 9.3 节）——看懂"外壳"。
6. **`main` / `run_*`**——看懂主机怎么调用。
7. **`lz4_batch_*_kernel`**（第 11 章）——最后看批量。
8. **Snappy**（第 8 章）——对照 LZ4 看差异，很快。

---

## 第 17 章 术语表

| 术语 | 解释 |
| --- | --- |
| **压缩 / Compress** | 把数据变短 |
| **解压 / Decompress** | 还原数据 |
| **无损 / Lossless** | 还原后完全一致 |
| **LZ77** | 一类"找重复"的压缩算法，LZ4/Snappy 都基于它 |
| **字面量 / Literal** | 原样写出的字节 |
| **匹配 / Match** | "复制前面某段"的指令 |
| **offset / 距离** | 往前回退多少字节 |
| **match length** | 复制多少字节 |
| **token** | LZ4 里描述一段的控制字节 |
| **chunk** | 一个独立压缩单元 |
| **哈希 / Hash** | 把任意数据映射成一个固定范围的数字（指纹） |
| **哈希表** | 存"指纹 → 上次出现位置"的数组 |
| **varint** | 变长整数编码（Snappy 开头用） |
| **主机 / Host** | CPU 侧 |
| **设备 / Device** | NPU 侧 |
| **GM** | 设备大内存（Global Memory） |
| **UB** | 计算单元旁的高速小缓存（Unified Buffer） |
| **核函数 / Kernel** | 在设备上并行执行的函数 |
| **block** | 核函数的一个并行实例 |
| **AIV** | 昇腾的向量计算核 |
| **MTE** | 数据搬运引擎 |
| **DataCopy** | 搬运数据的接口 |
| **TQue / TBuf** | 带同步的队列缓冲 / 普通缓冲 |
| **Ascend C** | 华为的设备端编程语言 |
| **bisheng** | 编译 Ascend C 的编译器 |
| **CANN** | 华为的整套 AI 软件栈（含驱动/编译器/库） |
| **ACL** | CANN 里管理内存、启动核函数的 C 接口 |
| **HLIF** | nvCOMP 的高层容器格式（magic/校验和/偏移表） |

---

## 第 18 章 测试与性能基准（tests / benchmarks）

项目里有两种"额外的程序"，它们**复用同一份设备代码**（`src/*_kernels.h`），
只是主机侧的目的不同：

| 目录 | 目的 | 对应 nvCOMP |
| --- | --- | --- |
| `tests/` | 证明"功能对" | `tests/test_lz4.cpp` 等 |
| `benchmarks/` | 证明"跑得快不快" | `benchmarks/benchmark_*_chunked.cu` |

### 18.1 测试程序怎么工作

`test_lz4` / `test_snappy` 的每一步都是同一个套路：

```
造数据 → 拷到设备 → 压缩 → 解压 → 拷回来 → 逐字节比对 → 记一次 成功/失败
```

- 输出缓冲先填 `0xAA`：如果某段没被写到，比对时一定失败（防止"碰巧相等"）。
- 每个检查用 `TCHECK(...)` 计数，最后打印 `N passed, M failed`。
- 覆盖：0~64 每个尺寸、各种边界尺寸、5 种数据形态、dtype 1/2/4、
  多 chunk、尺寸查询、批量 API。

```bash
# 四个叶子组合：算法 × 单/批
bash scripts/run_tests_lz4_single.sh
bash scripts/run_tests_lz4_batch.sh
bash scripts/run_tests_snappy_single.sh
bash scripts/run_tests_snappy_batch.sh
# 按算法聚合 / 全部
bash scripts/run_tests_lz4.sh
bash scripts/run_tests_snappy.sh
bash scripts/run_tests.sh
# 或直接跑某个程序
./build/test_lz4 --mode single|batch|both
./build/test_snappy --mode single|batch|both
```

### 18.2 基准程序怎么工作

`bench_lz4` / `bench_snappy`：

```
生成 128MB 数据 → 切成 64KB 的 chunk
→ 预热 3 次（让缓存/驱动进入稳定状态）
→ 计时 10 次（ACL 设备事件，只统计核函数执行时间）
→ 取平均 → 算压缩率、压缩 GB/s、解压 GB/s
→ 最后再做一次往返校验
```

关键指标：

| 指标 | 公式 | 含义 |
| --- | --- | --- |
| `ratio` | 压缩后字节 / 原始字节 | 越小压得越好 |
| `compGB/s` | 原始字节 / (压缩毫秒 × 1e6) | 压缩吞吐 |
| `decGB/s` | 原始字节 / (解压毫秒 × 1e6) | 解压吞吐 |

```bash
# 四个叶子组合：算法 × 单/批
bash scripts/run_bench_lz4_single.sh    128 10 3
bash scripts/run_bench_lz4_batch.sh     128 10 3
bash scripts/run_bench_snappy_single.sh 128 10 3
bash scripts/run_bench_snappy_batch.sh  128 10 3
# 按算法聚合 / 全部
bash scripts/run_bench_lz4.sh    128 10 3
bash scripts/run_bench_snappy.sh 128 10 3
bash scripts/run_bench.sh        128 10 3
# 或直接跑某个程序
./build/bench_lz4    128 10 3 1 --mode single|batch
./build/bench_snappy 128 10 3   --mode single|batch
```

### 18.3 为什么计时要用"设备事件"而不是 `time` 命令

主机 `time` 会把"申请内存、拷数据、启动开销、等待"全算进去，
而设备事件只夹在核函数前后，测到的才是**NPU 真正干活的时间**。
这跟 nvCOMP 用 CUDA events 是同一套做法。

### 18.4 新增一个测试用例很简单

在 `tests/test_lz4.asc` 的 `main` 里加一行：

```cpp
g_group = "mycase";
TCHECK(rt(gen_random(12345, 42, 255), 1));   // 12345 字节随机数据，dtype=1
```

`rt()` 会自动做完整的"压缩→解压→逐字节比对"。

---

## 附录 A 常量速查

| 常量 | 值 | 作用 |
| --- | --- | --- |
| `CHUNK` | 65536 | 每块最大未压缩字节 |
| `HASH_BITS` | 12 | 哈希表位宽 |
| `HASH_SIZE` | 4096 | 哈希表项数 |
| `MAXCOMP` | 76522 | 压缩输出上限 |
| `SLOT` | 76544 | 每块 GM 输出槽（32B 对齐） |
| `MAX_LAUNCH` | 32 | 单次 launch 最大 block 数 |
| `MAXB` | 256 | 批量单次最大 item 数 |
| `MAX_COPY_LENGTH` | 64 | Snappy 单次拷贝上限 |
| `MAX_COPY_DISTANCE` | 32768 | Snappy 最大回溯 |

## 附录 B 函数索引（LZ4）

| 函数 | 位置 | 作用 |
| --- | --- | --- |
| `brev32` | 设备 | 32 位比特反转 |
| `lz4hash` | 设备 | 4 字节 → 哈希下标 |
| `rd32` | 设备 | 从 UB 读 4 字节 |
| `eqRange<TS>` | 设备 | 比较 TS 字节 |
| `lz4CompressT<TS>` | 设备 | **压缩核心** |
| `emitSeq` | 设备 | 写一个序列 |
| `lz4SizeOnly` | 设备 | 只算解压大小 |
| `lz4Decompress` | 设备 | **解压核心** |
| `lz4_compress_kernel` | 核 | 单文件压缩 |
| `lz4_decompress_kernel` | 核 | 单文件解压 |
| `lz4_size_kernel` | 核 | 尺寸查询 |
| `lz4_batch_compress_kernel` | 核 | 批量压缩 |
| `lz4_batch_decompress_kernel` | 核 | 批量解压 |
| `run_decode` | 主机 | `--decode` |
| `run_size` | 主机 | `--size` |
| `run_batch` | 主机 | `--batch` |
| `main` | 主机 | CLI 分派 |

Snappy 一一对应（`snapHash` / `emitLiteral` / `emitCopy` / `snapCompress` / `snapSizeOnly` / `snapDecompress` 等）。
