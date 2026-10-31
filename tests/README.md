# tests/ —— 正确性测试说明

> 这个目录回答一个问题：**移植到 NPU 上的 LZ4 / Snappy，压完再解，能不能一个字节都不差地还原？**
>
> 而且**单 buffer 路径**和**指针数组批量路径**是**分开测试、可以分开运行**的。

---

## 1. 两种路径、三种运行模式

| 路径 | 用的 kernel | 数据模型 |
| --- | --- | --- |
| **single**（非批量） | `lz4_compress_kernel` / `lz4_decompress_kernel` / `lz4_size_kernel` | 一整块连续 buffer，按 64 KiB 切分 |
| **batch**（批量） | `lz4_batch_compress_kernel` / `lz4_batch_decompress_kernel` / `lz4_batch_size_kernel` | 指针数组 + 每项可变长度（可互不连续）；单项 ≤16 MiB，内部按 64 KiB 分块 |

运行模式：

```bash
./build/test_lz4 --mode single     # 只测非批量
./build/test_lz4 --mode batch      # 只测批量
./build/test_lz4 --mode both       # 两者都测（默认）
# Snappy 同样：./build/test_snappy --mode single|batch|both
```

> 两种路径用的是**同一份设备端算法**（`../src/*_kernels.h`），
> 但 kernel 外壳、寻址方式、输出布局完全不同，所以必须分别验证。

---

## 2. 文件清单

| 文件 | 作用 |
| --- | --- |
| `test_common.h` | 极简测试框架（`TCHECK` / 计数 / 汇总）+ 数据生成器 |
| `test_lz4.asc` | LZ4 测试（`build/test_lz4`） |
| `test_snappy.asc` | Snappy 测试（`build/test_snappy`） |
| `README.md` | 本文件 |

---

## 3. "正确"是怎么定义的

无损压缩只有一条标准：**解压结果 == 原始输入，长度相同、逐字节相同。**

```
原始数据 ──NPU压缩──► 压缩流 ──NPU解压──► 还原数据 ──逐字节比较──► 原始数据
```

压缩和解压都真的在 NPU 上跑（`<<<>>>` 启动核函数），比较在主机侧用 `memcmp`。

---

## 4. 非批量路径的三个测试函数

### 4.1 `rt(in, dtype)` —— 往返

```
① numChunks = ceil(n / CHUNK)，n==0 时按 1 块
② 申请 dIn / dComp / dSizes / dOut / dProd
③ dOut 全部填 0xAA（毒化），其余清零；数据拷入 dIn
④ lz4_compress_kernel → 同步 → lz4_decompress_kernel → 同步
⑤ 拷回 dOut（数据）和 dProd（每块实际产出长度）
⑥ 按 dProd 拼接：recon = 块0前op0 + 块1前op1 + ...
⑦ 判定：recon.size()==n && (n==0 || memcmp(recon,in,n)==0)
```

三个关键点：

- **两个判据**：长度抓"多/少产出"，`memcmp` 抓"长度对但内容错"；
- **0xAA 毒化**：漏写的字节一定比对失败，避免"填 0 恰好相等"；
- **按实际长度拼接**：`dOut` 每块固定 CHUNK，最后一块常不满，直接比会误判。

### 4.2 `size_query(in, dtype)`

压缩后只跑 `lz4_size_kernel`（只解析、不写输出），
判定第 c 块大小 == `min(CHUNK, n - c*CHUNK)`。

### 4.3 覆盖组（single）

| 组 | 内容 |
| --- | --- |
| `single/tiny` | 0~64 逐个 |
| `single/sizes` | 边界尺寸（65534~65537、`CHUNK+1`、`2*CHUNK+7`、300000 等） |
| `single/shapes` | 全 0 / 重复 / 文本 / 结构化 / 随机 / 小值域随机 |
| `single/typed` | dtype 1/2/4，含奇数长度 |
| `single/size` | 尺寸查询 |

---

## 5. 批量路径的四个测试函数

### 5.1 `batch_rt(items, dtype)` —— 批量往返

```
每项单独 aclrtMalloc（互不连续）
构造 输入指针 / 输出指针 / 长度 / 容量 数组 → 拷到设备
lz4_batch_compress_kernel → lz4_batch_decompress_kernel
逐项判定：实际长度 == 原始长度，且内容 memcmp == 0（0 字节项跳过 memcmp）
```

**dtype 1 / 2 / 4 都要跑**（`batch/d1`、`batch/d2`、`batch/d4`）。

### 5.2 `batch_size_query(items, dtype)`

批量压缩后跑 `lz4_batch_size_kernel`，
逐项判定"查询到的解压大小" == 该项原始长度。
对应 nvCOMP 的 `nvcompBatchedLZ4GetDecompressSizeAsync`。

### 5.3 `cross_check(data, dtype)` —— 单/批一致性

同一份数据（长度 ≤ CHUNK）分别走：

- 非批量：1 块压缩
- 批量：1 个 item 压缩

断言 **压缩长度相同、压缩字节完全相同**。
用于证明两条路径没有语义分叉。

### 5.4 `batch/maxb` —— 边界

构造 **MAXB（256）个**变长 item（含 0 字节项），跑批量往返。
验证"单次 launch 上限"附近的指针/长度数组处理。

### 5.5 覆盖组（batch）

| 组 | 内容 |
| --- | --- |
| `batch/d1` / `d2` / `d4` | 19 个变长 item（0,1,2,...,65536），三种 dtype |
| `batch/size` | 批量尺寸查询（dtype 1 和 4） |
| `batch/maxb` | 256 个变长 item |
| batch/big | 3 个 >64 KiB 的 item（100K/200K/300K），验证内部多块 |
| `batch/cross` | 单/批一致性（3 组数据，含 dtype 4） |

---

## 6. 判定与计分

```cpp
static int g_pass = 0, g_fail = 0;
static const char* g_group = "";
#define TCHECK(cond) ...            // 真则 g_pass++，否则 g_fail++ 并打印
int test_summary(const char* suite); // 打印 N passed / M failed，返回退出码
```

- 退出码：**全过 0，有失败 1**；
- 失败打印 `[FAIL] 组名 文件:行 表达式`；
- 程序开头打印 `[test_lz4] mode=single|batch|both`，避免看错模式。

---

## 6.1 输出格式

每个用例打印一行，包含数据形态、单元数、输入/压缩字节、压缩率：

```
[ OK ] grp=single/sizes     chunks=1      launch=1  n=65535    comp=65788     ratio=1.0039  comp_ms=   0.000 comp_GBs=    -- dec_ms=      -- dec_GBs=    --  data=runs(3)
[ OK ] grp=batch/d1         items=19      launch=1  n=145338   comp=8142      ratio=0.0560  comp_ms=   0.000 comp_GBs=    -- dec_ms=      -- dec_GBs=    --  data=19 items, 0..65536B (random/runs/text/zeros)
```

| 字段 | 含义 |
| --- | --- |
| `grp` | 测试组 |
| `chunks` / `items` | single 的分块数 / batch 的 item 数 |
| `launch` | 使用的 block 数（single）或子批次数（batch） |
| `n` | 输入总字节 |
| `comp` | 压缩后总字节（`size` 组为查询到的解压大小） |
| `ratio` | `comp / n`（空输入显示 `--`） |
| `comp_ms` / `comp_GBs` | 正确性测试不计时，恒为 `0.000` / `--`；吞吐只在 `benchmarks/` 输出 |
| `data` | 数据形态；batch 还会给出 item 范围；size/cross 会给出 expected/两侧压缩长度 |

`--quiet` 只打印失败用例和最终汇总。

最终汇总：

```
== lz4: 121 passed, 0 failed ==
   aggregate: n=2980960 comp=508400 ratio=0.1705
RESULT: PASS
```

> `aggregate` 汇总所有**实际压缩**用例（不含 size/cross），便于快速看整体压缩率。

## 7. 当前覆盖规模（实测）

| 程序 | single | batch | both |
| --- | --- | --- | --- |
| `test_lz4` | **121 项** | **10 项** | 131 项 |
| `test_snappy` | **104 项** | **7 项** | 111 项 |

> batch 的"项数"少，但每项内部覆盖多个 item（19 个或 256 个），
> 所以实际数据组合并不少。

---

## 8. 这些用例分别能抓什么 bug

| 可能的 bug | 被谁抓到 |
| --- | --- |
| 匹配长度 / offset 算错 | `shapes`（repeat/text/structured/random） |
| 字面量长度扩展 15/255 边界错 | `sizes` 中 >15、>255、>65535 的尺寸 |
| LZ4 末尾 5 字节 / `len-12` 规则错 | `tiny`（0~64）+ `sizes` |
| 最后一块长度处理错 | `CHUNK+1`、`2*CHUNK+7`、300000 |
| 32B 对齐导致尾部丢字节 | 1000/1001/1003/65537 等非对齐长度 |
| 输出漏写 | `0xAA` 毒化 + 长度判据 |
| 空输入崩溃 | `tiny` 的 n=0、batch 的 0 字节项 |
| **批量指针/长度数组错位** | `batch/d1..d4` |
| **批量输出槽覆盖相邻项** | `batch/maxb`、`batch/d1..d4`（0 字节与大 item 混排） |
| **批量尺寸查询错** | `batch/size` |
| **单/批语义分叉** | `batch/cross` |
| **批量 + dtype 2/4** | `batch/d2`、`batch/d4` |

---

## 9. 怎么运行

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /workspace/ql/wqr/nvcomp_ascend

bash scripts/build.sh          # 编译全部目标

# 四个叶子组合：算法 × 单/批
bash scripts/run_tests_lz4_single.sh
bash scripts/run_tests_lz4_batch.sh
bash scripts/run_tests_snappy_single.sh
bash scripts/run_tests_snappy_batch.sh
#  -> logs/test_{lz4,snappy}_{single,batch}.log

# 按算法聚合（两种模式）
bash scripts/run_tests_lz4.sh
bash scripts/run_tests_snappy.sh

# 全部
bash scripts/run_tests.sh

# 或手动
./build/test_lz4    --mode single
./build/test_lz4    --mode batch
./build/test_snappy --mode single
./build/test_snappy --mode batch
```

---

## 10. 怎么加一个用例

非批量：

```cpp
g_group = "mycase";
TCHECK(rt(gen_random(12345, 42, 255), 1));   // 12345 字节，dtype=1
```

批量：

```cpp
std::vector<std::vector<uint8_t>> items;
items.push_back(gen_text(1000));
items.push_back(gen_random(65536, 7, 255));
items.push_back(gen_zeros(0));
TCHECK(batch_rt(items, 4));                  // dtype=4
```

---

## 11. `tests/` 和 `scripts/run_all.sh` 的分工

| | `tests/`（本目录） | `scripts/run_all.sh` |
| --- | --- | --- |
| 验证 | **自洽**：NPU 压 → NPU 解 → 比原数据 | **互操作**：NPU ↔ 官方 lz4 / google-snappy |
| 依赖 | 只依赖本项目 | 需先编译 `ref/tools/ref_*` |
| 覆盖 | single + batch（分开） | 单文件、dtype、batch、size |

一句话：`tests/` 保证"自己前后一致"，`run_all.sh` 保证"和外部世界一致"。

---

## 12. 已知局限

- 不做 fuzz / 损坏流测试；
- 单设备号（device 0）、单 stream；
- 批量单次 launch **最多 256 项**（`MAXB`），更大批需主机分批；
- 单 item 长度 **≤ 16 MiB**（`CHUNK`）；
- 不校验压缩率（正确性与压缩率无关）。

---

## 13. 一句话总结

> single 与 batch 两套 kernel 各自做"压缩→解压→按实际长度拼接→长度相等 + 逐字节 memcmp"，
> 用 0xAA 毒化防漏写；batch 额外验证 dtype 1/2/4、尺寸查询、256 项边界、
> 以及"单/批对同一数据必须产出相同压缩流"。


---

## 新增测试分组（整改后）

| 分组 | 内容 | 数量 |
| --- | --- | --- |
| `single/sizes` | 边界长度 7/8/15/16/17/31/32/33/63/64/65/…/131071/131072/131073、跨 chunk | 扩展 |
| `batch/sizes` | batchSize = 1, 3, 31, 32, 33, 255, 256, **257**（257 由主机自动分批） | 8 |
| `rand/single` | 固定种子随机长度 / 内容 / 重复度，覆盖 dtype 1/2/4 | 8（`--rand` 可调） |
| `rand/batch` | 固定种子随机 item 数（1–20）与长度 | 8 |
| `invalid` | 缺失/截断 token、literal、offset，offset=0，offset>已产出，输出溢出 | 8 |

判定原则：

- 只比较实际有效数据范围；
- 设备返回长度必须 ≤ 实际分配容量；
- 设备返回 `-1`（损坏流）一律 FAIL；
- 不再把超长输出 `clamp` 后继续比较。

`test_lz4 --mode single --quiet` 与 `--mode batch --quiet` 分别覆盖；`--rand N` 控制随机迭代数。
