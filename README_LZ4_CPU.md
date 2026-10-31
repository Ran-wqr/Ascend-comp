# LZ4 的 CPU 参考实现对比（鲲鹏 vs 昇腾 NPU）

本文档说明：**同一批字节数据上，单核 CPU 的官方 LZ4 与本项目 NPU LZ4 内核的正确性和性能对比**，
以及怎么复现、怎么看、公平性边界在哪。

> 浮点（FP16/BF16/FP32）那条线（LZ4 平面分离 vs HANS）见 [`README_FP.md`](README_FP.md)。

---

## 0. 一句话结论

- 正确性：三方**全部逐字节一致**；另外跑了上游官方 `make test`，输出 `All tests completed`。
- 性能强烈依赖**数据规模与类型**（单核 CPU vs NPU kernel-only，median）：
  - 2 MiB：`t_big_random` CPU **10.0 GB/s** vs NPU **0.39**（≈26×）；`t_big_struct` CPU **7.1** vs NPU **1.81**（≈4×）
  - **1 GB（更代表真实吞吐）**：`enwik9` 压缩 **NPU 0.86 GB/s vs CPU 0.36（NPU 快 2.4×）**；
    `large_random` CPU **8.96** vs NPU **0.39**（CPU 快 23×）；`large_struct` CPU **1.72** vs NPU **0.78**
  - 解压：CPU 在三个 1 GB 文件上全部领先（1.5–13×）
- ratio：随机/文本/重复/零数据三方一致；**结构化数据上 NPU 略差**（0.0357 vs CPU 0.0322），
  原因是两者的哈希函数与匹配选择不同（不是数学错误）。

---

## 1. 对比的是哪三个实现

| 代号 | 来源 | 提交 | 说明 |
| --- | --- | --- | --- |
| `lz4cpu_up` | 上游 [lz4/lz4](https://github.com/lz4/lz4) **v1.9.4** | `5ff83968` | **主对照**：官方参考实现，BSD-2-Clause |
| `lz4cpu_kp` | [kunpengcompute/lz4](https://github.com/kunpengcompute/lz4)（BoostKit/KZL 的 LZ4 子库，`dev-1.9.4`） | `9b09896f` | 附加列：鲲鹏 NEON/预取优化版。注意其 commit 里有"增大 skip 步长换取压缩速度"，**会改变 ratio** |
| `lz4npu` | 本项目 `src/lz4_ascend_kernels.h` | 当前工作区 | 昇腾 Ascend C 内核，64 KiB 分块 |

上游与鲲鹏版都从 `gitcode.com/boostkit/lz4.git` 获取（GitHub 在本机不可达），
其中"上游"取的是**第一个鲲鹏优化 commit 的父提交**，内容即 LZ4 v1.9.4，无 KZL 标记（已校验）。

```bash
ref/lz4_up_src/   # 5ff83968, 0 处 KZL_MEMCPY
ref/lz4_kp_src/   # 9b09896f, NEON vld1q_u8/vst1q_u8 + prefetch
```

---

## 2. 测试数据（12 个字节向量）

与字节流单测共用同一批文件，无需另造数据：

| 文件 | 字节 | 特征 |
| --- | --- | --- |
| `t_empty.bin` | 0 | 空 |
| `t_1byte.bin` | 1 | 单字节 |
| `t_tiny.bin` | 23 | 极短字符串 |
| `t_small.bin` | 1000 | 小规模规律字节 |
| `t_repeat.bin` | 65536 | `ABCDABCD…` 高重复 |
| `t_zeros.bin` | 65536 | 全零 |
| `t_chunkboundary.bin` | 65569 | 跨 64 KiB 块边界 |
| `t_text.bin` | 225000 | 英文文本重复 |
| `t_random.bin` | 300000 | 均匀随机（不可压） |
| `t_struct.bin` | 348890 | 小整数 + 文本混合 |
| `t_big_struct.bin` | 2 MiB | 多块结构化 |
| `t_big_random.bin` | 2 MiB | 多块随机 |

生成：`python3 scripts/gen_testdata.py`（固定种子，已有）。

---

## 3. 怎么跑

```bash
# 1) 构建：两个 CPU 静态库 + 两个 CPU 基准 + NPU 文件模式基准
bash scripts/build.sh                 # 产出 build/bench_lz4_file
bash scripts/build_ref_lz4_cpu.sh     # 产出 build/bench_lz4_cpu_{up,kp}
                                      # 首次会克隆 ref/lz4_{kp,up}_src（约 1 分钟）

# 2) 一键对比（12 文件 × 3 实现）
bash scripts/run_bench_lz4_cpu.sh 10 3        # 参数：iters warmup（默认 CPU 绑核 2）
bash scripts/run_bench_lz4_cpu.sh 10 3 5      # 第三个参数可换绑核

# 3) 汇总表（已自动打印；也可离线重跑）
python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu.csv
```

> `scripts/build.sh` 会 `rm -rf build/`，因此重建后 CPU 基准二进制不在了；
> 再跑一次 `bash scripts/build_ref_lz4_cpu.sh` 即可（`run_bench_lz4_cpu.sh` 也会自动补建）。
> 直接跑 NPU 二进制前需要 `source /usr/local/Ascend/cann-9.0.0/set_env.sh`。

单文件调试：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh   # NPU 二进制需要 CANN 运行时
# CPU（上游 / 鲲鹏）
taskset -c 2 ./build/bench_lz4_cpu_up data/t_big_random.bin lz4cpu_up  10 3
taskset -c 2 ./build/bench_lz4_cpu_kp data/t_big_random.bin lz4cpu_kp  10 3

# NPU（kernel-only + e2e）
./build/bench_lz4_file data/t_big_random.bin 1 10 3
```

参数：`<file> <dtype> [iters] [warmup] [csv]`（NPU 的 dtype=1 表示按字节匹配）。

产物：

| 文件 | 内容 |
| --- | --- |
| `logs/bench_lz4_cpu.csv` | 36 行（12 文件 × 3 实现），18 列 |
| `logs/bench_lz4_cpu.log` | 汇总表 + 逐次运行日志 |
| `logs/bench_lz4_cpu.env` | CPU/编译器/两个 commit/编译 flags |
| `logs/build_ref_lz4_cpu.log` | 两个库的编译日志 |

---

## 4. 参数与计时口径（关键）

| 项 | CPU（up / kp） | NPU（lz4npu） |
| --- | --- | --- |
| API | `LZ4_compress_default` / `LZ4_decompress_safe` | 本项目内核（同块 API 语义） |
| 块大小 | **64 KiB** | **64 KiB**（`CHUNK`） |
| 线程 | **单线程**，`taskset -c 2` 绑核 | 单 stream，`min(块数,32)` 个 block |
| 计时 | `clock_gettime(CLOCK_MONOTONIC)` 包住整轮块循环 | ACL Event 包住 kernel |
| 预热 | 3 轮 compress+decompress | 同 |
| 采样 | 10 次，取 mean/median/min/max/sd | 同 |
| 小文件处理 | `reps = max(1, 1 MiB / size)`，单样本内重复并除以 reps | 不适用 |
| `e2e_ms` | 空（CPU 无搬运） | H2D + (压缩+解压) + D2H，单独一列 |
| `comp_bytes` | 各块 `LZ4_compress_default` 输出之和（无容器头） | 各 chunk 压缩长度之和（无容器头） |

**为什么 NPU 主指标用 kernel-only**：CPU 是纯计算、没有数据搬运；NPU 的 H2D/D2H 是部署成本，
不应算进"算法谁快"。所以主表用 kernel-only，E2E 单独给（CSV 的 `e2e_ms` 列，日志也会打印 h2d/kern/d2h 分解）。

---

## 5. 结果（iters=10，warmup=3，median）

### 5.1 代表性大文件（2 MiB）

| 文件 | ratio (up / kp / npu) | 压缩 GB/s (up / kp / npu) | 解压 GB/s (up / kp / npu) | NPU E2E |
| --- | --- | --- | --- | --- |
| `t_big_random` | 1.0039 / 1.0039 / 1.0039 | **10.03 / 12.74 / 0.39** | **33.18 / 33.39 / 2.02** | 7.03 ms（0.30 GB/s） |
| `t_big_struct` | 0.0322 / **0.0303** / 0.0357 | **7.14 / 7.30 / 1.81** | **10.02 / 12.94 / 1.96** | 2.80 ms（0.75 GB/s） |

### 5.2 全量（12 文件）

| 文件 | in(B) | up_r | kp_r | npu_r | up_c | kp_c | npu_c | up_d | kp_d | npu_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t_empty | 0 | 0 | 0 | 0 | – | – | – | – | – | – |
| t_1byte | 1 | 2.0000 | 2.0000 | 2.0000 | 0.01 | 0.01 | ~0 | – | – | – |
| t_tiny | 23 | 1.0870 | 1.0870 | 1.0870 | 0.23 | 0.23 | ~0 | – | – | – |
| t_small | 1000 | 0.2690 | 0.2690 | 0.2690 | 2.50 | 3.33 | 0.02 | – | 10.00 | 0.03 |
| t_repeat | 65536 | 0.0041 | 0.0041 | 0.0041 | 16.38 | 16.38 | 0.06 | 3.64 | 2.89 | 0.04 |
| t_zeros | 65536 | 0.0041 | 0.0041 | 0.0041 | 15.60 | 15.60 | 0.06 | 3.64 | 2.90 | 0.12 |
| t_chunkboundary | 65569 | 0.0085 | 0.0085 | 0.0085 | 14.25 | 14.57 | 0.06 | 38.57 | 21.15 | 0.06 |
| t_text | 225000 | 0.0049 | 0.0049 | 0.0049 | 15.41 | 15.52 | 0.20 | 6.03 | 7.33 | 0.19 |
| t_random | 300000 | 1.0039 | 1.0039 | 1.0039 | 10.95 | 15.87 | 0.06 | 38.96 | 39.47 | 0.29 |
| t_struct | 348890 | 0.3519 | 0.3513 | **0.4032** | 0.79 | 0.95 | 0.15 | 3.42 | 5.45 | 0.29 |
| t_big_struct | 2097152 | 0.0322 | **0.0303** | 0.0357 | 7.14 | 7.30 | 1.81 | 10.02 | 12.94 | 1.96 |
| t_big_random | 2097152 | 1.0039 | 1.0039 | 1.0039 | 10.03 | 12.74 | 0.39 | 33.18 | 33.39 | 2.02 |

`c`/`d` = 压缩/解压吞吐（GB/s，由 median 换算）。压缩耗时胜者：**CPU 12 / NPU 0**。

---

### 5.3 大数据集：5 × 1 GB（推荐用于吞吐结论）

```bash
bash scripts/run_bench_lz4_cpu_large.sh 3 1      # 参数：iters warmup（默认 3/1，每个文件 1 GB）
python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu_large.csv
```

数据（`data/large/`，**5 个 × 1,000,000,000 B = 5 GB**，sha256 见 `data/large/manifest.json`）：

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `enwik9.bin` | mattmahoney.net（enwik9.zip，322,592,222 B） | 真实维基百科 XML，文本 |
| `large_struct_1G.bin` | 生成器 | 每 64 KiB：48 KiB 重复记录 + 16 KiB 新噪声 |
| `large_binary_1G.bin` | 生成器 | 每 4 KiB 页：2 KiB 模板 + 2 KiB 噪声（二进制/DB 类） |
| `large_mixed_1G.bin` | 生成器 | 250 MB 文本 + 250 MB 记录 + 250 MB 随机 + 250 MB 全零 |
| `large_random_1G.bin` | 生成器（固定种子） | 不可压缩对照 |

> **单文件如何突破 4 GiB**：NPU 内核用 32 位偏移，所以 `bench_lz4_file` 把文件按
> **1 GiB 窗口**分片处理（每个窗口的 `chunks × SLOT` 始终 < 4 GiB）。分片点在 64 KiB 边界，
> 而 LZ4 块之间没有状态共享，因此**压缩字节流与 ratio 和"一次性处理"完全相同**；
> 输出里的 `windows=N` 会显示实际分片数。当前上限设为 8 GiB（约 3 倍显存）。

实测（iters=3，warmup=1，median）：

| 文件 | ratio (up / kp / npu) | 压缩 GB/s (up / kp / npu) | 解压 GB/s (up / kp / npu) | NPU E2E |
| --- | --- | --- | --- | --- |
| enwik9 | 0.5102 / 0.5075 / **0.5489** | **0.36 / 0.44 / 0.86** | 2.28 / 3.05 / 1.55 | 1.88 s（0.53 GB/s） |
| large_binary | 0.5086 / 0.5086 / 0.5108 | **4.37 / 6.64 / 0.65** | 12.55 / 11.74 / 2.03 | 2.10 s（0.48 GB/s） |
| large_mixed | 0.5114 / 0.5107 / 0.5225 | **1.04 / 1.21 / 0.71** | 4.26 / 4.55 / 1.95 | 1.98 s（0.51 GB/s） |
| large_random | 1.0039 / 1.0039 / 1.0039 | **8.96 / 10.51 / 0.39** | 20.17 / 20.15 / 2.03 | 3.12 s（0.32 GB/s） |
| large_struct | 0.4699 / 0.4698 / 0.4699 | **1.73 / 1.67 / 0.78** | 6.94 / 7.27 / 1.55 | 1.99 s（0.50 GB/s） |

**压缩胜者：CPU 4 / NPU 1**（唯一赢的是 enwik9，见下）。

### 5.4 单个 4 GB 文件

```bash
python3 scripts/gen_large_testdata.py --only text4g --size 4000000000
bash scripts/run_bench_lz4_cpu_4g.sh 3 1        # 约 3 分钟
python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu_4g.csv
```

`data/large/large_4G.bin` = **4,000,000,000 B**（enwik9 文本重复到 4 GB，
sha256 `bd46a0faa398d04e…`）。实测（iters=3，warmup=1，median，4 个 1 GiB 窗口）：

| ratio (up / kp / npu) | 压缩 GB/s (up / kp / npu) | 解压 GB/s (up / kp / npu) | NPU E2E |
| --- | --- | --- | --- |
| 0.5101 / 0.5075 / **0.5488** | **0.36 / 0.44 / 0.86** | 2.28 / 3.04 / 1.55 | 7.48 s（0.53 GB/s） |

结论与 1 GB enwik9 一致且更稳：**NPU 压缩 0.86 GB/s，是 CPU 单线程的 2.4 倍**；
解压 CPU 仍领先（2.28 vs 1.55）。ratio 差异仍来自哈希/匹配策略（+7.6%）。

**重要结论（和 2 MiB 不同）**

1. **NPU 压缩在 enwik9 上反超 CPU**：0.86 vs 0.36 GB/s（2.4×）。原因是 NPU 用 32 个 AIV block
   并行处理 16384 个 chunk，而 CPU 是单线程；文本匹配长、哈希探测少，并行度能吃满。
2. **CPU 在随机/结构化上仍领先**：随机数据几乎无匹配、纯扫描带宽（23×）；结构化 2.2×。
3. **解压 CPU 全面领先**（1.5–13×），NPU 解压受串行依赖链限制。
4. **NPU 的 ratio 略差**：enwik9 0.5489 vs 0.5102（+7.6%），来自不同哈希/匹配策略。
5. **E2E**：enwik9 上 NPU 1.88 s 甚至快于 CPU 单线程 comp+dec 的 3.23 s；随机数据 CPU 仍快约 19×。

### 5.5 三个 5 GB 单文件（文本 / 结构化 / 随机）

```bash
python3 scripts/gen_large_testdata.py --size 5000000000 \
    --only text5g --only struct5g --only random5g
bash scripts/run_bench_lz4_cpu_5g.sh 3 1       # 约 7 分钟
python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu_5g.csv
```

| 文件 | 字节 | 形态 | sha256（前 16 位） |
| --- | ---: | --- | --- |
| `large_5G_text.bin` | 5,000,000,000 | enwik9 真实文本重复 | `50bf96fe1b2c5a66` |
| `large_5G_struct.bin` | 5,000,000,000 | 每 64 KiB：48 KiB 记录 + 16 KiB 噪声 | `759de6667b6d8cd3` |
| `large_5G_random.bin` | 5,000,000,000 | 确定性随机（不可压缩） | `2944bdbaf46844e9` |

实测（iters=3，warmup=1，median；每个文件 5 个 1 GiB 窗口）：

| 文件 | ratio (up / kp / npu) | 压缩 GB/s (up / kp / npu) | 解压 GB/s (up / kp / npu) | NPU E2E |
| --- | --- | --- | --- | --- |
| text | 0.5101 / 0.5075 / **0.5488** | **0.36 / 0.44 / 0.86** | 2.28 / 3.04 / 1.55 | 9.37 s |
| struct | 0.4699 / 0.4698 / 0.4699 | **1.72 / 1.67 / 0.78** | 6.94 / 7.27 / 1.55 | 9.94 s |
| random | 1.0039 / 1.0039 / 1.0039 | **9.49 / 10.92 / 0.39** | 19.80 / 19.83 / 2.03 | 15.61 s |

结论与 1 GB / 4 GB 完全一致：**文本上 NPU 压缩领先 2.4 倍**；结构化/随机 CPU 领先；
解压 CPU 全面领先。5 GB 只让数字更稳（sd 更小），不改变方向。

## 6. 怎么解读

1. **谁快取决于数据**：小文件（2 MiB，缓存内）CPU 压缩快 4–26×；
   但 **1 GB 的 enwik9 上 NPU 压缩反超 2.4×**（32 block 并行 vs CPU 单线程）。
   随机/结构化数据 CPU 仍领先，解压 CPU 全部领先。不要用单一数据集下结论。
2. **小文件不要看 NPU 吞吐**：64 KiB 以下 NPU 每次 launch+同步约 **1 ms** 固定开销，
   0.06 GB/s 是开销主导；2 MiB 已能反映算法，**1 GB 最接近真实吞吐**。
3. **CPU 小文件数字偏乐观**：文件在 L1/L2 里，16 GB/s 级别的压缩速度是缓存效应；
   1 GB 行的数字（0.36–8.96 GB/s）才是 HBM/内存带宽下的真实值。
4. **ratio 不是完全一致**：
   - 随机/文本/重复/零/边界：三方完全相同（0.0039 差异级）；
   - 结构化：NPU 0.0357 / 0.4032 vs CPU 0.0322 / 0.3519。原因：NPU 用
     `brev32(key)+(key^0xc375)` 哈希、CPU 用 LZ4 官方哈希，且匹配选择/最小匹配规则不同；
     两者都是**合法 LZ4 块**，只是"压缩策略"不同。
   - 鲲鹏版在 `t_big_struct` 上 ratio 反而更好（0.0303 vs 0.0322），说明它的 skip 调整
     并非单向变差；差异都如实列在表里。
5. **鲲鹏优化版 vs 上游**：大文件压缩快 **27%**（12.74 vs 10.03 GB/s，随机）和 2%（结构），
   小文件基本持平；`t_struct` 上 0.95 vs 0.79 GB/s。

---

## 7. 公平性

**保证**

- 同一份文件、同一 64 KiB 块、同一 iters/warmup、同一统计口径（median）；
- CPU 单线程 + `taskset` 绑核；NPU 单 stream；
- 两边都**不含容器头**，`comp_bytes` 都是分块压缩长度之和；
- 三方每文件都做**逐字节**往返校验，失败即非零退出；
- NPU 主指标 kernel-only，E2E 单列，不与 CPU 混算；
- 记录 CPU 型号/核数/编译 flags/两个 commit（`logs/bench_lz4_cpu.env`）。

**已知差异（会影响解读）**

| 差异 | 影响 |
| --- | --- |
| CPU 数据在缓存里，NPU 从 HBM 读 | 小文件 CPU 偏快；2 MiB 行更公平 |
| CPU 单核 2.9 GHz，NPU 是 AI Core 标量单元 | 不是同一种执行模型，结论是"当前实现谁快"，不是"硬件上限" |
| NPU 每次 launch ~1 ms 固定成本 | 小文件吞吐失真 |
| 鲲鹏版改了 skip 步长 | 其 ratio 与上游不同属预期 |
| CPU 只测单线程 | 多核 CPU 吞吐（未测）会更高，但那是另一维度 |

**未覆盖**

- CPU 多线程 / 多 NUMA 扩展性；
- 大文件（>2 MiB）与冷缓存（未做 cache flush）；
- KAE 硬件压缩（本机无 `/dev/hisi_zip*`，不可用）；
- H2D/D2H 与计算重叠（NPU 单 stream 串行）。

---

## 8. 正确性证据

1. **逐文件位级校验**：CPU 两个实现与 NPU 都在 `bench_*` 内部做
   `compress → decompress → memcmp(原文件, 解压结果)`，36/36 全部 `OK`。
2. **官方测试套件**：对上游 v1.9.4 执行 `make -C ref/lz4_up_src test`，
   末尾输出 `All tests completed`（419 行日志，包含 frame/块/错误路径用例）。
3. **ratio 交叉核对**：同一文件上游 CLI 的 `lz4 -b1` 与本 harness 量级一致
   （本 harness 64 KiB 块，CLI 用 256 KiB，故绝对值不同）。

---

## 9. 目录与文件

```
ref/lz4_up_src/                 # 上游 v1.9.4（含 .git，便于核对 commit）
ref/lz4_kp_src/                 # 鲲鹏 dev-1.9.4
benchmarks/bench_lz4_cpu.c      # CPU 基准（两个库共用一份源码）
benchmarks/bench_lz4_file.asc   # NPU 单文件基准（kernel-only + e2e）
scripts/build_ref_lz4_cpu.sh    # 克隆/切换/编译两个库 + 两个 CPU 基准
scripts/run_bench_lz4_cpu.sh    # 12 文件 × 3 实现 → CSV/日志
scripts/summarize_lz4_cpu.py    # side-by-side 表
data/large/                    # 3 × 1 GB：enwik9 + struct + random（manifest 锁哈希）
scripts/gen_large_testdata.py  # 生成/校验 1 GB 集（enwik9 自动下载）
scripts/run_bench_lz4_cpu_large.sh  # 1 GB 版 CPU vs NPU
logs/bench_lz4_cpu{,_large}.{csv,log,env}
```

---

## 10. 后续可以做什么

1. **CPU 多线程**：块级并行（每线程独立 64 KiB 块），看鲲鹏多核扩展；与 NPU 的 32 block 对比。
2. **冷缓存口径**：每轮前 flush L2/L3，得到 HBM 带宽受限下的 CPU 数字，更接近 NPU 的内存条件。
3. **更大文件**（64 MiB+）与 NUMA 绑定，减少缓存与调度噪声。
4. NPU 侧：向量化哈希/匹配扩展，把 0.39–1.81 GB/s 往上推，再与 CPU 单核比较。
