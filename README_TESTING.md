# 四种测试怎么跑：浮点 LZ4 / HANS / 字节 LZ4（NPU）/ CPU 字节 LZ4

本文只讲"**怎么测、怎么看、是否公平**"。原理与优化结论见
[`README.md`](README.md)、[`README_FP.md`](README_FP.md)、[`README_LZ4_CPU.md`](README_LZ4_CPU.md)。

---

## 0. 先看这张速查表

| 我想测 | 用哪个程序 | 测试数据 | 一条命令 | 结果 |
| --- | --- | --- | --- | --- |
| 浮点 LZ4（NPU） | `build/bench_fp lz4` / `lz4raw` | `data/fp/<dtype>/*.bin` | `bash scripts/run_bench_fp.sh 10 3` | `logs/bench_fp.csv` |
| HANS（NPU，官方算子） | `build/bench_fp hans` 或 `hans/bench_hans` | 同上 | 同上 / `bash scripts/run_bench_hans.sh 10 3` | `logs/bench_fp.csv` / `logs/bench_hans.csv` |
| 字节 LZ4（NPU，合成数据） | `build/bench_lz4` | 运行时内存生成 | `bash scripts/run_bench.sh 128 10 3` | `logs/bench_lz4_*.{log,csv}` |
| 字节 LZ4（NPU，单文件） | `build/bench_lz4_file` | `data/t_*.bin` | `./build/bench_lz4_file data/t_text.bin 1 10 3` | stdout；第 5 个参数传 `csv` 可得 CSV 行 |
| **CPU 字节 LZ4** | `build/bench_lz4_cpu_{up,kp}` | `data/t_*.bin` | `bash scripts/run_bench_lz4_cpu.sh 10 3` | `logs/bench_lz4_cpu.csv` |
| **CPU vs NPU（1 GB）** | 同上 | `data/large/*.bin`（5 × 1 GB = 5 GB） | `bash scripts/run_bench_lz4_cpu_large.sh 3 1` | `logs/bench_lz4_cpu_large.csv` |
| **CPU vs NPU（单个 4 GB）** | 同上 | `data/large/large_4G.bin` | `bash scripts/run_bench_lz4_cpu_4g.sh 3 1` | `logs/bench_lz4_cpu_4g.csv` |
| **CPU vs NPU（3 × 5 GB）** | 同上 | `data/large/large_5G_*.bin` | `bash scripts/run_bench_lz4_cpu_5g.sh 3 1` | `logs/bench_lz4_cpu_5g.csv` |
| 正确性（字节 LZ4/Snappy） | `build/test_lz4` / `test_snappy` | 合成 + 边界 + 随机 | `bash scripts/run_tests.sh` | 366 项全 PASS |

---

## 1. 共同约定（先读，避免误读）

```bash
# 所有 NPU 程序都需要 CANN 运行时
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /workspace/ql/wqr/nvcomp_ascend
bash scripts/build.sh          # 8 个 ASC 目标
```

1. **两条数据线，不要横比**：
   - **字节线**：`data/t_*.bin`（12 个向量）+ 运行时合成数据；
   - **浮点线**：`data/fp/`（50 个 FP16/BF16/FP32 文件）。
   两条线的 ratio 绝对值不可直接比较（数据分布完全不同）。
2. **每个基准都做位级校验**：`compress → decompress → memcmp`，只有 `verify=OK` 的数字才算数；
   任何失败都会让进程返回非 0。
3. **时间看 median**：每个数据集都输出 mean/median/min/max/sd；对比时用 `*_median`。
   `sd/mean > 5%` 说明这次测量不稳（机器噪声），别下结论。
4. **`ratio` 越低越好**；`GB/s = in_bytes / (ms × 1e6)`。
5. **warmup/iters**：默认 `iters=10, warmup=3`；同一轮对比里两边必须相同。

---

## 2. 浮点 LZ4 怎么测

浮点线有两个 LZ4 变体，**名字很像，别混**：

| codec | 含义 | 块大小 |
| --- | --- | --- |
| `lz4raw` | 原始 LZ4，无变换 | 64 KiB |
| `lz4` | **字节平面分离**（FP16/BF16 两平面、FP32 四平面）+ LZ4 | 32 KiB |

### 2.1 单文件

```bash
# 平面分离版
./build/bench_fp lz4    bf16 data/fp/bf16/bf16_weights_1MiB.bin 5 2
# 原始版（对照）
./build/bench_fp lz4raw bf16 data/fp/bf16/bf16_weights_1MiB.bin 5 2
```

命令格式：

```
bench_fp <lz4raw|lz4|hans> <fp16|bf16|fp32> <file.bin> [iters=10] [warmup=3] [csv]
```

输出（真实样本）：

```
bf16_weights_1MiB.bin   lz4  bf16  524288  1048576  0.9685  3.20  3.20  3.200  3.209  0.004  1.13  1.126  1.115  1.131  OK
```

逐列：

| 列 | 含义 |
| --- | --- |
| `dataset` | 文件名 |
| `codec` / `dtype` | `lz4` / `bf16` |
| `elems` / `in_bytes` | 元素数 / 原始字节数 |
| `ratio` | 压缩后 / 原始 |
| `c_mean c_med c_min c_max c_sd` | 压缩耗时（ms）的均值/中位/最小/最大/标准差 |
| `d_mean d_med d_min d_max` | 解压耗时（ms） |
| `OK` | 位级校验通过 |

### 2.2 全量（50 个文件 × 3 个 codec）

```bash
bash scripts/run_bench_fp.sh 10 3      # 参数：iters warmup
python3 scripts/summarize_fp.py logs/bench_fp.csv --size 8
```

- 跑之前会自动执行 `gen_fp_data.py --verify`（重算 50 个文件的 sha256），**不一致直接中止**；
- 产物：`logs/bench_fp.csv`（150 行）、`logs/bench_fp.log`（含 `[env]` 与逐次日志）；
- 耗时约 7–8 分钟。

汇总表列含义：`raw_r`/`plane_r`/`hans_r` 三种 codec 的 ratio，`best` 是三者最小，
`*_c` 是压缩耗时中位数（ms），最后一行为总体计数。

### 2.3 输出怎么读（浮点专属注意事项）

- **1 MiB 不要当主结论**：HANS 每核有 8448 B 定长尾部，1 MiB 下 ratio 会 >1。主看 **8 MiB** 行。
- **`fp32_weights` 不能引用**：它是 BF16 转来的，低 16 位尾数为 0，ratio 0.4866 是假象；
  原生 fp32 看 `f32nat`（0.9864）。
- 浮点线的 comp/dec 是**算子/内核时间**，不含 H2D/D2H。

---

## 3. HANS 怎么测

HANS 是华为官方闭源算子（CANN 9.0.0 的 aclnn 接口），本工程**只调用、不重写**。

### 3.1 推荐：走统一基准（与 LZ4 同一二进制、同一口径）

```bash
./build/bench_fp hans bf16 data/fp/bf16/bf16_weights_1MiB.bin 5 2
```

真实输出：

```
bf16_weights_1MiB.bin   hans bf16  524288  1048576  0.7716  0.23  0.23  0.225  0.238  0.005  0.05  0.046  0.045  0.047  OK
```

列与 §2.1 完全相同（`ratio` 0.7716，压缩 0.23 ms）。

### 3.2 只测 HANS 的独立基线

```bash
bash scripts/build_hans.sh
bash scripts/run_bench_hans.sh 10 3       # 先校验数据集，再跑 50 个文件
```

- 产物：`logs/bench_hans.csv`（50 行，17 列）、`logs/bench_hans.log`；
- 这是"只有 HANS"的历史基线，用于和统一基准交叉核对（ratio 应一致，耗时允许 ±噪声）。

### 3.3 硬约束（不满足会跳过，不是失败）

```
elems % 64 == 0   且   elems >= 32768
```

不满足时程序打印并**以 0 退出**：

```
[skip] HANS needs elems%64==0 && elems>=32768 (got 12345)
```

本工程的 1 MiB / 8 MiB 文件都满足（fp16/bf16 1 MiB = 524288 elems，fp32 = 262144 elems）。

### 3.4 `comp_bytes` 是怎么算的（决定 ratio 是否公平）

HANS 输出 4 个张量，我们**不用分配上界**，而是：

```
comp_bytes = mantissa + fixed_used + var_used + pdf(1024 B)
```

其中 `fixed_used / var_used` 从 `fixed` 前 512 B 的算子头里读每核实际写入量。
这样 HANS 不会因为"按上界分配"而被算得更差。

### 3.5 参数

`statistic=true`（每次自统计 pdf）、`reshuff=false`、单 stream。
`statistic=false`（复用 pdf）与 `reshuff=true` **未测**。

---

## 4. 字节 LZ4（NPU）怎么测

### 4.1 正确性

```bash
bash scripts/run_tests.sh
# 或单独：
./build/test_lz4 --mode both            # single 159 + batch 42
./build/test_lz4 --mode single --quiet
./build/test_lz4 --mode batch  --rand 20   # 随机化迭代次数
```

输出结尾：

```
== lz4: 159 passed, 0 failed ==
RESULT: PASS
```

覆盖：边界长度、全零/重复/文本/结构/随机、dtype 1/2/4、批量 1..257、非法流 8 例、
固定种子随机化。

### 4.2 合成数据性能（`bench_lz4`）

```bash
./build/bench_lz4 128 10 3 1 --mode single
#                  MB iters warmup dtype --mode single|batch|both
#                                             [--item uniform|mixed]  (batch 用)
bash scripts/run_bench.sh 128 10 3     # 两个算法、两种模式一起跑
```

输出（真实样本，8 MB 便于展示）：

```
dataset           chunks     in(MB)   comp(MB)    ratio   comp(ms)  compGB/s    dec(ms)   decGB/s verify
single/zeros         128       8.00       0.03   0.0041      4.313      1.94      2.113      3.97     OK
stats/single/zeros comp_ms mean=4.313 med=4.313 min=4.313 max=4.314 sd=0.001 | dec_ms mean=2.113 ...
e2e/single/zeros h2d=2.547ms kern=6.454ms d2h=2.681ms total=11.682ms GB/s=0.72
```

三层输出：

| 行 | 含义 |
| --- | --- |
| `single/...` / `batch/...` | **kernel-only**：压缩/解压内核时间与吞吐 |
| `stats/...` | 逐次采样的 mean/median/min/max/sd |
| `e2e/...` | 端到端：H2D + 内核 + D2H 分段与总吞吐 |

CSV：`logs/bench_lz4_<mode>_d<dtype>.csv`（21 列，含 e2e 分解）。
`--item mixed` 会让批量 item 长度不等（uniform 是等长）。

### 4.3 单文件性能（`bench_lz4_file`，供 CPU 对比用）

```bash
./build/bench_lz4_file data/t_text.bin 1 10 3
#                      file  dtype(1|2|4) iters warmup [csv]
```

真实输出：

```
t_small.bin   lz4npu   1000   0.2690   0.039   0.034   0.046   0.005   0.029   0.025   0.021 OK  (e2e=0.114ms h2d=0.013 kern=0.082 d2h=0.019)
```

它输出与 CPU 基准**同构**的 18 列 CSV（含 `e2e_ms`），所以两者能直接合并对比。

> 为什么不用 `bench_lz4` 做 CPU 对比？因为 `bench_lz4` 的数据是内存里生成的，
> 不是这 12 个文件，无法保证"同一份字节"。

---

## 5. CPU 字节 LZ4 怎么测

### 5.1 构建（一次即可）

```bash
bash scripts/build_ref_lz4_cpu.sh
```

它做三件事：

1. 克隆/切到 `ref/lz4_kp_src`（鲲鹏 `dev-1.9.4`）与 `ref/lz4_up_src`（上游 v1.9.4）；
2. 用相同 flags（`-O3 -march=native -flto -fno-plt`）各编一个静态库；
3. 链接出 `build/bench_lz4_cpu_up` 与 `build/bench_lz4_cpu_kp`。

### 5.2 单文件

```bash
taskset -c 2 ./build/bench_lz4_cpu_up data/t_text.bin lz4cpu_up 10 3
taskset -c 2 ./build/bench_lz4_cpu_kp data/t_text.bin lz4cpu_kp 10 3
#             file  codec-label  iters warmup [blockKB=64] [csv]
```

真实输出：

```
t_small.bin   lz4cpu_up   1000   0.2690   0.000   0.000   0.000   0.000   0.000   0.000   0.000 OK  (block=65536B reps=1048)
```

- 小文件会显示 `0.000 ms`：因为单样本内重复了 1048 次再平均，数值低于显示精度。
  看 CSV（`%.4f`）或汇总里的 GB/s，不要以为"没测到"。
- `reps` = `max(1, 1 MiB / size)`，只影响小于 1 MiB 的输入。

### 5.3 全量对比（12 文件 × 3 实现）

```bash
bash scripts/run_bench_lz4_cpu.sh 10 3
python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu.csv
```

产物：`logs/bench_lz4_cpu.csv`（36 行）、`.log`、`.env`（CPU/编译器/两个 commit/flags）。

汇总表：

```
dataset                    in(B)    up_r   kp_r   npu_r |   up_c    kp_c   npu_c | ...
t_big_random.bin         2097152  1.0039 1.0039 1.0039 |  10.03   12.74    0.39 | ...
```

`up` = 上游 v1.9.4（**主对照**），`kp` = 鲲鹏优化版（附加列），`npu` = 本工程内核；
`*_c` / `*_d` 是由 median 换算的 GB/s。

### 5.4 大数据集（5 × 1 GB / 单个 4 GB / 3 × 5 GB）

```bash
python3 scripts/gen_large_testdata.py          # 首次：下载 enwik9 + 生成两个 1 GB 对照
python3 scripts/gen_large_testdata.py --verify # 只校验 sha256
bash scripts/run_bench_lz4_cpu_large.sh 3 1    # iters=3, warmup=1
python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu_large.csv
```

- `data/large/`：5 个 **1 GB** + 1 个 **4 GB** + 3 个 **5 GB** 单文件（共 24 GB），哈希锁定：
  真实文本 enwik9、结构化、二进制页、混合段、随机；5 GB 那组是 text/struct/random；
- 默认 iters 比 12 向量那组小（每个文件 1 GB），全套约 1.5 分钟；
- **为什么需要它**：1 GB 同时消除"CPU 缓存效应"和"NPU launch 开销"，得到真实吞吐；
- 实测会看到 **NPU 在 enwik9 压缩上反超 CPU**（0.86 vs 0.36 GB/s），
  而随机/结构化数据 CPU 领先、解压 CPU 全面领先（详见 `README_LZ4_CPU.md` §5.3）。

另外两组单文件（同一 harness，自动 1 GiB 窗口分片）：

```bash
bash scripts/run_bench_lz4_cpu_4g.sh 3 1     # 单个 4 GB
bash scripts/run_bench_lz4_cpu_5g.sh 3 1     # 3 × 5 GB（text / struct / random）
```

### 5.5 输出怎么读

- **看 2 MiB 两行**（`t_big_random` / `t_big_struct`）：小文件 CPU 数字受缓存影响偏乐观；
- NPU 小文件吞吐被 **约 1 ms 的 launch+同步固定开销**支配（64 KiB 只有 0.06 GB/s），不代表算法；
- `kp` 的 ratio 与 `up` 不完全相同是**预期**的（鲲鹏版调整了 skip 步长）。

---

## 6. 测试集是否一样？公平吗？

### 6.1 三组对比各自的"同一测试集"范围

| 对比组 | 用的数据 | 是否同一份字节 | 块大小 | 并发 | 计时口径 | 组内可横比 |
| --- | --- | --- | --- | --- | --- | --- |
| 浮点：`lz4raw` vs `lz4` vs `hans` | `data/fp` 的**同一个文件** | ✅ 同一文件、同一块 device 内存 | 64K / 32K / 整张量 | 单 stream | 算子/内核 | ✅（但注意块大小不同） |
| HANS 独立基线 vs 统一基准 | 同上 50 个文件 | ✅ | 整张量 | 单 stream | 算子 | ✅（ratio 应一致，耗时 ±噪声） |
| 字节：NPU 合成数据 | 运行时生成（**不落盘**） | —（同一次运行内同一块内存） | 64K | 单 stream | kernel-only + e2e | 仅同参数内可比 |
| 字节：**CPU vs NPU** | `data/t_*.bin` 的**同一个文件** | ✅ 同一文件 | 都用 **64K** | CPU 单线程绑核 / NPU 单 stream | CPU wall-clock / NPU kernel-only | ✅（这是主对比） |
| 字节：**CPU vs NPU（1 GB）** | `data/large` 的**同一个文件** | ✅ 同一文件 | 都用 **64K** | 同上 | 同上 | ✅（吞吐结论用这组） |
| 浮点 vs 字节（跨线） | 不同数据 | ❌ | — | — | — | ❌ 只能比方法/量级 |

### 6.2 脚本/代码强制的公平性

1. **哈希锁定**：浮点线跑前 `gen_fp_data.py --verify` 重算 50 个文件 sha256，不一致直接中止。
2. **同一个二进制**：浮点三种 codec 都是 `build/bench_fp`；字节 CPU 对比里 NPU 也走同一套
   参数解析/统计/校验（只有库不同）。
3. **同一组参数**：同一轮里 iters/warmup 相同。
4. **同一验证**：每个文件、每个 codec 都做逐字节比对。
5. **失败不出数**：ACL/计时/校验失败 → 非零退出，不产出吞吐。
6. **HANS 不吃亏**：ratio 用算子头里的实际写入量，不用分配上界；`statistic=true` 把统计成本也算进去。

### 6.3 已知差异（会影响解读，但不是作弊）

| 差异 | 影响 | 建议 |
| --- | --- | --- |
| 浮点 LZ4 平面用 **32 KiB**、raw 用 64 KiB | 平面路径每块"首字面量"开销翻倍，repeat/sparse 反而变差 | 看 `plane vs raw` 时记住这点 |
| HANS 是整张量 + 熵编码，LZ4 是分块字典 | 算法类别不同 | 这是有意的"强弱对照"，不是同算法比实现 |
| HANS 每核 8448 B 尾部 | 1 MiB 输入 ratio 会 >1 | 主结论只看 8 MiB |
| CPU 数据在 L1/L2，NPU 从 HBM 读 | 小文件 CPU 偏快 | 看 2 MiB 行 |
| NPU 每次 launch ~1 ms | 小文件吞吐失真 | 同上 |
| CPU 只测单线程 | 多核 CPU 会更快（未测） | 不要外推到多线程 |
| 鲲鹏版改 skip 步长 | ratio 与上游不同 | 表里已单独列出 |

### 6.4 一句话结论

- **组内公平**：同一文件、同一块大小（字节线）、同一 iters、同一统计、同一校验。
- **跨组不公平也不该比**：浮点 50 文件 vs 字节 12 向量是两批完全不同分布的数据；
  它们的 ratio 只能各自内部比较。

---

## 7. CSV 字段字典

### 7.1 浮点（`logs/bench_fp.csv`，18 列）

```
dataset, codec, dtype, elems, in_bytes, comp_bytes, ratio,
comp_ms_{mean,median,min,max,sd}, dec_ms_{mean,median,min,max,sd}, verify
```

### 7.2 字节文件 / CPU（`logs/bench_lz4_cpu.csv`，18 列）

```
dataset, codec, dtype, in_bytes, comp_bytes, ratio,
comp_ms_{mean,median,min,max,sd}, dec_ms_{mean,median,min,max,sd}, e2e_ms, verify
```

> 与 7.1 的区别：没有 `elems`，多了 `e2e_ms`（CPU 行为空）。

### 7.3 NPU 合成基准（`logs/bench_lz4_<mode>_d<dtype>.csv`，21 列）

```
dataset, items, in_bytes, comp_bytes, ratio,
comp_ms_{mean,median,min,max,std}, dec_ms_{mean,median,min,max,std},
e2e_h2d_ms, e2e_kern_ms, e2e_d2h_ms, e2e_total_ms, e2e_gbs, verify
```

### 7.4 HANS 独立基线（`logs/bench_hans.csv`，17 列）

```
dataset, dtype, elems, in_bytes, comp_bytes, ratio,
comp_ms_{mean,median,min,max,sd}, dec_ms_{mean,median,min,max,sd}, verify
```

---

## 8. 常见误读

1. **"NPU 只有 0.06 GB/s"** → 那是 64 KiB 文件被 launch 开销支配；看 2 MiB，吞吐结论看 1 GB
   （1 GB 的 enwik9 上 NPU 压缩可达 0.86 GB/s，比 CPU 单线程还快）。
2. **"浮点 LZ4 比 HANS 好/差"** → 要看哪类数据：真实权重 HANS 胜，平滑/稀疏/长游程 LZ4 胜。
3. **"fp32 平面分离 0.49，完胜 HANS"** → 那是 BF16 转换导致低 16 位为 0，看 `f32nat`。
4. **"1 MiB 上 HANS ratio > 1"** → 每核固定尾部导致，主指标用 8 MiB。
5. **"CPU 0.000 ms"** → 低于显示精度，看 CSV 或 GB/s。
6. **"`dtype=2` 就是 FP16"** → 不是。LZ4 的 `dtype` 只是匹配步长；浮点语义在 `lz4_fp_*`。
7. **"ratio 一样就说明实现一样"** → 不同哈希也能压出同样大小，反之亦然；以位级校验 + 官方互操作为准。
8. **"bench_lz4 和 bench_lz4_file 是一回事"** → 前者用内存合成数据，后者读单个文件。

---

## 9. 脚本清单（按用途）

| 用途 | 脚本 |
| --- | --- |
| 编译全部 NPU 目标 | `scripts/build.sh` |
| 编译 CPU 参考（上游/鲲鹏） | `scripts/build_ref_lz4_cpu.sh` |
| 编译 HANS 基线 | `scripts/build_hans.sh` |
| 编译官方 lz4/snappy 互操作工具 | `scripts/build_ref.sh` |
| 正确性（366 项） | `scripts/run_tests.sh`（或 `run_tests_{lz4,snappy}_{single,batch}.sh`） |
| 互操作（201 项，三态判定） | `scripts/run_all.sh [--strict]` |
| 字节性能（合成） | `scripts/run_bench.sh [MB] [iters] [warmup]` |
| 浮点：LZ4 vs HANS | `scripts/run_bench_fp.sh [iters] [warmup]` |
| HANS 单独基线 | `scripts/run_bench_hans.sh [iters] [warmup]` |
| CPU vs NPU（字节） | `scripts/run_bench_lz4_cpu.sh [iters] [warmup] [core]` |
| 浮点数据生成/校验 | `scripts/gen_fp_data.py [--verify] [--sizes 1 8 64]` |
| 汇总 | `scripts/summarize_fp.py`、`scripts/summarize_lz4_cpu.py` |
| 一键回归 | `scripts/regress.sh [--with-bench]` |
| 压缩产物基线 | `scripts/check_baseline.sh save\|verify` |

### 9.1 LZ4 性能脚本（重构版，数据集可预设）

完整说明见 [`README_BENCH_LZ4.md`](README_BENCH_LZ4.md)。

| 目标 | 脚本 |
| --- | --- |
| 字节 single | `bash scripts/bench_lz4_byte_single.sh synthetic 10 3 all 128` |
| 字节 batch | `bash scripts/bench_lz4_byte_batch.sh 10 3 all 128 mixed` |
| CPU（up+kp） | `bash scripts/bench_cpu_lz4.sh vec12 10 3` |
| 浮点 LZ4（raw+plane） | `bash scripts/bench_lz4_fp.sh fp8m 10 3 both` |
| HANS | `bash scripts/bench_hans_fp.sh fp8m 10 3` |
| CPU vs NPU（字节） | `bash scripts/bench_cpu_vs_lz4.sh vec12 10 3` |
| HANS vs LZ4（浮点） | `bash scripts/bench_hans_vs_lz4.sh fp8m 10 3` |
| 数据集预设 | `bash scripts/bench_cpu_vs_lz4.sh --help` |
