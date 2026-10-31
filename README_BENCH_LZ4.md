# LZ4 性能测试脚本（重构版）

7 个职责单一的脚本 + 1 个共享数据集配置。旧的 `run_bench_*` 全部保留可用（Snappy 未改动）。

| # | 脚本 | 测什么 | 数据 | 默认 |
|---|---|---|---|---|
| 1 | `bench_lz4_byte_single.sh` | NPU LZ4 **single**（字节） | 合成 或 文件 | synthetic 128 MiB |
| 2 | `bench_lz4_byte_batch.sh` | NPU LZ4 **batch**（字节） | 合成 item | 128 MiB, mixed |
| 3 | `bench_cpu_lz4.sh` | **CPU** LZ4（up + kp 一个脚本） | 文件 | `vec12` |
| 4 | `bench_lz4_fp.sh` | NPU LZ4 **浮点**（lz4raw + lz4） | FP 文件 | `fp8m` |
| 5 | `bench_hans_fp.sh` | **HANS**（浮点） | FP 文件 | `fp8m` |
| 6 | `bench_cpu_vs_lz4.sh` | **CPU(up/kp) vs NPU**（字节，同数据集） | 文件 | `vec12` |
| 7 | `bench_hans_vs_lz4.sh` | **HANS vs LZ4(raw/plane)**（浮点，同数据集） | FP 文件 | `fp8m` |

共享配置：`scripts/lz4_bench_common.sh`（数据集预设表）。

---

## 1. 数据集怎么选 / 加 / 改

### 方式 A：用内置预设（最常用）

| 预设 | 内容 | 文件数 |
|---|---|---|
| `vec12` | `data/t_*.bin` | 12（KB–2 MiB） |
| `large1g` | `data/large/*_1G.bin` | 5 × 1 GB |
| `large4g` | `data/large/large_4G.bin` | 1 × 4 GB |
| `large5g` | `data/large/large_5G_*.bin` | 3 × 5 GB |
| `largeall` | `data/large/*.bin` | 9（24 GB） |
| `fp1m` / `fp8m` / `fpall` | `data/fp/...` | 25 / 25 / 50 |
| `fp16_8m` / `bf16_8m` / `fp32_8m` | 单 dtype 的 8 MiB 集 | 9 / 9 / 7 |

```bash
bash scripts/bench_cpu_vs_lz4.sh vec12 10 3      # 12 个向量
bash scripts/bench_cpu_vs_lz4.sh large1g 3 1     # 5 个 1 GB
bash scripts/bench_hans_vs_lz4.sh fp16_8m 10 3   # 只看 fp16 的 8 MiB
```

### 方式 B：临时指定 glob（不改文件）

```bash
PATTERN="data/t_big_*.bin" bash scripts/bench_cpu_vs_lz4.sh vec12 5 1
```

### 方式 C：永久新增一个预设

编辑 `scripts/lz4_bench_common.sh`，加一行：

```bash
[mycorpus]="data/mycorpus/*.bin"
```

之后所有脚本都能用 `mycorpus`。
若数据在 `data/fp` 或 `data/large` 下，脚本会自动做 manifest 哈希校验；
其它路径默认可复现，不校验。

### 查可用预设

```bash
bash scripts/bench_cpu_vs_lz4.sh --help    # 任意脚本都支持 --help
```

---

## 2. 每个脚本的参数与输出

### 1) 字节 single

```bash
bash scripts/bench_lz4_byte_single.sh [preset=synthetic] [iters=10] [warmup=3] [dtype=all] [MB=128]
```

- `preset=synthetic`：内存生成，`MB` MiB/每种 dtype；`dtype=all|1|2|4`；
- `preset=vec12|large*`：逐文件跑单缓冲内核（dtype 固定 1）；
- 输出：`logs/bench_lz4_byte_single.log`；合成模式的 CSV 是 `logs/bench_lz4_single_d{1,2,4}.csv`，
  文件模式是 `logs/bench_lz4_byte_single_<preset>.csv`。

### 2) 字节 batch

```bash
bash scripts/bench_lz4_byte_batch.sh [iters=10] [warmup=3] [dtype=all] [MB=128] [item=uniform|mixed]
```

- batch item 是**生成**的（`uniform` 每项 64 KiB；`mixed` 1–65536 B 变长）；
- 输出：`logs/bench_lz4_byte_batch.log` + `logs/bench_lz4_batch_d{1,2,4}.csv`；
- 说明：batch 没有"文件清单"驱动，因为 batch item 是相互独立的内存 buffer，不是文件。

### 3) CPU LZ4（up + kp 一个脚本）

```bash
bash scripts/bench_cpu_lz4.sh [preset=vec12] [iters=10] [warmup=3] [cpu_core=2]
```

- 两个库都用块 API、64 KiB 块、单线程、`taskset` 绑核；
- 输出：`logs/bench_cpu_lz4_<preset>.csv` + `.log`，并打印 ratio / 压缩 GB/s / 解压 GB/s。

### 4) 浮点 LZ4（lz4raw + lz4）

```bash
bash scripts/bench_lz4_fp.sh [preset=fp8m] [iters=10] [warmup=3] [codecs=both|raw|plane]
```

- `raw` = `lz4raw`（无预处理，64 KiB）；`plane` = `lz4`（字节平面分离，32 KiB）；`both` 默认；
- dtype 从路径 `data/fp/<dtype>/` 自动识别；
- 输出：`logs/bench_lz4_fp_<preset>.csv/.log`。

### 5) HANS（浮点）

```bash
bash scripts/bench_hans_fp.sh [preset=fp8m] [iters=10] [warmup=3]
```

- 与 #4 同一个二进制 `build/bench_fp`、同一套计时/校验，只是 codec=hans；
- 输出：`logs/bench_hans_fp_<preset>.csv/.log`。

### 6) CPU vs NPU（字节）

```bash
bash scripts/bench_cpu_vs_lz4.sh [preset=vec12] [iters=10] [warmup=3] [cpu_core=2]
```

- 三个实现：`lz4cpu_up` / `lz4cpu_kp` / `lz4npu`；
- 同一份文件、同一 64 KiB 块、同一 iters/warmup；
- 输出：`logs/bench_cpu_vs_lz4_<preset>.csv/.log`
  （NPU 的 `e2e_ms` 单独一列，主指标是 kernel-only）。

### 7) HANS vs LZ4（浮点）

```bash
bash scripts/bench_hans_vs_lz4.sh [preset=fp8m] [iters=10] [warmup=3]
```

- 三个 codec：`lz4raw` / `lz4` / `hans`，同一份 FP 文件、同一二进制；
- 输出：`logs/bench_hans_vs_lz4_<preset>.csv/.log`，末尾附 side-by-side 表。

---

## 3. 结果怎么看

表里三列核心指标（每行一个数据集）：

| 列 | 含义 |
|---|---|
| `*_r` | **ratio = 压缩后 / 原始**，越低越好 |
| `cGB/s` | 压缩吞吐 = `in_bytes / comp_ms_median / 1e6` |
| `dGB/s` | 解压吞吐 = `in_bytes / dec_ms_median / 1e6` |
| `best` | 同一数据集里 ratio 最小的 codec |

CSV 里还有 `mean / median / min / max / sd`：**对比优化用 median**；
`sd/mean > 5%` 说明这次测量不稳，先别下结论；`verify` 必须全是 `OK`。

---

## 4. 公平性

| 保证 | 做法 |
|---|---|
| 同一份数据 | 同一 dtype 下所有 codec 读**同一个文件**；`data/fp` `data/large` 跑前重算 sha256 |
| 同一计时 | 同一二进制/同一段代码、ACL Event、warmup + 多次采样、median |
| 同一边界 | 字节线 CPU 与 NPU 都用 **64 KiB 块**；CPU 单线程绑核 |
| 同一判定 | 逐字节往返；失败即非零，不产出吞吐 |
| 口径分开 | NPU 主指标 kernel-only，`e2e_ms`（H2D+kernel+D2H）单独列 |

已知差异（解读时注意）：FP 的 `lz4` 用 32 KiB 块（UB 限制）、HANS 是整张量、
HANS 每核 8448 B 尾部（1 MiB 下 ratio 会 >1，主看 8 MiB）、fp32 的 bf16-cast 权重不能引用
（看 `f32nat`）。

---

## 5. 按目标选脚本

| 我想… | 用 |
|---|---|
| 看 single 内核在不同数据上的吞吐 | #1（`synthetic`，`dtype=all`） |
| 看 batch 内核 + `uniform/mixed` | #2 |
| 只要 CPU 的 up/kp 数字 | #3 |
| 看 LZ4 预处理（lz4raw vs lz4）值不值 | #4 |
| 只要 HANS 的基线 | #5 |
| CPU 与 NPU 谁快（同数据） | #6（`vec12` 或 `large1g/4g/5g`） |
| 浮点上 HANS 与 LZ4 谁 ratio 好/谁快 | #7 |

---

## 6. 与旧脚本的关系

| 旧脚本 | 状态 |
|---|---|
| `run_bench_lz4_single.sh` / `_batch.sh` | 保留，等价于 #1 synthetic / #2 |
| `run_bench_lz4_cpu.sh` | 保留，仍支持 `BENCH_PATTERN/BENCH_CSV/BENCH_LOG`；#6 就是它的预设封装 |
| `run_bench_lz4_cpu_large.sh` / `_4g.sh` / `_5g.sh` | 保留；等价于 #6 的 `large1g/4g/5g` |
| `run_bench_fp.sh` | 保留；等价于 #7（`fp8m` 起） |
| `run_bench_hans.sh` | 保留（走独立 `hans/bench_hans`，用于交叉核对） |
| `run_bench.sh` / `run_bench_snappy*.sh` | **未改动**，Snappy 线保持原样 |

---

## 7. 快速自检

```bash
# 语法 + 帮助
for s in scripts/bench_*.sh; do bash -n "$s"; done
bash scripts/bench_cpu_vs_lz4.sh --help

# 小规模冒烟（每个 1-2 分钟）
bash scripts/bench_lz4_byte_single.sh synthetic 2 1 1 8
bash scripts/bench_lz4_byte_batch.sh 2 1 1 8 mixed
bash scripts/bench_cpu_lz4.sh vec12 1 0
bash scripts/bench_lz4_fp.sh fp16_8m 1 0 both
bash scripts/bench_hans_fp.sh fp16_8m 1 0
bash scripts/bench_cpu_vs_lz4.sh vec12 1 0
bash scripts/bench_hans_vs_lz4.sh fp16_8m 1 0
```
