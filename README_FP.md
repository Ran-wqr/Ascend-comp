# FP16 / BF16 / FP32 测试说明（LZ4 vs HANS）

本文档只讲**浮点数据**这一条线：数据从哪来、怎么跑 LZ4 和 HANS 的正确性/性能、
参数是什么、结果怎么看、公平性边界在哪。

> 只关心字节流（非浮点）的任务，请看根目录 `README.md` 与 `tests/README.md`。

---

## 0. 一句话现状

在同一份固定数据、同一个二进制、同样的计时与校验下（8 MiB，iters=10，warmup=3）：

| 代码 | 最佳 ratio 次数 | 说明 |
| --- | --- | --- |
| `lz4raw`（原始 LZ4，64 KiB 块） | 12 / 25 | 长游程、稀疏、全零、不可压缩数据 |
| `lz4`（字节平面分离，32 KiB 块） | 5 / 25 | 平滑/连续数值数据 |
| `hans`（官方 CANN 算子） | 8 / 25 | 真实权重、高斯分布、宽指数 |

**没有一方全面胜出**。真实权重（最重要的一类）目前 HANS 领先；吞吐 HANS 领先 10–100 倍。

---

## 1. 测试数据集

### 1.1 位置与规模

```
data/fp/
├── manifest.json          # 每个文件的 dtype/elems/bytes/sha256/source/note
├── fp16/  bf16/  fp32/
│   ├── <dtype>_<class>_1MiB.bin
│   └── <dtype>_<class>_8MiB.bin
```

- 共 **50 个文件 / 225 MiB**
- 每个 dtype：8 个 class × {1 MiB, 8 MiB} = 72 MiB
- 额外的 `fp32_f32nat_{1,8}MiB.bin`（仅 fp32，见 1.4）

### 1.2 8 类数据

| class | 怎么造 | 想测什么 |
| --- | --- | --- |
| `weights` | 真实 Qwen3-32B BF16 权重切片 | 最贴近实战的权值分布 |
| `gauss` | N(0, 0.02) 高斯 | 典型初始化/激活分布 |
| `uniform` | 均匀随机**字节** | 不可压缩对照组（下界） |
| `smooth` | 正弦 + 线性斜坡 | 连续/平滑数值，指数位高度重复 |
| `zeros` | 全 +0.0 | 长游程极限 |
| `repeat` | 一条 512 元素的行重复 | 跨行重复（LZ4 的强项） |
| `sparse` | 95% 为零 + 稀疏高斯 | 稀疏张量 |
| `wideexp` | 指数跨 2^-40 … 2^40 | 指数分散、尾数随机 |

### 1.3 尺寸与 HANS 约束

HANS 要求 `elems % 64 == 0` 且 `elems >= 32768`，因此：

| dtype | 1 MiB | 8 MiB |
| --- | --- | --- |
| fp16 / bf16 | 524288 elems | 4194304 elems |
| fp32 | 262144 elems | 2097152 elems |

三种 dtype 的**字节数相同**、元素数按 2:2:4 不同——这正是真实张量的表示方式，
ratio 与吞吐都按字节计算，所以可以直接横向比。

### 1.4 真实权重是怎么来的（重要）

- 源：`/workspace/ql/wqr/ENEC_910C/models/BF16/Qwen3-32B/model-00001-of-00017.safetensors`
- 张量：`model.layers.0.mlp.down_proj.weight`（BF16，5120×25600），取前 N 个 BF16 值
- `bf16`：原样
- `fp32`：`u16 << 16`，**精确无损**（但低 16 位尾数为 0，见 1.5）
- `fp16`：round-to-nearest-even，0 个溢出（极少数次正规值有舍入，manifest 里记录）
- `f32nat`：高 16 位取真实权重，低 16 位用**确定性伪随机**填充 → 模拟原生 fp32 权重的尾数噪声

### 1.5 为什么必须有 `f32nat`

`fp32_weights` 是 BF16 转来的，低 16 位尾数全 0，LZ4 平面分离能把它压到 0.4866——
**这是数据假象，不能当结论**。原生 fp32 请只看 `f32nat`（0.9864）。

### 1.6 生成与校验

```bash
python3 scripts/gen_fp_data.py                  # 生成（幂等，固定种子 20261028）
python3 scripts/gen_fp_data.py --verify         # 只校验：重算 sha256，与 manifest 比对
python3 scripts/gen_fp_data.py --sizes 1 8 64   # 追加 64 MiB（按需，默认只有 1/8 MiB）
python3 scripts/gen_fp_data.py --no-weights     # 只造合成数据（没有权重文件时）
```

`--verify` 会重新生成到临时目录再逐文件比 sha256，**任何不一致就非零退出**。
所有跑分脚本在开始前都会自动执行它——这就是"每次测试集一致"的机制。

> 注意：`--verify` 需要约 45 秒（要重新生成 225 MiB 做比对），属于预期开销。

---

## 2. 相关文件

| 路径 | 作用 |
| --- | --- |
| `scripts/gen_fp_data.py` | 数据生成 + 哈希校验 |
| `src/lz4_fp_ascend_kernels.h` | LZ4 平面分离设备核（32 KiB/块） |
| `src/lz4_ascend_kernels.h` | 原始 LZ4 设备核（64 KiB/块），`lz4raw` 用它 |
| `benchmarks/bench_fp.asc` → `build/bench_fp` | **统一基准/校验二进制**，三种 codec |
| `scripts/run_bench_fp.sh` | 一键跑完 50 个文件 × 3 个 codec |
| `scripts/summarize_fp.py` | 从 CSV 生成对比表 |
| `hans/bench_hans.cpp` → `hans/bench_hans` | 只测 HANS 的独立基线（同计时口径） |
| `scripts/build_hans.sh` / `scripts/run_bench_hans.sh` | 上面那个独立基线的构建/运行 |
| `logs/bench_fp.csv` / `.log` | 统一对比结果 |

---

## 3. 快速开始（三条命令）

```bash
# 0) 构建（会一起产出 build/bench_fp）
bash scripts/build.sh

# 1) 一键跑完：50 文件 × {lz4raw, lz4, hans}
bash scripts/run_bench_fp.sh 10 3      # 参数：iters warmup

# 2) 只看 8 MiB 的 side-by-side
python3 scripts/summarize_fp.py logs/bench_fp.csv --size 8
```

耗时参考：`run_bench_fp.sh 10 3` 约 **7.5 分钟**（含 45 s 数据校验）。

---

## 4. 正确性怎么测

### 4.1 浮点数据：每条基准都内建位级校验

`bench_fp` 对**每个文件、每个 codec** 都做完整往返：

```
压缩 → 解压 → 按字节 memcmp(原文件, 解压结果) → 不一致即 verify=FAIL 且进程非零退出
```

判据是**逐字节相等**（对浮点就是位级相等，NaN/±0 也不会被放过）。

单文件快速校验（只跑 1 次、不预热）：

```bash
# LZ4 原始
./build/bench_fp lz4raw bf16 data/fp/bf16/bf16_weights_8MiB.bin 1 0
# LZ4 平面分离
./build/bench_fp lz4    bf16 data/fp/bf16/bf16_weights_8MiB.bin 1 0
# HANS
./build/bench_fp hans   bf16 data/fp/bf16/bf16_weights_8MiB.bin 1 0
```

末尾看到 `OK` 即通过；`FAIL` 时进程返回 1，并在 stderr 打印 `bad=`（多少字节不同）、`first=`（第一个不一致的偏移）。

批量校验（任意一个失败，整体 FAIL）：

```bash
bash scripts/run_bench_fp.sh 1 0     # iters=1、warmup=0，最快的一轮全量校验
```

### 4.2 已有字节流单测（不覆盖浮点数据集）

`build/test_lz4`（single 159 项 / batch 42 项）与 `build/test_snappy`（single 134 / batch 31）
覆盖 dtype=1/2/4 的边界长度、随机化、非法流、批量等，但用的是**合成字节数据**，
不是 `data/fp/` 里的浮点文件。两者互补：

```bash
bash scripts/run_tests.sh          # 字节流单测全量
bash scripts/check_baseline.sh verify   # 非浮点路径的压缩产物哈希未变
```

> 现状：浮点路径**没有**独立的 `test_lz4fp` 单元测试，正确性由 `bench_fp` 每次运行的
> 位级校验保证（当前 150/150 全 OK）。这是已知缺口，见第 9 节。

---

## 5. 性能怎么测

### 5.1 单文件

```bash
./build/bench_fp <codec> <dtype> <file> [iters=10] [warmup=3] [csv]
# codec: lz4raw | lz4 | hans
# dtype: fp16 | bf16 | fp32
```

例：

```bash
./build/bench_fp lz4  fp32 data/fp/fp32/fp32_f32nat_8MiB.bin 10 3
./build/bench_fp hans fp32 data/fp/fp32/fp32_f32nat_8MiB.bin 10 3
```

### 5.2 全量 + 汇总

```bash
bash scripts/run_bench_fp.sh 10 3
python3 scripts/summarize_fp.py logs/bench_fp.csv --size 8
```

### 5.3 计时口径（关键）

| 项 | 做法 |
| --- | --- |
| 输入搬运 | **不计时**：数据在 warmup 之前一次性 H2D，后续所有迭代复用同一块设备内存 |
| 压缩计时 | 每次迭代：`recordEvent(start)` → 压缩算子/内核 → `recordEvent(end)` → 同步 → 取差值 |
| 解压计时 | 同上，单独一轮 |
| 预热 | 默认 3 轮 compress+decompress，热身结束后才开始采样 |
| 采样 | 默认 10 次，逐次记录，输出 mean / **median** / min / max / sd |
| 校验 | 计时全部结束后才 D2H 比对，不影响计时 |
| 失败处理 | ACL 调用、事件计时、校验任一失败 → 打印错误并返回非零，**不产出吞吐数字** |

所以表里的 `comp_ms` / `dec_ms` 是**算子/内核时间**，不含 H2D/D2H。
（`bench_lz4`/`bench_snappy` 里另有 kernel-only 与 E2E 两种口径，浮点这条线目前只有前者。）

### 5.4 ratio 口径（关键）

| codec | `comp_bytes` 怎么算 |
| --- | --- |
| `lz4raw` | 各 64 KiB 块压缩长度之和（不含任何容器头） |
| `lz4` | 各 32 KiB 块压缩长度之和（不含变换元数据/容器头） |
| `hans` | `mantissa + fixed_used + var_used + pdf(1024 B)` |

HANS 的 `fixed/var` 是**按上界分配**的，但我们不用分配量，而是从 `fixed` 前 512 B 的
算子头里读每核实际写入量（`header[8+core]`、`header[64+core]`）求和——避免把 HANS 算差。

`ratio = comp_bytes / in_bytes`，**越低越好**。

### 5.5 参数一览

| 参数 | LZ4 raw | LZ4 平面 | HANS |
| --- | --- | --- | --- |
| 块大小 | 64 KiB | 32 KiB | 整张量（算子内部分核） |
| 匹配元素步长 | `dtype`（2 或 4） | 1（平面内逐字节） | — |
| 并行 | `min(块数, 32)` 个 block | 同左 | 算子内部 |
| 压缩参数 | 哈希表 4096、`brev32(key)+(key^0xc375)` | 同左 | `statistic=true`（自统计 pdf） |
| 解压 | 标准 LZ4 block | LZ4 block + 逆平面重排 | `reshuff=false` |
| pdf | — | — | 256×int32，每次调用重新统计 |

---

## 6. 结果怎么看

### 6.1 CSV 列（`logs/bench_fp.csv`，18 列）

```
dataset, codec, dtype, elems, in_bytes, comp_bytes, ratio,
comp_ms_mean, comp_ms_median, comp_ms_min, comp_ms_max, comp_ms_sd,
dec_ms_mean, dec_ms_median, dec_ms_min, dec_ms_max, dec_ms_sd, verify
```

- `ratio`：压缩后/压缩前，**越低越好**
- `*_ms_median`：**对比优化时优先看这个**（比 mean 抗抖动）
- `*_ms_sd`：标准差；`sd/mean > 5%` 说明该次测量不稳定，先别下结论
- `verify`：必须全为 `OK`

换算吞吐：`GB/s ≈ in_bytes / (ms × 1e6)`。
例：8 MiB / 24.9 ms ≈ 0.34 GB/s。

### 6.2 side-by-side 表

```bash
python3 scripts/summarize_fp.py logs/bench_fp.csv --size 8
```

输出列含义：

| 列 | 含义 |
| --- | --- |
| `raw_r` / `plane_r` / `hans_r` | 三种 codec 的 ratio |
| `best` | 三者中 ratio 最小的那个 |
| `raw_c` / `plane_c` / `hans_c` | 压缩耗时中位数（ms） |
| `verify` | 三者全部 OK 才显示 OK |

最后一行 `best ratio count` 是总体计数。

### 6.3 当前实测（8 MiB，iters=10，warmup=3）

| 数据集 | LZ4 raw | LZ4 平面 | HANS | 最佳 |
| --- | --- | --- | --- | --- |
| bf16 / weights | 1.0033 | 0.9680 | **0.7732** | HANS |
| fp16 / weights | 1.0036 | 1.0029 | **0.9478** | HANS |
| fp32 / f32nat | 1.0039 | 0.9864 | **0.8975** | HANS |
| bf16 / gauss | 1.0032 | 0.9647 | **0.7653** | HANS |
| bf16 / smooth | 0.4852 | **0.2704** | 0.6344 | 平面 |
| fp16 / smooth | 0.8920 | **0.4426** | 0.7548 | 平面 |
| fp32 / smooth | 1.0039 | **0.6433** | 0.8346 | 平面 |
| bf16 / zeros | **0.0041** | 0.0042 | 0.5953 | raw |
| bf16 / repeat | **0.0197** | 0.0349 | 0.7906 | raw |
| bf16 / sparse | **0.1659** | 0.2464 | 0.7054 | raw |
| bf16 / uniform | **1.0039** | 1.0040 | 1.0250 | raw |
| fp32 / wideexp | 1.0039 | 1.0040 | **0.9937** | HANS |

计数：**lz4raw 12 / lz4(平面) 5 / hans 8**（n=25）。

| 耗时（8 MiB，median） | LZ4 raw | LZ4 平面 | HANS |
| --- | --- | --- | --- |
| 压缩 | 4.0–13.9 ms | 8.5–27.1 ms | 0.05–0.88 ms |
| 解压 | 1.1–4.5 ms | 6.2–9.8 ms | 0.05–0.12 ms |

> 单机单次结果，用于相对比较；不同机器/负载下绝对值会变，`median` 才可横向比。

---

## 7. 公平性：保证了什么

**强保证（脚本/代码强制）**

1. **同一份字节**：同一 dtype 下三种 codec 读同一个文件的同一块 device 内存，不存在"各生成一份"。
2. **同一个二进制**：`lz4raw` / `lz4` / `hans` 都由 `build/bench_fp` 执行，参数解析、warmup/iters、事件计时、统计、校验是同一段代码。
3. **同一组参数**：同一轮里 iters/warmup 完全相同。
4. **测试集可复现**：`manifest.json` 记录 50 个文件的 sha256；每次跑之前 `--verify` 重新生成比对，不一致直接中止。
5. **失败即失败**：ACL/计时/校验任一失败 → 非零退出，不产生数字；`run_bench_fp.sh` 汇总里出现 FAIL 即整体非零。
6. **HANS 不吃亏**：`comp_bytes` 用算子头里的实际写入量，不是分配上界；`statistic=true` 把统计成本也算进去（对 HANS 更严格而不是更宽松）。

**需要你知道的差异（不是作弊，但会影响解读）**

| 差异 | 影响 | 建议 |
| --- | --- | --- |
| LZ4 raw 用 64 KiB 块，LZ4 平面用 32 KiB 块（UB 放不下第三个 64 KiB 缓冲） | 平面路径每块"首字面量"开销翻倍，repeat/sparse 反而变差 | 看 `plane vs raw` 时记住这一点；下一步做 64 KiB 平面路径 |
| LZ4 是分块字典压缩，HANS 是整张量 + 熵编码 | 算法类别不同，属"各自自然配置"对比 | 若要比"同块大小"，需等 LZ4 支持 64 KiB 平面 |
| HANS 有每核 8448 B 定长尾部 | 1 MiB 输入下 ratio 会 >1（如 fp16/weights 1.0204） | **主结论只看 8 MiB**，1 MiB 只作压力用例 |
| LZ4 的 ratio 不含容器头；HANS 的含自身元数据（pdf 1 KiB + 尾部） | 对 LZ4 略有利 | 差 1 KiB 量级，8 MiB 下 <0.02% |
| 只测了 `statistic=true, reshuff=false` | 未覆盖 pdf 复用、`reshuff=true` | 需要模型级场景时补测 |

**没有保证的**

- 不是同一算法复杂度级别的比较（字典 vs 熵编码），这是**有意的**：HANS 就是要作为"更强但更重"的基线。
- 未做多卡/多 stream/并发、未做长稳；单次 `iters=10`。
- 未包含 H2D/D2H（这是算子级对比，不是端到端）。

---

## 8. 常见问题

**Q：为什么 `fp32_weights` 的平面 ratio 0.4866 不能引用？**
A：它由 BF16 转换而来，低 16 位尾数全 0，相当于给 LZ4 送了 16 位常数。原生 fp32 看 `f32nat`（0.9864）。

**Q：为什么 LZ4 平面分离有时比 raw 还差？**
A：主要因为块大小从 64 KiB 降到 32 KiB，每个块都要重新付一次"首个字面量"；
`repeat`/`sparse` 这类高度重复的数据对此最敏感。

**Q：为什么 HANS 在 1 MiB 上 ratio 会大于 1？**
A：HANS 每核有 8448 B 定长尾部，1 MiB 输入下元数据占比过高。所以主表用 8 MiB。

**Q：想加自己的数据怎么办？**
A：把文件放进 `data/fp/<dtype>/`，然后用 `./build/bench_fp <codec> <dtype> <file>` 单跑即可。
若要纳入"固定测试集"，把生成逻辑加进 `scripts/gen_fp_data.py` 并重新生成 manifest。

**Q：HANS 能压任意长度吗？**
A：不能。必须 `elems % 64 == 0` 且 `elems >= 32768`；不满足时 `bench_fp` 会打印
`[skip] HANS needs elems%64==0 && elems>=32768` 并以 0 退出（不是失败）。

---

## 9. 当前限制（如实清单）

1. **没有 `.lz4fp` 容器格式**：平面分离只在内存里做往返，没有落盘格式、没有 CLI、没有与官方 LZ4 的互操作。
2. **浮点路径没有独立单测**：靠 `bench_fp` 每次运行的位级校验（150/150 OK）。
3. **平面路径仍用逐字节 shuffle**：压缩耗时约 +80%、解压约 +100%，是当前主要性能损失。
4. **只测了 HANS 的一种配置**（`statistic=true, reshuff=false`）。
5. **1 MiB 不进主结论**（HANS 尾部开销）。

下一步计划（按收益）：向量化 shuffle + 64 KiB 平面块 → 位平面/指数差分等针对尾数熵的变换 →
容器 v2 + 浮点单测。
