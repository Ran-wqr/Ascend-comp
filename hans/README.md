# HANS baseline（官方 CANN 算子，非自研实现）

HANS 是华为在 Ascend 上的闭源无损压缩算子。本项目**不重新实现**它，而是直接调用
CANN 9.0.0 提供的 aclnn 接口，把它作为 LZ4 优化的对照基线。

## 1. 接口与环境

| 项 | 值 |
| --- | --- |
| CANN | 9.0.0（`aclnnop/aclnn_hans_encode.h`、`aclnnop/aclnn_hans_decode.h`） |
| 库 | `libopapi_math.so`（另有 `es_math` 图模式接口） |
| 设备 | npu-smi 报告 `Ascend910`，dav-2201 |
| dtype | `ACL_FLOAT16` / `ACL_BF16` / `ACL_FLOAT` |
| 形态 | `encode(input, pdf:int32[1,256], statistic, reshuff) -> (pdf, mantissa, fixed, var)`；`decode(mantissa, fixed, var, pdf, reshuff) -> out` |

### 硬约束（来自 torch_npu 文档 + kernel 头信息）

1. `elems % 64 == 0` 且 `elems >= 32768`；
2. `mantissa` 字节数 = `elems * (elemSize - 1)`（精确）；
3. `fixed` 字节数 ≥ 512（前 512 B 是算子头信息）；
4. `fixed + var` 字节数 ≥ `input + input/64 + 8448 * Vector核数`；
5. `reshuff=true` 时 `var` 失效，且 `fixed` 需容纳整个压缩上界。

## 2. 压缩字节口径

```
compressed_bytes = mantissa_bytes                      (精确)
                 + Σ header[8 + core]   (fixed_used)   (每个 core 的实际写入量)
                 + Σ header[64 + core]  (var_used)
                 + 256 * 4              (pdf = 1 KiB)
ratio = compressed_bytes / input_bytes
```

`header` 位于 `fixed` 的前 512 B，字段含义见
`opp/.../ascendc/hans_encode/hans_encode_base.h`。这样得到的是**实际使用字节**，
不是"按上界分配"的字节，避免把 HANS 算得比实际更差。

## 3. 构建与运行

```bash
bash scripts/build_hans.sh            # g++ + aclnn（与 Ascend C kernel 构建分开）
bash scripts/run_bench_hans.sh 10 3   # iters=10 warmup=3，先校验数据集再跑
```

输出：

- `logs/bench_hans.log`：可读表格
- `logs/bench_hans.csv`：结构化结果（17 列，含 mean/median/min/max/sd 与 verify）

脚本在运行前执行 `python3 scripts/gen_fp_data.py --verify`，**测试集哈希不一致直接中止**。

## 4. 固定测试集

`data/fp/`，48 个文件、216 MiB，`manifest.json` 记录每个文件的
dtype / elems / bytes / sha256 / source / note。

- 8 类数据：`weights`（真实 Qwen3-32B BF16 权重切片）、`gauss`、`uniform`、
  `smooth`、`zeros`、`repeat`、`sparse`、`wideexp`
- 每类 1 MiB + 8 MiB
- 三种 dtype 的**字节数相同**（1/8 MiB），元素数按 2:2:4 自然不同
- fp32 由 BF16 精确转换（左移 16 位），fp16 转换 0 个溢出（极小值落入 fp16 次正规）

## 5. Baseline 结果（8 MiB，iters=10，warmup=3，median）

| dtype | 数据集（8 MiB） | HANS | LZ4 raw | LZ4 平面 | 最佳 |
| --- | --- | --- | --- | --- | --- |
| bf16 | weights | **0.7732** | 1.0033 | 0.9680 | HANS |
| bf16 | gauss | **0.7653** | 1.0032 | 0.9647 | HANS |
| bf16 | smooth | 0.6344 | 0.4852 | **0.2704** | 平面 |
| bf16 | zeros | 0.5953 | **0.0041** | 0.0042 | raw |
| bf16 | repeat | 0.7906 | **0.0197** | 0.0349 | raw |
| bf16 | sparse | 0.7054 | **0.1659** | 0.2464 | raw |
| bf16 | wideexp | **0.9703** | 1.0039 | 1.0039 | HANS |
| bf16 | uniform | 1.0250 | **1.0039** | 1.0040 | raw |
| fp16 | weights | **0.9478** | 1.0036 | 1.0029 | HANS |
| fp16 | smooth | 0.7548 | 0.8920 | **0.4426** | 平面 |
| fp16 | gauss | **0.9446** | 1.0039 | 1.0035 | HANS |
| fp32 | f32nat（原生仿真） | **0.8975** | 1.0039 | 0.9864 | HANS |
| fp32 | gauss | **0.8915** | 1.0039 | 0.9848 | HANS |
| fp32 | smooth | 0.8346 | 1.0039 | **0.6433** | 平面 |
| fp32 | wideexp | **0.9937** | 1.0039 | 1.0040 | HANS |
| fp32 | weights（bf16 cast） | 0.8975 | 0.9967 | 0.4866 | **不作结论**¹ |

¹ fp32 权重由 BF16 转换，低 16 位尾数为 0 → 0.4866 是数据假象；原生 fp32 看 `f32nat`。

25 个 8 MiB 数据集最佳 ratio 计数：**LZ4 raw 12 / LZ4 平面 5 / HANS 8**。

> 三列都由**同一二进制** `build/bench_fp` 产生（codec = `lz4raw` / `lz4` / `hans`），
> 输入字节、warmup/iters、ACL Event 计时、统计与位级校验完全一致。

### 吞吐（8 MiB，median）

| dtype | 数据 | HANS comp | HANS dec | LZ4(平面) comp | LZ4(平面) dec |
| --- | --- | --- | --- | --- | --- |
| bf16 | weights | 0.67 ms (12.5 GB/s) | 0.10 ms (84 GB/s) | 24.9 ms (0.34 GB/s) | 8.7 ms (0.97 GB/s) |
| fp16 | weights | 0.63 ms (13.4 GB/s) | 0.12 ms (70 GB/s) | 27.0 ms (0.31 GB/s) | 8.3 ms (1.01 GB/s) |
| fp32 | f32nat | 0.50 ms (16.7 GB/s) | 0.07 ms (115 GB/s) | 26.0 ms (0.32 GB/s) | 8.4 ms (1.00 GB/s) |

吞吐差距 10–100 倍，主要来自 LZ4 的串行 LZ77 解析（AIV 标量单元）与逐字节平面重排；
平面重排还把 LZ4 压缩耗时抬高约 2 倍（8 MiB bf16 weights：13.9 ms → 24.9 ms）。
这是下一阶段优化的主战场：向量化 shuffle、更大的块、以及针对尾数熵的变换。

## 6. 结论（对后续优化的影响）

1. **1 MiB 不适合做主对比**：HANS 每个 core 有 8448 B 定长尾部，1 MiB 输入下
   `ratio > 1`（如 fp16/weights 1.0204）。主指标固定用 **8 MiB**，1 MiB 只作小尺寸压力用例。
2. **HANS 的 ratio 下限约为 `(elemSize-1)/elemSize`**（zeros：fp16/bf16 0.595、fp32 0.798），
   因为尾数原样存储。LZ4 在长游程上远胜（zeros 0.004）。
3. **真实权重上 LZ4 现有 typed 模式几乎不压缩（≈1.00）**，HANS 0.77–0.95。
   这正是"字节平面分离"要解决的问题：符号/指数平面高度可压缩，尾数平面近似噪声。
4. **吞吐差距巨大**：HANS 压缩 9–82 GB/s、解压 68–131 GB/s；现有 LZ4 是 0.4–2 GB/s。
   LZ4 的优化重点在向量化解析，而不是继续调哈希参数。

## 7. 统一 harness（P2）

`benchmarks/bench_fp.asc` 编译出的 `build/bench_fp` 同时支持三种 codec：

```bash
./build/bench_fp lz4raw bf16 data/fp/bf16/bf16_weights_8MiB.bin 10 3   # 原始 LZ4（64 KiB，无变换）
./build/bench_fp lz4    bf16 data/fp/bf16/bf16_weights_8MiB.bin 10 3   # 平面分离（32 KiB）
./build/bench_fp hans   bf16 data/fp/bf16/bf16_weights_8MiB.bin 10 3   # 官方 HANS
```

三者共用：输入读取、warmup/iters、ACL Event 计时、mean/median/min/max/sd、
位级校验、失败即非零退出。`scripts/run_bench_fp.sh` 负责哈希校验、逐文件调用与 CSV 汇总，
`scripts/summarize_fp.py` 生成 side-by-side 表。

## 8. 公平性契约（脚本强制）

- 运行前校验 `manifest.json` 的 sha256；
- 同一 dtype 下 LZ4 与 HANS 读取**同一份字节缓冲**；
- 使用相同的 `warmup / iters` 与相同的统计口径（median 主指标，附 mean/min/max/sd）；
- 结果分别写入 `logs/bench_hans.csv` 与 LZ4 的 CSV，报告中引用同一份 manifest 哈希；
- 计时失败 / 校验失败直接 FAIL，不产生吞吐数字。
