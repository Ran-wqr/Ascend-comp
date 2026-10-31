# nvcomp_ascend

把 **NVIDIA nvCOMP `branch-2.2`** 的 **LZ4 / Snappy** 无损压缩迁移到
**华为昇腾 910C / 910B（`dav-2201`）**，设备端使用 **Ascend C** 编写，
并在真实 NPU 上完成编译、运行、正确性验证与性能基准测试。

数据格式与官方实现**逐字节兼容**：LZ4 为标准 raw block，Snappy 为标准 block
（含 varint 前缀），可直接与 `lz4` / `google-snappy` 互相解压。

---

## 1. 能力与验证状态

| 能力 | 实现 | 验证 |
| --- | --- | --- |
| LZ4 压缩 / 解压 | ✅ NPU 设备核 | ✅ single + batch |
| LZ4 解压尺寸查询 | ✅ | ✅ single + batch |
| LZ4 类型化输入（1/2/4 字节对齐） | ✅ `dtype=1/2/4` | ✅ single + batch |
| Snappy 压缩 / 解压 | ✅ NPU 设备核 | ✅ single + batch |
| Snappy 解压尺寸查询 | ✅ | ✅ single + batch |
| 批量 API（指针数组、变长 item、每项独立输出） | ✅ | ✅ 含 dtype 1/2/4；单 item ≤16 MiB；>256 项主机侧自动分批 |

当前测试结果：

- `test_lz4`：single **159** 项、batch **42** 项，全部通过
- `test_snappy`：single **134** 项、batch **31** 项，全部通过
- 参考互操作套件（NPU ↔ 官方编解码器，`--strict`）：**全部 PASS，0 SKIP**
- 回归基线：`bash scripts/check_baseline.sh verify` → 15 份压缩产物 SHA-256 不变

> 未迁移：HLIF 容器格式、C API 字面签名、逐 chunk 状态上报/校验和。
> 详见 [`docs/design.md`](docs/design.md)。

> 面向使用者的完整说明（数据集、跑法、参数、结果解读、公平性）见 **[`README_FP.md`](README_FP.md)**。

> 字节流 LZ4 与 **CPU 官方实现（上游 v1.9.4 / 鲲鹏版）** 的对比见 **[`README_LZ4_CPU.md`](README_LZ4_CPU.md)**。
>
> **LZ4 性能测试脚本（7 个，含数据集预设）见 [`README_BENCH_LZ4.md`](README_BENCH_LZ4.md)。**

### 1.0 怎么测（四条路径）

浮点 LZ4 / HANS / 字节 LZ4（NPU）/ CPU 字节 LZ4 的**指令、输出解读、测试集一致性**见 **[`README_TESTING.md`](README_TESTING.md)**。

### 1.1 可靠性基线（本轮整改）

- **Batch 读写边界**：主机侧按 Kernel 的取整粒度（指针 `ceil(n/4)*4`、大小/容量 `ceil(n/8)*8`）
  分配并零初始化元数据；读回长度等于实际分配长度，不再多读 32 B。
- **解压边界检查**：LZ4 与 Snappy 解压器逐项校验 token/tag、长度扩展字节、offset 字节、
  offset≠0 且 ≤ 已产出长度、输出容量与整数溢出；非法流返回 `-1`，主机侧一律判失败。
  Snappy 额外校验 varint 前导与"声明长度 == 实际产出"。
- **输出长度**：不再静默 `clamp` 到 `CHUNK`；设备返回长度超出容量直接失败。
- **空输入语义**：
  - 单缓冲区 / 文件：LZ4 输出标准空块 `00`（1 字节），Snappy 输出 varint(0)（1 字节），
    二者都能被官方解码器接受；
  - Batch 中的空 item 表示为 **0 个块**（`clen = 0`，无数据），与"标准空块"区分，保持向后兼容。
- **Batch 规模**：`MAXB = 256` 是**单次 Kernel Launch 上限**；demo / tests / benchmark
  均在主机侧按 256 自动分批，因此 257 项等更大批量可正常工作（已测 1/3/31/32/33/255/256/257）。
- **错误处理**：所有 ACL 调用检查返回值并打印 `[ACL ERROR] 文件:行 API -> 错误码`；
  构建脚本启用 `set -e` + `pipefail`，编译失败不会打印 `[build] OK`。
- **测试判定**：互操作脚本区分 `PASS / FAIL / SKIP`，SKIP 不计入 PASS；
  `--strict` 下任何 SKIP 都返回非零。

---

### 1.2 浮点数据（FP16 / BF16 / FP32）与 HANS 基线

本轮新增了**浮点数据专用路径**，并用华为官方 HANS 算子（CANN 9.0.0 的 aclnn 接口）作为对照基线。

| 组件 | 位置 | 说明 |
| --- | --- | --- |
| 固定测试集 | `data/fp/` + `data/fp/manifest.json` | 50 个文件 / 225 MiB，sha256 锁定；8 类合成 + 真实 Qwen3-32B BF16 权重切片 + 原生 fp32 仿真 |
| 数据生成/校验 | `scripts/gen_fp_data.py` | 同种子字节级可复现；`--verify` 不一致直接失败 |
| LZ4 浮点路径 | `src/lz4_fp_ascend_kernels.h` | 字节平面分离（S-planes）+ 标准 LZ4 块，32 KiB 块 |
| 统一基准 | `benchmarks/bench_fp.asc` → `build/bench_fp` | **同一二进制**跑 `lz4raw` / `lz4` / `hans`，同输入、同 warmup/iters、同事件计时、同统计、同位级校验 |
| 一键对比 | `scripts/run_bench_fp.sh` | 先校验测试集哈希，再逐文件跑两种 codec，输出 `logs/bench_fp.csv` |
| 汇总 | `scripts/summarize_fp.py` | 从 CSV 生成 side-by-side 对比表 |
| HANS 基线 | `hans/` + `scripts/build_hans.sh` | 官方闭源算子，只调用不重写；含接口/口径/结论说明 |

```bash
python3 scripts/gen_fp_data.py            # 生成（幂等）
python3 scripts/gen_fp_data.py --verify   # 只校验哈希
python3 scripts/gen_fp_data.py --sizes 1 8 64   # 按需追加 64 MiB
bash scripts/run_bench_fp.sh 10 3         # LZ4 vs HANS，写入 logs/bench_fp.csv
python3 scripts/summarize_fp.py logs/bench_fp.csv --size 8
```

**8 MiB、iters=10、warmup=3 实测（ratio 越低越好；三个 codec 同一二进制）**

| 数据集 | LZ4 raw | LZ4 平面 | HANS | 最佳 |
| --- | --- | --- | --- | --- |
| bf16 / weights | 1.0033 | 0.9680 | **0.7732** | HANS |
| fp16 / weights | 1.0036 | 1.0029 | **0.9478** | HANS |
| fp32 / f32nat（原生仿真） | 1.0039 | 0.9864 | **0.8975** | HANS |
| fp32 / weights（bf16 cast） | 0.9967 | **0.4866** | 0.8975 | 不作结论¹ |
| bf16 / gauss | 1.0032 | 0.9647 | **0.7653** | HANS |
| bf16 / smooth | 0.4852 | **0.2704** | 0.6344 | LZ4 平面 |
| fp16 / smooth | 0.8920 | **0.4426** | 0.7548 | LZ4 平面 |
| fp32 / smooth | 1.0039 | **0.6433** | 0.8346 | LZ4 平面 |
| bf16 / zeros | **0.0041** | 0.0042 | 0.5953 | LZ4 raw |
| bf16 / repeat | **0.0197** | 0.0349 | 0.7906 | LZ4 raw |
| bf16 / sparse | **0.1659** | 0.2464 | 0.7054 | LZ4 raw |
| bf16 / uniform | **1.0039** | 1.0040 | 1.0250 | LZ4 raw |
| fp32 / wideexp | 1.0039 | 1.0040 | **0.9937** | HANS |

25 个 8 MiB 数据集上 **最佳 ratio 计数：LZ4 raw 12 / LZ4 平面 5 / HANS 8**。

| 指标（8 MiB，median） | LZ4 raw | LZ4 平面 | HANS |
| --- | --- | --- | --- |
| 压缩耗时 | 4.0–13.9 ms | 8.5–27.1 ms | 0.05–0.88 ms |
| 解压耗时 | 1.1–4.5 ms | 6.2–9.8 ms | 0.05–0.12 ms |

要点：

- **没有任何一方全面胜出**：HANS 赢在真实权重/高斯分布（尾数近噪声），LZ4 赢在游程/稀疏/均匀数据。
- **平面分离不是万能药**：它在 smooth 上大幅领先（fp32 1.0039 → 0.6433），
  但在 repeat/sparse 上反而更差（bf16 repeat 0.0197 → 0.0349）——因为 FP 路径用 32 KiB 块，
  每个块的"首个字面量"开销相对翻倍；同时它把压缩耗时抬高约 2 倍。
- **真实权重是 LZ4 的短板**：三种 dtype 都输给 HANS（0.968/1.003/0.986 vs 0.773/0.948/0.898）。
  下一步应针对"尾数平面近似噪声"这一熵下限做工作（位平面拆分、指数差分、或向量化 +
  更大的块），而不是继续单纯调哈希。
- **fp32 的 bf16 转换版不可用作结论**：低 16 位尾数为 0，0.4866 是数据假象；
  原生 fp32 以 `f32nat` 的 0.9864 为准。
- **1 MiB 不作主指标**：HANS 每 core 有 8448 B 定长尾部，1 MiB 下 ratio 会 >1。

## 2. 环境要求

| 项 | 值 |
| --- | --- |
| 设备 | Ascend 910（`dav-2201`），UB 192 KiB |
| CANN | 9.0.0（`/usr/local/Ascend/cann-9.0.0`） |
| 编译器 | `bisheng`（ASC 语言，`--npu-arch=dav-2201`） |
| 主机 | Linux aarch64，gcc/g++（编译参考工具用），python3 |

**每次新开 shell 都要先加载环境**：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

---

## 3. 目录结构

```
nvcomp_ascend/
├── CMakeLists.txt                 # ASC 工程：8 个可执行目标
├── common_ascend.h                # 主机侧工具：读/写文件、ACL_CHECK
├── README.md                      # 本文件（总览 / 编译 / 运行）
├── README_FP.md                   # FP16/BF16/FP32：LZ4 vs HANS 完整说明
├── README_LZ4_CPU.md              # 字节流 LZ4：CPU(官方/鲲鹏) vs NPU
├── src/
│   ├── lz4_ascend_kernels.h       # LZ4 设备端算法 + 5 个核函数（被 demo/tests/bench 复用）
│   ├── lz4_fp_ascend_kernels.h    # LZ4 浮点路径：字节平面分离（FP16/BF16/FP32）
│   ├── lz4_ascend.asc             # LZ4 demo：主机驱动 + CLI
│   ├── snappy_ascend_kernels.h    # Snappy 设备端算法 + 5 个核函数
│   └── snappy_ascend.asc          # Snappy demo：主机驱动 + CLI
├── hans/                          # HANS 基线（官方 aclnn 算子封装 + 说明）
├── tests/                         # 正确性测试（single / batch 分开）
│   ├── test_common.h              # 极简测试框架 + 数据生成器
│   ├── test_lz4.asc
│   ├── test_snappy.asc
│   └── README.md                  # 判定逻辑详解
├── benchmarks/                    # 性能基准（single / batch 分开）
│   ├── bench_common.h             # 设备事件计时 + 数据生成 + 报表
│   ├── bench_lz4.asc
│   ├── bench_snappy.asc
│   ├── bench_fp.asc               # 统一 FP 基准：lz4raw / lz4(平面) / hans
│   ├── bench_lz4_file.asc         # NPU 单文件基准（kernel-only + e2e）
│   ├── bench_lz4_cpu.c            # CPU 参考基准（上游 / 鲲鹏共用一份源码）
│   └── README.md                  # 测量方法详解
├── scripts/
│   ├── build.sh                   # 编译全部目标
│   ├── build_ref.sh               # 编译官方 lz4 / google-snappy 参考工具
│   ├── gen_testdata.py            # 生成单文件测试向量
│   ├── gen_batch.py               # 生成批量测试集（40 个变长 item）
│   ├── check_sizes.py             # 校验 --size 输出
│   ├── check_batch.py             # 校验批量参考解码结果
│   ├── run_tests_lz4_single.sh    # LZ4 / 非批量 正确性
│   ├── run_tests_lz4_batch.sh     # LZ4 / 批量   正确性
│   ├── run_tests_snappy_single.sh # Snappy / 非批量 正确性
│   ├── run_tests_snappy_batch.sh  # Snappy / 批量   正确性
│   ├── run_tests_lz4.sh           # LZ4 的两种模式
│   ├── run_tests_snappy.sh        # Snappy 的两种模式
│   ├── run_tests.sh               # 全部正确性测试
│   ├── run_bench_lz4_single.sh    # LZ4 / 非批量 性能（d1、d2、d4）
│   ├── run_bench_lz4_batch.sh     # LZ4 / 批量   性能（d1、d2、d4、mixed）
│   ├── run_bench_snappy_single.sh # Snappy / 非批量 性能
│   ├── run_bench_snappy_batch.sh  # Snappy / 批量   性能
│   ├── run_bench_lz4.sh           # LZ4 的两种模式
│   ├── run_bench_snappy.sh        # Snappy 的两种模式
│   ├── run_bench.sh               # 全部性能基准
│   ├── run_all.sh                 # 参考互操作四向验证（PASS/FAIL/SKIP + --strict）
│   ├── regress.sh                 # 一键回归：编译+单测+严格互操作(+基准)
│   ├── check_baseline.sh          # 压缩产物 SHA-256 基线 save/verify
│   ├── build_hans.sh              # 编译 HANS 基线（aclnn 宿主程序）
│   ├── build_ref_lz4_cpu.sh       # 编译上游/鲲鹏 LZ4 + 两个 CPU 基准
│   ├── gen_fp_data.py             # 生成/校验 50 个浮点测试文件（sha256 锁定）
│   ├── run_bench_fp.sh            # FP：LZ4(raw/平面) vs HANS
│   ├── run_bench_hans.sh          # HANS 单独基线
│   ├── run_bench_lz4_cpu.sh       # 字节流：CPU(up/kp) vs NPU
│   ├── summarize_fp.py            # FP 结果 side-by-side
│   └── summarize_lz4_cpu.py       # CPU/NPU 结果 side-by-side
├── ref/                           # 官方参考实现（第三方，用于交叉验证）
│   ├── lz4/                       # github.com/lz4/lz4 源码（参考 CLI/库）
│   ├── lz4_up_src/                # 上游 LZ4 v1.9.4（CPU 主对照）
│   ├── lz4_kp_src/                # kunpengcompute/lz4（鲲鹏优化版）
│   ├── snappy/                    # github.com/google/snappy 源码
│   └── tools/                     # ref_lz4(.cpp) / ref_snappy(.cpp) 及编译产物
├── docs/
│   ├── README.md                  # 初学者完全指南（逐文件/逐函数/逐行注释）
│   └── design.md                  # 设计说明：GPU SIMT → Ascend AIV 语义迁移
├── data/                          # 测试数据（可删，脚本重建）
│   ├── t_*.bin                    # 12 个字节向量（KB 级）
│   ├── batch/                     # 40 项变长批量
│   ├── fp/                        # 50 个 FP16/BF16/FP32 文件（225 MiB）
│   └── large/                     # 5×1GB + 1×4GB + 3×5GB：text/struct/random/4G/5G
├── logs/                          # 构建/运行日志（可删）
└── build/                         # CMake 产物（可删重建）
```

> **源码**只有 `src/`、`tests/`、`benchmarks/`、`scripts/`、`common_ascend.h`、
> `CMakeLists.txt`、`docs/`；`data/`、`logs/`、`build/`、`ref/*` 都可重新生成。

---

## 4. 编译

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /workspace/ql/wqr/nvcomp_ascend

# 编译本项目（8 个 ASC 目标）
bash scripts/build.sh

# 编译官方参考编解码器（做交叉验证时需要）
bash scripts/build_ref.sh
```

产物（`build/`）：

| 可执行文件 | 作用 |
| --- | --- |
| `lz4_demo` | LZ4 命令行工具：压缩/解压/尺寸/批量 |
| `snappy_demo` | Snappy 命令行工具 |
| `test_lz4` / `test_snappy` | 正确性测试（single / batch / both） |
| `bench_lz4` / `bench_snappy` | 合成数据性能基准（kernel-only + E2E） |
| `bench_fp` | 统一 FP 基准：`lz4raw` / `lz4`(平面) / `hans` |
| `bench_lz4_file` | NPU 单文件基准（给 CPU 对比用） |

另外两个由独立脚本构建（不在 CMake 里）：

| 可执行文件 | 作用 |
| --- | --- |
| `hans/bench_hans` | HANS 单独基线（`scripts/build_hans.sh`） |
| `build/bench_lz4_cpu_{up,kp}` | CPU 上游 / 鲲鹏基准（`scripts/build_ref_lz4_cpu.sh`） |
| `test_lz4` | LZ4 正确性测试（`--mode`） |
| `test_snappy` | Snappy 正确性测试（`--mode`） |
| `bench_lz4` | LZ4 性能基准（`--mode` / `--item`） |
| `bench_snappy` | Snappy 性能基准 |

日志：`logs/cmake.log`、`logs/make.log`。

> 手动编译等价于：`mkdir -p build && cd build && cmake .. && make -j4`。

---

## 5. 运行

### 5.1 正确性测试（single / batch 分开）

```bash
# 四个叶子组合（算法 × 单/批）
bash scripts/run_tests_lz4_single.sh
bash scripts/run_tests_lz4_batch.sh
bash scripts/run_tests_snappy_single.sh
bash scripts/run_tests_snappy_batch.sh

# 按算法聚合（两种模式）
bash scripts/run_tests_lz4.sh
bash scripts/run_tests_snappy.sh

# 全部
bash scripts/run_tests.sh

# 或直接调程序
./build/test_lz4    --mode single|batch|both
./build/test_snappy --mode single|batch|both
```

参数：`--mode single|batch|both`（默认 `both`）；`--quiet` 只打印失败与汇总；`--rand N` 设置随机化迭代次数（默认 8，固定种子可复现）。
日志：`logs/test_{lz4,snappy}_{single,batch}.log`。
退出码：全过 0，有失败 1。

### 5.2 性能基准（single / batch 分开）

```bash
# 四个叶子组合（算法 × 单/批）；参数：MB iters warmup
bash scripts/run_bench_lz4_single.sh 128 10 3     # d1 + d2 + d4
bash scripts/run_bench_lz4_batch.sh  128 10 3     # d1 + d2 + d4 + mixed
bash scripts/run_bench_snappy_single.sh 128 10 3
bash scripts/run_bench_snappy_batch.sh  128 10 3

# 按算法聚合
bash scripts/run_bench_lz4.sh 128 10 3
bash scripts/run_bench_snappy.sh 128 10 3

# 全部
bash scripts/run_bench.sh 128 10 3

# 或直接调程序
./build/bench_lz4    128 10 3 1 --mode single|batch
./build/bench_lz4    128 10 3 1 --mode batch --item mixed
./build/bench_snappy 128 10 3   --mode single|batch
```

参数：`[totalMB] [iters] [warmup] [dtype] [--mode ...] [--item uniform|mixed]`
（Snappy 无 `dtype`）。
日志：`logs/bench_*_{single,batch}*.log`。

每个数据集除主表外还会输出：

- `stats/<name>`：压缩/解压耗时的 **mean / median / min / max / 标准差**（基于逐次迭代样本）；
- `e2e/<name>`：**H2D + Kernel + D2H** 分段耗时与端到端吞吐（主机 `steady_clock` 与 ACL Event 分开计时）；
- `[env]`：NPU 型号（`npu-smi`）、CANN / Driver 版本、`CHUNK / MAXB / dtype / seed / mode`；
- `logs/bench_<algo>_<mode>_d<dtype>.csv`：上述指标的结构化 CSV。

> **测量口径**：逐次迭代采样会暴露预热（boost→稳态频率）与主机抢占两类干扰。
> 16 MB / 256 chunk 的 LZ4 batch 压缩在 `warmup=2, iters=2` 下会出现
> 8.637 ms 与 17.235 ms 的双峰样本；`warmup=5, iters=12` 后稳定在 8.614 ms
> （sd=0.002）。对比优化前后请使用 **median/min**，并保持 `warmup>=3, iters>=10`。

### 5.3 命令行工具（demo）

```bash
# LZ4：压缩 + 解压自检，并导出压缩块；dtype=1/2/4
./build/lz4_demo data/t_text.bin /tmp/t.lz4blocks 1

# 只解压外部（如官方参考编码器）产生的块
./build/lz4_demo --decode /tmp/t.lz4blocks /tmp/t.out

# 只查询每块解压后大小
./build/lz4_demo --size /tmp/t.lz4blocks /tmp/t.sizes

# 批量：list 文件每行一个输入路径（变长、互不连续的独立 item）
printf 'a.bin\nb.bin\nc.bin\n' > /tmp/list.txt
./build/lz4_demo --batch /tmp/list.txt 1

# Snappy 完全对称（无 dtype）
./build/snappy_demo data/t_text.bin /tmp/t.snappyblocks
./build/snappy_demo --decode /tmp/t.snappyblocks /tmp/t.out
./build/snappy_demo --size   /tmp/t.snappyblocks /tmp/t.sizes
./build/snappy_demo --batch  /tmp/list.txt
```

### 5.4 参考互操作验证

```bash
bash scripts/run_all.sh            # 普通模式：SKIP 不影响退出码
bash scripts/run_all.sh --strict   # 严格模式：任何 SKIP 都返回非零
# 对 12 个向量 × {lz4 d1,d2,d4, snappy} + 3 个批量用例做：
#   A NPU 往返   B NPU 产物 → 官方解码器
#   C 官方产物 → NPU 解压   D NPU 尺寸查询
# 状态：PASS=实际执行且通过，FAIL=执行失败，SKIP=未执行（含原因）
# 结果写入 logs/summary.log / logs/skips.log / logs/failures.log
```

### 5.5 一键回归

```bash
bash scripts/regress.sh                  # 编译 + 单元测试 + 严格互操作
bash scripts/regress.sh --with-bench     # 追加性能基准
bash scripts/check_baseline.sh save      # 记录当前压缩产物哈希
bash scripts/check_baseline.sh verify    # 与基线比对（编码若变化会报 MISMATCH）
```

### 5.6 指令速查

| 目标 | 命令 |
| --- | --- |
| 编译全部 | `bash scripts/build.sh` |
| 编译参考工具 | `bash scripts/build_ref.sh` |
| 正确性 LZ4 非批量 | `bash scripts/run_tests_lz4_single.sh` |
| 正确性 LZ4 批量 | `bash scripts/run_tests_lz4_batch.sh` |
| 正确性 Snappy 非批量 | `bash scripts/run_tests_snappy_single.sh` |
| 正确性 Snappy 批量 | `bash scripts/run_tests_snappy_batch.sh` |
| 性能 LZ4 非批量 | `bash scripts/run_bench_lz4_single.sh 128 10 3` |
| 性能 LZ4 批量 | `bash scripts/run_bench_lz4_batch.sh 128 10 3` |
| 性能 Snappy 非批量 | `bash scripts/run_bench_snappy_single.sh 128 10 3` |
| 一键回归 | `bash scripts/regress.sh` |
| 压缩产物基线 | `bash scripts/check_baseline.sh verify` |
| 浮点测试集生成 | `python3 scripts/gen_fp_data.py` |
| 浮点测试集校验 | `python3 scripts/gen_fp_data.py --verify` |
| LZ4 vs HANS（浮点） | `bash scripts/run_bench_fp.sh 10 3` |
| 浮点结果汇总 | `python3 scripts/summarize_fp.py logs/bench_fp.csv --size 8` |
| 编译 CPU 参考库 | `bash scripts/build_ref_lz4_cpu.sh` |
| LZ4 CPU vs NPU | `bash scripts/run_bench_lz4_cpu.sh 10 3` |
| CPU 对比汇总 | `python3 scripts/summarize_lz4_cpu.py logs/bench_lz4_cpu.csv` |
| 1 GB 数据集生成/校验 | `python3 scripts/gen_large_testdata.py [--verify]` |
| CPU vs NPU（1 GB） | `bash scripts/run_bench_lz4_cpu_large.sh 3 1` |
| CPU vs NPU（单个 4 GB） | `bash scripts/run_bench_lz4_cpu_4g.sh 3 1` |
| CPU vs NPU（3 × 5 GB） | `bash scripts/run_bench_lz4_cpu_5g.sh 3 1` |
| LZ4 性能（重构版入口） | `bash scripts/bench_cpu_vs_lz4.sh vec12 10 3` |
| 数据集预设列表 | `bash scripts/bench_cpu_vs_lz4.sh --help` |
| 性能 Snappy 批量 | `bash scripts/run_bench_snappy_batch.sh 128 10 3` |
| 全部正确性 / 性能 | `bash scripts/run_tests.sh` / `bash scripts/run_bench.sh 128 10 3` |
| 参考互操作 | `bash scripts/run_all.sh` |
| LZ4 单文件 | `./build/lz4_demo <file> [blocks] [dtype]` |
| LZ4 解码外部块 | `./build/lz4_demo --decode <blocks> <out>` |
| LZ4 尺寸查询 | `./build/lz4_demo --size <blocks> <sizes>` |
| LZ4 批量 | `./build/lz4_demo --batch <list> [dtype]` |

---

## 6. 运行时数据流

### 6.1 总览

```
主机 (CPU)                                  设备 (NPU, AIV)
──────────                                  ────────────────────────────────
read_file ─► hIn
aclrtMalloc ─► dIn / dComp / dSizes ...
aclrtMemcpy  H2D ─────────────► GM
                                 │  DataCopy (MTE2)
                                 ▼
                                UB ──► 设备端算法 ──► UB
                                 │  DataCopy (MTE3)
                                 ▼
                                GM
aclrtMemcpy  D2H ◄─────────────┘
memcmp / 统计 ─► 结果
```

- **GM**：设备大内存（Global Memory）；
- **UB**：每 block 约 184 KiB 可用（Unified Buffer）；
- **DataCopy**：GM↔UB 搬运（MTE 引擎）；
- **TQue 的 EnQue/DeQue**：在"搬运"和"计算"之间插同步。

### 6.2 单文件压缩 + 解压（`lz4_demo <file>`）

```
① numChunks = ceil(totalBytes / 64KiB)
② 申请设备内存
     dIn    : round32(totalBytes)
     dComp  : numChunks × SLOT        （每块 76544B 输出槽）
     dSizes : numChunks × 32          （每块压缩长度）
     dOut    : numChunks × 64KiB
     dProd  : numChunks × 32          （每块实际解压长度）
③ 数据拷入 dIn
④ lz4_compress_kernel<<<nb>>>
     每个 block 处理 c = bid, bid+nb, ...
       DataCopy dIn[c*64KiB ..] → UB
       lz4CompressT<dtype>       → UB
       DataCopy UB → dComp[c*SLOT]
       写压缩长度 → dSizes
⑤ lz4_decompress_kernel<<<nb>>>
       读 dSizes → DataCopy dComp 槽 → UB
       lz4Decompress → UB → dOut[c*64KiB]
       写实际长度 → dProd
⑥ 按 dProd 拼接各块，与原始数据逐字节比对 → ROUNDTRIP OK/FAIL
⑦ 导出块流：每块 [u32 长度][压缩字节]
```

### 6.3 批量（`--batch <list>` / `bench_* --mode batch`）

```
list.txt → 每项独立数据（单 item 最大 16 MiB）
构造（拷到 GM）：
   inPtrs[]     : 每项输入设备地址
   compPtrs[]   : 每项输出设备地址
   uncompBytes[]/outCapBytes[]
   blockSizes[] : 每项内部 64 KiB 块的压缩长度表（int32，MAX_BLOCKS/项）
lz4_batch_compress_kernel<<<nb>>>
   设备端先把指针数组 DataCopy 进 UB
   每个 block 分摊 item：
       把 item 按 64 KiB 切成 blocks 个块
       逐块：DataCopy → UB → 压缩 → DataCopy → compPtrs[i]+块偏移
       记录每块压缩长度到 blockSizes，累加 compBytes[i]
lz4_batch_decompress_kernel<<<nb>>>
       按 blockSizes 逐块解压，写到 outPtrs[i]+块偏移，累加 actualBytes[i]
逐项比对
```

> **单 item 上限 16 MiB**（nvCOMP LZ4 的同口径上限），内部自动按 64 KiB 分块；
> 每项还有一个 `blockSizes` 表，解压时据此定位各块。
> 单次 launch 最多 **256 项**（`MAXB`），更大批由主机循环。
> item ≤ 64 KiB 时只有 1 块，行为与旧版完全一致。

### 6.4 `--decode` / `--size`

- `--decode`：读入外部块流（`[u32 len][bytes]` 重复），只跑解压核 → 写出拼接结果；
- `--size`：读入块流，只跑尺寸核（解析 token/varint，不写输出）→ 每块 u32 大小。

### 6.5 关键常量

| 常量 | 值 | 含义 |
| --- | --- | --- |
| `CHUNK` | 65536 | 每块最大未压缩字节（64 KiB） |
| `HASH_SIZE` | 4096 | 压缩哈希表项数 |
| `MAXCOMP` | 76522 | 压缩输出上界 |
| `SLOT` | 76544 | 每块 GM 输出槽（32B 对齐） |
| `MAX_LAUNCH` | 32 | 单次 launch 最大 block 数 |
| `MAXB` | 256 | 批量单次最大 item 数 |
| `MAX_ITEM_BYTES` | 16777216 | 单 item 上限（16 MiB） |
| `MAX_BLOCKS` | 256 | 单 item 内部 64 KiB 块数上限 |
| Snappy `MAX_COPY_LENGTH` | 64 | 单次 copy 上限 |
| Snappy `MAX_COPY_DISTANCE` | 32768 | 最大回溯距离 |

---

## 7. 实测结果

### 7.1 正确性

| 程序 | single | batch |
| --- | --- | --- |
| `test_lz4` | 121 通过 | 9 通过 |
| `test_snappy` | 104 通过 | 6 通过 |

参考互操作：**51/51** 通过。

每个用例都会打印数据形态、单元数、输入/压缩字节与压缩率，例如：

```
[ OK ] grp=single/sizes     chunks=1      launch=1  n=65535    comp=65788     ratio=1.0039  comp_ms=   0.000 comp_GBs=    -- ...  data=runs(3)
[ OK ] grp=batch/d1         items=19      launch=1  n=145338   comp=8142      ratio=0.0560  comp_ms=   0.000 comp_GBs=    -- ...  data=19 items, 0..65536B
```

> 正确性测试不计时（`comp_ms=0.000`、吞吐为 `--`）；吞吐只在性能基准中输出。

### 7.2 性能（128 MiB / 5 iters / 2 warmup，压缩/解压 GB/s）

基准表包含 `items` 列（single 为 chunk 数、batch 为 item 数），并在表头前打印 item 规格：

```
[bench_lz4] totalMB=32 iters=3 warmup=1 dtype=1 chunks=512 mode=batch item=mixed
            item pattern=mixed items=543 min=1024B avg=61794B max=262144B launches=3 (<=256/launch)
dataset            items     in(MB)   comp(MB)    ratio   comp(ms)  compGB/s    dec(ms)   decGB/s verify
batch/zeros         1185      32.00       0.14   0.0043     40.811      0.82     19.942      1.68     OK
```

| 配置 | zeros | text | structured | random |
| --- | --- | --- | --- | --- |
| LZ4 d1 single | 1.95 / 3.99 | 1.88 / 1.76 | 1.01 / 1.85 | 0.39 / 2.04 |
| LZ4 d1 batch | 1.95 / 3.99 | 1.88 / 1.76 | 0.96 / 1.85 | 0.36 / 2.03 |
| LZ4 d1 batch/mixed | 0.84 / 1.72 | 0.81 / 0.75 | 0.41 / 0.79 | 0.16 / 0.87 |
| LZ4 d2 single | 2.06 / 1.37 | 1.98 / 2.04 | 1.27 / 1.82 | 0.66 / 2.04 |
| LZ4 d2 batch | 2.06 / 1.37 | 1.98 / 2.03 | 1.30 / 1.81 | 0.68 / 2.03 |
| LZ4 d4 single | 2.12 / 1.37 | 2.04 / 2.04 | 1.47 / 1.54 | 1.02 / 2.04 |
| LZ4 d4 batch | 2.12 / 1.37 | 2.04 / 2.03 | 1.49 / 1.54 | 1.04 / 2.03 |
| Snappy single | 2.00 / 3.72 | 1.93 / 1.73 | 1.24 / 1.71 | 0.45 / 2.04 |
| Snappy batch | 2.00 / 3.71 | 1.93 / 1.72 | 1.22 / 1.73 | 0.45 / 2.04 |

结论：

- uniform 批量与单路径基本持平（压缩差 0~8%）；
- mixed 批量（1K~256K 混合尺寸）因逐项固定开销变慢；
- `dtype` ??????????????????? d1/d2/d4 ?? 0.39 / 0.66 / 1.02 GB/s?
  structured ? ratio ?? 0.355 / 0.359 / 0.386?

---

## 8. 设计要点（简述）

- **算法语义迁移，不是语法翻译**：保留 LZ77 语义与数据格式，把 nvCOMP 的
  warp 协作改为 **chunk 级 block 并行**；不使用任何 CUDA 概念。
- **AIV 标量/向量分工**：串行的 LZ77 解析跑在设备端标量单元；批量搬运走
  `DataCopy`（MTE）；哈希表清零用向量 `Duplicate`。
- 设备代码集中在 `src/*_kernels.h`，demo / tests / benchmarks **共享同一份实现**。

详见 [`docs/design.md`](docs/design.md)。

---

## 9. 常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| `libascendcl.so: cannot open shared object file` | 没加载环境：`source .../set_env.sh` |
| `find_package(ASC)` 失败 | 同上；再 `rm -rf build && bash scripts/build.sh` |
| 编译报 `maybe need a '__ubuf__ half *'` | dav-c220 的向量算术只支持 half/float，字节运算需用标量 |
| 结果全 0 / 数据错 | 漏了 `EnQue/DeQue`（搬运与计算未同步） |
| 尾部数据丢失 | `DataCopy` 长度不是 32B 倍数 |
| 编译报 UB 超限 | 缓冲总量 > ~184 KiB，减小 `CHUNK`/`MAXB` |
| `batch too big (max 256)` | 批量超过 `MAXB`，拆批 |
| `item too large (max 16777216)` | 单 item 超过 `CHUNK` |
| `ACL ERROR ... 507035` | 核函数里队列不平衡（如对空队列 DeQue） |

---

## 10. 文档索引

| 文档 | 内容 |
| --- | --- |
| [`README.md`](README.md) | 本文件：结构 / 编译 / 运行 / 数据流 / 实测 |
| [`docs/README.md`](docs/README.md) | **初学者完全指南**：逐文件、逐函数、逐行注释 |
| [`docs/design.md`](docs/design.md) | 设计说明：GPU SIMT → Ascend AIV 的语义迁移与取舍 |
| [`tests/README.md`](tests/README.md) | 正确性判定逻辑、覆盖矩阵、能抓哪些 bug |
| [`benchmarks/README.md`](benchmarks/README.md) | 性能测量方法、single/batch 对比、注意事项 |

---

## 11. 限制

- 功能优先，未做性能调优（解析仍在 AIV 标量单元串行执行）。
- 分块固定 64 KiB（大 item 内部自动分块）；单 item ≤ 16 MiB；批量单次 ≤ 256 项（受 UB 限制）。
- 未实现 HLIF 容器格式、C API 字面签名、逐 chunk 状态上报与校验和。
- 基准为单设备号、单 stream；未测多 stream / 多设备扩展性。
- 正确性测试未包含损坏流 / fuzz。
