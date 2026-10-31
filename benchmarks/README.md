# benchmarks/ —— 性能基准说明

> 回答两个独立问题：**压缩率多少**、**压缩/解压有多快**。
> 单 buffer 路径和指针数组批量路径 **分开测量**，可以对同量数据做直接对比。

---

## 1. 两种模式

| 模式 | 用的 kernel | 数据组织 |
| --- | --- | --- |
| `--mode single` | `lz4_compress_kernel` / `lz4_decompress_kernel`（Snappy 同构） | 一整块连续 buffer，按 64 KiB 切分，输出到固定槽位 |
| `--mode batch` | `lz4_batch_compress_kernel` / `lz4_batch_decompress_kernel` | 指针数组 + 每项长度；每项内存在大 buffer 中 **带 padding（非连续）** |
| `--mode both` | 两者都跑 | 默认 |

批量模式的 `--item`：

| 取值 | 含义 |
| --- | --- |
| `uniform`（默认） | 每项 64 KiB（与 single 的分块完全一致） |
| `mixed` | 循环使用 1K~256K 混合尺寸（含 128K/256K 大 item），模拟真实批 |

> `uniform` 下两种模式处理的是**同一份数据、同样的 64 KiB 块**，
> 因此两行数字可以直接比较，差异只来自 API/寻址方式。

---

## 2. 运行

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /workspace/ql/wqr/nvcomp_ascend

# 分开跑（推荐）
./build/bench_lz4    128 10 3 1 --mode single
./build/bench_lz4    128 10 3 1 --mode batch
./build/bench_lz4    128 10 3 1 --mode batch --item mixed
./build/bench_snappy 128 10 3   --mode single
./build/bench_snappy 128 10 3   --mode batch

# 四个叶子组合：算法 × 单/批（参数：MB iters warmup）
bash scripts/run_bench_lz4_single.sh    128 10 3   # d1 + d2 + d4
bash scripts/run_bench_lz4_batch.sh     128 10 3   # d1 + d2 + d4 + mixed
bash scripts/run_bench_snappy_single.sh 128 10 3
bash scripts/run_bench_snappy_batch.sh  128 10 3

# 按算法聚合
bash scripts/run_bench_lz4.sh    128 10 3
bash scripts/run_bench_snappy.sh 128 10 3

# 全部
bash scripts/run_bench.sh 128 10 3
```

参数：`[totalMB] [iters] [warmup] [dtype] [--mode ...] [--item ...]`。
日志写到 `logs/bench_*.log`。

---

## 3. 测量方法（与 nvCOMP 一致）

1. 生成 `totalMB` 的大 buffer（5 种数据形态：zeros / repeat / text / structured / random）；
2. 按 64 KiB 切分（single）或按 item 尺寸切分（batch）；
3. **预热** `warmup` 次；
4. 用 **ACL 设备事件**（等价 CUDA events）在 `iters` 次迭代中分别计时压缩和解压，取平均；
5. 计算：`ratio = compBytes / inBytes`，`GB/s = inBytes / (ms × 1e6)`；
6. 每个数据集做一次**往返校验**（`verify=OK` 才说明该行数字有效）。

计时**只包含设备核函数执行**，不含 `aclrtMalloc` / H2D / D2H。

---

## 4. 批量模式的实现细节（为什么不是"单路径换皮"）

- 输入用**指针数组**（`__gm__ uint64_t[]`）表达，每项地址 = 大 buffer + 带 padding 的偏移；
- 每项长度可变，压缩长度写回 32B 槽位；
- 因单次 launch 最多 `MAXB=256` 项，基准会按 256 一组循环 launch；
- 因此批量模式的固定开销包括：指针数组搬运、每项 `DataCopy`、每项长度写回。

---

## 5. 实测样例（32 MiB / 3 iters / 1 warmup）

```
=== lz4 benchmark (chunk=65536B, dtype=1) ===
dataset            units     in(MB)   comp(MB)    ratio   comp(ms)  compGB/s    dec(ms)   decGB/s verify
single/zeros         512      32.00       0.13   0.0041     17.216      1.95      8.406      3.99     OK
batch/zeros          512      32.00       0.13   0.0041     17.219      1.95      8.409      3.99     OK
single/structured    512      32.00      11.64   0.3636     33.842      0.99     18.218      1.84     OK
batch/structured     512      32.00      11.64   0.3636     35.302      0.95     18.225      1.84     OK
single/random        512      32.00      32.13   1.0039     85.956      0.39     16.481      2.04     OK
batch/random         512      32.00      32.13   1.0039     91.986      0.36     16.488      2.04     OK
```

批量模式在表头前还会打印一行 item 规格，例如：

```
[bench_lz4] totalMB=32 iters=3 warmup=1 dtype=1 chunks=512 mode=batch item=mixed
            item pattern=mixed items=543 min=1024B avg=61794B max=262144B launches=3 (<=256/launch)
```

输出列含义：

> **列名随模式变化**：`--mode single` 显示 `chunks`，`--mode batch` 显示 `items`，
> `--mode both` 显示 `units` 并额外打印一行
> `note: units = chunks for single/, items for batch/`。
> 非批量路径内部同样按 64 KiB 切块并行处理，所以它有 chunk 数，只是没有 `item`。

| 列 | 含义 |
| --- | --- |
| `dataset` | `single/<形态>` 或 `batch/<形态>` |
| `chunks` / `items` / `units` | 处理单元数：single 为 64 KiB chunk 数；batch 为 item 数 |
| `in(MB)` | 原始数据量（MiB） |
| `comp(MB)` | 压缩后总量 |
| `ratio` | 压缩率 = comp / in |
| `comp(ms)` / `compGB/s` | 单次压缩设备耗时 / 吞吐（按原始字节） |
| `dec(ms)` / `decGB/s` | 单次解压设备耗时 / 吞吐 |
| `verify` | 该数据集往返校验结果 |

结论：

- **uniform 批量与单路径几乎等价**（压缩慢约 0~8%，来自指针数组/逐项搬运开销）；
- **mixed 批量明显变慢**（32 MiB 下 zeros 压缩 1.95 → 0.82 GB/s），因为混合尺寸下 item 更小、数量更多（本样本 543 项），
  每项固定开销占主导；
- Snappy 的 batch/single 差异更小（uniform 下基本持平）。

---

## 6. 注意事项

- `--item mixed` **单 item 最大 16 MiB**；超过 64 KiB 时内部自动按 64 KiB 分块（每块一个标准块）。
- 会创建大量小 item，内存占用随 item 数增长；超大 `totalMB` + mixed 可能耗尽设备内存。
- 默认 64 KiB chunk 与 nvCOMP `benchmark_*_chunked` 的默认值一致，可直接对比默认口径。
- 本项目 `CHUNK` 是编译期常量（受 UB 限制），**不支持运行时改 chunk 大小**；如需其他块大小需改 `CHUNK` 重编译。
- 基准是单设备号、单 stream；未测多 stream / 多设备扩展性。
- 若需要"端到端"（含 H2D/D2H）时间，需在主机侧另加计时，本基准不包含。


---

## 统计量、端到端与环境记录（整改后）

每个数据集输出：

1. **主表行**：`comp(ms)`、`compGB/s`、`dec(ms)`、`decGB/s`（均为多次迭代平均值）、`verify`。
2. `stats/<name>`：压缩与解压的 `mean / median / min / max / sd`（逐次迭代样本）。
3. `e2e/<name>`：`h2d`、`kern`、`d2h`、`total` 与端到端 GB/s。
   - `h2d` / `d2h` 为主机 `std::chrono::steady_clock` 计时；
   - `kern` 为 ACL Event 计时；
   - 两类指标分开记录，不混用。
4. `[env]` 行：时间戳、NPU 型号（`npu-smi`）、CANN / Driver 版本、`CHUNK`、`MAXB`、`dtype`、`seed`、模式。
5. CSV：`logs/bench_<algo>_<mode>_d<dtype>.csv`，含上述全部指标。

计时失败（`aclrtRecordEvent` / `aclrtSynchronizeEvent` / `aclrtEventElapsedTime` 返回错误）
时该数据集直接判 `FAIL`，不参与吞吐统计。

### E2E 预热

端到端一次只跑一遍，因此测量前会先做一次**不计时**的 comp+dec 预热：
主机的正确性校验循环会让设备空闲十几毫秒、时钟回落，若不预热，单次冷跑会偏慢约 2 倍
（实测 16 MB batch：预热前 `kern=23.466 ms`，预热后 `kern=12.852 ms`，
与单独测得的 `comp 8.612 + dec 4.214 = 12.826 ms` 一致）。

### 测量方法说明（重要）

逐次迭代采样会暴露两类真实干扰：

1. **预热不足**：前 1–2 次迭代运行在 boost 频率，之后进入稳态。实测 16 MB / 256 chunk
   的 LZ4 batch 压缩，`warmup=2, iters=2` 得到 8.637 与 17.235 ms 两个样本
   （均值 12.9、标准差 6.1）；改成 `warmup=5, iters=12` 后稳定在
   8.614 ms（标准差 0.002）。
2. **主机抢占**：`aclrtRecordEvent(start)` 与 Kernel launch 之间若被 OS 调度打断，
   该样本会包含设备空转时间，出现约 2 倍的双峰。

因此：

- 对比优化前后请用 **median**（或 **min**）而不是 mean；
- 建议 `warmup >= 3`、`iters >= 10`；
- 若 `sd / mean > 5%`，先排查主机负载，不要直接当作 Kernel 性能差异；
- 所有样本都写入 CSV，可复核。
