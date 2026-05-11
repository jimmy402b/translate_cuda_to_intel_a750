# HashNeRF Intel Arc XPU 算子融合优化设计

Date: 2026-05-08

## 背景

HashNeRF-pytorch（纯 PyTorch Instant-NGP 实现）已在 Intel Arc A750 + Windows 11 上成功运行，通过 `torch.xpu` 后端训练。当前速度 ~1.5 it/s，完成 50000 次迭代需 ~9 小时。

目标：将训练时间缩短至 30 分钟以内（对标 RTX 3060 纯 PyTorch 水平），约需 15-20x 提速。

## 核心约束

- **投入程度**：适度——可写优化代码，但不想手写底层 GPU kernel
- **技术能力**：对 GPU 编程不熟悉，愿意借助 Claude Code / LLM 辅助
- **优先目标**：速度大幅优于纯 PyTorch，不追求对标 tiny-cuda-nn 的极端性能
- **参考资源**：tiny-cuda-nn（NVIDIA 融合 kernel）、Intel oneAPI / IPEX 官方文档和示例

## 总体架构

```
Phase 1: torch.compile + IPEX
  目标 4-6 it/s, 改动 5-10 行, 耗时 < 1h
      │
      ▼
Phase 2: XPU Profiler 热点分析
  输出 top-5 热点表格 + chrome trace, 耗时 ~1h
      │
      ▼
Phase 3: 针对性 SYCL 融合 (AI 辅助翻译 tiny-cuda-nn → SYCL)
  目标 10-15 it/s, 写 2-3 个 custom op, 耗时 1-3 天
      │
      ▼
评估: 是否达到目标 A (<30min)? 不够则后续考虑方案 B
```

每一步有明确退出条件，不会无限投入。

---

## Phase 1: torch.compile + IPEX + BF16

### 原理

当前瓶颈是 kernel launch 次数过多——hash 编码 16 次 `nn.Embedding` + MLP 3 层 `nn.Linear` + 体渲染 pointwise 操作，每次 iteration 有几十个 kernel，launch overhead 和显存往返是主要耗时。

三管齐下：
1. **IPEX**（Intel Extension for PyTorch）——用 Intel 优化算子和图优化替换 PyTorch 原生后端
2. **torch.compile**——对关键路径做 kernel 融合
3. **BF16**——减半显存带宽需求，对 67MB hash 表 + 大量射线读写效果明显

### 具体改动

```python
import intel_extension_for_pytorch as ipex

# 模型 + 优化器用 IPEX 优化
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
model_fine, _ = ipex.optimize(model_fine, dtype=torch.bfloat16)

# 网络查询函数用 torch.compile
network_query_fn = torch.compile(network_query_fn, backend="xpu")
```

### 验证标准

- 代码不报错
- 速度 ≥ 4 it/s
- Loss 正常收敛

### 回退策略

- `torch.compile` XPU backend 不支持 → 只用 IPEX（3-5 行改动即可）
- BF16 导致 loss 发散 → 回退 FP32
- IPEX 本身有问题 → 跳过 Phase 1，直接 Phase 2

---

## Phase 2: XPU Profiler 热点分析

### 方法

用 PyTorch Profiler（`torch.profiler`）抓 10 个 iteration 的 XPU 执行记录。

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=5)
) as prof:
    # 跑 10 个 iteration
```

### 判断表

| 观察 | 含义 | 行动 |
|------|------|------|
| `aten::embedding` 占 40%+ | hash 编码是主瓶颈 | Phase 3 优先融 hash encoding |
| `aten::linear` / matmul 占 30%+ | MLP 瓶颈 | Phase 3 优先融 MLP |
| kernel launch overhead 占 20%+ | 融合不够 | 更激进的 manual fusion |
| XMX 利用率 < 30% | MLP 太小 | 增大 ray chunk size |
| 显存带宽利用率 > 80% | 带宽瓶颈 | 考虑 compact hash table 或 缓存 |

### 产出

- Top-5 热点 op 表格
- Chrome trace JSON（可视化时间线）

---

## Phase 3: SYCL 算子融合（SYCLomatic 优先 + AI 辅助调试）

### 核心思路

用 Intel 官方的 SYCLomatic 工具自动翻译 tiny-cuda-nn 的 CUDA kernel → DPC++，然后 Claude Code 分析翻译质量、修编译错误、验证正确性。

**角色分工：**
- **Claude Code**：跑 SYCLomatic、读懂翻译输出、修复编译错误、验证 kernel 正确性
- **用户**：执行命令（环境在本地）、看 benchmark 结果、做"继续 / 放弃"决策
- **用户不需要学 GPU 编程**，SYCL 语法、subgroup、shared memory 都由 Claude 处理

### 翻译流水线

```
tiny-cuda-nn CUDA kernel (参考源码)
        │
        ▼  SYCLomatic 自动翻译
        │    dpct --in-root=... --out-root=... grid.cu
        │    自动处理: warp→subgroup, shared mem→local mem, <<<>>>→nd_range
        ▼
SYCL kernel (初稿, 可能有编译/逻辑错误)
        │
        ▼  Claude Code 分析 + 修复
        │    1. 读翻译输出，评估质量
        │    2. 修编译错误（API 不匹配、头文件等）
        │    3. 修逻辑错误（内存模型差异等）
        │    4. 标注不翻译的部分，手动补写
        ▼
SYCL kernel (可编译 + 正确)
        │
        ▼  PyTorch custom op 包装
        │    torch.utils.cpp_extension.load
        │    torch.autograd.Function (forward + backward)
        ▼
集成到 HashNeRF 替换对应模块
```

### 四步实施流程

| 步骤 | 内容 | 谁主导 | 产出 |
|------|------|--------|------|
| **Step A** | 跑 SYCLomatic 翻译 tiny-cuda-nn `grid.h` | Claude 指导, 用户执行 | 翻译后的 `.dp.cpp` 文件 |
| **Step B** | Claude 分析翻译质量、修复编译/逻辑错误 | Claude | 可编译的 DPC++ kernel |
| **Step C** | 包装为 PyTorch custom op + 正确性验证 | Claude 写代码 | `.cpp` + `setup.py`, 单测对比 |
| **Step D** | 集成到 HashNeRF + benchmark 对比 | Claude 改 Python, 用户跑训练 | 速度对比数据 |

### Priority 排序（由 Phase 2 结果决定）

| 优先级 | Kernel | 作用 | tiny-cuda-nn 参考 |
|--------|--------|------|-------------------|
| P0 | Fused Hash Encoding | 16 层 hash 查表 + concatenate → 单 kernel | `grid.h`, `encoding.h` |
| P1 | Fused MLP Forward | Linear → ReLU → Linear 融合 | `network.h` |
| P2 | Fused Ray Marching | 采样 + alpha 累积 融合 | `ray.h`（可选） |

### 退出条件

- SYCL kernel 遇到 Arc A750 硬件限制（如 shared memory 不足）→ 跳过该 kernel
- 翻译完成但加速 < 1.5x → 放弃，回退纯 PyTorch
- 累计加速达 10+ it/s → 目标达成

---

## 回退和止损

整个方案最大的风险是 Phase 3 的 SYCL kernel 翻译遇到难以调试的问题。如果发生：
- Phase 1 和 Phase 2 本身已经有产出（IPEX 加速 + 性能分析数据）
- 可以从 Phase 3 安全退出，把 Phase 1 的结果作为最终方案
- 任何时候可以手写 `train.sh` 脚本封装环境，方便复现

## 依赖

- Intel oneAPI Base Toolkit（DPC++ 编译器 + SYCLomatic）
- IPEX（`intel_extension_for_pytorch`）
- PyTorch Profiler（内置于 PyTorch）
- Intel PTI（可选，GPU 底层 profiler）
