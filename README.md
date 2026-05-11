# translate_cuda_to_intel_a750

将 NVIDIA CUDA NeRF 加速方案翻译为 Intel Arc GPU 可用的 SYCL 实现，让 Intel 显卡也能高效训练 NeRF。

## 目标

tiny-cuda-nn 是 NVIDIA Instant-NGP 的 CUDA 加速后端，通过算子融合将 NeRF 训练从数小时缩短到数分钟。本项目将其核心融合 kernel（hash encoding）翻译为 Intel SYCL/DPC++，使 Intel Arc 显卡获得同等加速效果。

### 阶段进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 纯 PyTorch baseline | ✅ 完成 | Arc A750 训练 50000 次，~1.85 it/s，PSNR 27-31 dB |
| Phase 1: torch.compile + BF16 | ✅ 完成 | 内置图优化 + 混合精度 |
| Phase 2: XPU Profiler | ✅ 完成 | 热点定位：hash 编码 68%，MLP 仅 6% |
| Phase 3: SYCL hash encoding 融合 | 🔄 即将开始 | 翻译 tiny-cuda-nn `grid.h` + `encoding.h` → SYCL |

## 硬件 & 环境

| 组件 | 规格 |
|------|------|
| GPU | Intel Arc A750 (8GB) |
| CPU | AMD Ryzen 5600 |
| OS | Windows 11 |
| Python | 3.14 |
| PyTorch | 2.11.0+xpu |
| oneAPI | Intel oneAPI Base Toolkit |

### 环境搭建

```bash
# 1. 安装 Intel oneAPI Base Toolkit + PyTorch XPU
# 参考: https://pytorch.org/get-started/locally/

# 2. 创建虚拟环境并安装依赖
python -m venv venv_xpu
source venv_xpu/Scripts/activate  # Windows
pip install configargparse imageio opencv-python tqdm matplotlib kornia pyvista imageio-ffmpeg

# 3. 验证 XPU 可用
python -c "import torch; print(torch.xpu.is_available())"
```

## Baseline 训练结果

### 纯 PyTorch (lego 场景, 50000 迭代)

| 指标 | 结果 |
|------|------|
| 训练速度 | ~1.85 it/s |
| 最终 PSNR | 27.2 dB |
| 最佳 PSNR | ~30.9 dB |
| 纯训练时长 | ~7.5 小时 |
| 训练总时长 | ~40 小时 (含渲染) |

### Phase 2 Profiler 结果

```
Hash 编码: ████████████████████████████████ 68%  (sort + embedding + scatter)
逐元素:    ██████████ 19%
MLP:       ███ 6%
其他:      ██ 7%
```

**Phase 3 方向：融合 hash 编码，MLP 非瓶颈。**

## XPU 适配 (相对原版 HashNeRF-pytorch 的修改)

### run_nerf.py
- `L30-35`: 设备优先级 xpu > cuda > cpu
- `L221`: HashEmbedder 转移到 XPU
- `L837`: `torch.set_default_device(device)`
- `L890`: `torch.autocast(device_type='xpu', dtype=torch.bfloat16)` 混合精度
- `L267`: `torch.compile(network_query_fn, backend="xpu")`
- 新增 `--N_iters` 参数

### run_nerf_helpers.py
- `L11`: `mse2psnr` lambda 设备感知
- `L246-251`: `get_rays` 设备感知

### utils.py
- `L9-12`: `BOX_OFFSETS` 静态变量 → `get_box_offsets(device)` 函数
- `L104-115`: `get_voxel_vertices` 中 tensor 显式指定设备

## 目录

```
HashNeRF-pytorch/
├── run_nerf.py              # 主训练脚本 (已修改)
├── run_nerf_helpers.py      # NeRF 模型 (已修改)
├── utils.py                 # 工具函数 (已修改)
├── hash_encoding.py         # Hash 编码
├── configs/                 # 训练配置
├── data/                    # 数据集 (symlink)
└── logs/                    # 训练日志 & checkpoint
docs/superpowers/specs/
└── 2026-05-08-hashnerf-xpu-optimization-design.md
```

## 设计文档

详见 [docs/superpowers/specs/2026-05-08-hashnerf-xpu-optimization-design.md](docs/superpowers/specs/2026-05-08-hashnerf-xpu-optimization-design.md)

## 参考

- 原始 HashNeRF-pytorch: [https://github.com/yashbhalgat/HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch)
- tiny-cuda-nn: [https://github.com/NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- Intel oneAPI: [https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
