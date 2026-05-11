# Phase 3: Fused Hash Encoding SYCL Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HashNeRF's 16-level Python-loop hash encoding (68% of training time) with a single fused DPC++ kernel, wrapped as a PyTorch custom op.

**Architecture:** One DPC++ kernel processes B points across all 16 levels in a single dispatch. Each work-item handles one point x one level: hash lookup → trilinear interpolation → write to output. Kernel is compiled via `torch.utils.cpp_extension.load()` and wrapped in `torch.autograd.Function` for drop-in replacement of `HashEmbedder.forward()`.

**Tech Stack:** Intel DPC++ (icx -fsycl), PyTorch XPU 2.11, torch.utils.cpp_extension, Python 3.14

---

## File Structure

| File | Purpose |
|------|---------|
| `HashNeRF-pytorch/sycl_ops/hash_encode_fwd.cpp` | DPC++ forward kernel + Python binding |
| `HashNeRF-pytorch/sycl_ops/hash_encode_bwd.cpp` | DPC++ backward kernel (if needed) |
| `HashNeRF-pytorch/sycl_ops/__init__.py` | Python load wrapper, autograd.Function |
| `HashNeRF-pytorch/hash_encoding.py` | Modify: add optional custom-op path in `HashEmbedder.forward()` |
| `HashNeRF-pytorch/tests/test_hash_encode.py` | Correctness test: custom op vs PyTorch reference |

---

### Task 1: Verify DPC++ compiler can build a minimal SYCL program

**Files:** Create `HashNeRF-pytorch/sycl_ops/test_dpcpp.cpp`

- [ ] **Step 1: Write a minimal SYCL test program**

Create `HashNeRF-pytorch/sycl_ops/test_dpcpp.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const int N = 1024;
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);

    {
        sycl::buffer<float> buf_a(a.data(), N);
        sycl::buffer<float> buf_b(b.data(), N);
        sycl::buffer<float> buf_c(c.data(), N);

        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                acc_c[i] = acc_a[i] + acc_b[i];
            });
        });
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (std::abs(c[i] - 3.0f) > 1e-5f) errors++;
    }
    std::cout << "Errors: " << errors << std::endl;
    return errors;
}
```

- [ ] **Step 2: Compile and run**

Run:
```bash
cmd.exe /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch\sycl_ops && icx -fsycl test_dpcpp.cpp -o test_dpcpp.exe && test_dpcpp.exe'
```

Expected output: `Device: Intel(R) Arc(TM) A750 Graphics` then `Errors: 0`

- [ ] **Step 3: Commit**

```bash
git add HashNeRF-pytorch/sycl_ops/
git commit -m "feat: verify DPC++ SYCL compilation on Intel Arc A750"
```

---

### Task 2: Download and analyze tiny-cuda-nn hash encoding kernel

**Files:** Create `docs/tiny-cuda-nn-grid-analysis.md`

- [ ] **Step 1: Fetch the full grid.h source**

```bash
curl -s "https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/include/tiny-cuda-nn/encodings/grid.h" -o /z/nerf_and_3dGS/docs/tiny-cuda-nn-grid.h
```

- [ ] **Step 2: Read and analyze the kernel_grid function**

Read `docs/tiny-cuda-nn-grid.h` and write analysis to `docs/tiny-cuda-nn-grid-analysis.md` covering:

1. **Kernel launch config**: `blockIdx.x` = element index, `blockIdx.y` = level index. Grid dims: `(ceil(num_elements/block_size), num_levels)`. Block dim: configurable, typically 128 or 256.
2. **Hash function**: `grid_index<N_POS_DIMS, HASH_TYPE>()` — uses prime multiplication and XOR, similar to Python `hash()` in `utils.py`.
3. **Trilinear interpolation**: 2^N_POS_DIMS (8 for 3D) corner lookups per level, each weighted by `(1-pos[dim])` or `pos[dim]`.
4. **Output layout**: `[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements]` — interleaved by features within levels, then by levels.
5. **Shared memory**: Not used in this kernel (each thread works independently on its element+level pair).
6. **Templates**: `T` (float/half), `N_POS_DIMS` (3), `N_FEATURES_PER_LEVEL` (2), `HASH_TYPE`, `GridType`, `InterpolationType`.

- [ ] **Step 3: Identify the subset we need to implement**

Our simplified version:
- `N_POS_DIMS=3`, `N_FEATURES_PER_LEVEL=2` (fixed, matches HashNeRF)
- `InterpolationType::Linear` only (no Smoothstep)
- `GridType::Hash` only (no Dense / Tiled)
- `HASH_TYPE` = standard prime-product hash (matching Python `utils.hash()`)
- **No gradients in first iteration** — we can implement backward with PyTorch autograd over the custom op, or add gradient kernel later
- **No `max_level_gpu`** — use uniform max_level for all points

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: analyze tiny-cuda-nn grid.h hash encoding kernel"
```

---

### Task 3: Write the fused hash encoding DPC++ forward kernel

**Files:** Create `HashNeRF-pytorch/sycl_ops/hash_encode_fwd.cpp`

Core algorithm (each work-item processes one point at one level):

```
For point i, level L:
  1. Compute resolution = base_res * b^L
  2. Scale position to grid coordinates: pos_grid = (pos - bbox_min) / (bbox_max - bbox_min) * resolution
  3. Split into integer (voxel index) and fractional (interpolation weight) parts
  4. For each of 8 corners of the voxel:
     a. Compute corner coordinate = voxel_min + corner_offset
     b. Hash corner coordinate → hash table index
     c. Lookup embedding[corner_idx]
     d. Accumulate: result += embedding * trilinear_weight
  5. Write result to output[point_idx * 32 + level * 2 + feature]
```

- [ ] **Step 1: Write the DPC++ kernel**

Create `HashNeRF-pytorch/sycl_ops/hash_encode_fwd.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <cmath>

// --- Hash function (matches Python utils.py hash()) ---
// primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
static constexpr uint32_t PRIMES[7] = {
    1, 2654435761u, 805459861u, 3674653429u,
    2097192037u, 1434869437u, 2165219737u
};

inline uint32_t hash_coords(int32_t x, int32_t y, int32_t z) {
    uint32_t result = 0;
    result ^= static_cast<uint32_t>(x) * PRIMES[0];
    result ^= static_cast<uint32_t>(y) * PRIMES[1];
    result ^= static_cast<uint32_t>(z) * PRIMES[2];
    return result;
}

// --- SYCL kernel ---
// Processes B points x 16 levels. Each work-item = one (point, level) pair.
// Grid: global_range = (B, 16); local_range = (1, 1) or (B, 1) depending on tuning.

torch::Tensor fused_hash_encode_forward(
    const torch::Tensor& points,           // [B, 3] float32
    const torch::Tensor& embeddings,       // [16 * T, 2] float32 (flattened 16 hash tables of size T)
    const torch::Tensor& bbox_min,         // [3] float32
    const torch::Tensor& bbox_max,         // [3] float32
    float base_resolution,                 // e.g. 16.0
    float b,                               // growth factor per level
    int log2_hashmap_size                  // e.g. 19
) {
    int64_t B = points.size(0);
    int T = 1 << log2_hashmap_size;
    constexpr int N_LEVELS = 16;
    constexpr int N_FEATURES = 2;

    auto output = torch::zeros({B, N_LEVELS * N_FEATURES}, points.options());
    auto keep_mask = torch::ones({B}, points.options().dtype(torch::kBool));

    // Accessor pattern for efficient GPU access
    auto points_acc = points.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
    auto embeddings_acc = embeddings.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
    auto output_acc = output.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
    auto bbox_min_acc = bbox_min.packed_accessor32<float, 1, torch::RestrictPtrTraits>();
    auto bbox_max_acc = bbox_max.packed_accessor32<float, 1, torch::RestrictPtrTraits>();

    // Get the current XPU stream from PyTorch
    sycl::queue q = sycl::ext::oneapi::get_queue_from_tensor(points);

    // Get raw pointers for the kernel
    const float* points_ptr = points.data_ptr<float>();
    const float* embeddings_ptr = embeddings.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    float bbox_min_x = bbox_min_acc[0];
    float bbox_min_y = bbox_min_acc[1];
    float bbox_min_z = bbox_min_acc[2];
    float bbox_size_x = bbox_max_acc[0] - bbox_min_acc[0];
    float bbox_size_y = bbox_max_acc[1] - bbox_min_acc[1];
    float bbox_size_z = bbox_max_acc[2] - bbox_min_acc[2];

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::range<2>(static_cast<size_t>(B), static_cast<size_t>(N_LEVELS)),
            [=](sycl::id<2> idx) {
                int64_t point_idx = idx[0];
                int level = idx[1];

                // Compute resolution for this level
                float resolution = base_resolution * sycl::pow(b, static_cast<float>(level));

                // Get point coordinates
                float px = points_ptr[point_idx * 3 + 0];
                float py = points_ptr[point_idx * 3 + 1];
                float pz = points_ptr[point_idx * 3 + 2];

                // Clamp to bounding box
                px = sycl::fmin(sycl::fmax(px, bbox_min_x), bbox_max_x);
                py = sycl::fmin(sycl::fmax(py, bbox_min_y), bbox_max_y);
                pz = sycl::fmin(sycl::fmax(pz, bbox_min_z), bbox_max_z);

                // Scale to grid coordinates
                float grid_x = (px - bbox_min_x) / bbox_size_x * resolution;
                float grid_y = (py - bbox_min_y) / bbox_size_y * resolution;
                float grid_z = (pz - bbox_min_z) / bbox_size_z * resolution;

                // Integer corner + fractional weights
                int32_t vx = static_cast<int32_t>(sycl::floor(grid_x));
                int32_t vy = static_cast<int32_t>(sycl::floor(grid_y));
                int32_t vz = static_cast<int32_t>(sycl::floor(grid_z));

                float fx = grid_x - static_cast<float>(vx);
                float fy = grid_y - static_cast<float>(vy);
                float fz = grid_z - static_cast<float>(vz);

                // Trilinear interpolation over 8 corners
                float feat0 = 0.0f;
                float feat1 = 0.0f;

                for (int corner = 0; corner < 8; corner++) {
                    int32_t cx = vx + ((corner >> 0) & 1);
                    int32_t cy = vy + ((corner >> 1) & 1);
                    int32_t cz = vz + ((corner >> 2) & 1);

                    // Trilinear weight
                    float wx = ((corner >> 0) & 1) ? fx : (1.0f - fx);
                    float wy = ((corner >> 1) & 1) ? fy : (1.0f - fy);
                    float wz = ((corner >> 2) & 1) ? fz : (1.0f - fz);
                    float weight = wx * wy * wz;

                    // Hash the corner coordinate
                    uint32_t h = hash_coords(cx, cy, cz);
                    uint32_t hash_idx = h & (static_cast<uint32_t>(T) - 1);

                    // Lookup embedding: embeddings[level * T + hash_idx]
                    int64_t emb_offset = static_cast<int64_t>(level) * T + static_cast<int64_t>(hash_idx);
                    feat0 += weight * embeddings_ptr[emb_offset * N_FEATURES + 0];
                    feat1 += weight * embeddings_ptr[emb_offset * N_FEATURES + 1];
                }

                // Write output: [point_idx, level*2 + 0] and [point_idx, level*2 + 1]
                output_ptr[point_idx * (N_LEVELS * N_FEATURES) + level * N_FEATURES + 0] = feat0;
                output_ptr[point_idx * (N_LEVELS * N_FEATURES) + level * N_FEATURES + 1] = feat1;
            });
    });

    q.wait();

    return output;
}

// --- Python binding ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_hash_encode_forward, "Fused hash encoding forward");
}
```

- [ ] **Step 2: Commit**

```bash
git add HashNeRF-pytorch/sycl_ops/hash_encode_fwd.cpp
git commit -m "feat: DPC++ fused hash encoding forward kernel"
```

---

### Task 4: Write the Python-side custom op wrapper

**Files:** Create `HashNeRF-pytorch/sycl_ops/__init__.py`, create `HashNeRF-pytorch/sycl_ops/setup.py`

- [ ] **Step 1: Write the autograd.Function wrapper**

Create `HashNeRF-pytorch/sycl_ops/__init__.py`:

```python
import torch
from torch.utils.cpp_extension import load
import os

_sycl_ops_path = os.path.dirname(os.path.abspath(__file__))

_fused_hash_encode = load(
    name="fused_hash_encode",
    sources=[os.path.join(_sycl_ops_path, "hash_encode_fwd.cpp")],
    extra_cflags=["-fsycl"],
    extra_ldflags=["-fsycl"],
    verbose=True,
)


class FusedHashEncode(torch.autograd.Function):
    """Fused 16-level hash encoding, drop-in replacement for HashEmbedder.forward().
    
    Input:
        points: [B, 3] float32 tensor
        embeddings: [16, T, 2] float32 tensor (16 hash tables of size T)
        bbox_min: [3] float32
        bbox_max: [3] float32
        base_resolution: float
        b: float (growth factor)
        log2_hashmap_size: int
    
    Returns:
        encoded: [B, 32] float32 (16 levels x 2 features concatenated)
    """
    
    @staticmethod
    def forward(ctx, points, embeddings, bbox_min, bbox_max,
                base_resolution, b, log2_hashmap_size):
        # Flatten embeddings: [16, T, 2] -> [16*T, 2]
        N_LEVELS, T, N_FEAT = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, N_FEAT).contiguous()
        
        output = _fused_hash_encode.forward(
            points, embeddings_flat,
            bbox_min, bbox_max,
            base_resolution, b, log2_hashmap_size
        )
        
        ctx.save_for_backward(points, embeddings, bbox_min, bbox_max)
        ctx.constants = (base_resolution, b, log2_hashmap_size)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Fallback: use PyTorch autograd through the forward computation.
        # For performance, we can add a DPC++ backward kernel later.
        # The forward is the critical path (training spends most time there).
        raise NotImplementedError(
            "Backward not implemented in SYCL yet. "
            "Use the pure-PyTorch HashEmbedder for training, "
            "or set requires_grad=False on positions."
        )
```

- [ ] **Step 2: Write the integrated forward method for HashEmbedder**

Modify `HashNeRF-pytorch/hash_encoding.py` to add a fast-path in `HashEmbedder.forward()`:

```python
# Add near the top of hash_encoding.py, after imports:
import os
_USE_SYCL = os.environ.get("HASHNERF_USE_SYCL", "0") == "1"
if _USE_SYCL:
    from sycl_ops import FusedHashEncode

# Add inside HashEmbedder.forward(), before the existing loop:
def forward(self, x):
    # SYCL fast path
    if _USE_SYCL:
        bbox_min, bbox_max = self.bounding_box
        encoded = FusedHashEncode.apply(
            x,
            torch.stack([e.weight for e in self.embeddings]),  # [16, T, 2]
            bbox_min, bbox_max,
            float(self.base_resolution),
            float(self.b),
            self.log2_hashmap_size
        )
        keep_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        return encoded, keep_mask
    
    # Original PyTorch path (unchanged)
    x_embedded_all = []
    for i in range(self.n_levels):
        ...
```

- [ ] **Step 3: Write correctness test**

Create `HashNeRF-pytorch/tests/test_hash_encode.py`:

```python
import torch
import sys
sys.path.insert(0, "..")

from hash_encoding import HashEmbedder
from sycl_ops import FusedHashEncode
import os
os.environ["HASHNERF_USE_SYCL"] = "0"


def test_fused_same_as_pytorch():
    """SYCL kernel must match pure PyTorch output to within 1e-4."""
    device = torch.device("xpu")
    
    # Setup identical to HashNeRF
    bounding_box = (torch.tensor([-1.5, -1.5, -1.5], device=device),
                    torch.tensor([1.5, 1.5, 1.5], device=device))
    
    embedder = HashEmbedder(bounding_box, n_levels=16, n_features_per_level=2,
                            log2_hashmap_size=19, base_resolution=16,
                            finest_resolution=512).to(device)
    
    # Random points
    B = 1024
    torch.manual_seed(42)
    points = torch.rand(B, 3, device=device) * 3.0 - 1.5  # [-1.5, 1.5]
    points.requires_grad_(False)
    
    # PyTorch reference
    ref_output, ref_mask = embedder(points)
    
    # SYCL output
    embeddings = torch.stack([e.weight for e in embedder.embeddings])
    sycl_output = FusedHashEncode.apply(
        points, embeddings,
        bounding_box[0], bounding_box[1],
        16.0, float(embedder.b), embedder.log2_hashmap_size
    )
    
    max_diff = (ref_output - sycl_output).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")
    
    assert max_diff < 1e-4, f"SYCL output differs from PyTorch! max_diff={max_diff}"
    print("PASSED: SYCL matches PyTorch")


if __name__ == "__main__":
    test_fused_same_as_pytorch()
```

- [ ] **Step 4: Commit**

```bash
git add HashNeRF-pytorch/sycl_ops/__init__.py HashNeRF-pytorch/hash_encoding.py HashNeRF-pytorch/tests/
git commit -m "feat: PyTorch custom op wrapper for fused hash encoding"
```

---

### Task 5: Compile, test correctness, and benchmark

**Files:** Modify `HashNeRF-pytorch/sycl_ops/__init__.py` (if needed for build fixes)

- [ ] **Step 1: Source oneAPI and run correctness test**

```bash
cmd.exe /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch && python tests/test_hash_encode.py'
```

Expected: `Max difference: < 1e-4` then `PASSED`

- [ ] **Step 2: If test fails, debug and fix**

- Compare intermediate values: print first 10 outputs from Python and SYCL
- Check hash function matches Python `utils.hash()`
- Check trilinear interpolation order matches Python `HashEmbedder.trilinear_interp()`
- Check embedding weight indexing matches

Fix, re-run until PASSED. Each fix → separate commit.

- [ ] **Step 3: Micro-benchmark**

Create benchmark script `HashNeRF-pytorch/tests/bench_hash_encode.py`:

```python
import torch
import sys, time
sys.path.insert(0, "..")
from hash_encoding import HashEmbedder
from sycl_ops import FusedHashEncode

device = torch.device("xpu")
bounding_box = (torch.tensor([-1.5, -1.5, -1.5], device=device),
                torch.tensor([1.5, 1.5, 1.5], device=device))

embedder = HashEmbedder(bounding_box).to(device)
embeddings = torch.stack([e.weight for e in embedder.embeddings])

sizes = [1024, 4096, 16384, 65536]
for B in sizes:
    points = torch.rand(B, 3, device=device)
    
    # Warmup
    for _ in range(10):
        embedder(points)
        FusedHashEncode.apply(points, embeddings, *bounding_box, 16.0, float(embedder.b), 19)
    torch.xpu.synchronize()
    
    # Benchmark PyTorch
    t0 = time.time()
    for _ in range(100):
        embedder(points)
    torch.xpu.synchronize()
    dt_pytorch = (time.time() - t0) / 100
    
    # Benchmark SYCL
    t0 = time.time()
    for _ in range(100):
        FusedHashEncode.apply(points, embeddings, *bounding_box, 16.0, float(embedder.b), 19)
    torch.xpu.synchronize()
    dt_sycl = (time.time() - t0) / 100
    
    print(f"B={B:6d} | PyTorch: {dt_pytorch*1000:.2f}ms | SYCL: {dt_sycl*1000:.2f}ms | Speedup: {dt_pytorch/dt_sycl:.2f}x")
```

- [ ] **Step 4: Run benchmark**

```bash
cmd.exe /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch && python tests/bench_hash_encode.py'
```

Expected: SYCL speedup >= 3x over PyTorch. If < 1.5x, evaluate whether to continue.

- [ ] **Step 5: Commit benchmark results**

```bash
git add HashNeRF-pytorch/tests/
git commit -m "test: add hash encoding correctness and benchmark tests"
```

---

### Task 6: Add backward pass and end-to-end training

**Files:** Create `HashNeRF-pytorch/sycl_ops/hash_encode_bwd.cpp`, modify `sycl_ops/__init__.py`

- [ ] **Step 1: Write the DPC++ backward kernel**

The backward pass computes gradients w.r.t. embeddings (hash table weights) for each corner of the voxel.
Same structure as forward but accumulates gradients instead of output.

Create `HashNeRF-pytorch/sycl_ops/hash_encode_bwd.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <torch/extension.h>

// (same hash function and constants as forward)

torch::Tensor fused_hash_encode_backward(
    const torch::Tensor& grad_output,     // [B, 32]
    const torch::Tensor& points,          // [B, 3]
    const torch::Tensor& embeddings,      // [16 * T, 2]
    const torch::Tensor& bbox_min,        // [3]
    const torch::Tensor& bbox_max,        // [3]
    float base_resolution,
    float b,
    int log2_hashmap_size
) {
    int64_t B = points.size(0);
    int T = 1 << log2_hashmap_size;
    constexpr int N_LEVELS = 16;
    constexpr int N_FEATURES = 2;

    auto grad_embeddings = torch::zeros_like(embeddings);

    const float* points_ptr = points.data_ptr<float>();
    const float* grad_out_ptr = grad_output.data_ptr<float>();
    const float* embeddings_ptr = embeddings.data_ptr<float>();
    float* grad_emb_ptr = grad_embeddings.data_ptr<float>();

    float bbox_min_x = bbox_min[0].item<float>();
    float bbox_min_y = bbox_min[1].item<float>();
    float bbox_min_z = bbox_min[2].item<float>();
    float bbox_size_x = bbox_max[0].item<float>() - bbox_min_x;
    float bbox_size_y = bbox_max[1].item<float>() - bbox_min_y;
    float bbox_size_z = bbox_max[2].item<float>() - bbox_min_z;

    sycl::queue q = sycl::ext::oneapi::get_queue_from_tensor(points);

    // Use atomic operations to accumulate gradient into shared hash table
    // For simplicity, use a scalar loop per (point, level, corner) — 
    // atomic contention is low because hash distributes indices well.

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::range<2>(static_cast<size_t>(B), static_cast<size_t>(N_LEVELS)),
            [=](sycl::id<2> idx) {
                int64_t point_idx = idx[0];
                int level = idx[1];

                float resolution = base_resolution * sycl::pow(b, static_cast<float>(level));

                float px = points_ptr[point_idx * 3 + 0];
                float py = points_ptr[point_idx * 3 + 1];
                float pz = points_ptr[point_idx * 3 + 2];

                px = sycl::fmin(sycl::fmax(px, bbox_min_x), bbox_max_x);
                py = sycl::fmin(sycl::fmax(py, bbox_min_y), bbox_max_y);
                pz = sycl::fmin(sycl::fmax(pz, bbox_min_z), bbox_max_z);

                float grid_x = (px - bbox_min_x) / bbox_size_x * resolution;
                float grid_y = (py - bbox_min_y) / bbox_size_y * resolution;
                float grid_z = (pz - bbox_min_z) / bbox_size_z * resolution;

                int32_t vx = static_cast<int32_t>(sycl::floor(grid_x));
                int32_t vy = static_cast<int32_t>(sycl::floor(grid_y));
                int32_t vz = static_cast<int32_t>(sycl::floor(grid_z));

                float fx = grid_x - static_cast<float>(vx);
                float fy = grid_y - static_cast<float>(vy);
                float fz = grid_z - static_cast<float>(vz);

                // Gradients w.r.t. each of 2 features from this (point, level)
                float go0 = grad_out_ptr[point_idx * (N_LEVELS * N_FEATURES) + level * N_FEATURES + 0];
                float go1 = grad_out_ptr[point_idx * (N_LEVELS * N_FEATURES) + level * N_FEATURES + 1];
                
                if (go0 == 0.0f && go1 == 0.0f) return;

                for (int corner = 0; corner < 8; corner++) {
                    int32_t cx = vx + ((corner >> 0) & 1);
                    int32_t cy = vy + ((corner >> 1) & 1);
                    int32_t cz = vz + ((corner >> 2) & 1);

                    float wx = ((corner >> 0) & 1) ? fx : (1.0f - fx);
                    float wy = ((corner >> 1) & 1) ? fy : (1.0f - fy);
                    float wz = ((corner >> 2) & 1) ? fz : (1.0f - fz);
                    float weight = wx * wy * wz;

                    uint32_t h = hash_coords(cx, cy, cz);
                    uint32_t hash_idx = h & (static_cast<uint32_t>(T) - 1);
                    int64_t emb_offset = static_cast<int64_t>(level) * T + static_cast<int64_t>(hash_idx);

                    // Atomic accumulate: grad += weight * grad_output
                    // Use atomic_ref for float (available in SYCL 2020)
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atm0(grad_emb_ptr[emb_offset * N_FEATURES + 0]);
                    atm0.fetch_add(weight * go0);

                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                        atm1(grad_emb_ptr[emb_offset * N_FEATURES + 1]);
                    atm1.fetch_add(weight * go1);
                }
            });
    });

    q.wait();
    return grad_embeddings;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &fused_hash_encode_backward, "Fused hash encoding backward");
}
```

- [ ] **Step 2: Update __init__.py to use the backward kernel**

Add backward support in `FusedHashEncode.backward()`:

```python
# In __init__.py, update the load call to include backward source:
_fused_hash_encode_bwd = load(
    name="fused_hash_encode_bwd",
    sources=[os.path.join(_sycl_ops_path, "hash_encode_bwd.cpp")],
    extra_cflags=["-fsycl"],
    extra_ldflags=["-fsycl"],
    verbose=True,
)

# Update FusedHashEncode.backward():
@staticmethod
def backward(ctx, grad_output):
    points, embeddings, bbox_min, bbox_max = ctx.saved_tensors
    base_resolution, b, log2_hashmap_size = ctx.constants
    
    embeddings_flat = embeddings.reshape(-1, 2).contiguous()
    
    grad_embeddings_flat = _fused_hash_encode_bwd.backward(
        grad_output, points, embeddings_flat,
        bbox_min, bbox_max,
        base_resolution, b, log2_hashmap_size
    )
    
    grad_embeddings = grad_embeddings_flat.reshape(16, -1, 2)
    # No gradient for points, bbox_min, bbox_max, base_resolution, b, log2_hashmap_size
    return None, grad_embeddings, None, None, None, None, None
```

- [ ] **Step 3: Full training run with SYCL enabled**

```bash
cmd.exe /c 'call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cd /d Z:\nerf_and_3dGS\HashNeRF-pytorch && set HASHNERF_USE_SYCL=1 && python run_nerf.py --config configs/lego.txt'
```

Run for 5000 iterations and compare:
- Training speed (it/s) vs Phase 1 baseline (~1.85 it/s)
- PSNR convergence vs Phase 0 baseline
- GPU memory usage

- [ ] **Step 4: Commit end-to-end integration**

```bash
git add HashNeRF-pytorch/sycl_ops/ HashNeRF-pytorch/hash_encoding.py
git commit -m "feat: add SYCL backward pass and end-to-end training integration"
```

---

### Task 7: Tune and document

**Files:** Create `docs/phase3-results.md`

- [ ] **Step 1: Tune work-group size**

Try work-group sizes: 32, 64, 128, 256. Adjust kernel launch config in `hash_encode_fwd.cpp`.  
Record the fastest configuration.

- [ ] **Step 2: Write results summary**

Create `docs/phase3-results.md`:

```markdown
# Phase 3 Results: Fused Hash Encoding SYCL Kernel

## Environment
- GPU: Intel Arc A750 (8GB)
- Compiler: Intel DPC++ 2026.0.0
- PyTorch: 2.11.0+xpu

## Micro-benchmark (hash encoding only)

| B (points) | PyTorch (ms) | SYCL (ms) | Speedup |
|------------|-------------|-----------|---------|
| 1024       | ...         | ...       | ...x    |
| 4096       | ...         | ...       | ...x    |
| 16384      | ...         | ...       | ...x    |
| 65536      | ...         | ...       | ...x    |

## End-to-end training benchmark (lego scene, 5000 iters)

| Metric | PyTorch baseline | SYCL fused | Delta |
|--------|-----------------|------------|-------|
| it/s   | ~1.85           | ...        | ...x  |
| PSNR   | ~27.3           | ...        | ...   |
| GPU mem | ... GB         | ... GB     | ...   |

## Conclusion

...
```

- [ ] **Step 3: Final commit**

```bash
git add docs/phase3-results.md
git commit -m "docs: Phase 3 results and benchmark data"
```

---

## Exit Criteria

- [ ] SYCL kernel compiles and runs on Arc A750
- [ ] Correctness: max difference vs PyTorch < 1e-4
- [ ] Speedup >= 1.5x in micro-benchmark (otherwise evaluate abandonment)
- [ ] End-to-end training PSNR within 1% of PyTorch baseline
- [ ] If speedup >= 3x → Phase 3 declared success, proceed to Phase 4 (MLP fusion) if needed
