# tiny-cuda-nn Grid Encoding Kernel Analysis

Analysis of `kernel_grid` from `tiny-cuda-nn/include/tiny-cuda-nn/encodings/grid.h` (NVlabs/tiny-cuda-nn, master branch).

> Source files analyzed:
> - `Z:\nerf_and_3dGS\docs\tiny-cuda-nn-grid.h` (the encoding kernel and surrounding class logic)
> - `Z:\nerf_and_3dGS\docs\tiny-cuda-nn-common_device.h` (helper functions: hash, position fract, resolution scaling)
> - `Z:\nerf_and_3dGS\HashNeRF-pytorch\utils.py` (Python reference hash implementation)
> - `Z:\nerf_and_3dGS\HashNeRF-pytorch\hash_encoding.py` (Python embedding + trilinear interpolation)

---

## 1. Kernel Launch Configuration

### 1.1 Grid and Block Dimensions

```cpp
// grid.h line 773-774
static constexpr uint32_t N_THREADS_HASHGRID = 512;
const dim3 blocks_hashgrid = { div_round_up(num_elements, N_THREADS_HASHGRID), m_n_levels, 1 };
```

- **Block size**: `(512, 1, 1)` -- 512 threads per block, 1D.
- **Grid size**: `(ceil(num_elements / 512), m_n_levels, 1)` -- 2D grid.
- **Total thread count**: up to `512 * ceil(B / 512) * L` where `B` = num_elements, `L` = num levels.

### 1.2 Thread-to-Work Mapping

```cpp
const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;   // element index  (0..B-1)
const uint32_t level = blockIdx.y;                            // level index    (0..L-1)
if (i >= num_elements) return;
```

Each thread handles exactly **one (element, level) pair**. The x-dimension of the grid distributes elements across blocks; the y-dimension iterates over levels. This design ensures that all threads in a block work on the same level, which keeps one hash table level in cache at a time -- a deliberate cache optimization noted in the source:

> "Idea: each block only takes care of _one_ hash level (but may iterate over multiple input elements). This way, only one level of the hashmap needs to fit into caches at a time."

### 1.3 Typical Launch Parameters

For HashNeRF with B=65536 rays * 192 samples = ~12.5M points and L=16 levels:
- `grid.x = ceil(12.5M / 512) = 24414`
- `grid.y = 16`
- Total threads launched: `24414 * 512 * 16 = 200 million`

---

## 2. Hash Function: `grid_index`

### 2.1 Call Chain

```
kernel_grid()
  -> grid_val(lambda) called for each corner
    -> grid_index<N_POS_DIMS, HASH_TYPE>(grid_type, hashmap_size, resolution, local_pos)
      -> grid_hash<N_POS_DIMS, HASH_TYPE>(pos_grid)       [if Hash + overflow]
        -> coherent_prime_hash<N_DIMS>(pos_grid)
          -> lcg_hash<N_DIMS, 7>(pos_grid, factors)
```

### 2.2 `grid_index` Logic

```cpp
// common_device.h lines 847-884
template <uint32_t N_DIMS, HashType HASH_TYPE>
__device__ uint32_t grid_index(const GridType grid_type, const uint32_t hashmap_size,
                               const uint32_t grid_resolution, const uvec<N_DIMS>& pos_grid) {
    uint32_t stride = 1;
    uint32_t index = 0;

    // If resolution is small enough not to overflow 32-bit multiplication:
    if (grid_resolution <= MAX_BASES[N_DIMS]) {
        // Compute dense index: pos_grid[0] + pos_grid[1]*res + pos_grid[2]*res^2
        for (uint32_t dim = 0; dim < N_DIMS; ++dim) {
            index += pos_grid[dim] * stride;
            stride *= grid_resolution;
        }
    } else {
        stride = 0xFFFFFFFF;  // sentinel: resolution too large, must use hash
    }

    // If hash table size < dense stride, switch to hash-based indexing
    if (grid_type == GridType::Hash && hashmap_size < stride) {
        index = grid_hash<N_DIMS, HASH_TYPE>(pos_grid);
    }

    return index % hashmap_size;
}
```

Key insight: The hash is only applied when the hash table is **smaller** than the dense grid. This is the defining characteristic of multi-resolution hash encoding -- coarse levels use dense indexing, fine levels use hash indexing. `MAX_BASES[3] = 0x659 = 1625` for N_DIMS=3, so any 3D grid with resolution > 1625 always takes the hash path.

### 2.3 Prime-Product Hash

```cpp
// common_device.h lines 781-791
template <uint32_t N_DIMS>
__device__ uint32_t coherent_prime_hash(const uvec<N_DIMS>& pos_grid) {
    constexpr uint32_t factors[7] = { 1u, 2654435761u, 805459861u, 3674653429u,
                                       2097192037u, 1434869437u, 2165219737u };
    return lcg_hash<N_DIMS, 7>(pos_grid, factors);
}

template <uint32_t N_DIMS, uint32_t N_PRIMES>
__device__ uint32_t lcg_hash(const uvec<N_DIMS>& pos_grid, const uint32_t primes[N_PRIMES]) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < N_DIMS; ++i) {
        result ^= pos_grid[i] * primes[i];    // XOR of (coordinate * prime)
    }
    return result;
}
```

For N_POS_DIMS=3: `hash = (x*1) ^ (y*2654435761) ^ (z*805459861)`, then `hash % hashmap_size`.

### 2.4 Comparison with Python `hash()` in `utils.py`

```python
# utils.py lines 15-26
def hash(coords, log2_hashmap_size):
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]
    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result
```

| Aspect | CUDA (`coherent_prime_hash`) | Python (`hash()`) |
|--------|------|--------|
| Primes | `[1, 2654435761, 805459861, ...]` | `[1, 2654435761, 805459861, ...]` (identical) |
| Hash formula | `(x*p0) ^ (y*p1) ^ (z*p2)` | `(x*p0) ^ (y*p1) ^ (z*p2)` (identical) |
| Modulo | `hash % hashmap_size` (general `%`) | `hash & ((1<<log2)-1)` (bitmask, power-of-2 only) |
| Overflow handling | Implicit mod-2^32 via uint32_t | Python big-integers (no overflow) |

Since `hashmap_size = 2^log2_hashmap_size` is always a power of 2 in our use case, the two are mathematically equivalent. Both use the **CoherentPrime** variant where the first factor is `1` (as opposed to `Prime` which uses `1958374283u` as the first factor). This is significant because `x*1 = x`, meaning the x-coordinate passes through unmodified, which helps with spatial coherence.

### 2.5 Available Hash Types in tiny-cuda-nn

| HashType | First factor | Description |
|----------|-------------|-------------|
| `Prime` | `1958374283u` | Original prime-product hash |
| `CoherentPrime` | `1u` | Same primes, but first factor=1 for spatial coherence |
| `ReversedPrime` | `2165219737u` | Primes in reverse order |
| `Rng` | (none) | PRNG-based hash using seed 1337 |
| `BaseConvert` | (none) | `sum(pos)*2531011`, repeated per dim |

**Our target:** `CoherentPrime` -- matches the Python hash exactly.

---

## 3. Trilinear (N-Linear) Interpolation

### 3.1 Position Fract (`pos_fract`)

Before interpolation, input positions are transformed:

```cpp
// common_device.h lines 1016-1029
template <typename F, typename FPRIME>
__device__ inline void pos_fract(const float input, float* pos, float* pos_derivative,
                                 uint32_t* pos_grid, float scale,
                                 F interpolation_fun, FPRIME interpolation_fun_derivative) {
    *pos = fmaf(scale, input, 0.5f);        // pos = scale * input + 0.5
    float tmp = floorf(*pos);                // integer part
    *pos_grid = (uint32_t)(int)tmp;         // grid cell index
    *pos -= tmp;                             // fractional part (0..1)
    *pos_derivative = interpolation_fun_derivative(*pos);
    *pos = interpolation_fun(*pos);          // apply smoothstep/identity
}
```

The **0.5 offset** is critical and comes from Appendix A of the Instant NGP paper. It staggers different resolution levels so that fractional coordinates don't spuriously align at integer scales.

For the `InterpolationType::Linear` path, `interpolation_fun = identity_fun` (which returns `val` unchanged), so `pos` simply contains the fractional coordinate `[0, 1)`.

For `InterpolationType::Smoothstep`, `pos = pos*pos*(3 - 2*pos)`, giving smoother interpolation at the cost of slightly more computation.

### 3.2 N-Linear Interpolation Loop

The kernel handles 2^N_POS_DIMS = 8 corners for 3D:

```cpp
// grid.h lines 146-163
tvec<T, N_FEATURES_PER_LEVEL, ...> result = {};

for (uint32_t idx = 0; idx < (1 << N_POS_DIMS); ++idx) {   // 0..7 for 3D
    float weight = 1;
    uvec<N_POS_DIMS> pos_grid_local;

    for (uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
        if ((idx & (1<<dim)) == 0) {
            weight *= 1 - pos[dim];           // weight contribution from this dim
            pos_grid_local[dim] = pos_grid[dim];        // lower corner
        } else {
            weight *= pos[dim];
            pos_grid_local[dim] = pos_grid[dim] + 1;    // upper corner
        }
    }

    result = fma((T)weight, grid_val(pos_grid_local), result);  // result += weight * grid_val
}
```

### 3.3 Corner Index to Binary Encoding

For 3D with `idx = 0..7`:

| idx | Binary | Corner offset (dx, dy, dz) | Weight factor |
|-----|--------|---------------------------|---------------|
| 0   | 000    | (0, 0, 0) | `(1-x)*(1-y)*(1-z)` |
| 1   | 001    | (1, 0, 0) | `x*(1-y)*(1-z)` |
| 2   | 010    | (0, 1, 0) | `(1-x)*y*(1-z)` |
| 3   | 011    | (1, 1, 0) | `x*y*(1-z)` |
| 4   | 100    | (0, 0, 1) | `(1-x)*(1-y)*z` |
| 5   | 101    | (1, 0, 1) | `x*(1-y)*z` |
| 6   | 110    | (0, 1, 1) | `(1-x)*y*z` |
| 7   | 111    | (1, 1, 1) | `x*y*z` |

### 3.4 Comparison with Python `trilinear_interp`

```python
# hash_encoding.py lines 32-56
def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3
    # step 1: interpolate along x-axis for each yz-pair
    c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
    c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
    c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
    c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]
    # step 2: interpolate along y-axis
    c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
    c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]
    # step 3: interpolate along z-axis
    c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]
    return c
```

| Aspect | CUDA kernel | Python |
|--------|------------|--------|
| Corner ordering | Generic bit-encoding loop over `idx` | Expanded 3-step x-y-z interpolation |
| Weight computation | `pos[dim]` (0..1 fractional, product across dims) | `(x-min)/(max-min)` (same idea) |
| Operations | `fma(weight, grid_val, result)` | `(1-w)*v0 + w*v1` |
| Mathematical equivalence | Exact (both compute weighted sum of 8 corners) | Same |

---

## 4. Output Layout

### 4.1 Interleaved (SoA) Layout

The kernel writes output in Structure-of-Arrays format:

```cpp
encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
```

For B elements, L=16 levels, F=2 features per level:

```
Memory layout (SoA, contiguous):
  Offset 0:          elem_0_level_0_feat_0, elem_1_level_0_feat_0, ..., elem_{B-1}_level_0_feat_0
  Offset B*1:        elem_0_level_0_feat_1, elem_1_level_0_feat_1, ..., elem_{B-1}_level_0_feat_1
  Offset B*2:        elem_0_level_1_feat_0, elem_1_level_1_feat_0, ..., elem_{B-1}_level_1_feat_0
  Offset B*3:        elem_0_level_1_feat_1, elem_1_level_1_feat_1, ..., elem_{B-1}_level_1_feat_1
  ...
  Offset B*31:       elem_0_level_15_feat_1, ..., elem_{B-1}_level_15_feat_1
```

This SoA layout is optimized for coalesced memory access: consecutive threads access consecutive memory addresses (good for GPU), and writes from different levels are separated by B-element strides.

### 4.2 Coalescing Rationale

When 512 threads in a block all write `encoded_positions[i + k*B]` for the same level `k`, thread `i` and thread `i+1` access addresses separated by 1 element (not B). This gives perfect coalescing. Without this layout, threads writing per-element would stride by `L*F` features, causing poor memory throughput.

### 4.3 Transpose to AoS for MLP Consumption

After the forward kernel, if the neural network expects Array-of-Structures (all features for one point contiguous), a transpose kernel converts SoA to AoS:

```
AoS layout: [elem_0_feat_0..31, elem_1_feat_0..31, ..., elem_{B-1}_feat_0..31]
```

This transformation is in `transpose_encoded_position` (lines 807-811).

### 4.4 Our Target Layout

For HashNeRF-pytorch, the Python code produces output of shape `[B, L*F]` = `[B, 32]` (Array of Structures). This is the format we should produce directly from our DPC++ kernel, **bypassing the SoA intermediate step**. Since we're writing a fused kernel that both hashes and interpolates, we can write directly to AoS layout:

```python
# Target: [B, 32] tensor
# encoded[b, level*F + f] = result for element b, level l, feature f
```

---

## 5. Shared Memory Usage

**None.** The `kernel_grid` forward pass uses no shared memory. Each thread works entirely independently:

1. Reads its input position from `positions_in` (global memory, but cached in L1)
2. Computes hash indices via integer arithmetic (registers only)
3. Reads grid values from `grid` (global memory, cached in L1)
4. Writes output to `encoded_positions` (global memory, coalesced writes)

This is ideal for our DPC++ port because it means we do not need to worry about:
- Shared memory allocation / bank conflicts
- `__syncthreads()` / SYCL `group_barrier()` calls
- Work-group-local memory management

---

## 6. Template Parameters and Their Values

### 6.1 Kernel Template Parameters

```cpp
template <typename T,                    // data type (float or __half)
          uint32_t N_POS_DIMS,           // input dimensionality (3 for 3D)
          uint32_t N_FEATURES_PER_LEVEL, // features per level (2 for NeRF)
          HashType HASH_TYPE>            // hash algorithm variant
```

### 6.2 Function Parameters

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `num_elements` | `uint32_t` | Batch size (number of input points) | 65536 rays * 192 samples |
| `num_grid_features` | `uint32_t` | `n_levels * N_FEATURES_PER_LEVEL` | 16 * 2 = 32 |
| `offset_table` | `ParamsOffsetTable` | Per-level offset into flat param array | `[0, 262144, 524288, ...]` |
| `base_resolution` | `uint32_t` | Coarsest resolution | 16 |
| `log2_per_level_scale` | `float` | log2 of per-level scale factor | `log2(b)` where b = exp(log(512/16)/15) |
| `max_level` | `float` | Max level as fraction of total features | 1.0 (all levels) |
| `max_level_gpu` | `float*` | Per-element max level (can be null) | null |
| `interpolation_type` | `InterpolationType` | Linear or Smoothstep | Linear |
| `grid_type` | `GridType` | Hash, Dense, or Tiled | Hash |
| `grid` | `T*` | Flat parameter array | Size = n_params * N_FEATURES_PER_LEVEL |
| `positions_in` | `MatrixView<const float>` | Input positions [N_POS_DIMS x num_elements] | (x,y,z) for each point |
| `encoded_positions` | `T*` | Output (SoA layout) | shape implied by indexing |
| `dy_dx` | `float*` | Jacobian output (can be null) | null for us |

### 6.3 Our Fixed Values

For our simplified DPC++ kernel:

| Parameter | Fixed Value | Reason |
|-----------|------------|--------|
| `N_POS_DIMS` | 3 | Only 3D inputs |
| `N_FEATURES_PER_LEVEL` | 2 | Matches HashNeRF |
| `HASH_TYPE` | `CoherentPrime` | Matches Python hash function |
| `GridType` | `Hash` | We don't need Dense or Tiled |
| `InterpolationType` | `Linear` | We don't need Smoothstep |
| `max_level_gpu` | `nullptr` | Uniform max_level for all points (simpler) |

---

## 7. Resolution Scaling and Level Capping

### 7.1 Resolution Computation

```cpp
// common_device.h lines 886-895
__host__ __device__ inline float grid_scale(uint32_t level, float log2_per_level_scale,
                                            uint32_t base_resolution) {
    return exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
}

__host__ __device__ inline uint32_t grid_resolution(float scale) {
    return (uint32_t)ceilf(scale) + 1;
}
```

For base_resolution=16, finest_resolution=512, 16 levels:
- `b = exp(ln(512/16) / (16-1)) = exp(ln(32)/15) = 32^(1/15) = 1.2599...`
- Level 0: `scale = 16 * 1 - 1 = 15`, `resolution = 16`
- Level 1: `scale = 16 * 1.2599 - 1 = 19.16`, `resolution = 21`
- Level 15: `scale = 16 * 32 - 1 = 511`, `resolution = 512`

The `-1.0f` in `grid_scale` is explained in the source: "The -1 means that base_resolution refers to the number of grid _vertices_ rather than the number of cells." This gives power-of-2-scaled parameter grids that fit better into cache lines.

### 7.2 Level Capping

```cpp
// grid.h lines 69-92
if (max_level_gpu) {
    max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
} else {
    max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;
}

if (level >= max_level + 1e-3f) {
    // Zero out output for this level
    return;
}
```

For our simplified version with `max_level_gpu = nullptr` and `max_level = 1.0`:
- `max_level = (1.0 * 32) / 2 = 16.0` -> all levels active.

We will hardcode this: all 16 levels run for all points.

---

## 8. Backward Pass (for Future Reference)

The backward pass consists of two kernels:

### 8.1 `kernel_grid_backward` -- Gradient w.r.t. grid parameters

- Each thread handles `N_FEATURES_PER_THREAD / N_FEATURES_PER_LEVEL` elements for one level
- Scatters gradients to hash table entries using atomic adds
- Mathematically: `dL/dgrid[index] += weight * dL/dy` for each corner

### 8.2 `kernel_grid_backward_input` -- Gradient w.r.t. input positions

- Simple reduction over all grid features: `dL/dx = sum_k dL/dy_k * dy_k/dx`
- Uses the Jacobian `dy_dx` computed in the forward pass

### 8.3 Our Strategy

For the first iteration, we skip the fused backward and rely on PyTorch autograd:
- The DPC++ forward kernel produces a `[B, 32]` tensor
- The grid parameters remain as PyTorch `nn.Embedding` modules (16 of them)
- `voxel_indices` and weights can be computed on CPU or via PyTorch operations
- PyTorch autograd handles the backward pass through the embedding lookup

This is slower than a fused backward kernel but correct and allows incremental development.

---

## 9. Subset to Implement (Simplified Forward Kernel)

### 9.1 What We Keep

| Component | Decision |
|-----------|----------|
| 2D grid-of-blocks layout (elements x levels) | Keep -- good cache behavior |
| `pos_fract` with identity function | Keep -- same as Linear interpolation |
| CoherentPrime hash | Keep -- matches Python |
| N-linear interpolation over 8 corners | Keep -- core algorithm |
| `grid_val` lookup from flat parameter array | Keep -- access pattern |

### 9.2 What We Simplify

| Component | tiny-cuda-nn | Our Version |
|-----------|-------------|-------------|
| N_POS_DIMS | Template parameter | Hardcoded to 3 |
| N_FEATURES_PER_LEVEL | Template parameter | Hardcoded to 2 |
| HASH_TYPE | Template parameter | Hardcoded to CoherentPrime |
| GridType | Template parameter | Hardcoded to Hash |
| InterpolationType | Template + switch | Hardcoded to Linear |
| Output layout | SoA interleaved | AoS `[B, 32]` directly |
| max_level_gpu | Optional per-element | Always uniform (all levels) |
| dy_dx gradient output | Optional | Removed (use autograd) |
| Smoothstep interpolation | Optional path | Removed |
| Nearest interpolation | Optional path | Removed |

### 9.3 Kernel Pseudocode

```cpp
// Simplified DPC++ forward kernel
void kernel_hash_encoding_forward(
    int B,                           // num_elements
    int L,                           // num_levels (16)
    int F,                           // features per level (2)
    const float* positions,          // [B, 3]
    const float* grid_params,        // flat: n_params * F
    const int* offset_table,         // [L+1] per-level offsets
    float base_resolution,           // 16
    float log2_per_level_scale,      // log2(b)
    float* output                    // [B, L*F] = [B, 32]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // element index
    int level = blockIdx.y;                            // level index
    if (i >= B) return;

    float pos[3];
    uint32_t pos_grid[3];

    float scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
    int resolution = (int)ceilf(scale) + 1;

    // pos_fract with identity function
    for (int dim = 0; dim < 3; ++dim) {
        float p = fmaf(scale, positions[i*3 + dim], 0.5f);
        float tmp = floorf(p);
        pos_grid[dim] = (uint32_t)(int)tmp;
        pos[dim] = p - tmp;   // identity function (no smoothstep)
    }

    int hashmap_size = offset_table[level+1] - offset_table[level];
    int param_offset = offset_table[level] * F;

    float result[2] = {0.0f, 0.0f};

    // 8 corners
    for (int idx = 0; idx < 8; ++idx) {
        float weight = 1.0f;
        uint32_t local_pos[3];

        for (int dim = 0; dim < 3; ++dim) {
            if ((idx & (1 << dim)) == 0) {
                weight *= 1.0f - pos[dim];
                local_pos[dim] = pos_grid[dim];
            } else {
                weight *= pos[dim];
                local_pos[dim] = pos_grid[dim] + 1;
            }
        }

        // coherent_prime_hash
        uint32_t hash = (local_pos[0] * 1u) ^ (local_pos[1] * 2654435761u) ^ (local_pos[2] * 805459861u);

        // grid_index
        uint32_t idx_in_table = hash % hashmap_size;

        // grid_val + fma accumulation
        int base = param_offset + idx_in_table * F;
        result[0] = fmaf(weight, grid_params[base], result[0]);
        result[1] = fmaf(weight, grid_params[base + 1], result[1]);
    }

    // Write to AoS output [B, L*F]
    output[i * (L * F) + level * F + 0] = result[0];
    output[i * (L * F) + level * F + 1] = result[1];
}
```

### 9.4 Key Differences from tiny-cuda-nn

1. **No SOA->AOS transpose needed.** We write directly to `[B, 32]` layout. This means thread `i` writes to `output[i*32 + level*2 + f]`, which is a strided write pattern (stride=32 between consecutive threads). This is less coalesced than tiny-cuda-nn's SOA writes but simpler for our first implementation. We can optimize later if needed.

2. **No `grid_index` dense-index fast path.** We always apply the hash, even for coarse levels where the dense grid would fit. For the first implementation, correctness is more important than the slight performance gain from dense indexing on coarse levels.

3. **Hardcoded `grid_val` inline.** Rather than the lambda + `tvec` abstraction, we directly index `grid_params[base + f]` and accumulate into local `float result[2]`.

---

## 10. Performance Considerations for Intel Arc A750

### 10.1 Memory Access Patterns

The kernel is memory-bound: for each (element, level) it performs:
- 1 read of 3 floats (position) -- cached, reused across levels
- 8 reads of 2 floats each (grid params) -- random access due to hash
- 1 write of 2 floats (output)

For B=65536 rays * 192 samples = 12.5M, L=16:
- Total grid reads: 12.5M * 16 * 8 * 2 * 4 bytes = **12.8 GB**
- Output writes: 12.5M * 16 * 2 * 4 bytes = **1.6 GB**
- Input reads: 12.5M * 3 * 4 = **0.15 GB**

The hash lookups cause irregular memory access, which is the main bottleneck. The Arc A750 has 512 GB/s memory bandwidth, so the theoretical minimum time is 14.55 GB / 512 GB/s = 28 ms.

### 10.2 Occupancy

With 512 threads per block, each thread uses ~30 registers (estimated), the Arc A750 (Xe-HPG architecture) should achieve good occupancy. The key constraint is the 8 hash lookups per corner causing memory latency.

### 10.3 L1/L2 Cache

The hash table size per level is `hashmap_size * 2 * 4` bytes = `2^19 * 8` = 4 MB. The Arc A750 has:
- L1: 192 KB per Xe-core (private)
- L2: 16 MB (shared)

The L2 cache is large enough to hold the entire hash table for a level (4 MB), which means repeated accesses to the same level from different threads in the same block can benefit from L2 hits. This is exactly why the grid-of-blocks layout (one level per block) is effective.

### 10.4 DPC++ Specifics

For Intel Arc, we should:
- Use `sycl::nd_range<2>` with the same 2D decomposition
- Use `group_id(0)` as element group, `group_id(1)` as level
- Use `local_id(0)` as thread within element group
- Avoid shared memory (none needed)
- Use `sycl::ext::intel::math::fma` for fused multiply-add
- Use 32-bit integer operations for hash computation (same as CUDA)

---

## 11. Source File Map

| File | Lines | Content |
|------|-------|---------|
| `docs/tiny-cuda-nn-grid.h` | 1854 | Full encoding class + kernel_grid + backward + initialization |
| `docs/tiny-cuda-nn-common_device.h` | 1233 | hash functions, pos_fract, resolution helpers |
| `HashNeRF-pytorch/utils.py` | 129 | Python hash(), get_voxel_vertices() |
| `HashNeRF-pytorch/hash_encoding.py` | 158 | HashEmbedder with trilinear_interp, per-level loop |
