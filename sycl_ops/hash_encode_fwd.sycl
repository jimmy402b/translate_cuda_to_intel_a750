#include <sycl/sycl.hpp>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Hash function -- MUST match Python utils.py::hash() exactly
//
// Python reference:
//   primes = [1, 2654435761, 805459861, ...]
//   xor_result = 0
//   for i in range(3):
//       xor_result ^= coords[..., i] * primes[i]
//   return ((1 << log2_hashmap_size) - 1) & xor_result
//
// This is the CoherentPrime variant from tiny-cuda-nn where the first
// factor is 1 (x-coordinate passes through unmodified), which helps
// with spatial coherence.
// ---------------------------------------------------------------------------
static constexpr uint32_t PRIMES[3] = {1, 2654435761u, 805459861u};

// ---------------------------------------------------------------------------
// Constants matching HashNeRF-pytorch configuration
// ---------------------------------------------------------------------------
static constexpr int N_LEVELS = 16;
static constexpr int N_FEATURES = 2;

// ---------------------------------------------------------------------------
// fused_hash_encode_forward
//
// Replaces the Python HashEmbedder.forward() which loops 16 levels in Python
// with individual nn.Embedding calls.  This kernel fuses all 16 levels into
// a single GPU dispatch, where each work-item processes one (point, level)
// pair independently.
//
// Parameters:
//   points           [B, 3] float32  -- 3D point positions
//   embeddings       [16 * T, 2] float32 -- flat concatenation of 16 hash
//                        tables, each of size T x 2
//   bbox_min         [3] float32 -- bounding box minimum corner
//   bbox_max         [3] float32 -- bounding box maximum corner
//   base_resolution  float -- coarsest resolution (e.g. 16.0)
//   b                float -- per-level growth factor;
//                        b = exp((log(finest)-log(base)) / (n_levels-1))
//   log2_hashmap_size int  -- log2 of hash table entries per level (e.g. 19)
//
// Returns:
//   [B, 32] float32 -- encoded features (16 levels x 2 features)
// ---------------------------------------------------------------------------
torch::Tensor fused_hash_encode_forward(
    const torch::Tensor& points,
    const torch::Tensor& embeddings,
    const torch::Tensor& bbox_min,
    const torch::Tensor& bbox_max,
    float base_resolution,
    float b,
    int log2_hashmap_size)
{
    // --- Input validation ---
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(embeddings.is_contiguous(), "embeddings must be contiguous");
    TORCH_CHECK(bbox_min.is_contiguous(), "bbox_min must be contiguous");
    TORCH_CHECK(bbox_max.is_contiguous(), "bbox_max must be contiguous");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "points must have shape [B, 3]");
    TORCH_CHECK(points.scalar_type() == torch::kFloat32,
                "points must be float32");
    TORCH_CHECK(embeddings.scalar_type() == torch::kFloat32,
                "embeddings must be float32");
    TORCH_CHECK(log2_hashmap_size > 0 && log2_hashmap_size <= 30,
                "log2_hashmap_size must be in [1, 30]");
    TORCH_CHECK(bbox_min.dim() == 1 && bbox_min.size(0) == 3,
                "bbox_min must have shape [3]");
    TORCH_CHECK(bbox_max.dim() == 1 && bbox_max.size(0) == 3,
                "bbox_max must have shape [3]");
    TORCH_CHECK(embeddings.dim() == 2 && embeddings.size(0) == N_LEVELS * (1LL << log2_hashmap_size) && embeddings.size(1) == N_FEATURES,
                "embeddings must have shape [16 * 2^log2_hashmap_size, 2]");

    // Runtime guards: N_LEVELS and N_FEATURES are compile-time constants.
    // If the Python constructor values change, this kernel must be rebuilt.
    TORCH_CHECK(embeddings.size(0) % N_LEVELS == 0,
                "embeddings dim 0 must be divisible by N_LEVELS (16)");
    TORCH_CHECK(embeddings.size(1) == N_FEATURES,
                "embeddings dim 1 must equal N_FEATURES (2)");

    const int64_t B = points.size(0);
    const int64_t T = 1LL << log2_hashmap_size;

    // --- Allocate output tensor ---
    // Output shape [B, 32] = [B, N_LEVELS * N_FEATURES], same device/dtype as input
    auto output = torch::empty({B, N_LEVELS * N_FEATURES}, points.options());

    // --- Get raw USM pointers ---
    // PyTorch XPU tensors are backed by device USM allocations on Intel GPU.
    // These raw pointers can be captured directly in SYCL kernels.
    const float* pts_ptr = points.data_ptr<float>();
    const float* emb_ptr = embeddings.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    const float* bmin_ptr = bbox_min.data_ptr<float>();
    const float* bmax_ptr = bbox_max.data_ptr<float>();

    // --- Get SYCL queue ---
    // Create a queue on the default GPU device (Intel Arc A750).
    // Task 5 (compile/test) will refine this to use PyTorch's XPU stream
    // for proper interop via c10::xpu::getCurrentXPUStream() or
    // sycl::ext::oneapi::get_queue_from_tensor() if available.
    sycl::queue q(sycl::gpu_selector_v);

    // --- Launch fused kernel ---
    // 2D parallel_for: dim0 = point index, dim1 = level index.
    // Each work-item handles exactly one (point, level) pair.
    // No shared memory needed -- all work-items are independent.
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(B, N_LEVELS), [=](sycl::id<2> idx) {
            const int point_idx = static_cast<int>(idx[0]);
            const int level = static_cast<int>(idx[1]);

            // ---------------------------------------------------------------
            // Step 1: Load 3D point position, clamp to bounding box
            // ---------------------------------------------------------------
            // Matching Python:
            //   xyz = torch.clamp(xyz, min=box_min, max=box_max)
            float px = pts_ptr[point_idx * 3 + 0];
            float py = pts_ptr[point_idx * 3 + 1];
            float pz = pts_ptr[point_idx * 3 + 2];

            px = sycl::fmin(sycl::fmax(px, bmin_ptr[0]), bmax_ptr[0]);
            py = sycl::fmin(sycl::fmax(py, bmin_ptr[1]), bmax_ptr[1]);
            pz = sycl::fmin(sycl::fmax(pz, bmin_ptr[2]), bmax_ptr[2]);

            // ---------------------------------------------------------------
            // Step 2: Compute grid resolution for this level
            // ---------------------------------------------------------------
            // Matching Python:
            //   resolution = torch.floor(self.base_resolution * self.b**i)
            // Note: this is DIFFERENT from tiny-cuda-nn's formula
            //   scale = exp2f(level*log2_b)*base_res - 1; res = ceil(scale)+1
            // We match Python exactly.
            const float resolution = sycl::floor(
                base_resolution * sycl::pow(b, static_cast<float>(level)));

            // ---------------------------------------------------------------
            // Step 3: Compute grid cell size (world units per voxel)
            // ---------------------------------------------------------------
            // Matching Python:
            //   grid_size = (box_max - box_min) / resolution
            const float gsx = (bmax_ptr[0] - bmin_ptr[0]) / resolution;
            const float gsy = (bmax_ptr[1] - bmin_ptr[1]) / resolution;
            const float gsz = (bmax_ptr[2] - bmin_ptr[2]) / resolution;

            // ---------------------------------------------------------------
            // Step 4: Scale position into grid coordinates;
            //         split into integer voxel index and fractional weight
            // ---------------------------------------------------------------
            // Matching Python:
            //   bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
            //   voxel_min_vertex = bottom_left_idx * grid_size + box_min
            //   weights = (x - voxel_min_vertex) / (voxel_max - voxel_min)
            //           = (x - box_min) / grid_size - bottom_left_idx
            const float sx = (px - bmin_ptr[0]) / gsx;
            const float sy = (py - bmin_ptr[1]) / gsy;
            const float sz = (pz - bmin_ptr[2]) / gsz;

            const float ix_f = sycl::floor(sx);
            const float iy_f = sycl::floor(sy);
            const float iz_f = sycl::floor(sz);

            const int ix = static_cast<int>(ix_f);
            const int iy = static_cast<int>(iy_f);
            const int iz = static_cast<int>(iz_f);

            // Fractional position within the voxel [0, 1)
            const float wx = sx - ix_f;
            const float wy = sy - iy_f;
            const float wz = sz - iz_f;

            // ---------------------------------------------------------------
            // Step 5: Trilinear interpolation over 8 voxel corners
            // ---------------------------------------------------------------
            // Corner index uses bit encoding (matches tiny-cuda-nn):
            //   bit 0 = x offset, bit 1 = y offset, bit 2 = z offset
            //
            // This produces the same mathematical result as Python's
            // 3-step x-y-z interpolation, because both compute:
            //   result = sum_{corner} weight * embedding(corner)
            // with weight = product over dims of (offset?w:1-w).
            //
            // Corner spatial offsets and weights (bit-encoded):
            //   corner  bits(x,y,z)  spatial offset [x,y,z]     weight
            //   0       000          [0,0,0]                    (1-wx)*(1-wy)*(1-wz)
            //   1       001          [1,0,0]  (x-bit set)       wx*(1-wy)*(1-wz)
            //   2       010          [0,1,0]  (y-bit set)       (1-wx)*wy*(1-wz)
            //   3       011          [1,1,0]  (x,y bits set)    wx*wy*(1-wz)
            //   4       100          [0,0,1]  (z-bit set)       (1-wx)*(1-wy)*wz
            //   5       101          [1,0,1]  (x,z bits set)    wx*(1-wy)*wz
            //   6       110          [0,1,1]  (y,z bits set)    (1-wx)*wy*wz
            //   7       111          [1,1,1]  (all bits set)    wx*wy*wz
            //
            // Note: spatial offsets here are sorted by bit encoding (L=x, M=y, H=z),
            // while Python's _BOX_OFFSETS uses [i for i in 0,1 for j in 0,1 for k in 0,1]
            // (z-major: offset 0→[0,0,0], 1→[0,0,1], ..., 4→[1,0,0], ..., 7→[1,1,1]).
            // The iteration order differs but the weighted sum is mathematically identical.
            // ---------------------------------------------------------------
            float r0 = 0.0f;  // accumulated feature 0
            float r1 = 0.0f;  // accumulated feature 1

            for (int corner = 0; corner < 8; ++corner) {
                // Extract corner offset from bit pattern
                const int dx = (corner >> 0) & 1;  // x offset
                const int dy = (corner >> 1) & 1;  // y offset
                const int dz = (corner >> 2) & 1;  // z offset

                // Trilinear interpolation weight
                const float weight = (dx ? wx : (1.0f - wx))
                                   * (dy ? wy : (1.0f - wy))
                                   * (dz ? wz : (1.0f - wz));

                // Corner grid coordinate = voxel_min + corner_offset
                const int cx = ix + dx;
                const int cy = iy + dy;
                const int cz = iz + dz;

                // -----------------------------------------------------------
                // Hash corner coordinate (CoherentPrime hash, matches Python)
                // -----------------------------------------------------------
                // Python reference:
                //   primes = [1, 2654435761, 805459861, ...]
                //   xor_result ^= coords[...,0] * 1
                //   xor_result ^= coords[...,1] * 2654435761
                //   xor_result ^= coords[...,2] * 805459861
                //   return ((1<<log2_hashmap_size)-1) & xor_result
                //
                // Cast to uint32_t before multiplication ensures well-defined
                // modulo-2^32 wrapping (matching CUDA/PyTorch int32 overflow).
                const uint32_t hash = (static_cast<uint32_t>(cx) * PRIMES[0])
                                    ^ (static_cast<uint32_t>(cy) * PRIMES[1])
                                    ^ (static_cast<uint32_t>(cz) * PRIMES[2]);

                // Bitwise AND is equivalent to modulo since T is power of 2
                const int hash_idx = static_cast<int>(hash & (static_cast<uint32_t>(T) - 1u));

                // -----------------------------------------------------------
                // Lookup embedding from flat table
                // -----------------------------------------------------------
                // Embeddings are concatenated: [level_0_table, level_1_table, ...]
                // Each level's table has T rows of 2 features.
                // embedding[level * T + hash_idx] has 2 features.
                const int emb_base = level * static_cast<int>(T) + hash_idx;
                const float e0 = emb_ptr[emb_base * N_FEATURES + 0];
                const float e1 = emb_ptr[emb_base * N_FEATURES + 1];

                // Accumulate weighted contribution
                r0 += weight * e0;
                r1 += weight * e1;
            }

            // ---------------------------------------------------------------
            // Step 6: Write output in AoS layout [B, 32]
            // ---------------------------------------------------------------
            // Output layout (matching Python torch.cat):
            //   [level_0_feat_0, level_0_feat_1, level_1_feat_0, level_1_feat_1,
            //    ..., level_15_feat_0, level_15_feat_1]
            const int out_base = point_idx * (N_LEVELS * N_FEATURES) + level * N_FEATURES;
            out_ptr[out_base + 0] = r0;
            out_ptr[out_base + 1] = r1;
        });
    });

    // Ensure kernel completes before returning the tensor to Python
    q.wait_and_throw();

    return output;
}

// ---------------------------------------------------------------------------
// PyTorch bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_hash_encode_forward,
          "Fused hash encoding forward (DPC++ kernel)");
}
