import torch
from torch.utils.cpp_extension import load
import os

_sycl_ops_path = os.path.dirname(os.path.abspath(__file__))

_fused_hash_encode = load(
    name="fused_hash_encode",
    sources=[os.path.join(_sycl_ops_path, "hash_encode_fwd.sycl")],
    with_sycl=True,
    extra_sycl_cflags=["-O2"],
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
        keep_mask: [B] bool (all True for valid points)
    """

    @staticmethod
    def forward(ctx, points, embeddings, bbox_min, bbox_max,
                base_resolution, b, log2_hashmap_size):
        # Flatten embeddings: [16, T, 2] -> [16*T, 2]
        N_LEVELS, T, N_FEAT = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, N_FEAT).contiguous()

        output = _fused_hash_encode.forward(
            points.contiguous(),
            embeddings_flat,
            bbox_min.contiguous(),
            bbox_max.contiguous(),
            base_resolution, b, log2_hashmap_size
        )

        # Compute keep_mask: True for points inside bounding box
        clamped = torch.clamp(points, min=bbox_min, max=bbox_max)
        keep_mask = torch.all(points == clamped, dim=-1)

        ctx.save_for_backward(points, embeddings, bbox_min, bbox_max)
        ctx.constants = (base_resolution, b, log2_hashmap_size)

        return output, keep_mask

    @staticmethod
    def backward(ctx, grad_output, _):
        # Fallback: raise NotImplementedError for now.
        # We'll add a DPC++ backward kernel in Task 6.
        # For training, the optimizer needs gradient w.r.t. embeddings.
        raise NotImplementedError(
            "Backward not implemented in SYCL yet. "
            "Task 6 will add the backward kernel."
        )
