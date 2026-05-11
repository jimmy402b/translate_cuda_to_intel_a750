import torch
import sys
import os

# Add parent directory to path so we can import hash_encoding and sycl_ops
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NOTE: Do NOT set HASHNERF_USE_SYCL=1 here.  We want the PyTorch reference
# path to be pure Python (HashEmbedder.forward takes the unmodified code path).
# We import sycl_ops directly to get the fused kernel.

from hash_encoding import HashEmbedder
from sycl_ops import FusedHashEncode


def test_fused_same_as_pytorch():
    """SYCL kernel must match pure PyTorch output to within 1e-4."""
    if not torch.xpu.is_available():
        print("SKIPPED: XPU device not available")
        return

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

    # PyTorch reference -- run with SYCL disabled so we get the pure-Python path
    import hash_encoding
    hash_encoding._USE_SYCL = False
    with torch.no_grad():
        ref_output, ref_mask = embedder(points)
    hash_encoding._USE_SYCL = True

    # SYCL output -- call the custom op directly (not via HashEmbedder.forward)
    # so we can test the kernel independently
    embeddings = torch.stack([e.weight for e in embedder.embeddings])
    sycl_output, sycl_mask = FusedHashEncode.apply(
        points, embeddings,
        bounding_box[0], bounding_box[1],
        16.0, float(embedder.b), embedder.log2_hashmap_size
    )

    max_diff = (ref_output - sycl_output).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")

    # Masks must match exactly (same floating-point comparison logic)
    mask_diff = (ref_mask != sycl_mask).sum().item()
    print(f"Mask mismatches: {mask_diff} / {B}")

    assert max_diff < 1e-4, f"SYCL output differs from PyTorch! max_diff={max_diff}"
    assert mask_diff == 0, f"SYCL mask differs from PyTorch! mismatches={mask_diff}"
    print("PASSED: SYCL matches PyTorch")


if __name__ == "__main__":
    test_fused_same_as_pytorch()
