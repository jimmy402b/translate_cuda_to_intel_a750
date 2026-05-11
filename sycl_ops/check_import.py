"""Test importing FusedHashEncode and report the exact error."""
import sys, os, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from sycl_ops import FusedHashEncode
    print("IMPORT OK: FusedHashEncode loaded successfully")
except Exception as e:
    print(f"IMPORT FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Also run the actual correctness test
from hash_encoding import HashEmbedder
import torch

if not torch.xpu.is_available():
    print("SKIPPED: XPU device not available")
    sys.exit(0)

device = torch.device("xpu")
bbox_min = torch.tensor([-1.5, -1.5, -1.5], device=device)
bbox_max = torch.tensor([1.5, 1.5, 1.5], device=device)
embedder = HashEmbedder(
    (bbox_min, bbox_max),
    n_levels=16, n_features_per_level=2,
    log2_hashmap_size=19, base_resolution=16, finest_resolution=512
).to(device)

B = 1024
torch.manual_seed(42)
points = torch.rand(B, 3, device=device) * 3.0 - 1.5

import hash_encoding
hash_encoding._USE_SYCL = False
with torch.no_grad():
    ref_output, ref_mask = embedder(points)

embeddings = torch.stack([e.weight for e in embedder.embeddings])
sycl_output, sycl_mask = FusedHashEncode.apply(
    points, embeddings,
    bbox_min, bbox_max,
    16.0, float(embedder.b), embedder.log2_hashmap_size
)

max_diff = (ref_output - sycl_output).abs().max().item()
print(f"Max difference: {max_diff:.6f}")

mask_diff = (ref_mask != sycl_mask).sum().item()
print(f"Mask mismatches: {mask_diff} / {B}")

if max_diff < 1e-4 and mask_diff == 0:
    print("PASSED: SYCL matches PyTorch")
else:
    print(f"FAILED: max_diff={max_diff}, mask_mismatches={mask_diff}")
    sys.exit(1)
