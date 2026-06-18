"""Re-export BOTH spike .ts files using the FIXED export_torchscript wrappers
(device-following batch vectors, CUDA-safe). Overwrites:
  spike/wrapper_traced.ts   (B=1)
  spike/wrapper_batched.ts  (B_MAX=8)
"""
from pathlib import Path
from catan_gnn.export_torchscript import export, export_batched

SPIKE = Path(__file__).resolve().parents[1] / "spike"
CKPT = Path("/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt")

import sys
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
export(checkpoint=CKPT, out_ts=SPIKE / "wrapper_traced.ts",
       hidden_dim=128, num_layers=4, device=dev)
export_batched(checkpoint=CKPT, out_ts=SPIKE / "wrapper_batched.ts",
               hidden_dim=128, num_layers=4, b_max=8, device=dev)
print(f"re-exported wrapper_traced.ts + wrapper_batched.ts on device={dev}")
