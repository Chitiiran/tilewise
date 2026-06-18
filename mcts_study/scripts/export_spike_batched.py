"""Export the spike net (az_iter_1.pt, 128x4) as a BATCHED .ts at B_MAX=8 for
the Rust batched-evaluator parity test. Sits next to wrapper_traced.ts."""
from pathlib import Path
from catan_gnn.export_torchscript import export_batched

SPIKE = Path(__file__).resolve().parents[1] / "spike"
CKPT = Path("/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt")

export_batched(checkpoint=CKPT, out_ts=SPIKE / "wrapper_batched.ts",
               hidden_dim=128, num_layers=4, b_max=8)
print(f"wrote {SPIKE/'wrapper_batched.ts'}")
