"""Verify sparse cache produces bit-identical items + forward passes vs dense.

Phase C of the sparse-cache rollout. Builds two tiny caches from the same
source (10 samples), compares __getitem__ outputs tensor-by-tensor, runs a
forward pass on a fresh GnnModel through both, and confirms outputs are
bit-equal.

If anything fails, prints WHICH check failed and exits non-zero. If
everything passes, prints DONE.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import torch

# Force determinism for the forward-pass test.
torch.manual_seed(0)

from catan_gnn.dataset import CachedDataset, CatanReplayDataset
from catan_gnn.gnn_model import GnnModel


# Use the seed_21M_partial run-dir which has a small number of seeds (5,515 games)
# so we don't load the whole 100k. We'll only iterate the first 10 anyway.
RUN_DIR = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study/runs/v3/data/v3_100k_lookahead_d500/seed_21M_partial")
N_SAMPLES = 10


class TruncatedSource(torch.utils.data.Dataset):
    """Wraps a CatanReplayDataset to expose only the first N positions.

    Avoids replaying all 177k positions when we only need 10 for verification.
    """
    def __init__(self, source, n):
        self._source = source
        self._n = min(n, len(source))
        # Forward _index so CachedDataset can pull seeds for the first N rows.
        if hasattr(source, "_index"):
            self._index = source._index.iloc[:self._n].reset_index(drop=True)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._source[i]


def make_source():
    """Build a CatanReplayDataset pointing at the small partial run, then
    truncate to the first N_SAMPLES rows."""
    full = CatanReplayDataset([RUN_DIR])
    return TruncatedSource(full, N_SAMPLES)


def build_cache(source, sparse: bool, name: str):
    """Build a cache with the given sparse flag, write to a temp file, load
    it, and return the dataset for inspection."""
    # Use mkdtemp so we get an empty dir; the cache file itself doesn't
    # exist yet (CachedDataset.__init__ would otherwise try to load it).
    tmpdir = Path(tempfile.mkdtemp(prefix=f"catan_cache_verify_{name}_"))
    cache_path = tmpdir / f"cache_{name}.pt"

    # Use chunk_size larger than N_SAMPLES so we always test the monolithic
    # save path here. (Chunked path is exercised by the real builder.)
    ds = CachedDataset(
        source=source, cache_path=cache_path, verbose=False,
        chunk_size=10**9, sparse=sparse,
    )
    print(f"  built {name}: {len(ds)} samples, sparse={ds._sparse}", flush=True)
    return ds, cache_path


def check_items_equal(dense_ds, sparse_ds):
    """Confirm __getitem__ produces identical tensors for every key."""
    print("\n=== C1: per-sample tensor equivalence ===", flush=True)
    for i in range(len(dense_ds)):
        d_data, d_v, d_p, d_l = dense_ds[i]
        s_data, s_v, s_p, s_l = sparse_ds[i]

        # Node features
        for key in ["hex", "vertex", "edge"]:
            if not torch.equal(d_data[key].x, s_data[key].x):
                print(f"  FAIL: sample {i}, {key}.x differs", flush=True)
                return False
        # Edge indices (these are the ones we deduplicated)
        for src, rel, dst in [("hex", "to", "vertex"), ("vertex", "to", "hex"),
                               ("vertex", "to", "edge"), ("edge", "to", "vertex")]:
            d_ei = d_data[src, rel, dst].edge_index
            s_ei = s_data[src, rel, dst].edge_index
            if not torch.equal(d_ei, s_ei):
                print(f"  FAIL: sample {i}, ({src}->{dst}).edge_index differs", flush=True)
                return False
        # Scalars + legal_mask
        if not torch.equal(d_data.scalars, s_data.scalars):
            print(f"  FAIL: sample {i}, scalars differ", flush=True); return False
        if not torch.equal(d_data.legal_mask, s_data.legal_mask):
            print(f"  FAIL: sample {i}, legal_mask differs", flush=True); return False
        # Targets
        if not torch.equal(d_v, s_v):
            print(f"  FAIL: sample {i}, value target differs", flush=True); return False
        if not torch.equal(d_p, s_p):
            print(f"  FAIL: sample {i}, policy target differs", flush=True); return False
        if not torch.equal(d_l, s_l):
            print(f"  FAIL: sample {i}, legal mask target differs", flush=True); return False
    print(f"  PASS: all {len(dense_ds)} samples have bit-identical tensors", flush=True)
    return True


def check_forward_equal(dense_ds, sparse_ds):
    """Confirm a fresh model produces identical outputs for both."""
    print("\n=== C2: forward-pass equivalence ===", flush=True)
    from torch_geometric.data import Batch
    torch.manual_seed(0)
    model = GnnModel(hidden_dim=32, num_layers=2)
    model.eval()

    for i in range(len(dense_ds)):
        d_data, _, _, _ = dense_ds[i]
        s_data, _, _, _ = sparse_ds[i]

        d_batch = Batch.from_data_list([d_data])
        s_batch = Batch.from_data_list([s_data])

        with torch.no_grad():
            d_value, d_policy = model(d_batch)
            s_value, s_policy = model(s_batch)

        if not torch.equal(d_value, s_value):
            diff = (d_value - s_value).abs().max().item()
            print(f"  FAIL: sample {i}, value output differs (max abs diff: {diff})", flush=True)
            return False
        if not torch.equal(d_policy, s_policy):
            diff = (d_policy - s_policy).abs().max().item()
            print(f"  FAIL: sample {i}, policy output differs (max abs diff: {diff})", flush=True)
            return False
    print(f"  PASS: forward outputs bit-identical for all {len(dense_ds)} samples", flush=True)
    return True


def check_storage_size(dense_ds, sparse_ds):
    """Verify sparse storage is genuinely smaller per item."""
    print("\n=== C0: per-item storage shape sanity ===", flush=True)
    d0 = dense_ds._items[0]
    s0 = sparse_ds._items[0]
    print(f"  dense item keys: {sorted(d0.keys())}")
    print(f"  sparse item keys: {sorted(s0.keys())}")
    expected_dense = {"hex_x", "vertex_x", "edge_x", "scalars", "legal_mask_attr",
                       "h2v_ei", "v2h_ei", "v2e_ei", "e2v_ei",
                       "value", "policy", "legal"}
    expected_sparse = {"hex_x", "vertex_x", "edge_x", "scalars", "legal_mask_attr",
                        "value", "policy", "legal"}
    if set(d0.keys()) != expected_dense:
        print(f"  FAIL: dense keys missing/extra (got {sorted(d0.keys())})")
        return False
    if set(s0.keys()) != expected_sparse:
        print(f"  FAIL: sparse keys missing/extra (got {sorted(s0.keys())})")
        return False
    print("  PASS: dense has all 12 keys; sparse has 8 (no edge_index)", flush=True)
    return True


def main():
    print("Loading source dataset...", flush=True)
    source = make_source()
    print(f"  source has {len(source)} positions", flush=True)
    if len(source) < N_SAMPLES:
        print(f"FAIL: need at least {N_SAMPLES} samples, got {len(source)}")
        return 1

    print("\nBuilding dense cache (control)...", flush=True)
    dense_ds, dense_path = build_cache(source, sparse=False, name="dense")

    print("\nBuilding sparse cache (test)...", flush=True)
    sparse_ds, sparse_path = build_cache(source, sparse=True, name="sparse")

    if not check_storage_size(dense_ds, sparse_ds):
        return 1
    if not check_items_equal(dense_ds, sparse_ds):
        return 1
    if not check_forward_equal(dense_ds, sparse_ds):
        return 1

    print("\nDONE — sparse cache verified equivalent to dense.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
