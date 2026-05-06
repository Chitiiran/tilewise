"""Build a CachedDataset (.pt) from one or more run dirs.

Standalone wrapper around `CachedDataset(source=CatanReplayDataset(...), cache_path=...)`
so we can pre-build the cache before training starts. Subsequent training runs
that point at the same `--cache-path` skip the build entirely.

Usage:
  python scripts/build_cache.py \
    --run-dirs runs/v3/data/v3_100k_lookahead_d500/seed_21M_partial \
    --run-dirs runs/v3/data/v3_100k_lookahead_d500/seed_21.1M_full \
    --cache-path runs/v3/cache_100k.pt

Note: --run-dirs accepts repeated values (one per --run-dirs flag) or a single
flag with multiple paths after it (nargs="+"). Both work.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from catan_gnn.dataset import CachedDataset, CatanReplayDataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", type=Path, nargs="+", action="extend", required=True,
                   help="One or more run directories to merge into the cache")
    p.add_argument("--cache-path", type=Path, required=True,
                   help="Output cache path (will be overwritten if exists)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if cache_path already exists")
    p.add_argument("--chunk-size", type=int, default=500_000,
                   help="Flush a chunk every N positions (default 500k). Lower "
                        "values reduce peak RAM during build at the cost of more "
                        "small files. 0 disables chunking (monolithic save).")
    args = p.parse_args()

    if args.cache_path.exists() and not args.force:
        print(f"Cache already exists at {args.cache_path}; pass --force to rebuild.", flush=True)
        return 0

    print(f"Building cache from {len(args.run_dirs)} run dir(s):", flush=True)
    for rd in args.run_dirs:
        print(f"  {rd}", flush=True)

    t0 = time.perf_counter()
    print("Constructing CatanReplayDataset (replays positions through engine)...", flush=True)
    source = CatanReplayDataset(args.run_dirs)
    t1 = time.perf_counter()
    print(f"  → {len(source)} positions ready in {t1 - t0:.1f}s", flush=True)

    print(f"Building cache + saving to {args.cache_path} (chunk_size={args.chunk_size})...", flush=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_size if args.chunk_size > 0 else 10**12  # effectively disabled
    cached = CachedDataset(source=source, cache_path=args.cache_path, chunk_size=chunk_size)
    t2 = time.perf_counter()
    print(f"  → {len(cached)} cached samples saved in {t2 - t1:.1f}s", flush=True)
    print(f"  total: {t2 - t0:.1f}s", flush=True)

    print(f"Cache file: {args.cache_path}  ({args.cache_path.stat().st_size / 1024**3:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
