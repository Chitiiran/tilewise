"""Post-cycle archival: move out-of-window raw parquet to the HDD, keep the
window on fast disk. Idempotent (breadcrumb), never deletes (moves). Runs only
after a cycle fully publishes. Spec 2026-06-13 §8."""
from __future__ import annotations

import shutil
from pathlib import Path


def archive_out_of_window(*, window_dirs, all_dirs, archive_root, rules_id) -> int:
    """Move parquet of dirs NOT in window_dirs to archive_root/<rules_id>/<dir>.
    Leaves an ARCHIVED.txt breadcrumb so re-runs skip already-moved dirs.
    Returns the count of dirs archived."""
    window = {Path(d).resolve() for d in window_dirs}
    archive_root = Path(archive_root)
    n = 0
    for d in all_dirs:
        d = Path(d)
        if d.resolve() in window:
            continue
        if (d / "ARCHIVED.txt").exists():
            continue
        dest = archive_root / rules_id / d.name
        dest.mkdir(parents=True, exist_ok=True)
        for pq in list(d.glob("*.parquet")):
            shutil.move(str(pq), str(dest / pq.name))
        (d / "ARCHIVED.txt").write_text(str(dest))
        n += 1
    return n
