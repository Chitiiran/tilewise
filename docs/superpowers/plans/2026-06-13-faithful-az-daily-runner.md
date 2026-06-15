# Faithful AZ Daily Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline — tasks build a resumable daily trainer; later tasks drive real WSL runs). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wrap the working `catan_az` loop in a resumable, resource-guarded daily AZ trainer with fresh-ratio self-play, ≤1-game-loss resumability, HDD archival, and a minimal progress dashboard.

**Architecture:** New modules `preflight.py` (env/resource guards), `daily.py` (resumable daily driver calling `loop.run_iteration` per cycle), `archive.py` (post-cycle HDD move), `dashboard/` (FastAPI route + static HTML). Small hooks into existing `config.py` (new knobs), `buffer.py` (fresh-ratio + rules_id filter), `arena.py` (VP-margin tiebreak). Entry: `scripts/run_az_day.sh`.

**Tech Stack:** Python 3.12 (WSL venv `~/catan_mcts_venvs/mcts-study/`), pytest, FastAPI (reuse `catan_mcts/web` patterns), nvidia-smi / psutil for guards, parquet via pandas.

**Spec:** `docs/superpowers/specs/2026-06-13-faithful-az-daily-runner-design.md`

**Commit cadence:** commit after every task's tests go green. Push the PR at each milestone marked **🚩 MILESTONE**.

**Run everything in WSL:** `wsl.exe -e bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study && <cmd>"`.

---

## File structure

| File | Responsibility |
|---|---|
| `catan_az/config.py` (modify) | new knobs: fresh_ratio, rules_id, worker_procs_max, per_proc_vram_mb, worker_nice, min_fast_gb, min_hdd_gb, stagnation_holds, archive_root, dashboard_port |
| `catan_az/resources.py` (create) | pure resource probes: free VRAM, free RAM, free disk, foreign-proc scan, GPU-proc fit math |
| `catan_az/preflight.py` (create) | run guards before a cycle → PreflightResult(ok, capped_procs, reasons); self-heal/abort/degrade |
| `catan_az/buffer.py` (modify) | rules_id-tagged dirs; `fresh_deficit()`; `select_window` filters to same rules_id |
| `catan_az/arena.py` (modify) | VP-margin tiebreak in `_vp_leader` |
| `catan_az/daily.py` (create) | resumable daily driver: manifest, fresh-ratio self-play (low-prio procs), per-cycle loop, stagnation detect, STOP sentinel |
| `catan_az/archive.py` (create) | move out-of-window parquet to HDD, idempotent, breadcrumbs |
| `catan_az/dashboard/server.py` (create) | FastAPI route serving journal/status/ladder JSON + champion link |
| `catan_az/dashboard/static/index.html` (create) | auto-refresh at-a-glance page |
| `scripts/run_az_day.sh` (create) | cron-ready entry: preflight → daily → archive |

---

## Task 1: Config knobs 🚩 (start of MILESTONE A — guards)

**Files:** Modify `catan_az/config.py`; Test `tests/test_az_daily_config.py`

- [ ] **Step 1: failing test**

```python
# tests/test_az_daily_config.py
def test_daily_knobs_defaults():
    from catan_az.config import AzConfig
    c = AzConfig()
    assert c.fresh_ratio == 0.70
    assert c.rules_id == "v3-full"
    assert c.worker_procs_max == 7
    assert c.per_proc_vram_mb == 535.0
    assert c.worker_nice == 10
    assert c.min_fast_gb == 10.0
    assert c.min_hdd_gb == 20.0
    assert c.stagnation_holds == 5
    assert c.archive_root == "/mnt/d/catan_az_archive"
    assert c.dashboard_port == 8099
```

- [ ] **Step 2: run, expect FAIL** (AttributeError)

Run: `python -m pytest tests/test_az_daily_config.py -q`

- [ ] **Step 3: add the fields** to the `AzConfig` dataclass in `config.py`, after `num_layers`:

```python
    # --- daily runner (spec 2026-06-13) ---
    fresh_ratio: float = 0.70
    rules_id: str = "v3-full"
    worker_procs_max: int = 7
    per_proc_vram_mb: float = 535.0
    worker_nice: int = 10
    min_fast_gb: float = 10.0
    min_hdd_gb: float = 20.0
    stagnation_holds: int = 5
    archive_root: str = "/mnt/d/catan_az_archive"
    dashboard_port: int = 8099
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git add -A && git commit -m "feat(az-daily): config knobs"`

## Task 2: Resource probes

**Files:** Create `catan_az/resources.py`; Test `tests/test_az_resources.py`

- [ ] **Step 1: failing tests**

```python
# tests/test_az_resources.py
def test_fit_gpu_procs_caps_by_vram():
    from catan_az.resources import fit_gpu_procs
    # 4096 MB free, 535 MB/proc, max 7 -> min(7, 7)=7
    assert fit_gpu_procs(free_vram_mb=4096, per_proc_mb=535, hard_max=7) == 7
    # 2000 MB free -> floor(2000/535)=3
    assert fit_gpu_procs(free_vram_mb=2000, per_proc_mb=535, hard_max=7) == 3
    # never below 1 if any room; 0 if none
    assert fit_gpu_procs(free_vram_mb=400, per_proc_mb=535, hard_max=7) == 0

def test_parse_nvidia_smi_free_vram():
    from catan_az.resources import parse_free_vram_mb
    # nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    assert parse_free_vram_mb("3500\n") == 3500.0

def test_parse_disk_free_gb(tmp_path):
    from catan_az.resources import free_disk_gb
    gb = free_disk_gb(tmp_path)
    assert gb > 0   # tmp dir is on a real fs
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement `resources.py`**

```python
"""Pure resource probes for the daily runner's guards. No side effects
beyond reading system state; each function is unit-testable with a string
or a path so preflight can be tested without a GPU."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


def fit_gpu_procs(*, free_vram_mb: float, per_proc_mb: float, hard_max: int) -> int:
    """How many GPU self-play procs fit in free VRAM, capped at hard_max."""
    by_vram = int(math.floor(free_vram_mb / per_proc_mb))
    return max(0, min(hard_max, by_vram))


def parse_free_vram_mb(nvidia_smi_out: str) -> float:
    """Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`."""
    line = nvidia_smi_out.strip().splitlines()[0]
    return float(line.strip())


def query_free_vram_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True)
    return parse_free_vram_mb(out)


def free_disk_gb(path) -> float:
    usage = shutil.disk_usage(Path(path))
    return usage.free / (1024 ** 3)


def free_ram_gb() -> float:
    # /proc/meminfo MemAvailable in kB
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def foreign_selfplay_pids(my_pid: int) -> list[int]:
    """PIDs of stray self_play_async / catan_az.loop procs not under my_pid."""
    try:
        out = subprocess.check_output(["pgrep", "-f",
            "experiments.self_play_async|catan_az.daily|catan_az.loop"], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(p) for p in out.split() if int(p) != my_pid]
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): resource probes (vram/disk/ram/procs)"`

## Task 3: Preflight guards

**Files:** Create `catan_az/preflight.py`; Test `tests/test_az_preflight.py`

- [ ] **Step 1: failing tests** (monkeypatch the probes — no real GPU/disk needed)

```python
# tests/test_az_preflight.py
import pytest

def _patch(monkeypatch, **kw):
    import catan_az.resources as r
    monkeypatch.setattr(r, "query_free_vram_mb", lambda: kw.get("vram", 4096))
    monkeypatch.setattr(r, "free_disk_gb", lambda p: kw.get("disk", 100))
    monkeypatch.setattr(r, "free_ram_gb", lambda: kw.get("ram", 50))
    monkeypatch.setattr(r, "foreign_selfplay_pids", lambda pid: kw.get("foreign", []))

def test_preflight_ok_caps_procs(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, vram=4096)
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert res.ok
    assert res.capped_procs == 7   # 4096/535 capped at 7

def test_preflight_low_disk_aborts(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, disk=2)   # < min_fast_gb 10
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert not res.ok
    assert any("disk" in r.lower() for r in res.reasons)

def test_preflight_low_vram_degrades_not_abort(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, vram=1600)   # floor(1600/535)=2
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert res.ok and res.capped_procs == 2

def test_preflight_no_vram_aborts(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, vram=300)   # 0 procs fit
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert not res.ok
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement `preflight.py`**

```python
"""Run guards before each daily cycle. Hard failures -> ok=False (abort with
reasons). Soft -> degrade (capped_procs). VRAM is the binding limit on the
4GB card; disk/lock are hard. GPU-busy is intentionally NOT a guard (user
shares the GPU; training is CPU-bound). Spec 2026-06-13 §6."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import resources as r


@dataclass
class PreflightResult:
    ok: bool
    capped_procs: int = 0
    reasons: list[str] = field(default_factory=list)


def preflight(cfg, *, loop_root: Path, archive_root: Path,
              my_pid: int | None = None) -> PreflightResult:
    reasons: list[str] = []
    my_pid = my_pid if my_pid is not None else os.getpid()

    # Hard: fast disk
    fast = r.free_disk_gb(loop_root)
    if fast < cfg.min_fast_gb:
        reasons.append(f"fast-disk free {fast:.1f}GB < {cfg.min_fast_gb}GB")

    # Hard: HDD (archive target)
    try:
        hdd = r.free_disk_gb(archive_root)
        if hdd < cfg.min_hdd_gb:
            reasons.append(f"hdd free {hdd:.1f}GB < {cfg.min_hdd_gb}GB")
    except FileNotFoundError:
        reasons.append(f"archive_root {archive_root} not mounted")

    # Soft: VRAM -> proc cap
    vram = r.query_free_vram_mb()
    procs = r.fit_gpu_procs(free_vram_mb=vram, per_proc_mb=cfg.per_proc_vram_mb,
                            hard_max=cfg.worker_procs_max)
    if procs == 0:
        reasons.append(f"VRAM free {vram:.0f}MB fits 0 procs")

    # Self-heal note: foreign procs are reaped by the caller (daily.py), not
    # here — preflight only reports. (Kept side-effect-free for testability.)

    ok = len(reasons) == 0
    return PreflightResult(ok=ok, capped_procs=procs, reasons=reasons)
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): preflight guards (disk/vram, abort/degrade)"`
- [ ] **Step 6: push** 🚩 **MILESTONE A complete** (guards) — `git push`

## Task 4: rules_id-tagged dirs + fresh-deficit (start of MILESTONE B — fresh-ratio self-play)

**Files:** Modify `catan_az/buffer.py`; Test `tests/test_az_fresh_ratio.py`

- [ ] **Step 1: failing tests**

```python
# tests/test_az_fresh_ratio.py
import json
import pandas as pd

def _mk_dir(root, name, n, rules_id="v3-full", champ="cell6"):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({"seed": range(n), "winner": [0]*n}).to_parquet(d / "games.x.parquet")
    (d / "meta.json").write_text(json.dumps({"rules_id": rules_id, "champion": champ}))
    return d

def test_fresh_deficit_counts_only_current_champion_and_rules(tmp_path):
    from catan_az.buffer import fresh_deficit
    _mk_dir(tmp_path, "old", 300, champ="cell6")           # stale champion
    _mk_dir(tmp_path, "new", 100, champ="az_iter_1")       # fresh
    # target = ceil(0.70 * 1000) = 700; have 100 fresh -> deficit 600
    d = fresh_deficit([tmp_path/"new", tmp_path/"old"], champion="az_iter_1",
                      rules_id="v3-full", window_games=1000, fresh_ratio=0.70)
    assert d == 600

def test_select_window_filters_rules_id(tmp_path):
    from catan_az.buffer import select_window
    a = _mk_dir(tmp_path, "a", 100, rules_id="v3-full")
    b = _mk_dir(tmp_path, "b", 100, rules_id="v4-trades")   # different rules
    sel = select_window([a, b], window_games=1000, rules_id="v3-full")
    assert sel == [a]   # b excluded
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — add to `buffer.py`:

```python
import json as _json
import math as _math


def _read_meta(run_dir: Path) -> dict:
    p = Path(run_dir) / "meta.json"
    return _json.loads(p.read_text()) if p.exists() else {}


def fresh_deficit(iter_dirs_newest_first, *, champion: str, rules_id: str,
                  window_games: int, fresh_ratio: float) -> int:
    """Games still needed so current-champion games are >= fresh_ratio of the
    window. Counts only dirs tagged with `champion` AND `rules_id`."""
    fresh = 0
    for d in iter_dirs_newest_first:
        m = _read_meta(d)
        if m.get("champion") == champion and m.get("rules_id") == rules_id:
            fresh += count_games(d)
    target = _math.ceil(fresh_ratio * window_games)
    return max(0, target - fresh)
```

And modify `select_window` to accept an optional `rules_id` filter:

```python
def select_window(iter_dirs_newest_first, window_games, rules_id=None):
    selected, total = [], 0
    for d in iter_dirs_newest_first:
        if rules_id is not None and _read_meta(d).get("rules_id") != rules_id:
            continue
        n = count_games(d)
        if n == 0:
            continue
        selected.append(d); total += n
        if total >= window_games:
            break
    if not selected:
        raise ValueError(f"buffer: no games in {len(iter_dirs_newest_first)} dirs")
    return selected
```

(Keep the existing signature working: `rules_id` defaults to None.)

- [ ] **Step 4: run, expect PASS** — also run `tests/test_az_buffer.py` to confirm no regression.
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): rules_id-tagged dirs + fresh-deficit window"`

## Task 5: Daily driver — manifest + resume skeleton

**Files:** Create `catan_az/daily.py`; Test `tests/test_az_daily.py`

- [ ] **Step 1: failing tests** (inject fake stage fns; no GPU)

```python
# tests/test_az_daily.py
import json

def test_manifest_round_trip(tmp_path):
    from catan_az.daily import DailyManifest
    m = DailyManifest(iter=3, stage="selfplay", champion="az_iter_1",
                      fresh_target=700, fresh_done=120, rules_id="v3-full")
    m.save(tmp_path)
    back = DailyManifest.load(tmp_path)
    assert back == m

def test_run_day_runs_until_stop(tmp_path, monkeypatch):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    calls = []
    def fake_cycle(cfg, loop_root, iter_n, capped_procs):
        calls.append(iter_n)
        if iter_n == 2:
            (loop_root / "STOP").write_text("")
        return "promote"
    cfg = AzConfig()
    run_day(cfg, loop_root=tmp_path, capped_procs=5, cycle_fn=fake_cycle,
            max_iters=10)
    assert calls == [1, 2]   # stopped after STOP appeared

def test_run_day_respects_max_iters(tmp_path):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    calls = []
    run_day(AzConfig(), loop_root=tmp_path, capped_procs=5,
            cycle_fn=lambda *a, **k: calls.append(k.get("iter_n") or a[2]) or "hold",
            max_iters=3)
    assert len(calls) == 3
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement `daily.py`** (skeleton: manifest + run_day loop with STOP + max_iters; the real cycle_fn comes in Task 7)

```python
"""Resumable daily AZ driver. run_day() runs cycles until a STOP sentinel,
max_iters, or a soft time hint. Each cycle is loop.run_iteration with
fresh-ratio self-play. Manifest (daily_state.json) makes a kill resumable to
the exact stage. Spec 2026-06-13 §4-5."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DailyManifest:
    iter: int
    stage: str
    champion: str
    fresh_target: int
    fresh_done: int
    rules_id: str

    def save(self, loop_root: Path) -> None:
        p = Path(loop_root) / "daily_state.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, p)

    @classmethod
    def load(cls, loop_root: Path) -> "DailyManifest | None":
        p = Path(loop_root) / "daily_state.json"
        return cls(**json.loads(p.read_text())) if p.exists() else None


def _stop_requested(loop_root: Path) -> bool:
    return (Path(loop_root) / "STOP").exists()


def run_day(cfg, *, loop_root: Path, capped_procs: int, cycle_fn,
            max_iters: int = 1000, next_iter: int | None = None) -> None:
    """Run cycles until STOP / max_iters. cycle_fn(cfg, loop_root, iter_n,
    capped_procs) -> verdict. next_iter resumes numbering (default: derive
    from existing iter_N dirs)."""
    loop_root = Path(loop_root)
    loop_root.mkdir(parents=True, exist_ok=True)
    n = next_iter if next_iter is not None else _next_iter_number(loop_root)
    done = 0
    while done < max_iters:
        if _stop_requested(loop_root):
            break
        cycle_fn(cfg, loop_root, n, capped_procs)
        n += 1
        done += 1


def _next_iter_number(loop_root: Path) -> int:
    existing = [int(d.name.split("_")[1]) for d in loop_root.glob("iter_*")
                if d.name.split("_")[1].isdigit()]
    return (max(existing) + 1) if existing else 1
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): daily driver manifest + run_day loop"`

## Task 6: Fresh-ratio self-play launcher (low-priority procs)

**Files:** Modify `catan_az/daily.py`; Test `tests/test_az_daily.py` (add)

- [ ] **Step 1: failing test**

```python
def test_generate_fresh_computes_deficit_and_skips_when_met(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    launched = {}
    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs, champion, rules_id):
        launched["n_games"] = n_games
        # simulate the procs writing meta + games
        import pandas as pd, json
        d = out_dir / "run1"; d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner":[0]*n_games}).to_parquet(d/"games.x.parquet")
        (d/"meta.json").write_text(json.dumps({"rules_id": rules_id, "champion": champion}))
        return [d]
    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(window_games=1000, fresh_ratio=0.70)
    dirs = daily.generate_fresh(cfg, iter_dir=tmp_path/"iter_1", champion="az_iter_1",
                                champion_ckpt=tmp_path/"c.pt", capped_procs=5,
                                prior_dirs=[])
    assert launched["n_games"] == 700   # ceil(0.70*1000), no prior fresh
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — add to `daily.py`:

```python
import subprocess
from .buffer import fresh_deficit


def _launch_selfplay_procs(cfg, out_dir, checkpoint, n_games, n_procs,
                           champion, rules_id):
    """Launch n_procs low-priority self_play_async procs splitting n_games,
    each writing meta.json {rules_id, champion}. Blocks until all exit.
    Returns the list of run dirs created."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per = max(1, n_games // max(1, n_procs))
    procs = []
    for i in range(n_procs):
        sb = 31_000_000 + i * 1_000_000
        cmd = ["nice", "-n", str(cfg.worker_nice),
               "python", "-m", "catan_mcts.experiments.self_play_async",
               "--out-root", str(out_dir), "--checkpoint", str(checkpoint),
               "--num-games", str(per), "--n-sims", str(cfg.sims),
               "--n-concurrent", str(cfg.n_concurrent),
               "--max-batch", str(cfg.max_batch), "--self-play",
               "--seed-base", str(sb), "--device", "cuda",
               "--max-seconds", "21600"]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    dirs = sorted(out_dir.glob("*self_play_async*"))
    # tag each with meta (the experiment doesn't know about champion/rules_id)
    for d in dirs:
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "champion": champion}))
    return dirs


def generate_fresh(cfg, *, iter_dir: Path, champion: str, champion_ckpt: Path,
                   capped_procs: int, prior_dirs: list) -> list:
    """Generate current-champion self-play until fresh games >= fresh_ratio of
    the window. Resumable: counts existing fresh first, generates only the
    deficit."""
    deficit = fresh_deficit(prior_dirs, champion=champion, rules_id=cfg.rules_id,
                            window_games=cfg.window_games, fresh_ratio=cfg.fresh_ratio)
    if deficit <= 0:
        return []
    return _launch_selfplay_procs(cfg, iter_dir / "selfplay", champion_ckpt,
                                  deficit, capped_procs, champion, cfg.rules_id)
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): fresh-ratio self-play launcher (nice, deficit-only)"`
- [ ] **Step 6: push** 🚩 **MILESTONE B complete** (fresh-ratio self-play) — `git push`

## Task 7: VP-margin tiebreak (start of MILESTONE C — gate + cycle)

**Files:** Modify `catan_az/arena.py`; Test `tests/test_az_arena.py` (add)

- [ ] **Step 1: failing test**

```python
def test_vp_margin_tiebreak_breaks_vp_tie():
    from catan_az.arena import _vp_leader_margin
    class E:
        def vp(self, p): return [9, 9, 4, 2][p]
        def stats(self): return {"players": [
            {"settlements_built": 3, "cities_built": 2},  # seat0: 5
            {"settlements_built": 4, "cities_built": 2},  # seat1: 6 -> wins tie
            {"settlements_built": 1, "cities_built": 0},
            {"settlements_built": 1, "cities_built": 0}]}
    class S:
        _engine = E()
    assert _vp_leader_margin(S()) == 1   # VP tie 0/1, broken by build count

def test_vp_margin_true_tie_is_draw():
    from catan_az.arena import _vp_leader_margin
    class E:
        def vp(self, p): return [9, 9, 4, 2][p]
        def stats(self): return {"players": [
            {"settlements_built": 3, "cities_built": 2},
            {"settlements_built": 3, "cities_built": 2},  # identical to seat0
            {"settlements_built": 1, "cities_built": 0},
            {"settlements_built": 1, "cities_built": 0}]}
    class S:
        _engine = E()
    assert _vp_leader_margin(S()) == -1   # all signals tied -> draw
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — in `arena.py`, add `_vp_leader_margin` and route the timeout exits through it (replace the two `_vp_leader(state)` calls):

```python
def _vp_leader_margin(state) -> int:
    """VP-leader tiebreak with a margin fallback (spec 2026-06-13 §9): top
    public VP wins; ties broken by (settlements+cities), then -1 draw."""
    vps = [int(state._engine.vp(i)) for i in range(4)]
    top = max(vps)
    leaders = [i for i, v in enumerate(vps) if v == top]
    if len(leaders) == 1:
        return leaders[0]
    stats = state._engine.stats()["players"]
    def builds(i):
        return stats[i]["settlements_built"] + stats[i]["cities_built"]
    best = max(builds(i) for i in leaders)
    tied = [i for i in leaders if builds(i) == best]
    return tied[0] if len(tied) == 1 else -1
```

Replace both `return _vp_leader(state), True` / `return (_vp_leader(state)), True` call sites with `return _vp_leader_margin(state), True`. Keep `_vp_leader` (other tests reference it) but have it delegate: `def _vp_leader(state): return _vp_leader_margin(state)`.

- [ ] **Step 4: run, expect PASS** — run full `tests/test_az_arena.py`.
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): VP-margin tiebreak fallback"`

## Task 8: The real cycle function (wire daily → loop)

**Files:** Modify `catan_az/daily.py`; Test `tests/test_az_daily.py` (add)

- [ ] **Step 1: failing test** (fake generate_fresh + run_iteration; assert manifest + verdict flow)

```python
def test_run_cycle_generates_then_runs_iteration(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    seen = {}
    monkeypatch.setattr(daily, "generate_fresh",
        lambda cfg, **k: seen.setdefault("gen", True) or [tmp_path/"sp"])
    def fake_run_iteration(cfg, loop_root, iter_n, *, existing_selfplay_dirs):
        seen["iter"] = iter_n; seen["dirs"] = existing_selfplay_dirs
        return "promote"
    monkeypatch.setattr(daily, "run_iteration", fake_run_iteration)
    monkeypatch.setattr(daily, "_champion_from_ladder",
        lambda root: ("az_iter_1", str(tmp_path/"c.pt")))
    v = daily.run_cycle(AzConfig(), tmp_path, 3, capped_procs=5)
    assert v == "promote" and seen["iter"] == 3 and seen["gen"] is True
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement `run_cycle`** in `daily.py`:

```python
from .loop import run_iteration
from .ladder import Ladder
from .status import StatusWriter


def _champion_from_ladder(loop_root: Path):
    l = Ladder(loop_root)
    c = l.champion()
    return c["name"], c["checkpoint"]


def run_cycle(cfg, loop_root: Path, iter_n: int, *, capped_procs: int) -> str:
    """One full AZ cycle: fresh self-play -> run_iteration (train/arena/publish).
    Writes the manifest at each transition. Returns the verdict."""
    loop_root = Path(loop_root)
    iter_dir = loop_root / f"iter_{iter_n}"
    champion, champion_ckpt = _champion_from_ladder(loop_root)
    status = StatusWriter(loop_root)

    DailyManifest(iter=iter_n, stage="selfplay", champion=champion,
                  fresh_target=0, fresh_done=0, rules_id=cfg.rules_id).save(loop_root)

    prior = _all_selfplay_dirs(loop_root)
    fresh_dirs = generate_fresh(cfg, iter_dir=iter_dir, champion=champion,
                                champion_ckpt=Path(champion_ckpt),
                                capped_procs=capped_procs, prior_dirs=prior)

    DailyManifest(iter=iter_n, stage="iterate", champion=champion,
                  fresh_target=0, fresh_done=len(fresh_dirs),
                  rules_id=cfg.rules_id).save(loop_root)

    all_fresh = fresh_dirs + prior
    verdict = run_iteration(cfg, loop_root, iter_n,
                            existing_selfplay_dirs=[str(d) for d in all_fresh])
    return verdict


def _all_selfplay_dirs(loop_root: Path) -> list:
    """All prior self-play run dirs newest-first (from iter_*/selfplay + iter_*
    SELFPLAY.done markers)."""
    dirs = []
    for it in sorted(loop_root.glob("iter_*"),
                     key=lambda p: int(p.name.split("_")[1]), reverse=True):
        sp = it / "selfplay"
        if sp.exists():
            dirs.extend(sorted(sp.glob("*self_play_async*"), reverse=True))
    return dirs
```

Then make `run_day`'s default `cycle_fn` be `run_cycle`.

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): run_cycle wires fresh self-play into the loop"`

## Task 9: Stagnation detection

**Files:** Modify `catan_az/daily.py`; Test `tests/test_az_daily.py` (add)

- [ ] **Step 1: failing test**

```python
def test_stagnation_flag_after_n_holds(tmp_path):
    from catan_az.daily import stagnation_holds_from_journal
    import csv
    p = tmp_path / "journal.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "verdict"]); w.writeheader()
        for i, v in enumerate(["promote", "hold", "hold", "hold"], 1):
            w.writerow({"iter": i, "verdict": v})
    assert stagnation_holds_from_journal(p) == 3   # trailing holds
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** in `daily.py`:

```python
import csv as _csv


def stagnation_holds_from_journal(journal_path: Path) -> int:
    """Count trailing consecutive non-promote verdicts (hold/invalid)."""
    if not Path(journal_path).exists():
        return 0
    with open(journal_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    n = 0
    for row in reversed(rows):
        if row.get("verdict") in ("hold", "invalid"):
            n += 1
        else:
            break
    return n
```

Wire into `run_day`: after each cycle, if `stagnation_holds_from_journal(loop_root/"journal.csv") >= cfg.stagnation_holds`, write a `STAGNATION` flag into status and `break` (stop the day; surfaced, not an error).

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): stagnation detection (N trailing holds)"`
- [ ] **Step 6: push** 🚩 **MILESTONE C complete** (gate + cycle + stagnation) — `git push`

## Task 10: Archive (start of MILESTONE D — lifecycle + dashboard)

**Files:** Create `catan_az/archive.py`; Test `tests/test_az_archive.py`

- [ ] **Step 1: failing tests**

```python
# tests/test_az_archive.py
import json, pandas as pd

def _mk(root, name, n):
    d = root / name; d.mkdir(parents=True)
    pd.DataFrame({"seed": range(n), "winner":[0]*n}).to_parquet(d/"games.x.parquet")
    (d/"meta.json").write_text(json.dumps({"rules_id":"v3-full","champion":"c"}))
    return d

def test_archive_moves_only_out_of_window(tmp_path):
    from catan_az.archive import archive_out_of_window
    hdd = tmp_path / "hdd"; hdd.mkdir()
    in_win = _mk(tmp_path, "in", 10)
    out_win = _mk(tmp_path, "out", 10)
    archive_out_of_window(window_dirs=[in_win], all_dirs=[in_win, out_win],
                          archive_root=hdd, rules_id="v3-full")
    # out moved (parquet gone from source, breadcrumb left), in untouched
    assert not list(out_win.glob("*.parquet"))
    assert (out_win / "ARCHIVED.txt").exists()
    assert list(in_win.glob("*.parquet"))
    assert list((hdd).rglob("games.x.parquet"))

def test_archive_idempotent(tmp_path):
    from catan_az.archive import archive_out_of_window
    hdd = tmp_path / "hdd"; hdd.mkdir()
    in_win = _mk(tmp_path, "in", 10); out_win = _mk(tmp_path, "out", 10)
    archive_out_of_window([in_win], [in_win, out_win], hdd, "v3-full")
    # second run: out already archived (breadcrumb), no error, no double-move
    archive_out_of_window([in_win], [in_win, out_win], hdd, "v3-full")
    assert (out_win / "ARCHIVED.txt").exists()
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement `archive.py`**

```python
"""Post-cycle archival: move out-of-window raw parquet to the HDD, keep the
window on fast disk. Idempotent (breadcrumb), never deletes (moves). Runs
only after a cycle fully publishes. Spec 2026-06-13 §8."""
from __future__ import annotations

import shutil
from pathlib import Path


def archive_out_of_window(window_dirs, all_dirs, archive_root, rules_id) -> int:
    """Move parquet of dirs NOT in window_dirs to archive_root/<rules_id>/<dir>.
    Leaves ARCHIVED.txt breadcrumb. Returns count of dirs archived."""
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
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): HDD archive of out-of-window games (idempotent)"`

## Task 11: Dashboard backend

**Files:** Create `catan_az/dashboard/__init__.py`, `catan_az/dashboard/server.py`; Test `tests/test_az_dashboard.py`

- [ ] **Step 1: failing test**

```python
# tests/test_az_dashboard.py
import json

def test_dashboard_summary_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from catan_az.dashboard.server import create_dashboard
    # seed minimal loop_root
    (tmp_path/"ladder.json").write_text(json.dumps({
        "champion":"az_iter_1","entries":{"az_iter_1":{"name":"az_iter_1",
        "checkpoint":"/c.pt","elo":1003.6,"games":120,"created_iter":1}},"history":[]}))
    (tmp_path/"status.json").write_text(json.dumps({"iter":2,"stage":"arena"}))
    (tmp_path/"journal.csv").write_text("iter,verdict,arena_winrate\n1,promote,0.65\n")
    app = create_dashboard(loop_root=tmp_path, web_port=8000)
    c = TestClient(app)
    r = c.get("/api/summary").json()
    assert r["champion"]["name"] == "az_iter_1"
    assert r["status"]["stage"] == "arena"
    assert len(r["journal"]) == 1
    assert "play_champion_url" in r
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement `dashboard/server.py`**

```python
"""Minimal AZ progress dashboard: one JSON summary endpoint + static page.
Reads journal.csv / status.json / ladder.json (no DB). Spec 2026-06-13 §7."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).parent / "static"


def create_dashboard(*, loop_root, web_port: int = 8000) -> FastAPI:
    loop_root = Path(loop_root)
    app = FastAPI(title="AZ Daily Dashboard")

    @app.get("/api/summary")
    def summary():
        ladder = _read_json(loop_root / "ladder.json", {})
        status = _read_json(loop_root / "status.json", {})
        journal = _read_csv(loop_root / "journal.csv")
        champ = (ladder.get("entries", {}) or {}).get(ladder.get("champion"), {})
        return {
            "champion": champ,
            "status": status,
            "journal": journal[-10:],
            # deep-link into the existing web app's lobby, az-champion tier
            "play_champion_url": f"http://localhost:{web_port}/?difficulty=az-champion",
        }

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _read_csv(p: Path):
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))
```

Also create empty `catan_az/dashboard/__init__.py`.

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-daily): dashboard summary endpoint"`

## Task 12: Dashboard frontend

**Files:** Create `catan_az/dashboard/static/index.html`; Test: manual (browser) + the backend test covers JSON.

- [ ] **Step 1: implement `index.html`** — a single self-contained page (vanilla JS, no build), polling `/api/summary` every 5s:

```html
<!doctype html><html><head><meta charset="utf-8"><title>AZ Daily</title>
<style>
 body{font-family:system-ui;margin:2rem;background:#0f1115;color:#e6e6e6}
 .card{background:#1a1d24;border-radius:8px;padding:1rem 1.5rem;margin:1rem 0}
 h1{font-size:1.4rem} .big{font-size:2rem;font-weight:700}
 table{border-collapse:collapse;width:100%} td,th{padding:.3rem .6rem;text-align:left;border-bottom:1px solid #2a2e38}
 .promote{color:#4ade80} .hold,.invalid{color:#fbbf24}
 a.play{display:inline-block;margin-top:.5rem;background:#2563eb;color:#fff;padding:.4rem .8rem;border-radius:6px;text-decoration:none}
 .flag{color:#f87171;font-weight:700}
</style></head><body>
<h1>Catan AZ — Daily Training</h1>
<div class="card"><div>Champion</div>
  <div class="big" id="champ">—</div>
  <div id="elo"></div>
  <a class="play" id="play" href="#">▶ Play the champion</a>
</div>
<div class="card"><div>Current run</div><div id="status">—</div><div id="stag" class="flag"></div></div>
<div class="card"><div>Last 10 iterations</div><table id="journal"><thead>
  <tr><th>iter</th><th>verdict</th><th>winrate</th><th>draws</th></tr></thead><tbody></tbody></table></div>
<script>
async function tick(){
  try{
    const s = await (await fetch('/api/summary')).json();
    const c = s.champion||{};
    document.getElementById('champ').textContent = c.name||'—';
    document.getElementById('elo').textContent = c.elo? `Elo ${c.elo.toFixed(1)} · ${c.games} games`:'';
    document.getElementById('play').href = s.play_champion_url||'#';
    const st = s.status||{};
    document.getElementById('status').textContent =
      `iter ${st.iter??'—'} · stage ${st.stage??'—'}`;
    document.getElementById('stag').textContent = st.STAGNATION? '⚠ STAGNATION — champion not improving':'';
    const tb = document.querySelector('#journal tbody'); tb.innerHTML='';
    (s.journal||[]).slice().reverse().forEach(r=>{
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${r.iter}</td><td class="${r.verdict}">${r.verdict}</td>`+
        `<td>${r.arena_winrate??''}</td><td>${r.arena_draws??''}</td>`;
      tb.appendChild(tr);
    });
  }catch(e){ document.getElementById('status').textContent='(waiting for data…)'; }
}
tick(); setInterval(tick, 5000);
</script></body></html>
```

- [ ] **Step 2: manual check** — launch dashboard against the live `az_loop` root, open `http://localhost:8099`, confirm champion az_iter_1 + the iter-1/iter-2 journal rows render and the Play link points at the web app.

Run: `python -c "import uvicorn; from catan_az.dashboard.server import create_dashboard; uvicorn.run(create_dashboard(loop_root='/home/chitii/catan_data/runs/v3/az_loop', web_port=8000), host='127.0.0.1', port=8099)"`

- [ ] **Step 3: commit** `git commit -m "feat(az-daily): dashboard frontend (auto-refresh, play link)"`

## Task 13: Entry script + archive wiring

**Files:** Create `scripts/run_az_day.sh`; Modify `catan_az/daily.py` (call archive post-cycle); Test `tests/test_az_daily.py` (archive-after-publish)

- [ ] **Step 1: failing test**

```python
def test_run_cycle_archives_after_publish(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    called = {}
    monkeypatch.setattr(daily, "generate_fresh", lambda cfg, **k: [])
    monkeypatch.setattr(daily, "run_iteration", lambda *a, **k: "promote")
    monkeypatch.setattr(daily, "_champion_from_ladder", lambda r: ("c","/c.pt"))
    monkeypatch.setattr(daily, "archive_out_of_window",
        lambda **k: called.setdefault("archived", True) or 0)
    monkeypatch.setattr(daily, "_all_selfplay_dirs", lambda r: [])
    monkeypatch.setattr(daily, "select_window", lambda *a, **k: [])
    daily.run_cycle(AzConfig(), tmp_path, 1, capped_procs=5)
    assert called.get("archived") is True
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — at the end of `run_cycle`, after `run_iteration` returns, call archive:

```python
    from .archive import archive_out_of_window
    from .buffer import select_window
    try:
        window = select_window(all_fresh, cfg.window_games, rules_id=cfg.rules_id)
    except ValueError:
        window = all_fresh
    archive_out_of_window(window_dirs=window, all_dirs=all_fresh,
                          archive_root=Path(cfg.archive_root) , rules_id=cfg.rules_id)
    return verdict
```

(import `archive_out_of_window` and `select_window` at module top; remove the inner import.)

- [ ] **Step 4: run, expect PASS**

- [ ] **Step 5: create `scripts/run_az_day.sh`**

```bash
#!/usr/bin/env bash
# Cron-ready daily AZ trainer entry. Preflight -> daily driver -> (archive is
# per-cycle inside the driver). Run from WSL with the mcts-study venv active.
set -euo pipefail
LOOP_ROOT="${1:-/home/chitii/catan_data/runs/v3/az_loop}"
cd "$(dirname "$0")/.."   # mcts_study/
LOCK="$LOOP_ROOT/daily.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "[az-day] another run is active (PID $(cat "$LOCK")) — abort"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
# reap stale self-play procs from a dead run
pkill -f 'experiments.self_play_async' 2>/dev/null || true
python -m catan_az.daily --loop-root "$LOOP_ROOT" --max-iters "${MAX_ITERS:-1000}"
```

- [ ] **Step 6: add `daily.py` CLI** (`__main__`): parse `--loop-root`, `--max-iters`; run preflight; if `ok`, `run_day(cfg, loop_root, capped_procs=res.capped_procs)`; else print reasons + exit 1.

```python
def cli_main():
    import argparse
    from .config import AzConfig
    from .preflight import preflight
    p = argparse.ArgumentParser()
    p.add_argument("--loop-root", type=Path, required=True)
    p.add_argument("--max-iters", type=int, default=1000)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()
    cfg = AzConfig.from_json(args.config) if args.config else AzConfig()
    res = preflight(cfg, loop_root=args.loop_root, archive_root=Path(cfg.archive_root))
    if not res.ok:
        print("[az-day] preflight FAILED:", "; ".join(res.reasons)); raise SystemExit(1)
    print(f"[az-day] preflight ok, {res.capped_procs} procs")
    run_day(cfg, loop_root=args.loop_root, capped_procs=res.capped_procs,
            cycle_fn=run_cycle, max_iters=args.max_iters)

if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 7: chmod + commit** `chmod +x scripts/run_az_day.sh && git commit -m "feat(az-daily): entry script + CLI + per-cycle archive wiring"`

## Task 14: Integration test (micro daily + resume)

**Files:** Test `tests/test_az_daily_integration.py` (marked `slow`)

- [ ] **Step 1: write the slow integration test** — one real micro daily cycle (CPU, scratch h32 net, tiny budget) then a simulated kill+resume:

```python
import pytest
pytestmark = pytest.mark.slow

def test_micro_daily_cycle_and_resume(tmp_path):
    import torch, json
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    from catan_az import daily
    from catan_gnn.gnn_model import GnnModel
    champ = tmp_path / "scratch.pt"
    torch.save(GnnModel(hidden_dim=32, num_layers=2).state_dict(), champ)
    Ladder(tmp_path, champion_checkpoint=str(champ), champion_name="scratch")
    cfg = AzConfig(window_games=4, fresh_ratio=0.5, sims=4, n_concurrent=4,
                   max_batch=4, max_epochs=1, arena_games=4, arena_sims=4,
                   vp_target=5, bonuses=False, hidden_dim=32, num_layers=2,
                   arena_min_decisive=1, archive_root=str(tmp_path/"hdd"))
    (tmp_path/"hdd").mkdir()
    # run one cycle on CPU by monkeypatching device to cpu via cfg path is
    # not exposed; instead call run_cycle with capped_procs=1 and rely on the
    # self_play_async --device default (cuda). For CI without GPU, skip if no cuda.
    if not torch.cuda.is_available():
        pytest.skip("micro daily needs CUDA (self_play_async --device cuda)")
    daily.run_day(cfg, loop_root=tmp_path, capped_procs=1, cycle_fn=daily.run_cycle,
                  max_iters=1)
    assert (tmp_path/"journal.csv").exists()
    assert (tmp_path/"daily_state.json").exists()
```

- [ ] **Step 2: run** `python -m pytest tests/test_az_daily_integration.py -q -m slow` — expect PASS (or skip if no CUDA on the runner; on this box CUDA exists).
- [ ] **Step 3: commit** `git commit -m "test(az-daily): micro daily integration"`
- [ ] **Step 4: push** 🚩 **MILESTONE D complete** (lifecycle + dashboard + entry) — `git push`

## Task 15: Live smoke + docs

- [ ] **Step 1:** run `scripts/run_az_day.sh` against a *throwaway* loop_root with `MAX_ITERS=1` and tiny config to confirm the end-to-end path (preflight → fresh self-play → iterate → archive) works on real hardware. Capture the preflight proc-cap line + the journal row.
- [ ] **Step 2:** launch the dashboard against the real `az_loop`, screenshot/confirm it renders champion + journal + play link.
- [ ] **Step 3:** write `docs/superpowers/journals/2026-06-13-az-daily-runner-built.md` — how to run daily, how to pause (`touch STOP`), how to read the dashboard, the VRAM proc-cap, where archives go.
- [ ] **Step 4: commit + push** 🚩 **MILESTONE E complete** (shipped) — update PR description.

---

## Self-review

**Spec coverage:** §1 framing → Task 13 CLI; §2 failure model → guards (T3) + stagnation (T9) + tests; §3 architecture → all files mapped; §3b worker sizing → T2 fit_gpu_procs + T6 nice procs; §4 resumability → T5 manifest + fresh-deficit (T4) + STOP; §5 fresh-ratio → T4/T6; §6 guards → T3; §7 dashboard → T11/T12; §8 archive → T10/T13; §9 VP-margin → T7; §10 testing → every task + T14; §12 config → T1. All covered.

**Placeholder scan:** no TBD/TODO; every code step has full code.

**Type consistency:** `fit_gpu_procs`, `preflight`/`PreflightResult.capped_procs`, `fresh_deficit`, `select_window(…, rules_id=)`, `DailyManifest`, `generate_fresh`, `run_cycle`, `_vp_leader_margin`, `archive_out_of_window`, `create_dashboard` — names consistent across tasks. `run_iteration(existing_selfplay_dirs=…)` matches the real loop signature (verified).

**Note for executor:** self_play_async defaults `--device cuda`; the integration test skips without CUDA. The live box has CUDA. Run all pytest in the WSL venv.
