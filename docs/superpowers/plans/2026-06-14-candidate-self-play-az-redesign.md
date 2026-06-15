# Candidate Self-Play AZ Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline — the final tasks drive a real multi-day WSL run). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the name-based "fresh" mechanism with candidate self-play + a `gen_iter`-tagged dynamic window (reset on promotion), 65% promote bar on a 300-game arena, max-10-iters terminal, and generator/champion terminology that makes the stale-data bug impossible to repeat.

**Architecture:** Self-play uses the LATEST CANDIDATE net (iter-1 → champion fallback). Each iteration generates exactly 1000 games tagged `{generator_name, gen_iter, rules_id}`. The training window = all dirs with `gen_iter >= last_promotion_iter` (reset on promotion). The arena (300 games) gates promotion at >65%. Builds on the `catan_az` package.

**Tech Stack:** Python 3.12 (WSL venv `~/catan_mcts_venvs/mcts-study/`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-candidate-self-play-az-redesign.md`

**Run commands in WSL:** `wsl.exe -e bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study && <cmd>"`. Commit from the repo root `cd /c/dojo/catan_bot/.claude/worktrees/az-bots`.

**Commit after each green task. Push at the 🚩 milestones.**

---

## File structure

| File | Change |
|---|---|
| `catan_az/config.py` | games_per_iter→1000, arena_games→300, promote_threshold→0.65, +max_iters_per_model=10; retire fresh_ratio/window_games (keep as ignored for back-compat) |
| `catan_az/buffer.py` | +`gen_window(dirs, *, gen_iter_min, rules_id)` (replaces fresh_deficit/select_window for the new path); +`own_iter_games(iter_dir, gen_iter)` for the quota |
| `catan_az/daily.py` | generator selection, gen_iter-tagged self-play, dynamic window, reset-on-promotion state, terminal-at-10; rename champion→generator where it means the self-play net |
| `catan_az/ladder.py` | +`last_promotion_iter` tracking (window reset boundary) |
| `catan_az/progress.py` | row gains `generator` + `window_iters` |
| tests | new test files per task |

---

## Task 1: Config — new knobs 🚩 (start MILESTONE A — mechanism core)

**Files:** Modify `catan_az/config.py`; Test `tests/test_az_redesign_config.py`

- [ ] **Step 1: failing test**

```python
# tests/test_az_redesign_config.py
def test_redesign_config_values():
    from catan_az.config import AzConfig
    c = AzConfig()
    assert c.games_per_iter == 1000
    assert c.arena_games == 300
    assert c.arena_games % 4 == 0          # 4 rotations
    assert c.promote_threshold == 0.65
    assert c.max_iters_per_model == 10
```

- [ ] **Step 2: run, expect FAIL** — `python -m pytest tests/test_az_redesign_config.py -q`

- [ ] **Step 3: edit `config.py`** — change three defaults + add one field:

```python
    games_per_iter: int = 1000         # candidate self-play, always generated
    arena_games: int = 300             # 65% bar needs the statistical power
    promote_threshold: float = 0.65    # strictly-greater; demand clear improvement
    max_iters_per_model: int = 10      # stop after 10 iters w/o promotion -> user
```

(Leave `fresh_ratio`, `window_games` in place but unused by the new path — old
tests still import them; a later cleanup task removes them.)

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): config — 1000 games, 300 arena, 0.65 bar, max-10"`

## Task 2: gen_iter window selection (buffer.py)

**Files:** Modify `catan_az/buffer.py`; Test `tests/test_az_gen_window.py`

- [ ] **Step 1: failing tests**

```python
# tests/test_az_gen_window.py
import json
import pandas as pd

def _mk(root, name, n, gen_iter, rules_id="v3-full", gen="az_iter_1"):
    d = root / name; d.mkdir(parents=True)
    pd.DataFrame({"seed": range(n), "winner": [0]*n}).to_parquet(d/"games.x.parquet")
    (d/"meta.json").write_text(json.dumps(
        {"rules_id": rules_id, "generator_name": gen, "gen_iter": gen_iter}))
    return d

def test_gen_window_includes_only_at_or_after_boundary(tmp_path):
    from catan_az.buffer import gen_window
    a = _mk(tmp_path, "a", 100, gen_iter=2)
    b = _mk(tmp_path, "b", 100, gen_iter=3)
    c = _mk(tmp_path, "c", 100, gen_iter=1)   # before boundary -> excluded
    sel = gen_window([a, b, c], gen_iter_min=2, rules_id="v3-full")
    assert set(sel) == {a, b}

def test_gen_window_filters_rules_id(tmp_path):
    from catan_az.buffer import gen_window
    a = _mk(tmp_path, "a", 100, gen_iter=2, rules_id="v3-full")
    b = _mk(tmp_path, "b", 100, gen_iter=2, rules_id="v4-trades")
    sel = gen_window([a, b], gen_iter_min=1, rules_id="v3-full")
    assert sel == [a]

def test_gen_window_legacy_dirs_excluded(tmp_path):
    """Dirs without gen_iter (old {champion,rules_id}) are treated as gen_iter=0
    -> excluded by any boundary >=1 (so old stale data can't pollute)."""
    from catan_az.buffer import gen_window
    d = tmp_path / "legacy"; d.mkdir()
    import pandas as pd, json
    pd.DataFrame({"seed":[1],"winner":[0]}).to_parquet(d/"games.x.parquet")
    (d/"meta.json").write_text(json.dumps({"rules_id":"v3-full","champion":"x"}))
    sel = gen_window([d], gen_iter_min=1, rules_id="v3-full")
    assert sel == []

def test_own_iter_games_counts_only_this_gen_iter(tmp_path):
    from catan_az.buffer import own_iter_games
    a = _mk(tmp_path, "a", 60, gen_iter=4)
    b = _mk(tmp_path, "b", 40, gen_iter=4)
    c = _mk(tmp_path, "c", 99, gen_iter=3)   # different iter -> not counted
    assert own_iter_games([a, b, c], gen_iter=4) == 100
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — add to `buffer.py` (reuse existing `_read_meta`, `count_games`):

```python
def gen_window(dirs, *, gen_iter_min: int, rules_id: str) -> list:
    """Dirs whose gen_iter >= gen_iter_min AND rules_id matches. Missing
    gen_iter (legacy dirs) = 0, so they're excluded by any boundary >= 1.
    This is the recency-based window (replaces name-based fresh selection;
    fixes the 2026-06-14 stale-data bug)."""
    out = []
    for d in dirs:
        m = _read_meta(d)
        if m.get("rules_id") != rules_id:
            continue
        if int(m.get("gen_iter", 0)) >= gen_iter_min:
            out.append(d)
    return out


def own_iter_games(dirs, *, gen_iter: int) -> int:
    """Games generated specifically in `gen_iter` (the quota counter — counting
    ONLY this iteration's own games is what kills the bug where other iters'
    games satisfied the quota)."""
    total = 0
    for d in dirs:
        if int(_read_meta(d).get("gen_iter", -1)) == gen_iter:
            total += count_games(d)
    return total
```

(Match the real `own_iter_games` signature to the test: it's called
`own_iter_games(dirs, gen_iter=...)` — keyword. Update the test call to
`own_iter_games([a,b,c], gen_iter=4)`.)

- [ ] **Step 4: run, expect PASS** + run `tests/test_az_fresh_ratio.py` (old path still imports cleanly).
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): gen_iter window + own-iter quota (buffer)"`

## Task 3: last_promotion_iter in ladder (reset boundary)

**Files:** Modify `catan_az/ladder.py`; Test `tests/test_az_ladder.py` (add)

- [ ] **Step 1: failing test**

```python
def test_last_promotion_iter_tracks_promotions(tmp_path):
    from catan_az.ladder import Ladder
    l = Ladder(tmp_path, champion_checkpoint="/c.pt", champion_name="seed")
    assert l.last_promotion_iter() == 0          # fresh -> 0
    l.register_candidate("az_iter_5", "/5.pt", created_iter=5)
    l.promote("az_iter_5", promoted_at_iter=5)
    assert l.last_promotion_iter() == 5
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — in `ladder.py`:
  - In `__init__` fresh-ladder dict, add `"last_promotion_iter": 0`.
  - Add accessor:

```python
    def last_promotion_iter(self) -> int:
        return self._data.get("last_promotion_iter", 0)
```

  - Change `promote` signature to `def promote(self, name, *, promoted_at_iter=None):`
    and inside it set `self._data["last_promotion_iter"] = promoted_at_iter` when
    given (keep back-compat: default None leaves it unchanged). Existing callers
    pass no kwarg, so they keep working.

- [ ] **Step 4: run, expect PASS** + `tests/test_az_ladder.py` full.
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): ladder tracks last_promotion_iter (window reset boundary)"`
- [ ] **Step 6: push** 🚩 **MILESTONE A** (config + window + reset state)

## Task 4: generator selection (latest candidate / champion fallback) — start MILESTONE B (daily mechanism)

**Files:** Modify `catan_az/daily.py`; Test `tests/test_az_generator.py`

- [ ] **Step 1: failing tests**

```python
# tests/test_az_generator.py
def test_generator_iter1_is_champion(tmp_path):
    from catan_az.daily import select_generator
    # no candidate exists yet -> champion
    name, ckpt = select_generator(tmp_path, iter_n=1,
                                  champion=("seed", "/seed.pt"))
    assert name == "seed" and ckpt == "/seed.pt"

def test_generator_iter_n_is_latest_candidate(tmp_path):
    from catan_az.daily import select_generator
    # iter (N-1) produced a candidate checkpoint
    cand = tmp_path / "iter_4" / "training"
    cand.mkdir(parents=True)
    (cand / "checkpoint_best.pt").write_bytes(b"x")
    name, ckpt = select_generator(tmp_path, iter_n=5,
                                  champion=("seed", "/seed.pt"))
    assert name == "cand_iter_4"
    assert ckpt.endswith("iter_4/training/checkpoint_best.pt")

def test_generator_falls_back_to_champion_if_no_candidate(tmp_path):
    from catan_az.daily import select_generator
    # iter 5 but iter_4 has no checkpoint -> champion fallback
    name, ckpt = select_generator(tmp_path, iter_n=5,
                                  champion=("seed", "/seed.pt"))
    assert name == "seed"
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** in `daily.py`:

```python
def select_generator(loop_root, iter_n: int, champion):
    """Net that generates THIS iteration's self-play. iter-1 (or no prior
    candidate) -> champion; else iter (N-1)'s trained candidate (promoted or
    not — self-play always uses the LATEST net, the canonical-AZ separation of
    'who self-plays' from 'who's crowned'). Returns (generator_name, ckpt_path)."""
    prev = Path(loop_root) / f"iter_{iter_n - 1}" / "training" / "checkpoint_best.pt"
    if iter_n > 1 and prev.exists():
        return f"cand_iter_{iter_n - 1}", str(prev)
    return champion   # (name, ckpt) tuple
```

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): generator = latest candidate (champion fallback)"`

## Task 5: gen_iter-tagged self-play + own-iter quota

**Files:** Modify `catan_az/daily.py` (`_launch_selfplay_procs` meta + `generate_fresh`→`generate_iter_games`); Test `tests/test_az_generate_iter.py`

- [ ] **Step 1: failing test**

```python
# tests/test_az_generate_iter.py
import json, pandas as pd

def test_generate_iter_games_tags_gen_iter_and_quota(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    launched = {}
    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id):
        launched["n"] = n_games; launched["gi"] = gen_iter
        d = out_dir / "run1"; d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner":[0]*n_games}).to_parquet(d/"games.x.parquet")
        (d/"meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name, "gen_iter": gen_iter}))
        return [d]
    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(games_per_iter=1000)
    dirs = daily.generate_iter_games(cfg, iter_dir=tmp_path/"iter_5",
                                     generator=("cand_iter_4", tmp_path/"c.pt"),
                                     gen_iter=5, capped_procs=5, prior_dirs=[])
    assert launched["n"] == 1000 and launched["gi"] == 5
    assert len(dirs) == 1

def test_generate_iter_resumes_only_own_deficit(tmp_path, monkeypatch):
    """Existing games from a DIFFERENT gen_iter do NOT reduce this iter's quota
    (the exact bug). Only this iter's own gen_iter games do."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    # prior: 900 games from gen_iter=3 (should NOT count toward iter-5 quota)
    p = tmp_path / "old"; p.mkdir()
    pd.DataFrame({"seed": range(900), "winner":[0]*900}).to_parquet(p/"games.x.parquet")
    (p/"meta.json").write_text(json.dumps({"rules_id":"v3-full","generator_name":"x","gen_iter":3}))
    asked = {}
    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs, generator_name, gen_iter, rules_id):
        asked["n"] = n_games; return []
    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(games_per_iter=1000)
    daily.generate_iter_games(cfg, iter_dir=tmp_path/"iter_5",
                              generator=("cand_iter_4", tmp_path/"c.pt"),
                              gen_iter=5, capped_procs=5, prior_dirs=[p])
    assert asked["n"] == 1000   # full quota, NOT 100
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** in `daily.py`:
  - Change `_launch_selfplay_procs` signature to take `generator_name, gen_iter,
    rules_id` and write `meta.json {rules_id, generator_name, gen_iter}`; also
    pass arch/vp flags (already there from the prior fix). Dir name stays from
    `--out-root`; the meta carries identity.
  - Replace `generate_fresh` with:

```python
def generate_iter_games(cfg, *, iter_dir, generator, gen_iter, capped_procs,
                        prior_dirs) -> list:
    """Generate exactly cfg.games_per_iter NEW games with `generator` for this
    iteration. Resumable: counts only THIS iteration's own gen_iter games and
    generates the remainder (so a kill resumes toward the quota; other iters'
    games never satisfy it — the 2026-06-14 bug)."""
    from .buffer import own_iter_games
    gen_name, gen_ckpt = generator
    have = own_iter_games(prior_dirs, gen_iter=gen_iter)
    deficit = max(0, cfg.games_per_iter - have)
    if deficit <= 0:
        return []
    dirs = _launch_selfplay_procs(cfg, Path(iter_dir) / "selfplay", Path(gen_ckpt),
                                  deficit, capped_procs, gen_name, gen_iter,
                                  cfg.rules_id)
    produced = sum(__import__("catan_az.buffer", fromlist=["count_games"]).count_games(d) for d in dirs)
    if produced == 0:
        raise RuntimeError(f"self-play produced 0 games for iter {gen_iter} "
                           f"(generator {gen_name}, ckpt {gen_ckpt})")
    return dirs
```

(Keep the cleaner explicit `from .buffer import count_games` at module top rather
than the `__import__` line — use the top-level import.)

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): gen_iter-tagged self-play + own-iter quota (kills the bug)"`

## Task 6: run_cycle rewrite (generator → gen-window → train → arena → reset-publish → terminal)

**Files:** Modify `catan_az/daily.py` (`run_cycle`, `run_day` terminal); Test `tests/test_az_redesign_cycle.py`

- [ ] **Step 1: failing tests** (fakes for stages)

```python
# tests/test_az_redesign_cycle.py
def test_run_cycle_uses_generator_and_gen_window(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    Ladder(tmp_path, champion_checkpoint=str(tmp_path/"c.pt"), champion_name="seed")
    (tmp_path/"c.pt").write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(daily, "select_generator",
                        lambda root, iter_n, champion: seen.setdefault("gen", ("seed", "/c.pt")) or ("seed","/c.pt"))
    monkeypatch.setattr(daily, "generate_iter_games",
                        lambda cfg, **k: seen.setdefault("gi", k["gen_iter"]) or [tmp_path/"sp"])
    monkeypatch.setattr(daily, "_all_selfplay_dirs", lambda r: [])
    monkeypatch.setattr(daily, "gen_window", lambda dirs, **k: seen.setdefault("win_min", k["gen_iter_min"]) or dirs)
    monkeypatch.setattr(daily, "run_iteration", lambda cfg, root, n, *, existing_selfplay_dirs: "hold")
    monkeypatch.setattr(daily, "archive_out_of_window", lambda **k: 0)
    v = daily.run_cycle(AzConfig(), tmp_path, 3, capped_procs=5)
    assert v == "hold" and seen["gi"] == 3
    assert seen["win_min"] == 0   # last_promotion_iter starts at 0

def test_run_day_stops_after_max_iters_without_promotion(tmp_path):
    import csv
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    # journal pre-seeded with 9 holds; one more cycle = 10 -> STOP
    jp = tmp_path/"journal.csv"
    with open(jp,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["iter","verdict"]); w.writeheader()
        for i in range(1,10): w.writerow({"iter":i,"verdict":"hold"})
    calls=[]
    def fake_cycle(cfg, root, n, procs):
        calls.append(n)
        with open(jp,"a",newline="") as f:
            csv.DictWriter(f,fieldnames=["iter","verdict"]).writerow({"iter":n,"verdict":"hold"})
        return "hold"
    run_day(AzConfig(max_iters_per_model=10), loop_root=tmp_path, capped_procs=5,
            cycle_fn=fake_cycle, max_iters=50, next_iter=10)
    assert len(calls) == 1   # hit 10 holds-since-promotion -> stopped
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: rewrite `run_cycle`** in `daily.py` to:

```python
def run_cycle(cfg, loop_root, iter_n, capped_procs) -> str:
    loop_root = Path(loop_root)
    iter_dir = loop_root / f"iter_{iter_n}"
    ladder = Ladder(loop_root)
    champion = (ladder.champion()["name"], ladder.champion()["checkpoint"])
    boundary = ladder.last_promotion_iter()      # window reset point

    generator = select_generator(loop_root, iter_n, champion)
    DailyManifest(iter=iter_n, stage="selfplay", champion=champion[0],
                  fresh_target=cfg.games_per_iter, fresh_done=0,
                  rules_id=cfg.rules_id).save(loop_root)

    prior = _all_selfplay_dirs(loop_root)
    new_dirs = generate_iter_games(cfg, iter_dir=iter_dir, generator=generator,
                                   gen_iter=iter_n, capped_procs=capped_procs,
                                   prior_dirs=prior)

    all_dirs = new_dirs + prior
    window = gen_window(all_dirs, gen_iter_min=boundary, rules_id=cfg.rules_id)

    DailyManifest(iter=iter_n, stage="iterate", champion=champion[0],
                  fresh_target=cfg.games_per_iter, fresh_done=len(new_dirs),
                  rules_id=cfg.rules_id).save(loop_root)

    verdict = run_iteration(cfg, loop_root, iter_n,
                            existing_selfplay_dirs=[str(d) for d in window])

    # on promote, run_iteration's PUBLISH already promoted; set the reset boundary
    if verdict == "promote":
        Ladder(loop_root).promote(f"az_iter_{iter_n}", promoted_at_iter=iter_n)

    archive_out_of_window(window_dirs=window, all_dirs=all_dirs,
                          archive_root=Path(cfg.archive_root), rules_id=cfg.rules_id)
    _append_progress_row(loop_root, iter_n, generator[0], new_dirs, window)
    DailyManifest(iter=iter_n, stage="done", champion=champion[0],
                  fresh_target=cfg.games_per_iter, fresh_done=len(new_dirs),
                  rules_id=cfg.rules_id).save(loop_root)
    return verdict
```

NOTE: `run_iteration`'s existing PUBLISH already calls `ladder.promote(name)`
without `promoted_at_iter`. To set the boundary, the simplest is the extra
`Ladder(loop_root).promote(..., promoted_at_iter=iter_n)` call above — but
`promote` appends history each call. **Fix:** make `promote` idempotent on the
boundary — if `name` is already champion, only update `last_promotion_iter`
(don't re-append history). Add that guard in `ladder.promote`. Add a test:
`test_promote_idempotent_sets_boundary`.

  - Then in `run_day`, replace the `stagnation_holds` stop with a
    **holds-since-last-promotion** terminal:

```python
        holds = _holds_since_promotion(loop_root)
        if holds >= cfg.max_iters_per_model:
            StatusWriter(loop_root).stage(n-1, "max_iters_reached",
                                          MAX_ITERS=True, holds=holds)
            break
```

  with helper `_holds_since_promotion(loop_root)` = count journal rows with
  verdict != "promote" since the last "promote" row (or since start).

- [ ] **Step 4: run, expect PASS** + full `tests/test_az_daily.py` (fix any
  call-site breaks from the `generate_fresh`→`generate_iter_games` rename).
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): run_cycle = generator->gen-window->arena->reset; max-10 terminal"`

## Task 7: progress row gains generator + window_iters

**Files:** Modify `catan_az/progress.py`, `catan_az/daily.py` (`_append_progress_row`); Test `tests/test_az_progress_log.py` (add)

- [ ] **Step 1: failing test**

```python
def test_progress_row_has_generator_and_window_iters(tmp_path):
    from catan_az.progress import append_progress
    append_progress(tmp_path, iter_n=5, champion="seed", generator="cand_iter_4",
                    new_games=1000, window_iters=[3,4,5], window_dirs=3,
                    verdict="hold", winrate=0.6, draw_rate=0.27)
    text = (tmp_path / "PROGRESS.md").read_text()
    assert "cand_iter_4" in text and "1000" in text and "3,4,5" in text
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** — add `generator` + `window_iters` params to
  `append_progress` (header + row), and update `_append_progress_row` in
  `daily.py` to compute `window_iters` from the window dirs' `gen_iter` tags
  (via `_read_meta`) and pass the generator name through.

- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** `git commit -m "feat(az-redesign): PROGRESS.md row gains generator + window_iters"`
- [ ] **Step 6: push** 🚩 **MILESTONE B** (full mechanism)

## Task 8: full suite + micro integration

**Files:** Test `tests/test_az_redesign_integration.py` (slow)

- [ ] **Step 1: run the whole az suite** — `python -m pytest tests/test_az_*.py -q -m "not slow"`; fix any breakage from renames (expected: a few call sites). All green before continuing.

- [ ] **Step 2: write the slow integration test** — a 2-iteration micro run
  (games_per_iter=4, sims=4, arena_games=4, h32/L2 scratch champion, CPU) that
  asserts: iter-1 self-plays with the champion; iter-2 self-plays with iter-1's
  candidate (generator name == `cand_iter_1`); each iter's meta.json carries the
  right `gen_iter`; the window for iter-2 includes both iters (boundary 0).

```python
import pytest
pytestmark = pytest.mark.slow

def test_two_iter_micro_candidate_selfplay(tmp_path):
    import torch, json, glob
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    from catan_az import daily
    from catan_gnn.gnn_model import GnnModel
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (self_play_async --device cuda)")
    champ = tmp_path/"scratch.pt"
    torch.save(GnnModel(hidden_dim=32, num_layers=2).state_dict(), champ)
    Ladder(tmp_path, champion_checkpoint=str(champ), champion_name="scratch")
    cfg = AzConfig(games_per_iter=4, sims=4, n_concurrent=4, max_batch=4,
                   max_epochs=1, arena_games=4, arena_sims=4, arena_min_decisive=1,
                   arena_max_draw_rate=1.0, vp_target=5, bonuses=False,
                   hidden_dim=32, num_layers=2, archive_root=str(tmp_path/"hdd"),
                   max_iters_per_model=10)
    (tmp_path/"hdd").mkdir()
    daily.run_day(cfg, loop_root=tmp_path, capped_procs=1,
                  cycle_fn=daily.run_cycle, max_iters=2)
    metas = [json.loads(open(m).read()) for m in
             glob.glob(str(tmp_path/"iter_*/selfplay/*/meta.json"))]
    gens = {m["generator_name"] for m in metas}
    assert "scratch" in gens                         # iter-1 used champion
    assert any(g.startswith("cand_iter_1") for g in gens)  # iter-2 used candidate
```

- [ ] **Step 3: run** `python -m pytest tests/test_az_redesign_integration.py -q -m slow` — expect PASS.
- [ ] **Step 4: commit** `git commit -m "test(az-redesign): full suite green + 2-iter candidate-selfplay integration"`
- [ ] **Step 5: push** 🚩 **MILESTONE C** (verified)

## Task 9: launch the real redesigned run

- [ ] **Step 1:** fresh loop root (keep the old one as history): the ladder's
  champion stays az_iter_1 (best real net). Start the redesigned daily run:
  `./scripts/run_az_day.sh /home/chitii/catan_data/runs/v3/az_loop` — preflight,
  then iter-N (next number) self-plays with the latest candidate / champion.
- [ ] **Step 2:** confirm the first iteration's `meta.json` has `gen_iter` set and
  PROGRESS.md shows `generator` + `new_games=1000` (NOT 0 — the bug is dead).
- [ ] **Step 3:** journal the launch + update the PR. 🚩 **MILESTONE D** (running).

---

## Self-review

**Spec coverage:** §2 decisions → games_per_iter/arena/bar/max-iters (T1), generator=latest-candidate (T4), gen_iter window + reset (T2/T3/T6), keep-all (no gating code = keep-all by default ✓), terminology generator/champion + gen_iter (T2/T4/T5 meta), 300-arena (T1), max-10 terminal (T6). §3 naming → meta `{generator_name, gen_iter, rules_id}` (T5), gen_iter window (T2). §7 loopholes → quota counts own gen_iter only (T5 test), reset-on-promotion (T3/T6), legacy dirs excluded (T2 test). §8 testing → every task + T8 integration. §9 migration → legacy gen_iter=0 exclusion (T2). All covered.

**Placeholder scan:** none — every code step has full code. (The `__import__` line in T5 is flagged in-text to use the top-level import instead.)

**Type consistency:** `gen_window(dirs, *, gen_iter_min, rules_id)`, `own_iter_games(dirs, *, gen_iter)`, `select_generator(loop_root, iter_n, champion)->(name,ckpt)`, `generate_iter_games(cfg, *, iter_dir, generator, gen_iter, capped_procs, prior_dirs)`, `ladder.promote(name, *, promoted_at_iter=None)`, `ladder.last_promotion_iter()`, `append_progress(..., generator, window_iters, ...)` — consistent across tasks.

**One known follow-up:** `run_iteration`'s PUBLISH already calls `promote(name)`; Task 6 adds an idempotent `promote(..., promoted_at_iter=iter_n)` — the idempotency guard (don't double-append history) is called out in T6 step 3 with its own test. Don't skip it.
