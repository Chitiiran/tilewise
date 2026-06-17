"""Pre-launch hardening fixes from the 2026-06-14 deep inspection.

Each test pins one confirmed faultline that would corrupt data, waste the
6-day run, or produce a wrong verdict. See
docs/superpowers/journals/2026-06-14-az-pipeline-deep-inspection.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---- self-play floor ratio is config-driven (2026-06-17 long-tail timeouts) --

def test_selfplay_floor_ratio_is_config_driven():
    """The M4 floor must come from cfg.selfplay_floor_ratio, not a hardcoded 0.8.

    2026-06-17: iter_7 at deadline=2400 produced 712/994 healthy games (72%) but
    the hardcoded 0.8 floor (800) rejected them. The 282 timed-out games were NOT
    stragglers — they were healthy LONG games (376-566 moves, same range as the
    healthy 170-575) cut by the wall-clock deadline under concurrency contention.
    Lowered the floor to 0.7 so a ~72% healthy yield trains instead of stalling."""
    from catan_az.config import AzConfig
    cfg = AzConfig()
    assert cfg.selfplay_floor_ratio == 0.7


def test_generate_iter_games_floor_uses_config(tmp_path, monkeypatch):
    """generate_iter_games must compute its reject-floor from selfplay_floor_ratio.
    With ratio=0.7 and quota=1000, 712 own-iter games must PASS (>=700), where the
    old 0.8 floor (800) raised RuntimeError."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig

    monkeypatch.setattr(daily, "_launch_selfplay_procs",
                        lambda *a, **k: [tmp_path / "d0"])
    # 712 own-iter games available after the (mocked) launch
    monkeypatch.setattr(daily, "own_iter_games", lambda dirs, gen_iter: 712,
                        raising=False)
    # buffer.own_iter_games is imported inside the function — patch there too
    import catan_az.buffer as buffer
    monkeypatch.setattr(buffer, "own_iter_games", lambda dirs, gen_iter: 712)

    cfg = AzConfig(games_per_iter=1000, selfplay_floor_ratio=0.7)
    # 712 >= int(0.7*1000)=700 -> must NOT raise
    daily.generate_iter_games(cfg, iter_dir=tmp_path, generator=("g", "c.pt"),
                              gen_iter=7, capped_procs=1, prior_dirs=[])

    cfg_strict = AzConfig(games_per_iter=1000, selfplay_floor_ratio=0.8)
    with pytest.raises(RuntimeError):  # 712 < 800
        daily.generate_iter_games(cfg_strict, iter_dir=tmp_path,
                                  generator=("g", "c.pt"), gen_iter=7,
                                  capped_procs=1, prior_dirs=[])


# ---- per-game deadline must clear real all-4-seat self-play (2026-06-16) -----

def test_selfplay_game_deadline_clears_measured_game_cost():
    """The per-game wall-clock deadline must exceed the MEASURED cost of a real
    all-4-seat self-play game at concurrency=24, sims=200.

    2026-06-16: iter_7 came back 994/994 timed_out (0 winners, all VP=0) because
    the deadline was 600s but a no-deadline probe measured these games at
    median 1078s, max 1316s wall-clock (24/24 finished healthy 10VP). iter_6's
    1322-move tail implies an even longer worst case. 600s cut 23/24 healthy
    games. Verified vs iter_6 timestamps (~47s/game throughput x24 ≈ 1100s).
    The deadline must sit above the measured max with headroom so it only ever
    fires on a genuinely stuck game, never a healthy long one."""
    from catan_az.config import AzConfig
    cfg = AzConfig()
    MEASURED_MAX_S = 1316          # observed worst case, 24-game probe 2026-06-16
    assert cfg.selfplay_game_deadline_seconds == 2400.0, (
        "per-game deadline must be 2400s (≈1.8x measured max 1316s)")
    assert cfg.selfplay_game_deadline_seconds > MEASURED_MAX_S, (
        f"deadline {cfg.selfplay_game_deadline_seconds}s must exceed measured "
        f"max {MEASURED_MAX_S}s or it cuts healthy games (the iter_7 failure)")


# ---- self-play worker time-cap is config-driven (2026-06-15 cap-vs-quota) ----

def test_selfplay_worker_max_seconds_is_config_driven(tmp_path, monkeypatch):
    """The per-worker wall-clock cap must come from cfg.selfplay_worker_max_seconds
    (was a hardcoded 21600 that raced the 1000-game quota and tripped M4 at
    731/1000). Both the --max-seconds arg and the p.wait() budget use it."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig

    captured = {"cmds": []}

    class _FakeProc:
        pid = 1
        returncode = 0

        def __init__(self, cmd):
            captured["cmds"].append(cmd)

        def wait(self, timeout=None):
            captured["wait_timeout"] = timeout

    monkeypatch.setattr(daily.subprocess, "Popen", lambda cmd, **k: _FakeProc(cmd))
    cfg = AzConfig(selfplay_worker_max_seconds=32400)
    out = tmp_path / "sp"
    daily._launch_selfplay_procs(cfg, out, tmp_path / "c.pt", n_games=10,
                                 n_procs=1, generator_name="g", gen_iter=6,
                                 rules_id="v3-full")
    cmd = captured["cmds"][0]
    assert "--max-seconds" in cmd
    assert cmd[cmd.index("--max-seconds") + 1] == "32400"
    # p.wait budget must exceed the worker cap (so we don't kill a healthy worker)
    assert captured["wait_timeout"] > 32400


# ---- resume must offset seed-base so deficit games are FRESH boards ----------

def test_deficit_resume_offsets_seed_base(tmp_path, monkeypatch):
    """2026-06-15 resume bug: deficit-resume re-launched workers from the SAME
    seed-base into NEW dirs, so they replayed the already-done boards
    (duplicates). The deficit launch must offset each worker's seed-base past the
    games already produced, so the remaining games use fresh boards."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig

    cmds = []

    class _FakeProc:
        pid = 1
        returncode = 0

        def __init__(self, cmd):
            cmds.append(cmd)

        def wait(self, timeout=None):
            pass

    monkeypatch.setattr(daily.subprocess, "Popen", lambda cmd, **k: _FakeProc(cmd))
    cfg = AzConfig()
    out = tmp_path / "sp"
    # 731 games already produced -> resume offset
    daily._launch_selfplay_procs(cfg, out, tmp_path / "c.pt", n_games=269,
                                 n_procs=7, generator_name="g", gen_iter=6,
                                 rules_id="v3-full", seed_offset=731)
    bases = [int(c[c.index("--seed-base") + 1]) for c in cmds]
    # worker 0's base = base(gen6,0) + 731 -> past its done seeds, fresh boards
    assert bases[0] == daily._worker_seed_base(gen_iter=6, i=0) + 731
    assert bases[1] == daily._worker_seed_base(gen_iter=6, i=1) + 731
    # offset stays well within each worker's 1M block (no cross-worker collision)
    assert all(b % 1_000_000 < 100_000 for b in bases)


# ---- M1: seed-base must vary per iteration ----------------------------------

def test_selfplay_seed_base_varies_by_iteration():
    """M1 (HIGH): the seed-base must include the gen_iter term, else every
    iteration replays the identical board set and the run can only memorise a
    fixed ~1000 boards — scientifically worthless. daily.py:117 had
    `sb = 31_000_000 + i * 1_000_000` with no iter term."""
    from catan_az.daily import _worker_seed_base
    # same worker index, different iterations -> different seed bases
    assert _worker_seed_base(gen_iter=1, i=0) != _worker_seed_base(gen_iter=2, i=0)
    # different workers within one iter -> different (already true)
    assert _worker_seed_base(gen_iter=1, i=0) != _worker_seed_base(gen_iter=1, i=1)
    # no collision across a realistic grid (10 iters x 7 workers x 1000 games)
    seen = set()
    for it in range(1, 11):
        for i in range(7):
            base = _worker_seed_base(gen_iter=it, i=i)
            block = range(base, base + 1000)
            assert seen.isdisjoint(block), f"seed collision iter={it} worker={i}"
            seen.update(block)


# ---- M2: record_arena must be idempotent per candidate ----------------------

def test_record_arena_idempotent_per_candidate(tmp_path):
    """M2 (HIGH): a crash after record_arena._save() but before PUBLISH.done
    re-runs the whole PUBLISH block on resume, double-applying the Elo delta and
    games count. record_arena must be idempotent keyed by the (iter-unique)
    candidate name."""
    from catan_az.ladder import Ladder
    L = Ladder(tmp_path, champion_checkpoint="/c.pt", champion_name="champ")
    L.register_candidate("az_iter_1", "/cand.pt", created_iter=1)
    L.record_arena("az_iter_1", "champ", wins_a=70, wins_b=45, draws=5)
    elo_after_first = L.entry("az_iter_1")["elo"]
    games_after_first = L.entry("az_iter_1")["games"]
    # resume re-applies the SAME arena -> must be a no-op
    L.record_arena("az_iter_1", "champ", wins_a=70, wins_b=45, draws=5)
    assert L.entry("az_iter_1")["elo"] == elo_after_first
    assert L.entry("az_iter_1")["games"] == games_after_first


def test_record_arena_idempotent_survives_reload(tmp_path):
    """The applied-arena guard must persist to disk (the crash/resume case
    constructs a fresh Ladder from ladder.json)."""
    from catan_az.ladder import Ladder
    L = Ladder(tmp_path, champion_checkpoint="/c.pt", champion_name="champ")
    L.register_candidate("az_iter_1", "/cand.pt", created_iter=1)
    L.record_arena("az_iter_1", "champ", wins_a=70, wins_b=45, draws=5)
    elo = L.entry("az_iter_1")["elo"]
    # fresh Ladder (simulates resume after crash) re-applying same arena
    L2 = Ladder(tmp_path)
    L2.record_arena("az_iter_1", "champ", wins_a=70, wins_b=45, draws=5)
    assert L2.entry("az_iter_1")["elo"] == elo


# ---- M3: arena results.jsonl resume must survive a torn final line ----------

def test_arena_resume_skips_torn_jsonl_line(tmp_path):
    """M3 (HIGH): a crash mid-write leaves a half-written final line in
    results.jsonl; the resume reader must skip it (mirror analytics.py) instead
    of raising JSONDecodeError and poison-pilling the iteration forever."""
    from catan_az.arena import _read_arena_results
    p = tmp_path / "results.jsonl"
    p.write_text(
        json.dumps({"seed": 1, "winner_role": "cand", "timed_out": False}) + "\n"
        + json.dumps({"seed": 2, "winner_role": "champ", "timed_out": True}) + "\n"
        + '{"seed": 3, "winner_role":'  # torn final line (crash mid-write)
    )
    done = _read_arena_results(p)
    assert set(done.keys()) == {1, 2}      # torn line skipped, not fatal
    assert done[2]["timed_out"] is True


# ---- M4: partial-worker death must fail loud, not train on a fraction -------

def test_generate_iter_games_raises_on_partial_production(tmp_path, monkeypatch):
    """M4 (HIGH): if self-play produces well under the quota (a CUDA-OOM race
    killed some of the 7 workers), generate_iter_games must RAISE — not let a
    30/1000 iteration silently train on a fraction of the data."""
    import json as _json

    import pandas as pd

    import catan_az.daily as daily
    from catan_az.config import AzConfig

    def fake_launch_partial(cfg, out_dir, checkpoint, n_games, n_procs,
                            generator_name, gen_iter, rules_id, seed_offset=0):
        d = out_dir / "run1"
        d.mkdir(parents=True)
        # only 100 of the requested ~1000 games came back (workers died)
        pd.DataFrame({"seed": range(100), "winner": [0] * 100}
                     ).to_parquet(d / "games.x.parquet")
        (d / "meta.json").write_text(_json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch_partial)
    cfg = AzConfig(games_per_iter=1000)
    with pytest.raises(RuntimeError, match="partial worker death"):
        daily.generate_iter_games(cfg, iter_dir=tmp_path / "iter_5",
                                  generator=("cand_iter_4", tmp_path / "c.pt"),
                                  gen_iter=5, capped_procs=5, prior_dirs=[])


# ---- count_games must not double-count shards after a crash -----------------

def test_count_games_dedups_seed_shards(tmp_path):
    """LOW but cheap: count_games globs games*.parquet; after a crash both
    per-seed shards and a _remainder/compacted shard can coexist, double-counting
    the same seed. Dedup by seed and drop phantom winner=-1 rows."""
    import pandas as pd
    from catan_az.buffer import count_games
    d = tmp_path
    # same seed in two shards (per-seed + compacted) + a phantom aborted game
    pd.DataFrame({"seed": [1, 2], "winner": [0, 1]}).to_parquet(d / "games.abc.parquet")
    pd.DataFrame({"seed": [2, 3], "winner": [1, -1]}).to_parquet(d / "games.seed=2.parquet")
    # unique real games: seeds {1,2,3} minus phantom seed-3 winner=-1 -> {1,2} = 2
    assert count_games(d) == 2
