"""Pre-launch hardening fixes from the 2026-06-14 deep inspection.

Each test pins one confirmed faultline that would corrupt data, waste the
6-day run, or produce a wrong verdict. See
docs/superpowers/journals/2026-06-14-az-pipeline-deep-inspection.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


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
                            generator_name, gen_iter, rules_id):
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
