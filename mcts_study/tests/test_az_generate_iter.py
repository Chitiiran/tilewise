"""gen_iter-tagged self-play + own-iter quota (spec §4 step 2)."""
from __future__ import annotations

import json

import pandas as pd
import pytest


def test_generate_iter_games_tags_gen_iter_and_quota(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    launched = {}

    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id, seed_offset=0):
        launched["n"] = n_games
        launched["gi"] = gen_iter
        d = out_dir / "run1"
        d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner": [0] * n_games}
                     ).to_parquet(d / "games.x.parquet")
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(games_per_iter=1000)
    dirs = daily.generate_iter_games(cfg, iter_dir=tmp_path / "iter_5",
                                     generator=("cand_iter_4", tmp_path / "c.pt"),
                                     gen_iter=5, capped_procs=5, prior_dirs=[])
    assert launched["n"] == 1000 and launched["gi"] == 5
    assert len(dirs) == 1


def test_generate_iter_resumes_only_own_deficit(tmp_path, monkeypatch):
    """Existing games from a DIFFERENT gen_iter do NOT reduce this iter's quota
    (the exact bug). Only this iter's own gen_iter games do."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    p = tmp_path / "old"
    p.mkdir()
    pd.DataFrame({"seed": range(900), "winner": [0] * 900}).to_parquet(p / "games.x.parquet")
    (p / "meta.json").write_text(json.dumps(
        {"rules_id": "v3-full", "generator_name": "x", "gen_iter": 3}))
    asked = {}

    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id, seed_offset=0):
        asked["n"] = n_games
        d = out_dir / "r"
        d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner": [0] * n_games}).to_parquet(d / "games.g.parquet")
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name, "gen_iter": gen_iter}))
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(games_per_iter=1000)
    daily.generate_iter_games(cfg, iter_dir=tmp_path / "iter_5",
                              generator=("cand_iter_4", tmp_path / "c.pt"),
                              gen_iter=5, capped_procs=5, prior_dirs=[p])
    assert asked["n"] == 1000   # full quota, NOT 100 (old gen_iter=3 ignored)


# ---- Task 7 (final review): production data-quality gate ---------------------
#
# `run_cycle` always calls `run_iteration(..., existing_selfplay_dirs=[...
# window])` — the exact branch loop.py's `_check_data_quality` EXEMPTS (that
# exemption is correct for operator-salvage). The production self-play
# completion point is actually HERE, in generate_iter_games, after the
# own-iter floor check passes. These tests pin the gate at that point.

def _healthy_launch(n_games_field="n_games"):
    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id, seed_offset=0):
        d = out_dir / "run_healthy"
        d.mkdir(parents=True)
        pd.DataFrame({
            "seed": range(n_games),
            "winner": [i % 4 for i in range(n_games)],
            "length_in_moves": [200 + i for i in range(n_games)],
            "timed_out": [False] * n_games,
        }).to_parquet(d / "games.x.parquet")
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
        return [d]
    return fake_launch


def _degenerate_launch():
    """Own-iter yield is high enough to clear the floor check (count_games()
    drops winner==-1 rows, so a PURE all-timeout batch would trip the floor
    first, not the data-quality gate) but the TIMEOUT RATE among all produced
    rows exceeds the degeneracy threshold — the case this gate exists for."""
    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id, seed_offset=0):
        d = out_dir / "run_degenerate"
        d.mkdir(parents=True)
        n_decisive = int(n_games * 0.75)   # 75% decisive, 25% timeouts: clears
        n_timeout = n_games - n_decisive   # the 70% floor, exceeds the 20% gate
        df = pd.DataFrame({
            "seed": range(n_games),
            "winner": [i % 4 for i in range(n_decisive)] + [-1] * n_timeout,
            "length_in_moves": [200 + i for i in range(n_decisive)]
                              + [999_999] * n_timeout,
            "timed_out": [False] * n_decisive + [True] * n_timeout,
        })
        df.to_parquet(d / "games.x.parquet")
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
        return [d]
    return fake_launch


def test_generate_iter_games_writes_data_quality_json_for_healthy_dirs(
        tmp_path, monkeypatch):
    """Healthy own-iter self-play -> data_quality.json is written in the
    iteration dir and generate_iter_games does NOT raise."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig

    monkeypatch.setattr(daily, "_launch_selfplay_procs", _healthy_launch())
    cfg = AzConfig(games_per_iter=100)
    iter_dir = tmp_path / "iter_9"
    dirs = daily.generate_iter_games(cfg, iter_dir=iter_dir,
                                     generator=("cand_iter_8", tmp_path / "c.pt"),
                                     gen_iter=9, capped_procs=1, prior_dirs=[])
    assert len(dirs) == 1
    dq_path = iter_dir / "data_quality.json"
    assert dq_path.exists()
    dq = json.loads(dq_path.read_text())
    assert dq["verdict"] == "ok"
    assert dq["games"] == 100
    assert dq["length_p50"] > 0 and dq["length_p90"] > 0 and dq["length_max"] > 0


def test_generate_iter_games_raises_on_degenerate_dirs(tmp_path, monkeypatch):
    """All-timeout own-iter self-play -> RuntimeError, but data_quality.json is
    STILL written (diagnostic even on refusal, matching loop.py's convention)."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig

    monkeypatch.setattr(daily, "_launch_selfplay_procs", _degenerate_launch())
    cfg = AzConfig(games_per_iter=100)
    iter_dir = tmp_path / "iter_10"
    with pytest.raises(RuntimeError, match="degenerate"):
        daily.generate_iter_games(cfg, iter_dir=iter_dir,
                                  generator=("cand_iter_9", tmp_path / "c.pt"),
                                  gen_iter=10, capped_procs=1, prior_dirs=[])
    dq_path = iter_dir / "data_quality.json"
    assert dq_path.exists()
    dq = json.loads(dq_path.read_text())
    assert dq["verdict"] == "degenerate"
    assert dq["timeouts"] == 25
    assert dq["games"] == 100


def test_generate_iter_games_gate_ignores_prior_iters_dirs(tmp_path, monkeypatch):
    """The gate covers only THIS iteration's NEW own-iter dirs, not prior_dirs
    (which already passed the gate in their own iteration) — a degenerate
    prior dir must not poison this iteration's fresh, healthy verdict, and
    vice versa a healthy prior dir must not mask this iteration's own
    degenerate data."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig

    # A prior iteration's dir was degenerate (already gated/handled in its own
    # iteration) — must NOT affect this iteration's data_quality.json.
    prior = tmp_path / "old_degenerate"
    prior.mkdir()
    pd.DataFrame({
        "seed": range(50), "winner": [-1] * 50,
        "length_in_moves": [999_999] * 50, "timed_out": [True] * 50,
    }).to_parquet(prior / "games.x.parquet")
    (prior / "meta.json").write_text(json.dumps(
        {"rules_id": "v3-full", "generator_name": "old", "gen_iter": 3}))

    monkeypatch.setattr(daily, "_launch_selfplay_procs", _healthy_launch())
    cfg = AzConfig(games_per_iter=100)
    iter_dir = tmp_path / "iter_11"
    daily.generate_iter_games(cfg, iter_dir=iter_dir,
                              generator=("cand_iter_10", tmp_path / "c.pt"),
                              gen_iter=11, capped_procs=1, prior_dirs=[prior])
    dq = json.loads((iter_dir / "data_quality.json").read_text())
    assert dq["verdict"] == "ok"
    assert dq["games"] == 100   # only this iter's own new dir, not prior's 50
