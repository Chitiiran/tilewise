"""Arena pausability: chunked plan, PAUSE sentinel stops between chunks, resume
skips done seeds (results.jsonl). Mocks the Rust engine (no GPU)."""
import sys
import types
from pathlib import Path

import pytest

from catan_az import arena as A
from catan_az.config import AzConfig


def _fake_run_arena_games(cand, champ, pairs, sims, *a, **k):
    return [{"seed": int(seed), "rot": int(rot), "winner_seat": 0,
             "winner_role": "cand", "timed_out": False, "vp_margin": 3}
            for rot, seed in pairs]


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    """Hermetic stub: inject a fake catan_mcts_rs + dummy .ts export for the
    duration of each test, then restore (monkeypatch undoes both)."""
    fake = types.ModuleType("catan_mcts_rs")
    fake.run_arena_games = _fake_run_arena_games
    monkeypatch.setitem(sys.modules, "catan_mcts_rs", fake)
    import catan_gnn.export_torchscript as ex
    monkeypatch.setattr(
        ex, "export",
        lambda **k: (Path(k["out_ts"]).write_text("ts"), Path(k["out_ts"]))[1])
    yield


def _cfg(**kw):
    return AzConfig(engine="rust", arena_games=16, arena_sims=4, hidden_dim=32,
                    num_layers=2, vp_target=10, bonuses=True,
                    arena_max_draw_rate=1.0, arena_min_decisive=0,
                    promote_threshold=0.55, **kw)


def test_arena_chunked_completes(tmp_path, monkeypatch):
    # avoid real .ts export
    monkeypatch.setattr(A, "_run_arena_rust", A._run_arena_rust)
    out = tmp_path / "arena"
    res = A._run_arena_rust(
        candidate_ckpt=tmp_path / "c.pt", champion_ckpt=tmp_path / "h.pt",
        cfg=_cfg(arena_chunk_games=4), out_dir=out, seed_base=30_000_000,
        results_path=out.joinpath("results.jsonl") if False else _mk(out),
        done={})
    lines = (out / "results.jsonl").read_text().splitlines()
    assert len(lines) == 16   # all games played


def _mk(out):
    out.mkdir(parents=True, exist_ok=True)
    return out / "results.jsonl"


def test_arena_pause_then_resume(tmp_path):
    out = tmp_path / "arena"; out.mkdir(parents=True)
    rp = out / "results.jsonl"
    (out / "PAUSE").write_text("")
    # M4: a paused arena RAISES (does NOT return a partial verdict).
    with pytest.raises(A.ArenaPaused):
        A._run_arena_rust(candidate_ckpt=tmp_path / "c.pt",
                          champion_ckpt=tmp_path / "h.pt",
                          cfg=_cfg(arena_chunk_games=4), out_dir=out,
                          seed_base=30_000_000, results_path=rp, done={})
    n0 = len(rp.read_text().splitlines()) if rp.exists() else 0
    assert n0 == 0, "PAUSE present from start -> no games played, no verdict"

    (out / "PAUSE").unlink()
    done = A._read_arena_results(rp)
    A._run_arena_rust(candidate_ckpt=tmp_path / "c.pt",
                      champion_ckpt=tmp_path / "h.pt",
                      cfg=_cfg(arena_chunk_games=4), out_dir=out,
                      seed_base=30_000_000, results_path=rp, done=done)
    lines = rp.read_text().splitlines()
    assert len(lines) == 16   # resume completes the plan
    # no duplicate seeds
    import json
    seeds = [json.loads(x)["seed"] for x in lines]
    assert len(seeds) == len(set(seeds))


def test_arena_paused_sentinel(tmp_path):
    assert not A._arena_paused(tmp_path)
    (tmp_path / "PAUSE").write_text("")
    assert A._arena_paused(tmp_path)
