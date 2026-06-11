"""Elo ladder + champion registry (spec §2 PUBLISH)."""
from __future__ import annotations

import json

import pytest


@pytest.fixture()
def ladder(tmp_path):
    from catan_az.ladder import Ladder
    return Ladder(tmp_path, champion_checkpoint="/ckpts/cell6.pt",
                  champion_name="cell6")


def test_fresh_ladder_seeds_champion_at_1000(ladder):
    assert ladder.champion()["name"] == "cell6"
    assert ladder.champion()["elo"] == 1000.0
    assert ladder.champion()["checkpoint"] == "/ckpts/cell6.pt"


def test_arena_result_moves_ratings_symmetrically(ladder):
    ladder.register_candidate("iter1", "/ckpts/iter1.pt", created_iter=1)
    before_champ = ladder.entry("cell6")["elo"]
    before_cand = ladder.entry("iter1")["elo"]
    ladder.record_arena("iter1", "cell6", wins_a=79, wins_b=41, draws=0)
    after_champ = ladder.entry("cell6")["elo"]
    after_cand = ladder.entry("iter1")["elo"]
    gain = after_cand - before_cand
    loss = before_champ - after_champ
    assert gain > 0
    assert gain == pytest.approx(loss)   # zero-sum update


def test_promote_flips_champion_and_keeps_history(ladder):
    ladder.register_candidate("iter1", "/ckpts/iter1.pt", created_iter=1)
    ladder.promote("iter1")
    assert ladder.champion()["name"] == "iter1"
    # Old champion remains on the ladder (humans can still play it).
    assert ladder.entry("cell6") is not None
    assert len(ladder.history()) == 1
    assert ladder.history()[0]["promoted"] == "iter1"


def test_persistence_round_trip(tmp_path):
    from catan_az.ladder import Ladder
    l1 = Ladder(tmp_path, champion_checkpoint="/c.pt", champion_name="seed")
    l1.register_candidate("x", "/x.pt", created_iter=2)
    l1.record_arena("x", "seed", wins_a=70, wins_b=50, draws=0)
    l2 = Ladder(tmp_path)   # reload from disk; no re-seeding
    assert l2.entry("x")["elo"] == l1.entry("x")["elo"]
    assert l2.champion()["name"] == "seed"


def test_atomic_write_leaves_old_file_on_partial_write(tmp_path):
    from catan_az.ladder import Ladder
    l1 = Ladder(tmp_path, champion_checkpoint="/c.pt", champion_name="seed")
    good = (tmp_path / "ladder.json").read_text()
    # Simulate a crash mid-write: tmp file exists, rename never happened.
    (tmp_path / "ladder.json.tmp").write_text('{"corrupt": tru')
    l2 = Ladder(tmp_path)
    assert l2.champion()["name"] == "seed"
    assert json.loads(good)["champion"] == "seed"
