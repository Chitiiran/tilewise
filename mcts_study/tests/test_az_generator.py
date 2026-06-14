"""Generator selection: latest candidate / champion fallback (spec §4 step 1)."""
from __future__ import annotations


def test_generator_iter1_is_champion(tmp_path):
    from catan_az.daily import select_generator
    name, ckpt = select_generator(tmp_path, iter_n=1,
                                  champion=("seed", "/seed.pt"))
    assert name == "seed" and ckpt == "/seed.pt"


def test_generator_iter_n_is_latest_candidate(tmp_path):
    from catan_az.daily import select_generator
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
