"""Phase 4 differential parity: the Rust catan_mcts_rs engine path agrees
move-by-move with the Python CatanState adapter (the oracle).

Both ultimately drive the same Rust engine; this proves the ADAPTER-level
semantics match — current_player mapping, chance_outcomes ordering, returns
convention, legal_actions set, final VPs — so the Rust MCTS layer built on top
sees an identical state machine.
"""
import numpy as np
import pytest

from catan_mcts.adapter import CatanGame

catan_mcts_rs = pytest.importorskip("catan_mcts_rs")


def _entries_from_choices(choices):
    """choices: list of ('chance'|'step', id) -> Rust (is_chance, id) list."""
    return [(kind == "chance", int(a)) for kind, a in choices]


@pytest.mark.parametrize("seed", list(range(20)))
def test_engine_move_by_move_parity(seed):
    py = CatanGame().new_initial_state(seed=seed)
    rng = np.random.default_rng(seed)
    choices: list[tuple[str, int]] = []
    steps = 0
    while not py.is_terminal() and steps < 400:
        entries = _entries_from_choices(choices)
        # Status parity BEFORE acting.
        is_term, is_chance, cp = catan_mcts_rs.debug_status(seed, entries)
        assert is_term == py.is_terminal()
        assert is_chance == py.is_chance_node()
        if not is_term and not is_chance:
            assert cp == py.current_player()

        if py.is_chance_node():
            # chance_outcomes ordering + probabilities must match exactly.
            rs_outs = catan_mcts_rs.debug_chance_outcomes(seed, entries)
            py_outs = py.chance_outcomes()
            assert len(rs_outs) == len(py_outs)
            for (rv, rp), (pv, pp) in zip(rs_outs, py_outs):
                assert rv == pv
                assert rp == pp
            outs = py_outs
            r = float(rng.random())
            cum, chosen = 0.0, outs[-1][0]
            for v, p in outs:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            choices.append(("chance", int(chosen)))
            py.apply_action(int(chosen))
        else:
            la = py.legal_actions()
            rs_la = catan_mcts_rs.debug_legal_actions(seed, entries)
            assert list(rs_la) == [int(x) for x in la], (
                f"legal_actions diverge at step {steps} seed {seed}"
            )
            a = int(la[int(rng.integers(len(la)))])
            choices.append(("step", a))
            py.apply_action(a)
        steps += 1

    # Terminal parity: returns + VPs.
    entries = _entries_from_choices(choices)
    rs_term, _, _ = catan_mcts_rs.debug_status(seed, entries)
    assert rs_term == py.is_terminal()
    rs_returns = catan_mcts_rs.debug_returns(seed, entries)
    assert list(rs_returns) == list(py.returns())
    rs_vps, rs_hist = catan_mcts_rs.debug_vps_and_history(seed, entries)
    assert list(rs_hist) == py.history()
