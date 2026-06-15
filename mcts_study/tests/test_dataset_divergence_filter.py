"""Build-time filter dropping un-replayable positions (recorder<->replay
divergence, 2026-06-14). For some self-play games the stored action_history
reproduces FEWER gated decisions for a player than the recorder logged move
rows (a determinism divergence between self-play generation and replay, most
likely chance/robber handling). Those move rows can never be replayed, so the
dataset must drop them at build time instead of crashing training on them.

Path A (2026-06-14): exclude the ~0.05% divergent positions; root-cause the
divergence (or move to full-observation recording) in a later cycle.
"""
from __future__ import annotations


def test_replayable_decision_counts_counts_per_player():
    """The helper replays a game once and returns, per player, how many gated
    decisions the action_history actually reproduces."""
    from catan_gnn.dataset import replayable_decision_counts
    from catan_bot import _engine
    # A real short game: play random to terminal, recording the history the
    # recorder WOULD store, then confirm the helper reproduces the same counts.
    import random
    e = _engine.Engine(7)
    rng = random.Random(7)
    history = []
    gated = {0: 0, 1: 0, 2: 0, 3: 0}
    steps = 0
    while not e.is_terminal() and steps < 50000:
        if e.is_chance_pending():
            outs = e.chance_outcomes()
            v = int(outs[0][0])
            history.append(v | 0x8000_0000)
            e.apply_chance_outcome(v)
            steps += 1
            continue
        legal = e.legal_actions()
        cp = int(e.current_player())
        a = rng.choice(legal)
        if len(legal) > 1:
            gated[cp] += 1
        history.append(int(a))
        e.step(int(a))
        steps += 1
    counts = replayable_decision_counts(seed=7, history=history)
    # The helper, replaying the SAME history deterministically, must reproduce
    # exactly the gated counts we observed during generation.
    assert counts == gated


def test_filter_drops_out_of_range_move_rows():
    """A move row whose move_index >= the replayable count for its player is
    dropped; in-range rows are kept."""
    from catan_gnn.dataset import _drop_divergent_rows
    import pandas as pd
    # game seed=1: player 0 replayably has 3 decisions (mi 0,1,2 valid; 3+ not)
    moves = pd.DataFrame({
        "seed": [1, 1, 1, 1, 1],
        "current_player": [0, 0, 0, 0, 1],
        "move_index": [0, 1, 2, 3, 0],   # p0 mi=3 is out of range -> dropped
    })
    counts_by_seed = {1: {0: 3, 1: 1, 2: 0, 3: 0}}
    kept = _drop_divergent_rows(moves, counts_by_seed)
    # p0 rows mi 0,1,2 kept; mi 3 dropped; p1 mi 0 kept => 4 rows
    assert len(kept) == 4
    assert not ((kept["current_player"] == 0) & (kept["move_index"] == 3)).any()
