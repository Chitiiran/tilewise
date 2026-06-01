# tests/test_async_mcts.py
import numpy as np
import torch
from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator
from catan_mcts.async_mcts import AsyncMcts
from catan_mcts import ACTION_SPACE_SIZE


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def _leaf_state(seed=42):
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=seed)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    return state


async def test_search_returns_visit_counts():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(0))
        state = _leaf_state()
        visits = await mcts.search(state, n_sims=16)
        assert visits.shape == (ACTION_SPACE_SIZE,)
        assert 0 < int(visits.sum()) <= 15
        legal = set(state.legal_actions())
        assert all(visits[a] == 0 for a in range(ACTION_SPACE_SIZE) if a not in legal)
    finally:
        await ev.stop()


def test_dirichlet_noise_perturbs_root_priors():
    # With dirichlet_eps>0, root child priors are mixed with Dirichlet noise:
    #   p' = (1-eps)*p + eps*eta.  They should differ from the originals, still
    #   sum to ~1, and stay non-negative.
    from catan_mcts.async_mcts import Node
    mcts = AsyncMcts(evaluator=None, c=1.4, rng=np.random.default_rng(0),
                     dirichlet_alpha=0.8, dirichlet_eps=0.25)
    # Build a fake root with 4 children carrying priors.
    root = Node.__new__(Node)
    root.children = {}
    orig = {10: 0.1, 20: 0.2, 30: 0.3, 40: 0.4}
    for a, p in orig.items():
        ch = Node.__new__(ch_cls := Node)
        ch.prior = p
        root.children[a] = ch
    mcts._apply_root_noise(root)
    new = {a: root.children[a].prior for a in orig}
    # priors changed
    assert any(abs(new[a] - orig[a]) > 1e-6 for a in orig), "noise did not perturb priors"
    # still a valid distribution
    assert abs(sum(new.values()) - 1.0) < 1e-5
    assert all(p >= 0.0 for p in new.values())


def test_dirichlet_disabled_by_default_leaves_priors_unchanged():
    from catan_mcts.async_mcts import Node
    mcts = AsyncMcts(evaluator=None, c=1.4, rng=np.random.default_rng(0))
    assert mcts.dirichlet_eps == 0.0  # off by default (arena/eval stays deterministic)
    root = Node.__new__(Node)
    root.children = {}
    orig = {10: 0.25, 20: 0.25, 30: 0.5}
    for a, p in orig.items():
        ch = Node.__new__(Node)
        ch.prior = p
        root.children[a] = ch
    mcts._apply_root_noise(root)
    for a, p in orig.items():
        assert root.children[a].prior == p, "default (eps=0) must not change priors"


async def test_self_play_flag_enables_exploration():
    # play_one_async_game(self_play=True) must enable Dirichlet noise on the
    # MCTS (eps>0). Default (self_play=False) keeps it deterministic (eps==0).
    import catan_mcts.async_mcts as azm
    captured = {}
    orig_cls = azm.AsyncMcts
    class SpyMcts(orig_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["eps"] = self.dirichlet_eps
    azm.AsyncMcts = SpyMcts
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        # self_play=True -> eps>0
        await azm.play_one_async_game(
            game=game, seed=5, evaluator=ev, n_sims=4,
            rng=np.random.default_rng(5), max_steps=40, self_play=True)
        assert captured["eps"] > 0.0, "self_play=True did not enable Dirichlet noise"
        # default -> eps==0
        captured.clear()
        await azm.play_one_async_game(
            game=game, seed=6, evaluator=ev, n_sims=4,
            rng=np.random.default_rng(6), max_steps=40)
        assert captured["eps"] == 0.0, "default self_play must be deterministic (eps=0)"
    finally:
        azm.AsyncMcts = orig_cls
        await ev.stop()


def test_temperature_sample_tau_zero_is_argmax():
    from catan_mcts.async_mcts import temperature_sample
    visits = np.zeros(280, dtype=np.int32)
    visits[10] = 3
    visits[20] = 7   # max
    visits[30] = 5
    rng = np.random.default_rng(0)
    # tau=0 -> deterministic argmax, regardless of rng
    for _ in range(5):
        assert temperature_sample(visits, tau=0.0, rng=rng) == 20


def test_temperature_sample_tau_one_samples_per_visit_distribution():
    from catan_mcts.async_mcts import temperature_sample
    visits = np.zeros(280, dtype=np.int32)
    visits[10] = 10
    visits[20] = 90   # should be picked ~90% of the time at tau=1
    rng = np.random.default_rng(0)
    picks = [temperature_sample(visits, tau=1.0, rng=rng) for _ in range(400)]
    frac20 = sum(1 for p in picks if p == 20) / len(picks)
    assert 0.80 < frac20 < 1.0, f"tau=1 sampling off: action20 picked {frac20:.2f}"
    # both actions appear (it's sampling, not argmax)
    assert any(p == 10 for p in picks), "tau=1 never explored the minority action"


async def test_search_applies_root_noise_when_enabled():
    # search() must call _apply_root_noise after expanding the root, so with
    # eps>0 the root child priors are perturbed before the sim loop runs.
    from catan_mcts.async_mcts import Node
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(0),
                         dirichlet_alpha=0.8, dirichlet_eps=0.25)
        applied = {"n": 0}
        orig = mcts._apply_root_noise
        def spy(root):
            applied["n"] += 1
            return orig(root)
        mcts._apply_root_noise = spy
        await mcts.search(_leaf_state(), n_sims=8)
        assert applied["n"] == 1, "search did not apply root noise exactly once"
    finally:
        await ev.stop()


async def test_value_rotated_to_absolute_seat():
    # The GNN value head is ego-relative; _expand_and_evaluate must rotate it to
    # absolute-seat order so backup indexes by node.to_play correctly.
    import numpy as np
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(0))
        state = _leaf_state()
        leaf_mover = state.current_player()
        # Get the raw ego-relative value the evaluator produces for this leaf.
        ego_value, _ = await ev.eval_leaf(state)
        ego_value = np.asarray(ego_value, dtype=np.float32)
        # Drive _expand_and_evaluate on a fresh node for the same state.
        from catan_mcts.async_mcts import Node
        node = Node(state.clone())
        value_abs = await mcts._expand_and_evaluate(node)
        # value_abs[absolute_seat] must equal ego_value[(absolute_seat - leaf_mover) % 4]
        for seat in range(4):
            offset = (seat - leaf_mover) % 4
            assert abs(value_abs[seat] - ego_value[offset]) < 1e-5, (
                f"seat {seat}: abs={value_abs[seat]} != ego[{offset}]={ego_value[offset]}")
    finally:
        await ev.stop()


async def test_play_full_game_terminates_and_records():
    from catan_mcts.async_mcts import play_one_async_game
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        result = await play_one_async_game(
            game=game, seed=123, evaluator=ev, n_sims=8,
            rng=np.random.default_rng(123), max_steps=200000)
        assert result.terminal is True
        assert -1 <= result.winner <= 3
        assert result.length_in_moves > 0
        assert len(result.moves) > 0
        m = result.moves[0]
        assert m.visit_counts.shape == (ACTION_SPACE_SIZE,)
        assert m.legal_mask.shape == (ACTION_SPACE_SIZE,)
        # final_vp must populate for a finished game (not stay [0,0,0,0]).
        assert len(result.final_vp) == 4
        assert sum(result.final_vp) > 0, f"final_vp not populated: {result.final_vp}"
        # the winner should have the most VP (>= vp_target if a real win)
        if result.winner >= 0:
            assert result.final_vp[result.winner] == max(result.final_vp)
    finally:
        await ev.stop()


async def test_per_game_rng_reproducible():
    from catan_mcts.async_mcts import play_one_async_game
    async def run():
        ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                                 max_batch=4, window_ms=5)
        ev.start()
        try:
            res = await play_one_async_game(
                game=CatanGame(vp_target=10, bonuses=True), seed=999,
                evaluator=ev, n_sims=8, rng=np.random.default_rng(999),
                max_steps=200000)
            return res.action_history
        finally:
            await ev.stop()
    h1 = await run()
    h2 = await run()
    assert h1 == h2, "same seed produced different play"
