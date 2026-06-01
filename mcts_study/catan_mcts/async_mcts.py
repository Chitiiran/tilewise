# catan_mcts/async_mcts.py
"""Minimal async MCTS for batched self-play.

PUCT/UCB over a CatanState tree. The ONLY await is the leaf evaluation, which
suspends the coroutine so other games' leaves can batch. Matches the OpenSpiel
MCTSBot semantics we rely on: uct_c=1.4, priors from the policy head, value
from the value head (or state.returns() at terminals), 4-player per-seat backup,
argmax-visit final move. See spec 2026-05-30-batched-gnn-evaluator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from catan_mcts import ACTION_SPACE_SIZE


class Node:
    __slots__ = ("state", "to_play", "is_expanded", "children", "prior",
                 "visit_count", "value_sum")

    def __init__(self, state, prior: float = 0.0) -> None:
        self.state = state
        self.to_play = state.current_player() if not state.is_terminal() else -1
        self.is_expanded = False
        self.children: dict = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0


class AsyncMcts:
    def __init__(self, evaluator, c: float = 1.4, rng=None,
                 dirichlet_alpha: float = 0.8, dirichlet_eps: float = 0.0) -> None:
        self.ev = evaluator
        self.c = float(c)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.last_root_value = 0.0
        # AlphaZero root exploration: mix Dirichlet(alpha) noise into the root
        # child priors with weight eps. eps=0 (default) disables it, so arena/
        # eval search stays deterministic; self-play passes eps>0 to explore.
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)

    def _apply_root_noise(self, root: "Node") -> None:
        """Mix Dirichlet noise into the root children's priors (AlphaZero):
        p' = (1-eps)*p + eps*eta,  eta ~ Dirichlet(alpha). No-op if eps==0."""
        if self.dirichlet_eps <= 0.0 or not root.children:
            return
        actions = list(root.children.keys())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_eps
        for a, eta in zip(actions, noise):
            child = root.children[a]
            child.prior = (1.0 - eps) * child.prior + eps * float(eta)

    def _ucb_score(self, parent: "Node", child: "Node") -> float:
        q = (child.value_sum / child.visit_count) if child.visit_count else 0.0
        u = self.c * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return q + u

    def _select_child(self, node: "Node"):
        best_score, best_a, best_child = -float("inf"), None, None
        for a, child in node.children.items():
            s = self._ucb_score(node, child)
            if s > best_score:
                best_score, best_a, best_child = s, a, child
        return best_a, best_child

    async def _expand_and_evaluate(self, node: "Node"):
        # Resolve any run of chance nodes iteratively (Catan's setup phase can
        # chain many), sampling each via the per-game rng. No GPU calls here.
        while node.state.is_chance_node():
            outcomes = node.state.chance_outcomes()
            r = float(self.rng.random())
            cum, chosen = 0.0, outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            nxt = node.state.clone()
            nxt.apply_action(int(chosen))
            node.state = nxt
            node.to_play = nxt.current_player() if not nxt.is_terminal() else -1
        state = node.state
        value, priors = await self.ev.eval_leaf(state)
        if priors is None:
            # Terminal leaf: state.returns() is ALREADY absolute-seat-indexed.
            return np.asarray(value, dtype=np.float32)
        # Non-terminal GNN leaf: value is EGO-relative (value[offset] is the
        # value for player (leaf_mover + offset) % 4). Rotate to absolute-seat
        # order so backup can index by node.to_play uniformly.
        leaf_mover = state.current_player()
        value = np.asarray(value, dtype=np.float32)
        value_abs = np.empty(4, dtype=np.float32)
        for offset in range(4):
            value_abs[(leaf_mover + offset) % 4] = value[offset]
        for a, p in priors:
            child_state = state.clone()
            child_state.apply_action(int(a))
            node.children[a] = Node(child_state, prior=p)
        node.is_expanded = True
        return value_abs

    def _backup(self, path, value_vec: np.ndarray) -> None:
        for node in path:
            node.visit_count += 1
            if node.to_play >= 0:
                node.value_sum += float(value_vec[node.to_play])

    async def search(self, root_state, n_sims: int) -> np.ndarray:
        root = Node(root_state.clone())
        root_value = await self._expand_and_evaluate(root)
        self._apply_root_noise(root)  # AlphaZero exploration (no-op if eps==0)
        root.visit_count += 1
        if root.to_play >= 0:
            root.value_sum += float(root_value[root.to_play])
        for _ in range(n_sims - 1):
            node, path = root, [root]
            while node.is_expanded and node.children and not node.state.is_terminal():
                _, node = self._select_child(node)
                path.append(node)
            value_vec = await self._expand_and_evaluate(node)
            self._backup(path, value_vec)
        self.last_root_value = (root.value_sum / root.visit_count) if root.visit_count else 0.0
        out = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        for a, child in root.children.items():
            out[a] = child.visit_count
        return out

    def best_action(self, visit_counts: np.ndarray) -> int:
        return int(np.argmax(visit_counts))


def temperature_sample(visit_counts: np.ndarray, tau: float, rng) -> int:
    """AlphaZero move selection from visit counts.

    tau -> 0  : deterministic argmax (strong play).
    tau == 1  : sample proportional to visit counts.
    general   : sample proportional to N(a)^(1/tau) over visited actions.
    """
    if tau <= 1e-6:
        return int(np.argmax(visit_counts))
    visited = np.nonzero(visit_counts)[0]
    if visited.size == 0:
        return int(np.argmax(visit_counts))
    counts = visit_counts[visited].astype(np.float64)
    weights = counts ** (1.0 / tau)
    probs = weights / weights.sum()
    return int(rng.choice(visited, p=probs))


@dataclass
class RecordedMove:
    current_player: int
    move_index: int
    legal_mask: np.ndarray
    visit_counts: np.ndarray
    action_taken: int
    root_value: float


@dataclass
class GameResult:
    seed: int
    terminal: bool
    winner: int
    final_vp: list[int]
    length_in_moves: int
    action_history: list[int]
    moves: list = field(default_factory=list)  # list[RecordedMove]


async def play_one_async_game(*, game, seed: int, evaluator, n_sims: int,
                              rng, max_steps: int = 200_000,  # ~20x observed game length; timeout guard
                              self_play: bool = False,
                              dirichlet_alpha: float = 0.8,
                              dirichlet_eps: float = 0.25,
                              temp_moves: int = 30):
    # AlphaZero exploration is ON only for self_play (data generation): Dirichlet
    # root noise + temperature-sampled move choice (tau=1 for the first
    # temp_moves PER-PLAYER decisions, then tau->0 argmax). Arena/eval keep the
    # default deterministic greedy play (self_play=False -> eps=0, argmax).
    eps = dirichlet_eps if self_play else 0.0
    state = game.new_initial_state(seed=seed)
    mcts = AsyncMcts(evaluator=evaluator, c=1.4, rng=rng,
                     dirichlet_alpha=dirichlet_alpha, dirichlet_eps=eps)
    moves: list = []
    # PER-PLAYER move_index (NOT global): the dataset replay (catan_gnn.dataset)
    # counts multi-legal decisions for the recorded current_player and matches
    # move_index, so each seat's decisions must be numbered 0,1,2,... independently.
    move_index_by_player = [0, 0, 0, 0]
    steps = 0
    while not state.is_terminal() and steps < max_steps:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = float(rng.random())
            cum, chosen = 0.0, outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
            steps += 1
            continue
        legal = state.legal_actions()
        if len(legal) == 1:
            state.apply_action(int(legal[0]))
            steps += 1
            continue
        visit_counts = await mcts.search(state, n_sims=n_sims)
        cp = int(state.current_player())
        if self_play:
            # tau=1 (sample) for this player's first temp_moves decisions,
            # then tau=0 (argmax) to sharpen the endgame.
            tau = 1.0 if move_index_by_player[cp] < temp_moves else 0.0
            action = temperature_sample(visit_counts, tau=tau, rng=rng)
        else:
            action = mcts.best_action(visit_counts)
        legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        legal_mask[np.asarray(legal, dtype=np.int64)] = 1
        moves.append(RecordedMove(
            current_player=cp, move_index=move_index_by_player[cp],
            legal_mask=legal_mask, visit_counts=visit_counts,
            action_taken=int(action), root_value=float(mcts.last_root_value)))
        state.apply_action(int(action))
        move_index_by_player[cp] += 1
        steps += 1
    terminal = state.is_terminal()
    if terminal:
        rets = state.returns()
        winner = int(np.argmax(rets)) if max(rets) > 0 else -1
    else:
        winner = -1
    final_vp = [0, 0, 0, 0]
    try:
        stats = state._engine.stats()
        final_vp = [int(stats["players"][i]["vp_final"]) for i in range(4)]
    except Exception as exc:
        import warnings
        warnings.warn(f"stats() vp_final unavailable for seed={seed}: {exc}; "
                      f"final_vp will be zeros", stacklevel=2)
    return GameResult(seed=seed, terminal=terminal, winner=winner,
                      final_vp=final_vp,
                      length_in_moves=steps,  # total engine steps (incl. chance/forced) — matches recorder schema
                      action_history=state.history(), moves=moves)
