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
    def __init__(self, evaluator, c: float = 1.4, rng=None) -> None:
        self.ev = evaluator
        self.c = float(c)
        self.rng = rng if rng is not None else np.random.default_rng(0)

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
        out = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        for a, child in root.children.items():
            out[a] = child.visit_count
        return out

    def best_action(self, visit_counts: np.ndarray) -> int:
        return int(np.argmax(visit_counts))


from dataclasses import dataclass, field


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
    final_vp: list
    length_in_moves: int
    action_history: list
    moves: list = field(default_factory=list)


async def play_one_async_game(*, game, seed: int, evaluator, n_sims: int,
                              rng, max_steps: int = 200000):
    state = game.new_initial_state(seed=seed)
    mcts = AsyncMcts(evaluator=evaluator, c=1.4, rng=rng)
    moves: list = []
    move_index = 0
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
        action = mcts.best_action(visit_counts)
        legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        legal_mask[np.asarray(legal, dtype=np.int64)] = 1
        moves.append(RecordedMove(
            current_player=int(state.current_player()), move_index=move_index,
            legal_mask=legal_mask, visit_counts=visit_counts,
            action_taken=int(action), root_value=0.0))
        state.apply_action(int(action))
        move_index += 1
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
        final_vp = [int(x) for x in stats.get("final_vp", final_vp)]
    except Exception:
        pass
    return GameResult(seed=seed, terminal=terminal, winner=winner,
                      final_vp=final_vp, length_in_moves=steps,
                      action_history=state.history(), moves=moves)
