"""ValueQGnnBot — 1-ply value-Q deployment of the GNN (no tree).

Diagnosis (docs/superpowers/journals/2026-06-01-puregnn-plateau-diagnosis.md):
PureGnn's argmax-of-policy collapses the soft visit-count distribution to one
move, discarding the value information that distinguishes Catan's frequently
near-equal moves. D1 showed the VALUE head fits well while the policy head
plateaus. ValueQGnnBot exploits that: for each legal action it applies the move
on a cloned state, evaluates the child's value head, and picks the action that
maximises the CURRENT MOVER's value. One batched eval per legal child, no tree.

The GNN value head is ego-relative (value[0] is the leaf mover's value). After
applying action a, the child has its own mover; the parent mover's value in the
child's frame is child_value[(parent_mover - child_mover) % 4]. Terminal children
return absolute-seat returns(), so we index [parent_mover] directly.
"""
from __future__ import annotations

import numpy as np


class ValueQGnnBot:
    """Greedy 1-ply value-Q player over the GNN value head.

    Reuses the same BatchedGnnEvaluator the GnnMcts player uses, so it benefits
    from leaf batching across concurrent tournament games.
    """

    def __init__(self, evaluator) -> None:
        self.ev = evaluator

    async def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("ValueQGnnBot: no legal actions in non-terminal state")
        if len(legal) == 1:
            return int(legal[0])

        parent_mover = state.current_player()
        best_action, best_value = None, -np.inf
        for a in legal:
            child = state.clone()
            child.apply_action(int(a))
            if child.is_terminal():
                val = np.asarray(child.returns(), dtype=np.float32)
                mover_value = float(val[parent_mover])  # returns() is absolute-seat
            else:
                child_mover = child.current_player()
                ego, _ = await self.ev.eval_leaf(child)
                ego = np.asarray(ego, dtype=np.float32)
                mover_value = float(ego[(parent_mover - child_mover) % 4])
            if mover_value > best_value:
                best_value, best_action = mover_value, int(a)
        return best_action
