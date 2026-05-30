# catan_mcts/batched_evaluator.py
"""Async GNN evaluator that batches MCTS leaf evals across concurrent games.

Each eval() call parks a Future on a pending queue and suspends. A background
batcher coroutine drains the queue and runs ONE forward pass per batch, then
resolves all the parked Futures. See spec 2026-05-30-batched-gnn-evaluator.
"""
from __future__ import annotations

import asyncio
import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class BatchedGnnEvaluator:
    def __init__(self, model: GnnModel, device: str = "cpu",
                 max_batch: int = 64, window_ms: float = 5.0) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch = int(max_batch)
        self.window_s = float(window_ms) / 1000.0
        self._pending: list[tuple] = []   # (features, future)
        self._wakeup: asyncio.Event | None = None
        self._batcher_task: asyncio.Task | None = None
        self._stopped = False
        # active_game_count is set by the orchestrator each step; default huge
        # so the all-parked flush clause never fires spuriously in unit tests.
        self.active_game_count = 10 ** 9
        # Stats (health metric).
        self.total_batches = 0
        self.total_requests = 0

    def start(self) -> None:
        self._wakeup = asyncio.Event()
        self._batcher_task = asyncio.create_task(self._batcher_loop(), name="batcher")

    async def stop(self) -> None:
        self._stopped = True
        if self._wakeup is not None:
            self._wakeup.set()
        if self._batcher_task is not None:
            await self._batcher_task

    @torch.no_grad()
    def _run_forward(self, features_list):
        batch = Batch.from_data_list(features_list).to(self.device)
        v, logits = self.model(batch)
        v_np = v.cpu().numpy().astype(np.float32)
        l_np = logits.cpu().numpy().astype(np.float32)
        return v_np, l_np

    async def eval(self, state) -> tuple[np.ndarray, np.ndarray]:
        # Features built on the caller side (cheap, CPU).
        obs = state._engine.observation()
        features = state_to_pyg(obs)
        fut = asyncio.get_running_loop().create_future()
        self._pending.append((features, fut))
        self.total_requests += 1
        if self._wakeup is not None:
            self._wakeup.set()
        return await fut

    async def _batcher_loop(self):
        loop = asyncio.get_running_loop()
        while not self._stopped:
            if not self._pending:
                await self._wakeup.wait()
                self._wakeup.clear()
                continue
            # Decide whether to flush now or wait for more requests.
            first_arrival = loop.time()
            while not self._stopped:
                n = len(self._pending)
                flush_now = (
                    n >= self.max_batch
                    or n >= self.active_game_count
                )
                if flush_now:
                    break
                elapsed = loop.time() - first_arrival
                if elapsed >= self.window_s:
                    break  # window fired -> flush partial
                # Sleep a short slice to let more requests arrive.
                try:
                    await asyncio.wait_for(self._wakeup.wait(),
                                           timeout=self.window_s - elapsed)
                except asyncio.TimeoutError:
                    pass
                self._wakeup.clear()
            if not self._pending:
                continue
            drained = self._pending[: self.max_batch]
            self._pending = self._pending[self.max_batch :]
            feats = [f for f, _ in drained]
            v_np, l_np = self._run_forward(feats)
            self.total_batches += 1
            for i, (_, fut) in enumerate(drained):
                if not fut.done():
                    fut.set_result((v_np[i], l_np[i]))
