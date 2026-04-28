"""Quick measurement: how long does the Python 'common.play_one_game' loop take
with all-random bots, no MCTS, and the realistic chance-sampler? Compare against
the Rust-only legal[0] bench (~28us/game)."""
import time
import random

from catan_mcts.adapter import CatanGame
from catan_mcts.experiments.common import play_one_game


class _RandomBot:
    def __init__(self, seed):
        self._rng = random.Random(seed)
    def step(self, state):
        return self._rng.choice(state.legal_actions())


game = CatanGame()
results = []
for seed in [42, 99, 123, 7]:
    t0 = time.perf_counter()
    bots = {i: _RandomBot(seed + i + 1) for i in range(4)}
    outcome = play_one_game(
        game=game,
        bots=bots,
        seed=seed,
        chance_rng=random.Random(seed),
        recorded_player=None,
        recorder_game=None,
    )
    elapsed = time.perf_counter() - t0
    results.append((seed, outcome.length_in_moves, elapsed))
    print(f"seed={seed}: {outcome.length_in_moves} steps, {elapsed*1000:.1f} ms, winner={outcome.winner}")

# Aggregate
total_steps = sum(r[1] for r in results)
total_time = sum(r[2] for r in results)
print(f"\nTotal: {total_steps} steps in {total_time*1000:.1f} ms = {total_steps/total_time:.0f} steps/sec")
print(f"Per-step cost: {total_time/total_steps*1e6:.1f} us")
print(f"Per-game cost (avg): {total_time/len(results)*1000:.0f} ms")
