"""Dump NumPy RNG golden values for the Rust replica parity tests.

Covers the three consumers the MCTS oracle uses (random / dirichlet / choice),
the PCG64 seeding via SeedSequence, AND the stdlib Mersenne `random.Random`
(used by the arena game-level chance fast-path).

Prints JSON to stdout; the Rust tests embed these values. Run in the venv.
"""
from __future__ import annotations

import json
import random as pyrandom

import numpy as np

SEEDS = [0, 1, 777, 20_000_000]


def dump():
    out = {"pcg64": {}, "mt19937": {}}
    for seed in SEEDS:
        ss = np.random.SeedSequence(seed)
        state_words = [int(x) for x in ss.generate_state(4, dtype=np.uint64)]
        g = np.random.default_rng(seed)
        bg = g.bit_generator.state["state"]

        g_r = np.random.default_rng(seed)
        randoms = [float(g_r.random()) for _ in range(8)]

        g_d = np.random.default_rng(seed)
        dirichlet = [float(x) for x in g_d.dirichlet([0.8, 0.8, 0.8, 0.8])]

        g_g = np.random.default_rng(seed)
        gamma = [float(x) for x in g_g.standard_gamma(0.8, size=6)]

        g_n = np.random.default_rng(seed)
        normal = [float(x) for x in g_n.standard_normal(6)]

        g_c = np.random.default_rng(seed)
        choice = [int(g_c.choice([0, 2, 5], p=[0.2, 0.3, 0.5])) for _ in range(8)]

        out["pcg64"][str(seed)] = {
            "seedseq_state4": state_words,
            "pcg64_state": int(bg["state"]),
            "pcg64_inc": int(bg["inc"]),
            "random_f64": randoms,
            "dirichlet_0p8_x4": dirichlet,
            "standard_gamma_0p8_x6": gamma,
            "standard_normal_x6": normal,
            "choice_025_p235_x8": choice,
        }

        # stdlib Mersenne (arena game chance fast-path uses random.Random(seed)).
        r = pyrandom.Random(seed)
        out["mt19937"][str(seed)] = {
            "random_x8": [r.random() for _ in range(8)],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    dump()
