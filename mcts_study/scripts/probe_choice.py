"""Verify numpy Generator.choice(a, p) with replacement uses:
   idx = cumsum(p).searchsorted(next_double(), side='right')   (per draw)
Reproduce with our PCG64 random_f64.
"""
import numpy as np

PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645
MASK128 = (1 << 128) - 1
MASK64 = (1 << 64) - 1


class Pcg64:
    def __init__(self, seed):
        ss = np.random.SeedSequence(seed)
        w = [int(x) for x in ss.generate_state(4, dtype=np.uint64)]
        initstate = (w[0] << 64) | w[1]
        initseq = (w[2] << 64) | w[3]
        self.inc = ((initseq << 1) | 1) & MASK128
        self.state = 0
        self._step(); self.state = (self.state + initstate) & MASK128; self._step()

    def _step(self):
        self.state = (self.state * PCG_MULT + self.inc) & MASK128

    def next_u64(self):
        self._step(); s = self.state
        rot = s >> 122
        xored = ((s >> 64) ^ (s & MASK64)) & MASK64
        return ((xored >> rot) | (xored << ((-rot) & 63))) & MASK64

    def random_f64(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


def my_choice(p, items, n):
    cum = np.cumsum(p)
    cum[-1] = 1.0  # numpy sets the last cdf entry to 1 to avoid fp drift
    out = []
    pc = Pcg64(seed)
    for _ in range(n):
        r = pc.random_f64()
        idx = int(np.searchsorted(cum, r, side="right"))
        out.append(items[idx])
    return out


for seed in (0, 777, 20_000_000):
    items = [0, 2, 5]
    p = [0.2, 0.3, 0.5]
    mine = my_choice(p, items, 8)
    g = np.random.default_rng(seed)
    theirs = [int(g.choice(items, p=p)) for _ in range(8)]
    print(f"seed={seed} choice match={mine == theirs}")
    if mine != theirs:
        print("  mine: ", mine)
        print("  numpy:", theirs)
