"""Verify the full PCG64 pipeline in pure Python: init -> next_u64 -> random().

Confirmed mapping: initstate=(w0<<64)|w1, initseq=(w2<<64)|w3, srandom, then
XSL-RR-128 output. Reproduces numpy default_rng(seed).random() exactly.
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
        self._step()
        self.state = (self.state + initstate) & MASK128
        self._step()

    def _step(self):
        self.state = (self.state * PCG_MULT + self.inc) & MASK128

    def next_u64(self):
        # XSL-RR 128 -> 64: rot = state >> 122; hi^lo, rotr by rot.
        self._step()
        s = self.state
        rot = s >> 122
        xored = ((s >> 64) ^ (s & MASK64)) & MASK64
        return ((xored >> rot) | (xored << ((-rot) & 63))) & MASK64

    def random_f64(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


for seed in (0, 777, 20_000_000):
    p = Pcg64(seed)
    mine = [p.random_f64() for _ in range(5)]
    g = np.random.default_rng(seed)
    theirs = [float(g.random()) for _ in range(5)]
    ok = all(a == b for a, b in zip(mine, theirs))
    print(f"seed={seed} random() match={ok}")
    if not ok:
        print("  mine:  ", mine)
        print("  numpy: ", theirs)
