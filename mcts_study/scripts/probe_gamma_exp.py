"""Validate a faithful pure-Python port of numpy's standard_exponential
(ziggurat), standard_gamma, and dirichlet against numpy, using the EXACT
ziggurat tables fetched into scripts/ziggurat_constants.h.

Once this matches numpy bit-for-bit, the same logic + tables port to Rust.
"""
import math
import re
from pathlib import Path

import numpy as np

HDR = Path(__file__).resolve().parent / "ziggurat_constants.h"

PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645
MASK128 = (1 << 128) - 1
MASK64 = (1 << 64) - 1

# numpy distributions.c ziggurat constants (normal + exp R/inv_r).
ZIGGURAT_NOR_R = 3.6541528853610087963519472518
ZIGGURAT_NOR_INV_R = 0.27366123732975827203338247596
ZIGGURAT_EXP_R = 7.6971174701310497140434110269


def parse_array(text, name, is_hex):
    m = re.search(rf"{name}\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    body = m.group(1)
    toks = [t.strip() for t in body.split(",") if t.strip()]
    out = []
    for t in toks:
        t = t.rstrip("UL").rstrip("L").rstrip("U")
        if is_hex:
            out.append(int(t, 16))
        else:
            out.append(float(t))
    return out


text = HDR.read_text()
ki = parse_array(text, "ki_double", True)
wi = parse_array(text, "wi_double", False)
fi = parse_array(text, "fi_double", False)
ke = parse_array(text, "ke_double", True)
we = parse_array(text, "we_double", False)
fe = parse_array(text, "fe_double", False)
assert all(len(a) == 256 for a in (ki, wi, fi, ke, we, fe)), [len(a) for a in (ki, wi, fi, ke, we, fe)]


class Pcg64:
    def __init__(self, seed):
        w = [int(x) for x in np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64)]
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

    def next_double(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)


def log1p(x):
    return math.log1p(x)


def standard_exponential(bg):
    while True:
        ri = bg.next_u64()
        ri >>= 3
        idx = ri & 0xFF
        ri >>= 8
        x = ri * we[idx]
        if ri < ke[idx]:
            return x
        # standard_exponential_unlikely
        if idx == 0:
            return ZIGGURAT_EXP_R - log1p(-bg.next_double())
        if (fe[idx - 1] - fe[idx]) * bg.next_double() + fe[idx] < math.exp(-x):
            return x
        # else loop


def standard_normal(bg):
    while True:
        r = bg.next_u64()
        idx = r & 0xFF
        r >>= 8
        sign = r & 0x1
        rabs = (r >> 1) & 0x000FFFFFFFFFFFFF
        x = rabs * wi[idx]
        if sign & 0x1:
            x = -x
        if rabs < ki[idx]:
            return x
        if idx == 0:
            while True:
                xx = -ZIGGURAT_NOR_INV_R * log1p(-bg.next_double())
                yy = -log1p(-bg.next_double())
                if yy + yy > xx * xx:
                    return -(ZIGGURAT_NOR_R + xx) if ((rabs >> 8) & 0x1) else (ZIGGURAT_NOR_R + xx)
        else:
            if (fi[idx - 1] - fi[idx]) * bg.next_double() + fi[idx] < math.exp(-0.5 * x * x):
                return x


def standard_gamma(bg, shape):
    if shape == 1.0:
        return standard_exponential(bg)
    if shape == 0.0:
        return 0.0
    if shape < 1.0:
        while True:
            U = bg.next_double()
            V = standard_exponential(bg)
            if U <= 1.0 - shape:
                X = U ** (1.0 / shape)
                if X <= V:
                    return X
            else:
                Y = -math.log((1 - U) / shape)
                X = (1.0 - shape + shape * Y) ** (1.0 / shape)
                if X <= (V + Y):
                    return X
    b = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9 * b)
    while True:
        while True:
            X = standard_normal(bg)
            V = 1.0 + c * X
            if V > 0.0:
                break
        V = V * V * V
        U = bg.next_double()
        if U < 1.0 - 0.0331 * (X * X) * (X * X):
            return b * V
        if math.log(U) < 0.5 * X * X + b * (1.0 - V + math.log(V)):
            return b * V


def dirichlet(bg, alpha):
    # numpy "standard case": acc = sequential sum of gammas, invacc = 1/acc,
    # then each value *= invacc (multiply, NOT divide — differs by 1 ULP).
    # (alpha.max() >= 0.1 so the stick-breaking small-alpha path is not taken.)
    y = [standard_gamma(bg, a) for a in alpha]
    acc = 0.0
    for v in y:
        acc = acc + v
    invacc = 1.0 / acc
    return [v * invacc for v in y]


def bits(x):
    return x  # exact compare via ==


ok_all = True
for seed in (0, 777, 20_000_000):
    bg = Pcg64(seed)
    mine_e = [standard_exponential(bg) for _ in range(5)]
    theirs_e = [float(x) for x in np.random.default_rng(seed).standard_exponential(5)]
    me = all(a == b for a, b in zip(mine_e, theirs_e))

    bg = Pcg64(seed)
    mine_g = [standard_gamma(bg, 0.8) for _ in range(6)]
    theirs_g = [float(x) for x in np.random.default_rng(seed).standard_gamma(0.8, size=6)]
    mg = all(a == b for a, b in zip(mine_g, theirs_g))

    bg = Pcg64(seed)
    mine_n = [standard_normal(bg) for _ in range(6)]
    theirs_n = [float(x) for x in np.random.default_rng(seed).standard_normal(6)]
    mn = all(a == b for a, b in zip(mine_n, theirs_n))

    bg = Pcg64(seed)
    mine_d = dirichlet(bg, [0.8, 0.8, 0.8, 0.8])
    theirs_d = [float(x) for x in np.random.default_rng(seed).dirichlet([0.8] * 4)]
    md = all(a == b for a, b in zip(mine_d, theirs_d))

    print(f"seed={seed} exp={me} gamma={mg} normal={mn} dirichlet={md}")
    if not (me and mg and mn and md):
        ok_all = False
        if not md:
            print("  mine_d :", mine_d)
            print("  numpy_d:", theirs_d)

print("ALL MATCH" if ok_all else "MISMATCH — fix before Rust port")
