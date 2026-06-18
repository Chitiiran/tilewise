"""Reproduce numpy SeedSequence.generate_state(n, uint64), exact port."""
import numpy as np

MASK32 = (1 << 32) - 1
XSHIFT = 16
MULT_A = 0x931e8875
MULT_B = 0x58f38ded
MIX_MULT_L = 0xca01f9dd
MIX_MULT_R = 0x4973f715
INIT_A = 0x43b0d7e5
INIT_B = 0x8b51f9dd
POOL_SIZE = 4
DEFAULT_POOL_SIZE = 4


def _int_to_u32_words(n: int):
    if n == 0:
        return [0]
    words = []
    while n > 0:
        words.append(n & MASK32)
        n >>= 32
    return words


class SeedSeq:
    def __init__(self, seed):
        # assembled_entropy = get_assembled_entropy():
        #   run_entropy (the seed words) + spawn_key (empty).
        # If spawn_key present numpy pads run_entropy to pool_size first; with
        # empty spawn_key, assembled = run_entropy as-is.
        self.entropy = _int_to_u32_words(seed)
        self.pool = [0] * POOL_SIZE
        self._mix_entropy(self.pool, self.entropy)

    @staticmethod
    def _hashmix(value, hash_const):
        value = (value ^ hash_const) & MASK32
        hash_const = (hash_const * MULT_A) & MASK32
        value = (value * hash_const) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value, hash_const

    @staticmethod
    def _mix(x, y):
        result = ((x * MIX_MULT_L) - (y * MIX_MULT_R)) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    def _mix_entropy(self, mixer, entropy):
        hash_const = INIT_A
        for i in range(len(mixer)):
            if i < len(entropy):
                v = entropy[i]
            else:
                v = 0
            mixer[i], hash_const = self._hashmix(v, hash_const)
        # Mix all bits together so late bits can affect earlier bits.
        for i_src in range(len(mixer)):
            for i_dst in range(len(mixer)):
                if i_src != i_dst:
                    h, hash_const = self._hashmix(mixer[i_src], hash_const)
                    mixer[i_dst] = self._mix(mixer[i_dst], h)
        # Add any remaining entropy, mixing each new word with each pool word.
        for i_src in range(len(mixer), len(entropy)):
            for i_dst in range(len(mixer)):
                h, hash_const = self._hashmix(entropy[i_src], hash_const)
                mixer[i_dst] = self._mix(mixer[i_dst], h)

    def generate_state(self, n_words):
        hash_const = INIT_B
        out = []
        src_cycle = 0
        for _ in range(n_words):
            data_val = self.pool[src_cycle]
            data_val = (data_val ^ hash_const) & MASK32
            hash_const = (hash_const * MULT_B) & MASK32
            data_val = (data_val * hash_const) & MASK32
            data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
            out.append(data_val)
            src_cycle = (src_cycle + 1) % len(self.pool)
        return out


def state4_u64(seed):
    w32 = SeedSeq(seed).generate_state(8)
    return [w32[i] | (w32[i + 1] << 32) for i in range(0, 8, 2)]


for seed in (0, 1, 777, 20_000_000):
    mine = state4_u64(seed)
    theirs = [int(x) for x in np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64)]
    print(f"seed={seed} match={mine == theirs}")
    if mine != theirs:
        print("  mine: ", mine)
        print("  numpy:", theirs)
