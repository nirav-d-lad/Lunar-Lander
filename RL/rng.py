from __future__ import annotations

import numpy as np

class RNG:
    def __init__(self, ss:np.random.SeedSequence):
        self.ss = ss
        self.rng = np.random.default_rng(self.ss)

    def next_seed(self):
        child = self.ss.spawn(1)[0]
        seed = child.generate_state(1, dtype=np.uint32)[0]
        return int(seed)

class AlgoRNG:

    def __init__(self, seed:int):
        self.seed = seed

        master = np.random.SeedSequence(seed)
        children = master.spawn(4)

        self.train = RNG(children[0])
        self.test =  RNG(children[1])
        self.eval = RNG(children[2])
        self.record = RNG(children[3])