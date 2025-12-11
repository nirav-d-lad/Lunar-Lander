from __future__ import annotations

from RL import Algorithm
from x import state, state_interactions

import numpy as np

from numpy.typing import NDArray

from tqdm import tqdm

import gymnasium as gym

class LSTD(Algorithm):

    def run(self, episodes:int=500) -> NDArray:

        env = gym.make(self.env_name)

        k = len(self.Φ(self.S.low, 0))

        w = np.zeros(k, dtype=np.float128)

        B = self.Δ * np.eye(k)
        b = np.zeros(k)
        
        for m in tqdm(range(episodes), bar_format='{l_bar}{bar:20}{r_bar}'):
            
            terminated = False
            truncated = False
            s,_ = env.reset(seed=self.rng.train.next_seed())
            
            while not(terminated or truncated):

                a = self.π(s, w)

                sP, r, terminated, truncated, _ = env.step(a)

                φ = self.Φ(s,a)
                φP = self.Φ(sP, self.π(sP,w)) * (1-terminated)

                u = B @ φ
                v = (φ - self.γ*φP).T @ B
                
                B = B - np.outer(u, v) / (1 + v @ φ)
                b = b + r*φ

                s = sP

            wP = B @ b.T     
            wP = np.asarray(wP).ravel()
            w = (1-self.α(m))*w + self.α(m)*wP 


            mean, hw = self.evaluate_policy(w,m, rng=self.rng.test)
            score = mean - hw 
            tqdm.write(f"Episode{m:3d}:\t{score:.2f}\t({mean:3.2f} ± {hw:3.2f})")

        env.close()

if __name__ == "__main__":

    algo = LSTD(env_name="LunarLander-v3", x=state_interactions, er=(0,0,0), lr=(1,0,0), Δ=1e-7)
    algo.run(episodes=200)
    algo.plot()
    w = algo.best_policy()
    algo.record(w,"LSTD")