from RL import Algorithm, fmt

import gymnasium as gym

import numpy as np

from tqdm import tqdm

from x import state, state_interactions

class SARSATD0(Algorithm):

    def run(self, episodes:int = 100):

        env = gym.make(self.env_name)

        k = len(self.Φ(self.S.low, 0))

        w = np.zeros(k, dtype=np.float128)
    
        for m in tqdm(np.arange(episodes), bar_format=fmt.bar, desc="TRAIN"):

            terminated = False
            truncated = False
            s,_ = env.reset(seed=self.rng.train.next_seed())
            
            a = self.π(s, w)

            while not(terminated or truncated):
                
                sP, r, terminated, truncated, _ = env.step(a)
                
                aP = self.greedy(sP, w, self.ε(m), self.rng.train)
            
                Qgrad = self.Φ(s, a)
                Qgrad_norm = Qgrad / np.sqrt(sum(Qgrad**2))
                
                Q = np.dot(w, self.Φ(s, a))
                Qp = np.dot(w, self.Φ(sP, aP)) * (1 - terminated)

                δ = r + self.γ*Qp - Q

                w =  w + self.α(m) * δ * Qgrad_norm 
                
                s = sP
                a = aP 

            mean, hw = self.evaluate_policy(w, m, rng=self.rng.test)
            
            score = mean - hw 
            tqdm.write(f"Episode{m:3d}:\t{score:.2f}\t({mean:3.2f} ± {hw:3.2f})")

        env.close()

if __name__ == "__main__":

    algo = SARSATD0(env_name="LunarLander-v3", x=state_interactions, er=(0, 0, 0), lr=(0.1, 0.1, 0), γ=0.99, λ=0)
    algo.run(episodes=200)
    w = algo.best_policy()
    algo.plot()
    algo.record(w,"SARSATD0")