from __future__ import annotations
import shutil
import pandas as pd

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from tqdm import tqdm

import pickle
import matplotlib.ticker as ticker


import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from moviepy import (CompositeVideoClip, TextClip, VideoFileClip,
                     concatenate_videoclips)
from numpy.typing import NDArray

from RL import calc, fmt 
from RL.rng import AlgoRNG, RNG

@dataclass
class Action:

    cardinality:int
    actions:NDArray


@dataclass
class State:

    cardinality:int
    low:NDArray
    high:NDArray

    def __post_init__(self):
        self.range:NDArray = self.high - self.low

class Algorithm:

    def __init__(
            self, 
            env_name:str,
            x:Callable,
            γ:float = 0.999,
            Δ:float = 1e-6,
            λ:float = 0.5,
            lr:tuple[float,float,float] = (0.5,0.5,0),
            er:tuple[float,float,float] = (0.5,0.5,0),
            seed:int = 42) -> None:

        self.env_name:str = env_name
        self.x = x

        env = gym.make(self.env_name)

        self.A = Action(cardinality=env.action_space.n,
                        actions=np.arange(env.action_space.n))
        
        self.S = State(cardinality=len(env.observation_space.low), 
                       low=env.observation_space.low, 
                       high=env.observation_space.high)
        env.close()

        self.γ:float = γ
        self.λ = λ
        self.Δ = Δ
        self.lr:tuple[float,float,float] = lr
        self.er:tuple[float,float,float] = er
        
        self.rng:AlgoRNG = AlgoRNG(seed)

        self.policies:list[dict] = []

    def α(self, n:int) -> float:
        α_init, α_decay, α_min = self.lr
        return max(α_init / (n + 1)**α_decay, α_min)

    def ε(self, n:int) -> float:
        ε_init, ε_decay, ε_min = self.er
        return max(ε_init / (n + 1)**ε_decay, ε_min)

    def Φ(self, s:NDArray, a:int) -> NDArray:
        return np.hstack([self.x(s) * (a == b) for b in self.A.actions])

    def π(self, s:NDArray, w:NDArray) -> int:
        return np.argmax([np.dot(self.Φ(s, a), w) for a in self.A.actions]) 
    
    def greedy(self, s:NDArray, w:NDArray, p:float, g:RNG) -> int:
        if g.rng.random() < p:
            a = g.rng.choice(self.A.actions)
        else:
            a= self.π(s, w)

        return a
    
    def evaluate_policy(self, w:NDArray, m:int | None = None, episodes:int = 10, rng:RNG | None = None) -> tuple[float,float]:
        
        if rng is None:
            rng = self.rng.eval

        env = gym.make(self.env_name)
        
        G = []

        for _ in np.arange(episodes):
            
            terminated = False
            truncated = False
            Gm = 0
            s,_ = env.reset(seed = rng.next_seed())

            while not( terminated or truncated ):
                a = self.π(s, w)
                sP, r, terminated, truncated,_ = env.step(a)
                s = sP
                Gm += r

            G.append(Gm)

        env.close()
        mean, hw = calc.confidence_interval(G)

        score = mean - hw 

        if m is not None:
            self.policies.append({'episode':m, 'score': score, 'mean': mean, 'halfwidth': hw, 'w': w.copy()})

        return mean, hw
        
    def best_policy(self) -> NDArray:

        policies = pd.DataFrame(self.policies)
        policies.sort_values(by='score', ascending=False, inplace=True)
        
        best_score = -np.inf
        print(f"EVALUATING TOP 10 POLICES")
        print("\t\tScore\t(mean ± hw)")
        for i, p in policies.head(10).iterrows():
            
            mean, hw = self.evaluate_policy(p['w'], episodes=30)
            score = mean - hw
            
            print(f"Episode {i:3d}\t{score:.1f}\t({mean:.1f} ± {hw:.1f})")
            
            if best_score < score:
                best_score = score
                w_star = p['w']
                m_star = p['episode']

        print(f"SUPERLATIVE POLICY:\tEPISODE {m_star}")

        os.makedirs("policy", exist_ok=True)

        lr = "_".join(f"{x:.3f}" for x in self.lr)
        er = "_".join(f"{x:.3f}" for x in self.er)
        
        fn = f"{self.__class__.__name__}SCORE({score:0.1f})GAMMA({self.γ:.3f}LR({lr})ER({er}).pkl"

        with open(os.path.join("policy", fn), "wb") as f:
            pickle.dump(w_star, f)

        return w_star

    def record(self, w:NDArray, title:str, episodes:int=10) -> None:

        warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")
        os.makedirs("tmp", exist_ok=True)
    
        env = gym.make(self.env_name, render_mode="rgb_array")
        env = RecordVideo(env, video_folder="tmp", episode_trigger=lambda ep: True)

        G = []

        # Loop through episodes
        for i in tqdm(range(episodes), bar_format=fmt.bar, desc="REC EP"):
            

            Gm = 0
            s,_ = env.reset(seed=self.rng.record.next_seed())

            terminated = False
            truncated = False
            while not( terminated or truncated ):
                a = self.π(s, w)
                sP, r, terminated, truncated, _ = env.step(a)
                s = sP
                Gm += r
        
            G.append(Gm)

        mean, hw = calc.confidence_interval(G)
        score = mean - hw

        env.close()

        video_files = sorted([f for f in os.listdir("tmp") if f.endswith(".mp4")])
        clips = []
        for i, vf in enumerate(tqdm(video_files, bar_format=fmt.bar, desc="ANNOTATE EPISODES")):
            vc = VideoFileClip(f"tmp/{vf}", )
            
            lr = ",".join(f"{x:.3f}" for x in self.lr)
            er = ",".join(f"{x:.3f}" for x in self.er)
            
            text = (f"{title}\n"
                    + f"EP {i+1:2d} Reward: {G[i]:0.1f}\n" 
                    + f"Score: {score:0.1f} ({mean:0.1f} ± {hw:0.1f})\n"
                    + f"FV:{self.x.__name__}\n"
                    + f"gamma={self.γ:.3f},lambda={self.λ:.3f},Delta={self.Δ:.3f}"
                    + f"lr=({lr})\ner=({er})")
            
            tc = TextClip(text=text, 
                            font_size=12, 
                            color='white', 
                            method='caption',
                            vertical_align='left', horizontal_align='top',
                            size = (vc.w, vc.h))
            tc = tc.with_duration(vc.duration)
            
            cvc = CompositeVideoClip([vc, tc])
            clips.append(cvc)

        print("RECORDING: Combining clips")
        final_clip = concatenate_videoclips(clips)
        os.makedirs("video", exist_ok=True)
        fn = f"video/{title}_({score:0.1f})"
        final_clip.write_videofile(fn + ".mp4")
        final_clip.write_gif(fn + ".gif", fps=15)
        print("RECORDING: Complete")

        shutil.rmtree("tmp")


    def plot(self):

        policies = pd.DataFrame(self.policies)

        policies.sort_values(by='episode', ascending=True, inplace=True)
        p = policies.iloc[0]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Theme
        sns.set_theme(style="whitegrid")

        # Main line
        sns.lineplot(data=policies, x='episode', y='mean', label="$\\mu$", ax=ax)

        # Confidence interval shading
        ax.fill_between(
            policies['episode'],
            policies['mean'] - policies['halfwidth'],
            policies['mean'] + policies['halfwidth'],
            alpha=0.3,
            label='95% CI'
        )

        # Best policy row
        p = policies.sort_values(by='score', ascending=False).iloc[0]

        lr = ",".join(f"{x:.3f}" for x in self.lr)
        er = ",".join(f"{x:.3f}" for x in self.er)

        title = (f"{self.__class__.__name__}\n"
                + f"Best score Episode {p['episode']}: {p['score']:0.1f} ({p['mean']:0.1f} ± {p['halfwidth']:0.1f})\n" 
                + f"x:{self.x.__name__}\n"
                + f"$\\gamma$={self.γ:.3e}, $\\lambda$={self.λ:.3e}, $\\Delta$={self.Δ:.3e}\n"
                + f"lr=({lr}), er=({er})")

        # Axis labels and formatting
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_ylim(-600, 300)
        ax.grid()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

        # Legend and layout
        ax.legend(loc="lower right")
        fig.tight_layout()

        os.makedirs("plots", exist_ok=True)
        fn = f"plots/{self.__class__.__name__}_score{p['score']:0.1f}.svg"
        fig.savefig(fn)