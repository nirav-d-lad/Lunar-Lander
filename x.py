import numpy as np

from numpy.typing import NDArray

from RL.algorithm import State

import gymnasium as gym

env = gym.make("LunarLander-v3")

S = State(
    cardinality=len(env.observation_space.low),
    low=env.observation_space.low,
    high=env.observation_space.high
    )

def state(s:NDArray):
    s = s.astype(np.float128)
    return s

def state_interactions(s:NDArray):
    s = s.astype(np.float128)

    px, py, vx, vy, θ, ω, lf, rf = s

    interactions = np.array([
        px*py, px*vx, px*θ, px*ω, px*lf, px*rf,
        py*vx, py*vy, py*θ, py*ω, py*lf, py*rf,
        vx*vy, vx*θ, vx*ω, vx*lf, vx*rf,
        vy*θ, vy*ω, vy*lf, vy*rf,
        θ*ω, θ*lf, θ*rf,
        ω*lf, ω*rf,
        lf*rf])

    return np.hstack((s, interactions))
