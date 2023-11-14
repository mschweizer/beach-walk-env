from typing import Union

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn


class TrueEpisodeMonitor(Monitor):
    def __init__(self, env):
        super().__init__(env, info_keywords=("episode_end",))

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        if info.get("episode"):
            info["episode"]["is_success"] = info["is_success"]
            info["true_episode"] = info["episode"]
            del info["is_success"]
            del info["episode"]
        return observation, reward, done, info
