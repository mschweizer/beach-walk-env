from typing import Union

import gym.core
import numpy as np
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn

NUM_DIRECTIONS = 4


class CustomObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(self.env.width, self.env.height, 2),  # number of cells
            dtype="uint8",
        )

    def observation(self, observation):
        only_img = observation['image']
        only_obj_type_and_agent_dir = only_img[:, :, (0, 2)]
        return only_obj_type_and_agent_dir


class TrueEpisodeMonitor(Monitor):
    def __init__(self, env):
        super().__init__(env, info_keywords=("episode_end",))

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        if info.get("episode"):
            info["true_episode"] = info["episode"]
            del info["episode"]
        return observation, reward, done, info
