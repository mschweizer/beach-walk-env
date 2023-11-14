import gym.core
import numpy as np
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX

# the object types that are occurring in the environment,
# all other object types are not considered for encoding and will raise an assertion error
OBJECT_IDX_TO_IDX = {
    OBJECT_TO_IDX["wall"]: 0,
    OBJECT_TO_IDX["floor"]: 1,
    OBJECT_TO_IDX["goal"]: 2,
    OBJECT_TO_IDX["lava"]: 3,
    OBJECT_TO_IDX["agent"]: 4,
}


def encode_one_hot(obs, out_shape):
    out = np.zeros(out_shape, dtype='bool')
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            object_type_idx = obs[i, j]
            assert object_type_idx in OBJECT_IDX_TO_IDX, \
                "The object type with idx {} is not yet " \
                "taken into account for 1-hot-encoding.".format(object_type_idx)
            idx = OBJECT_IDX_TO_IDX[object_type_idx]
            out[idx, i, j] = 1
    return out


class OneHotObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a fully observable
    agent view as observation.
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(OBJECT_IDX_TO_IDX), obs_shape[0], obs_shape[1]),
            dtype='bool'
        )

    def observation(self, obs):
        return encode_one_hot(obs, self.observation_space.shape)
