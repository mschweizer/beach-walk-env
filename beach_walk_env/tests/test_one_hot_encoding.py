import numpy as np
from minigrid.wrappers import FullyObsWrapper

from beach_walk_env import BeachWalkEnv, CustomObsWrapper, OneHotObsWrapper
from beach_walk_env.one_hot_observation_wrapper import ONE_HOT_OBJECT_TO_LAYER_IDX


def create_one_hot_env(grid_size, agent_position):
    env = BeachWalkEnv(grid_size, agent_position)
    env = FullyObsWrapper(env)
    env = CustomObsWrapper(env)
    return OneHotObsWrapper(env)


def test_agent_encoded_at_start_position():
    grid_size = 6
    agent_position = (1, 2)
    one_hot_env = create_one_hot_env(grid_size, agent_position)

    initial_obs, _ = one_hot_env.reset()
    assert tuple(np.argwhere(initial_obs[ONE_HOT_OBJECT_TO_LAYER_IDX["agent"]] == 1)[0]) == agent_position


def test_agent_is_encoded_only_once():
    grid_size = 6
    agent_position = (1, 2)
    one_hot_env = create_one_hot_env(grid_size, agent_position)

    initial_obs, _ = one_hot_env.reset()
    assert len(np.argwhere(initial_obs[ONE_HOT_OBJECT_TO_LAYER_IDX["agent"]] == 1)) == 1
