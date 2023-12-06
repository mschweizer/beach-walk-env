import numpy as np
import pytest

from beach_walk_env import BeachWalkEnv


def test_putting_agent_outside_grid_raises_error():
    size = 6
    env = BeachWalkEnv(size=size)

    with pytest.raises(AssertionError):
        env.put_agent(size + 2, size + 2)


def test_putting_agent_at_new_position_changes_position_to_new_position():
    env = BeachWalkEnv()
    new_agent_position = [2, 2]
    env.put_agent(*new_agent_position)
    assert np.all([env.agent_pos, np.array(new_agent_position)])
