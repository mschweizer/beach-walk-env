import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.wrappers import TimeLimit
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from seals.util import AutoResetWrapper

from beach_walk_env.actions import Actions
from beach_walk_env.observation_wrapper import CustomObsWrapper
from beach_walk_env.one_hot_observation_wrapper import OneHotObsWrapper
from beach_walk_env.true_episode_monitor import TrueEpisodeMonitor
from beach_walk_env.water import Water


class BeachWalkEnv(MiniGridEnv):
    MiniGridEnv.metadata.update(
        {"render_fps": 5}
    )

    def __init__(
        self, 
        size=6, 
        agent_start_pos=(1, 2), 
        agent_start_dir=0, 
        max_steps=25, 
        wind_gust_probability=0.5,
        windy=True,
        wind_setting=None,
        reward=1., 
        penalty=-1., 
        discount=1., 
        **kwargs
        ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.wind_gust_probability = wind_gust_probability
        self.windy = windy
        self.wind_setting = wind_setting

        self.reward = reward
        self.penalty = penalty
        self.discount = discount

        self.reward_spec = {"reward": self.reward, "penalty": self.penalty, "discount": self.discount}

        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "Avoid the water and get to the green goal square."),
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            highlight=False,
            **kwargs
        )
        self.actions = Actions
        self.action_space = Discrete(len(self.actions))
        self.reward_range = (-1, 1)

        self.total_step_count = 0

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the waterfront
        self.grid.horz_wall(x=1, y=1, length=4, obj_type=Water)

        # Generate beach
        for x in range(1, 5):
            for y in range(2, 5):
                self.grid.set(x, y, Floor(color="yellow"))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        goal = Goal()
        self.put_obj(goal, 4, 2)

        self.mission = self._create_mission_statement()

    @staticmethod
    def _create_mission_statement():
        return "Reach the goal without falling into the water."

    def _normal_step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if action is None:
            return self.gen_obs(), reward, terminated, truncated, info

        self.step_count += 1

        # Turn agent in the direction it tries to move
        self.agent_dir = action
        
        ## where is the mechanism that prevent the agent from going beyond the grid??
        fwd_pos = self.front_pos
        
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            terminated = True
            reward = self._reward()
            info["episode_end"] = "success"
            info["is_success"] = True
        if fwd_cell is not None and fwd_cell.type == 'lava':
            # terminated = True
            reward = self._penalty()
            info["episode_end"] = "failure"
            info["is_success"] = False
        if self.step_count >= self.max_steps:
            truncated = True
            if "episode_end" not in info:
                info["episode_end"] = "timeout"
                info["is_success"] = False
        obs = self.gen_obs()
        
        return obs, reward, terminated, truncated, info

    def step(self, action):
        if self.windy:
            # if the wind effect overwrites the original action
            if self.wind_setting == "overwrite":
                if self._rand_float(0, 1) < self.wind_gust_probability:
                    action = self.action_space.sample()
                obs, reward, terminated, truncated, info = self._normal_step(action)
            # the wind blows after the action being taken
            elif self.wind_setting == "stack": 
                obs, reward, terminated, truncated, info = self._normal_step(action)
                if not terminated and not truncated:
                    if self._rand_float(0, 1) < self.wind_gust_probability:
                        action = self.action_space.sample()
                        obs, reward, terminated, truncated, info = self._normal_step(action)
            else:
                raise Exception("Wind Setting not supported")
        else:
            obs, reward, terminated, truncated, info = self._normal_step(action)
        return obs, reward, terminated, truncated, info

    def _reward(self):
        return self.reward * self.discount ** self.step_count

    def _penalty(self):
        return self.penalty * self.discount ** self.step_count

    def put_agent(self, i, j):
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        self.agent_pos = np.array((i, j))


def create_wrapped_beach_walk(size=6, agent_start_pos=(1, 2), agent_start_dir=0, max_steps=150,
                              wind_gust_probability=0.5, discount=1., **kwargs):
    env = BeachWalkEnv(size, agent_start_pos, agent_start_dir, max_steps, wind_gust_probability, discount=discount,
                       **kwargs)
    env = FullyObsWrapper(env)
    env = CustomObsWrapper(env)
    env = OneHotObsWrapper(env)
    return env


def create_fixed_horizon_beach_walk(size=6, agent_start_pos=(1, 2), agent_start_dir=0, horizon_length=25,
                                    wind_gust_probability=0.5, discount=1.0, **kwargs):
    env = create_wrapped_beach_walk(size=size, agent_start_pos=agent_start_pos, agent_start_dir=agent_start_dir,
                                    wind_gust_probability=wind_gust_probability, discount=discount, **kwargs)
    env = TrueEpisodeMonitor(env)
    env = AutoResetWrapper(env, discard_terminal_observation=False)
    env = TimeLimit(env, max_episode_steps=horizon_length)
    return env
