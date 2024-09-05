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
        wind_setting="stack",
        terminate_in_water=False,
        reward=1.,
        penalty=-1.,
        discount=1.,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.wind_gust_probability = wind_gust_probability
        self.wind_setting = wind_setting
        self.terminate_in_water = terminate_in_water

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
        self.current_episode_steps_in_water = 0

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

    def step(self, action):
        if action is None:
            return self.gen_obs(), 0., False, False, {}
        if self._wind_gust_occurs():
            obs, reward, terminated, truncated, info = self._wind_step(action)
        else:
            obs, reward, terminated, truncated, info = self._normal_step(action)
        if terminated or truncated:
            info = {**info, **self._finalize_episode()}
        return obs, reward, terminated, truncated, info

    def _finalize_episode(self):
        last_episode_steps_in_water = self.current_episode_steps_in_water
        self.current_episode_steps_in_water = 0
        return dict(steps_in_water=last_episode_steps_in_water)

    def _wind_gust_occurs(self):
        return self._rand_float(0, 1) < self.wind_gust_probability

    def _wind_step(self, action):
        # if the wind effect overwrites the original action
        if self.wind_setting == "overwrite":
            obs, reward, terminated, truncated, info = self._random_direction_step()
        # the wind blows after the action being taken
        elif self.wind_setting == "stack":
            obs, reward, terminated, truncated, info = self._normal_step(action)
            if not terminated and not truncated:
                obs, additional_reward, terminated, truncated, info = self._random_direction_step()
                reward += additional_reward
        else:
            raise Exception("Wind setting not supported")
        return obs, reward, terminated, truncated, info

    def _random_direction_step(self):
        action = self.action_space.sample()
        obs, reward, terminated, truncated, info = self._normal_step(action)
        return obs, reward, terminated, truncated, info

    def _normal_step(self, action):
        self.step_count += 1
        self._maybe_move_position(action)
        reward, terminated, move_info = self._handle_move()
        truncated, truncate_info = self._maybe_truncate_episode(move_info)
        return self.gen_obs(), reward, terminated, truncated, {**move_info, **truncate_info}

    def _maybe_move_position(self, action):
        if self._can_move_in_intended_direction(action):
            self.agent_pos = self.front_pos

    def _can_move_in_intended_direction(self, action):
        self.agent_dir = action  # Turn agent in the direction it tries to move
        fwd_cell = self.grid.get(*self.front_pos)  # Get the contents of the cell in front of the agent
        return fwd_cell is None or fwd_cell.can_overlap()

    def _handle_move(self):
        cell_type = self._get_current_cell_type()
        if cell_type == 'goal':
            reward, terminated, additional_info = self._treat_reaching_goal()
        elif cell_type == 'lava':
            reward, terminated, additional_info = self._treat_reaching_water()
        else:
            reward, terminated, additional_info = 0.0, False, {}
        if cell_type:
            additional_info["cell_type"] = cell_type if cell_type != 'lava' else 'water'
        return reward, terminated, additional_info

    def _get_current_cell_type(self):
        current_agent_cell = self.grid.get(*self.agent_pos)
        return current_agent_cell.type if current_agent_cell else None

    def _treat_reaching_goal(self):
        terminated = True
        return self._reward(), terminated, self._create_info_for_reaching_goal()

    def _reward(self):
        return self.reward * self.discount ** self.step_count

    @staticmethod
    def _create_info_for_reaching_goal():
        return dict(episode_end="success", is_success=True)

    def _treat_reaching_water(self):
        self.current_episode_steps_in_water += 1
        terminated = True
        if self.terminate_in_water:
            return self._penalty(), terminated, self._create_info_for_water_termination()
        return self._penalty(), not terminated, {}

    def _penalty(self):
        return self.penalty * self.discount ** self.step_count

    @staticmethod
    def _create_info_for_water_termination():
        return dict(episode_end="failure", is_success=False)

    def _maybe_truncate_episode(self, current_info):
        truncated = True
        if self._steps_remaining() or self._just_reached_terminal_state(current_info):
            return not truncated, {}
        return truncated, self._create_info_for_truncation()

    def _steps_remaining(self):
        return self.step_count < self.max_steps

    @staticmethod
    def _just_reached_terminal_state(info):
        return "episode_end" in info

    @staticmethod
    def _create_info_for_truncation():
        return dict(episode_end="timeout", is_success=False)

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
