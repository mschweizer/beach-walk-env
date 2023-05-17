from gym.spaces import Discrete
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Lava, Floor
from beach_walk_env.actions import Actions
from beach_walk_env.water import Water


class BeachWalkEnv(MiniGridEnv):

    metadata = {
        'video.frames_per_second': 5
    }

    def __init__(self, size=6, agent_start_pos=(1, 2), agent_start_dir=0, max_steps=150, wind_gust_probability=0.5,
                 **kwargs):
        self.mission = None
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.wind_gust_probability = wind_gust_probability

        max_steps = max_steps if max_steps else 4 * size * size

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )
        self.actions = Actions
        self.action_space = Discrete(len(self.actions))

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

    def step(self, action):
        if self._rand_float(0, 1) < self.wind_gust_probability:
            action = self.action_space.sample()

        self.step_count += 1

        reward = 0.0
        done = False

        # Turn agent in the direction it tries to move
        self.agent_dir = action

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            done = True
            reward = self._reward()
        if fwd_cell is not None and fwd_cell.type == 'lava':
            done = True

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}
