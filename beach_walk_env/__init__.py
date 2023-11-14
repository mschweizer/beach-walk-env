from gymnasium import register

from beach_walk_env.beach_walk import *
from beach_walk_env.observation_wrapper import CustomObsWrapper

register(
    id="BeachWalk-v0",
    entry_point="beach_walk_env:create_wrapped_beach_walk",
)

register(
    id="FixedHorizonBeachWalk-v0",
    entry_point="beach_walk_env:create_fixed_horizon_beach_walk",
)
