from gymnasium import register
from gymnasium.envs.registration import WrapperSpec
from minigrid.wrappers import FullyObsWrapper

from beach_walk_env.beach_walk import *
from beach_walk_env.observation_wrapper import CustomObsWrapper

register(
    id="BeachWalk-v0",
    entry_point="beach_walk_env:create_wrapped_beach_walk",

)

register(
    id="FixedHorizonBeachWalk-v0",
    entry_point="beach_walk_env:create_fixed_horizon_beach_walk"
)

register(
    id="BeachWalk-v0",
    entry_point="beach_walk_env:BeachWalkEnv",
    additional_wrappers=(
        WrapperSpec(
            name=FullyObsWrapper.class_name(),
            entry_point=f"beach_walk_env:FullyObsWrapper",
            kwargs={}
        ),
        WrapperSpec(
            name=CustomObsWrapper.class_name(),
            entry_point=f"beach_walk_env:CustomObsWrapper",
            kwargs={}
        )
    )
)