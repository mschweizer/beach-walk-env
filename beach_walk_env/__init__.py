from gym import register
from beach_walk_env.beach_walk import *

register(
    id="BeachWalk-v0",
    entry_point="beach_walk_env:create_wrapped_beach_walk",
)
