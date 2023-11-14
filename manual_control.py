#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
from gymnasium import Env
from minigrid.manual_control import ManualControl
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from beach_walk_env import BeachWalkEnv


class BeachWalkManualControl(ManualControl):
    def __init__(self, env: BeachWalkEnv, seed=None) -> None:
        assert isinstance(env.unwrapped, BeachWalkEnv)
        super().__init__(env, seed)
        self.env = env
        self.seed = seed
        self.closed = False

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": self.env.unwrapped.actions.left,
            "right": self.env.unwrapped.actions.right,
            "up": self.env.unwrapped.actions.up,
            "down": self.env.unwrapped.actions.down,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="BeachWalk gym environment to load",
        choices=gym.envs.registry.keys(),
        default="BeachWalk-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: Env = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    manual_control = BeachWalkManualControl(env, seed=args.seed)
    manual_control.start()
