#!/usr/bin/env python3
"""
Contains function to load FrozenLake environment from Gymnasium
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc: list of lists containing a custom description of the map
        map_name: string containing the pre-made map to load
        is_slippery: boolean to determine if the ice is slippery

    Returns:
        The loaded gymnasium environment
    """
    if desc is None and map_name is None:
        desc = gym.envs.toy_text.frozen_lake.generate_random_map(size=8)

    return gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
