#!/usr/bin/env python3
"""Module to load the FrozenLake environment."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the Gymnasium FrozenLake environment.

    Args:
        desc: custom description of the map
        map_name: pre-existing map name
        is_slippery: boolean determining if ice is slippery

    Returns:
        The FrozenLake environment instance
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
    return env
