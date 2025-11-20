"""
AeroFighters SNES Reinforcement Learning Package

A complete RL project for training agents to play AeroFighters-Snes
using Gym Retro and Stable-Baselines3.

Main components:
- env_wrapper: Environment creation, preprocessing, and reward shaping
- action_mapping: Discrete action space for SNES controls
- train_ppo: PPO training script with visualization
- evaluate: Evaluation and video recording
- utils: Helper functions for vectorization and logging
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from aerofighters_rl.env_wrapper import make_aerofighters_env
from aerofighters_rl.action_mapping import get_action_space, map_action

__all__ = [
    "make_aerofighters_env",
    "get_action_space",
    "map_action",
]
