"""
Action mapping for AeroFighters SNES

Maps discrete actions to SNES button arrays for Gym Retro.
Designed for shooter game controls with movement and shooting.
"""

import numpy as np
from typing import List


# SNES button indices for stable-retro
# These correspond to the button array expected by retro.make()
BUTTON_MAP = {
    'B': 0,       # Usually shoot/fire
    'Y': 1,       # Alternative button
    'SELECT': 2,
    'START': 3,
    'UP': 4,
    'DOWN': 5,
    'LEFT': 6,
    'RIGHT': 7,
    'A': 8,       # Alternative action
    'X': 9,       # Alternative action
    'L': 10,      # Left shoulder
    'R': 11,      # Right shoulder
}

# Define discrete action space
# Each action is [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
# Based on observation: B(0)=Bomb, Y(1)=Shoot
ACTIONS = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NO-OP
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SHOOT (Y)
    2: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # BOMB (B)
    3: [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # UP + SHOOT
    4: [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # DOWN + SHOOT
    5: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # LEFT + SHOOT
    6: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # RIGHT + SHOOT
    7: [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # UP-LEFT + SHOOT
    8: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # UP-RIGHT + SHOOT
    9: [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # DOWN-LEFT + SHOOT
    10: [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # DOWN-RIGHT + SHOOT
}


def get_action_space() -> int:
    """
    Get the size of the discrete action space.
    
    Returns:
        Number of discrete actions available
    """
    return len(ACTIONS)


def map_action(discrete_action: int) -> np.ndarray:
    """
    Map a discrete action to SNES button array.
    
    Args:
        discrete_action: Integer action from agent (0 to num_actions-1)
        
    Returns:
        numpy array of button states for Retro environment
        
    Raises:
        ValueError: If action is out of bounds
    """
    if discrete_action not in ACTIONS:
        raise ValueError(
            f"Invalid action {discrete_action}. Must be in range [0, {len(ACTIONS)-1}]"
        )
    
    return np.array(ACTIONS[discrete_action], dtype=np.uint8)


def get_action_meanings() -> List[str]:
    """
    Get human-readable meanings for each action.
    
    Returns:
        List of action descriptions
    """
    meanings = [
        "NO-OP",
        "SHOOT",
        "BOMB",
        "UP + SHOOT",
        "DOWN + SHOOT",
        "LEFT + SHOOT",
        "RIGHT + SHOOT",
        "UP-LEFT + SHOOT",
        "UP-RIGHT + SHOOT",
        "DOWN-LEFT + SHOOT",
        "DOWN-RIGHT + SHOOT",
    ]
    return meanings


if __name__ == "__main__":
    # Test action mapping
    print("Action Space Size:", get_action_space())
    print("\nAction Meanings:")
    for i, meaning in enumerate(get_action_meanings()):
        action_array = map_action(i)
        print(f"{i}: {meaning:15} -> {action_array}")
