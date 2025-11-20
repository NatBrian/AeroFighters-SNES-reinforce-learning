"""
Environment wrapper for AeroFighters SNES

Creates and wraps the Retro environment with:
- Frame preprocessing (grayscale, resize to 84x84)
- Frame stacking (4 frames)
- Frame skipping (repeat action for 4 frames)
- Discrete action space
- Reward shaping (score delta + survival bonus)
"""

import gym
import numpy as np
import cv2
import retro
from gym import spaces
from collections import deque
from typing import Tuple, Any, Dict

from aerofighters_rl.action_mapping import get_action_space, map_action


class PreprocessFrame(gym.ObservationWrapper):
    """
    Preprocesses frames to 84x84 grayscale.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
    
    def observation(self, obs):
        """Convert to grayscale and resize to 84x84"""
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


class FrameStack(gym.Wrapper):
    """
    Stack the last n frames together.
    """
    
    def __init__(self, env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        # Update observation space to stack frames
        low = np.repeat(env.observation_space.low[..., np.newaxis], n_frames, axis=-1)
        high = np.repeat(env.observation_space.high[..., np.newaxis], n_frames, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Fill the deque with the same frame
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Stack frames along the last dimension"""
        return np.stack(self.frames, axis=-1)


class FrameSkip(gym.Wrapper):
    """
    Repeat action for n frames and return max pooled frame.
    """
    
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        return obs, total_reward, done, info


class DiscreteActions(gym.ActionWrapper):
    """
    Converts discrete actions to SNES button arrays.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(get_action_space())
        # Store the button mapping
        self._actions = {i: map_action(i) for i in range(get_action_space())}
    
    def action(self, act):
        """Convert discrete action to button array"""
        # Ensure act is an integer
        if isinstance(act, np.ndarray):
            act = int(act.item())
        else:
            act = int(act)
        return self._actions[act]


class RewardShaping(gym.Wrapper):
    """
    Custom reward shaping for AeroFighters:
    - Score delta reward (main driver)
    - Survival bonus per step
    - Death penalty
    """
    
    def __init__(self, env, survival_bonus: float = 0.1, death_penalty: float = -10.0):
        super().__init__(env)
        self.survival_bonus = survival_bonus
        self.death_penalty = death_penalty
        self.prev_score = 0
        self.prev_lives = 3  # Assuming 3 starting lives
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_score = 0
        self.prev_lives = 3  # Default starting lives
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Get current score and lives from info
        current_score = info.get('score', 0)
        current_lives = info.get('lives', self.prev_lives)
        
        # Calculate score delta reward
        score_delta = (current_score - self.prev_score) / 10.0
        
        # Survival bonus
        survival_reward = self.survival_bonus
        
        # Death penalty
        death_reward = 0.0
        if current_lives < self.prev_lives:
            death_reward = self.death_penalty
        
        # Total shaped reward
        shaped_reward = score_delta + survival_reward + death_reward
        
        # Update previous values
        self.prev_score = current_score
        self.prev_lives = current_lives
        
        # Store original reward in info for logging
        info['original_reward'] = reward
        info['shaped_reward'] = shaped_reward
        info['score_delta'] = score_delta
        
        return obs, shaped_reward, done, info


def make_aerofighters_env(
    rank: int = 0,
    frame_skip: int = 4,
    frame_stack: int = 4,
    reward_shaping: bool = True,
) -> gym.Env:
    """
    Create and wrap the AeroFighters-Snes environment.
    
    Args:
        rank: Environment rank for seeding (used in vectorized envs)
        frame_skip: Number of frames to skip (action repeat)
        frame_stack: Number of frames to stack
        reward_shaping: Whether to apply custom reward shaping
        
    Returns:
        Wrapped gymnasium environment
    """
    
    # Create base retro environment (use ALL actions space, we'll convert our discrete actions)
    env = retro.make(
        game='AeroFighters-Snes',
        state='Level1.USA',  # Start directly at Level 1
        use_restricted_actions=retro.Actions.ALL,  # This gives us MultiBinary actions
        obs_type=retro.Observations.IMAGE
    )
    
    # Apply frame skipping
    if frame_skip > 1:
        env = FrameSkip(env, skip=frame_skip)
    
    # Apply discrete action mapping
    env = DiscreteActions(env)
    
    # Preprocess frames
    env = PreprocessFrame(env)
    
    # Stack frames
    if frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)
    
    # Apply reward shaping
    if reward_shaping:
        env = RewardShaping(env)
    
    # Seed the environment (gym 0.25 uses env.seed() not reset(seed=...))
    env.seed(rank)
    
    return env


if __name__ == "__main__":
    # Test environment creation
    print("Creating AeroFighters environment...")
    env = make_aerofighters_env()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    print("\nRunning random episode...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
