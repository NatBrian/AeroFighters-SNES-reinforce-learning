"""
Utility functions for AeroFighters RL project

Includes:
- Vectorized environment creation
- Logging setup
- Plotting functions
- Progress tracking
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gym


def make_vec_env(
    env_fn: Callable,
    n_envs: int = 8,
    use_subprocess: bool = True,
) -> gym.Env:
    """
    Create vectorized environments for parallel training.
    
    Args:
        env_fn: Function that creates a single environment
        n_envs: Number of parallel environments
        use_subprocess: If True, use SubprocVecEnv (faster), else DummyVecEnv
        
    Returns:
        Vectorized environment
    """
    
    # Create list of env creation functions with different seeds
    env_fns = [lambda i=i: env_fn(rank=i) for i in range(n_envs)]
    
    # Choose vectorization method
    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    
    return vec_env


def setup_logging_dirs(*dirs: str) -> None:
    """
    Create directories for logging if they don't exist.
    
    Args:
        *dirs: Directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def plot_training_curves(
    log_dir: str,
    save_path: str,
    title: str = "Training Progress"
) -> None:
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        save_path: Path to save the plot
        title: Plot title
    """
    try:
        from tbparse import SummaryReader
        
        reader = SummaryReader(log_dir)
        df = reader.scalars
        
        if df.empty:
            print("No training data found in logs")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot episode reward
        if 'rollout/ep_rew_mean' in df['tag'].values:
            reward_df = df[df['tag'] == 'rollout/ep_rew_mean']
            axes[0, 0].plot(reward_df['step'], reward_df['value'])
            axes[0, 0].set_title('Episode Reward Mean')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Plot episode length
        if 'rollout/ep_len_mean' in df['tag'].values:
            length_df = df[df['tag'] == 'rollout/ep_len_mean']
            axes[0, 1].plot(length_df['step'], length_df['value'])
            axes[0, 1].set_title('Episode Length Mean')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].grid(True)
        
        # Plot learning rate
        if 'train/learning_rate' in df['tag'].values:
            lr_df = df[df['tag'] == 'train/learning_rate']
            axes[1, 0].plot(lr_df['step'], lr_df['value'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True)
        
        # Plot FPS
        if 'time/fps' in df['tag'].values:
            fps_df = df[df['tag'] == 'time/fps']
            axes[1, 1].plot(fps_df['step'], fps_df['value'])
            axes[1, 1].set_title('Training Speed (FPS)')
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('FPS')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        plt.close()
        
    except ImportError:
        print("tbparse not installed. Install with: pip install tbparse")
    except Exception as e:
        print(f"Error plotting training curves: {e}")


class ProgressCallback(BaseCallback):
    """
    Custom callback for tracking and displaying training progress.
    Saves checkpoints and generates plots periodically.
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        plot_freq: int = 50000,
        plot_path: Optional[str] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.plot_freq = plot_freq
        self.plot_path = plot_path or "plots/"
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
    def _init_callback(self) -> None:
        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Save model checkpoint
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.save_path,
                f"ppo_aerofighters_{self.n_calls}_steps.zip"
            )
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Checkpoint saved: {checkpoint_path}")
                print(f"Timesteps: {self.n_calls:,}")
                print(f"{'='*60}\n")
        
        # Generate plots periodically
        if self.n_calls % self.plot_freq == 0 and self.n_calls > 0:
            plot_path = os.path.join(
                self.plot_path,
                f"training_progress_{self.n_calls}.png"
            )
            # Get log directory from logger
            log_dir = self.logger.dir if hasattr(self.logger, 'dir') else "logs/"
            plot_training_curves(log_dir, plot_path)
        
        return True


class EpisodeStatsCallback(BaseCallback):
    """
    Callback to collect episode statistics for reporting.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if any episode finished
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    # Get episode info
                    info = self.locals["infos"][i]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
                        self.episode_count += 1
                        
                        if self.verbose > 0 and self.episode_count % 10 == 0:
                            mean_reward = np.mean(self.episode_rewards[-100:])
                            mean_length = np.mean(self.episode_lengths[-100:])
                            print(f"\nEpisode {self.episode_count}")
                            print(f"  Mean reward (last 100): {mean_reward:.2f}")
                            print(f"  Mean length (last 100): {mean_length:.1f}")
        
        return True
    
        return True
    
    def get_stats(self) -> dict:
        """Get episode statistics"""
        if not self.episode_rewards:
            return {}
        
        return {
            "total_episodes": self.episode_count,
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
        }


def render_game_frame(
    env: gym.Env,
    observation: Optional[np.ndarray] = None,
    window_name: str = "AeroFighters Training",
    info_text: Optional[str] = None,
) -> None:
    """
    Render a game frame with high quality from the retro environment.
    This is a reusable rendering function for any RL algorithm.
    
    Args:
        env: The game environment (wrapped or unwrapped)
        observation: Optional observation to use as fallback if screen can't be retrieved
        window_name: Name of the OpenCV window
        info_text: Optional text to display on the frame
    """
    import cv2
    
    frame = None
    
    # Try to get the original screen from the environment
    # Unwrap to find the Retro environment
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        if hasattr(unwrapped_env, 'get_screen'):
            try:
                frame = unwrapped_env.get_screen()
                break
            except:
                pass
        unwrapped_env = unwrapped_env.env
    
    # If we found the retro env but get_screen didn't work
    if frame is None and hasattr(unwrapped_env, 'em'):
        try:
            frame = unwrapped_env.get_screen()
        except:
            pass
    
    # Fallback: Use observation if we can't get the original screen
    if frame is None and observation is not None:
        frame = observation
        # Handle different observation formats
        # Handle channel-first format (C, H, W)
        if len(frame.shape) == 3:
            if frame.shape[0] == 4:  # Stacked frames, channel first
                frame = frame[-1, :, :]
            elif frame.shape[0] == 1:
                frame = frame[0, :, :]
            # Handle channel-last format (H, W, C)
            elif frame.shape[2] == 4:  # Stacked frames, channel last
                frame = frame[:, :, -1]
        
        # Convert grayscale observation to BGR for display
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize to 512x448 (2x SNES resolution)
        frame = cv2.resize(frame, (512, 448), interpolation=cv2.INTER_NEAREST)
    else:
        # We got the high-quality frame!
        # It's usually RGB, convert to BGR for OpenCV
        if frame is not None:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize to 2x for visibility (SNES is 256x224 -> 512x448)
            frame = cv2.resize(frame, (512, 448), interpolation=cv2.INTER_NEAREST)
    
    if frame is not None:
        # Add info text if provided
        if info_text:
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


class RenderingCallback(BaseCallback):
    """
    Callback to render the environment during training.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.window_name = 'AeroFighters Training'
        
    def _on_step(self) -> bool:
        # Get observation from the training environment
        obs = self.locals.get('new_obs')
        observation = obs[0] if obs is not None else None
        
        # Get the environment (handle vectorized environments)
        env = None
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
        
        # Prepare info text
        info_text = f"Step: {self.num_timesteps}"
        
        # Use the reusable rendering function
        render_game_frame(
            env=env,
            observation=observation,
            window_name=self.window_name,
            info_text=info_text
        )
            
        return True
        
    def _on_training_end(self) -> None:
        import cv2
        cv2.destroyWindow(self.window_name)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test directory setup
    setup_logging_dirs("test_logs", "test_models", "test_plots")
    
    print("\nUtilities tested successfully!")
