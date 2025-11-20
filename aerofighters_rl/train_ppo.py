"""
Training script for PPO agent on AeroFighters-Snes

This is the main training script with:
- PPO algorithm from Stable-Baselines3
- Vectorized environments for parallel training
- TensorBoard logging
- Checkpoint saving
- Training visualization and progress tracking
"""

import argparse
import os
from datetime import datetime
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

from aerofighters_rl.env_wrapper import make_aerofighters_env
from aerofighters_rl.utils import (
    make_vec_env,
    setup_logging_dirs,
    ProgressCallback,
    EpisodeStatsCallback,
    RenderingCallback,
    plot_training_curves,
)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        
    Returns:
        Schedule function
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    
    return func


def train_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    learning_rate: float = 1e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    save_freq: int = 50_000,
    log_dir: str = "logs/",
    model_dir: str = "models/",
    plot_dir: str = "plots/",
    use_linear_schedule: bool = True,
    render: bool = False,
) -> PPO:
    """
    Train a PPO agent on AeroFighters-Snes.
    
    Args:
        total_timesteps: Total number of training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per environment before update
        batch_size: Minibatch size for optimization
        n_epochs: Number of optimization epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Gradient clipping
        save_freq: Save checkpoint every n steps
        log_dir: Directory for TensorBoard logs
        model_dir: Directory for model checkpoints
        plot_dir: Directory for plots
        use_linear_schedule: Use linear learning rate schedule
        render: Enable rendering during training
        
    Returns:
        Trained PPO model
    """
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir, f"ppo_{timestamp}")
    setup_logging_dirs(log_dir, model_dir, plot_dir)
    
    print("\n" + "="*80)
    print("AEROFIGHTERS SNES - PPO TRAINING")
    print("="*80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save frequency: {save_freq:,} steps")
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Rendering: {render}")
    print("="*80 + "\n")
    
    # Create vectorized environments
    print("Creating environments...")
    # Use DummyVecEnv if rendering to allow direct access to env.render()
    use_subprocess = not render
    
    env = make_vec_env(
        env_fn=make_aerofighters_env,
        n_envs=n_envs,
        use_subprocess=use_subprocess
    )
    print(f"[OK] Created {n_envs} parallel environments\n")
    
    # Learning rate schedule
    lr = linear_schedule(learning_rate) if use_linear_schedule else learning_rate
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=log_dir,
    )
    print("[OK] PPO model initialized\n")
    
    # Setup logger
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # Create callbacks
    callbacks = []
    
    progress_callback = ProgressCallback(
        save_freq=save_freq,
        save_path=model_dir,
        plot_freq=save_freq,
        plot_path=plot_dir,
        verbose=1
    )
    callbacks.append(progress_callback)
    
    episode_stats_callback = EpisodeStatsCallback(verbose=1)
    callbacks.append(episode_stats_callback)
    
    if render:
        rendering_callback = RenderingCallback(verbose=1)
        callbacks.append(rendering_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train the model
    print("Starting training...")
    print("Monitor progress in TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print("\nTraining in progress...\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "ppo_aerofighters_latest.zip")
    model.save(final_model_path)
    print(f"\n[OK] Final model saved: {final_model_path}")
    
    # Generate final plots
    final_plot_path = os.path.join(plot_dir, "training_final.png")
    plot_training_curves(log_dir, final_plot_path, title="Final Training Progress")
    
    # Print episode statistics
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    
    stats = episode_stats_callback.get_stats()
    if stats:
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Min/Max reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"Mean episode length: {stats['mean_length']:.1f}")
    
    print(f"\nModel saved to: {final_model_path}")
    print(f"Logs saved to: {log_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("\nView training progress:")
    print(f"  tensorboard --logdir={log_dir}")
    print("\nRun evaluation:")
    print(f"  python -m aerofighters_rl.evaluate --model-path {final_model_path}")
    print("="*80 + "\n")
    
    env.close()
    return model


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on AeroFighters-Snes"
    )
    
    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per environment before update (default: 2048)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size (default: 64)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every n steps (default: 50,000)"
    )
    
    # Directories
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Directory for logs (default: logs/)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Directory for models (default: models/)"
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/",
        help="Directory for plots (default: plots/)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during training (forces n-envs=1)"
    )
    
    args = parser.parse_args()
    
    # Force n_envs=1 if rendering is enabled
    n_envs = args.n_envs
    if args.render and n_envs > 1:
        print("WARNING: Rendering enabled, forcing n-envs=1 for visualization")
        n_envs = 1
    
    # Train the model
    train_ppo(
        total_timesteps=args.total_timesteps,
        n_envs=n_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        render=args.render,
    )


if __name__ == "__main__":
    main()
