"""
Evaluation script for trained AeroFighters agent

Loads a trained model and evaluates it on the game with:
- Visual rendering
- Episode statistics
- Video recording (optional)
- Performance metrics
"""

import argparse
import os
import time
import numpy as np
import cv2
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from aerofighters_rl.env_wrapper import make_aerofighters_env


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    record_video: bool = False,
    video_folder: str = "videos/",
    video_length: int = 10000,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to trained model .zip file
        n_episodes: Number of episodes to evaluate
        render: Enable rendering
        record_video: Record video of episodes
        video_folder: Directory to save videos
        video_length: Max steps per video
        deterministic: Use deterministic actions (no exploration)
        
    Returns:
        Dictionary of evaluation statistics
    """
    
    print("\n" + "="*80)
    print("AEROFIGHTERS SNES - MODEL EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"Rendering: {render}")
    print(f"Recording: {record_video}")
    print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PPO.load(model_path)
    print(f"[OK] Model loaded from {model_path}\n")
    
    # Create environment
    print("Creating environment...")
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
    
    # Create environment
    env = make_aerofighters_env(rank=0)
    
    print("[OK] Environment created\n")
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    print("Starting evaluation...\n")
    
    for episode in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle potential tuple return
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Video writer for this episode
        video_writer = None
        if record_video:
            video_path = os.path.join(video_folder, f"aerofighters_eval_ep{episode+1}.mp4")
            # Will initialize writer on first frame to get dimensions
        
        print(f"Episode {episode + 1}/{n_episodes}")
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            result = env.step(action)
            
            # Handle both 4-tuple (old gym) and 5-tuple (new gym/gymnasium) returns
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
                
            # Handle vectorized env returns (if any)
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, list):
                info = info[0]
            
            episode_reward += reward
            episode_length += 1
            
            # Render if enabled or recording
            if render or record_video:
                # Try to get high-quality screen
                frame = None
                
                # Unwrap to find Retro env
                current_env = env
                while hasattr(current_env, 'env'):
                    if hasattr(current_env, 'get_screen'):
                        frame = current_env.get_screen()
                        break
                    current_env = current_env.env
                
                # Fallback to observation
                if frame is None:
                    if len(obs.shape) == 3 and obs.shape[2] == 4: # Stacked
                        frame = obs[:, :, -1]
                    else:
                        frame = obs
                    
                    # Convert to BGR
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    # Convert RGB to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Resize
                frame = cv2.resize(frame, (512, 448), interpolation=cv2.INTER_NEAREST)
                
                # Record video frame
                if record_video:
                    if video_writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))
                    
                    video_writer.write(frame)
                
                # Show window
                if render:
                    cv2.imshow('AeroFighters Agent', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        done = True
            
            if done:
                score = info.get('score', 0)
                episode_scores.append(score)
                
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Length: {episode_length}")
                print(f"  Score: {score}")
                print(f"  Score: {score}")
                print()
                break
        
        if video_writer is not None:
            video_writer.release()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    if render:
        cv2.destroyAllWindows()
    
    env.close()
    
    # Calculate statistics
    stats = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_score': np.mean(episode_scores) if episode_scores else 0,
        'max_score': np.max(episode_scores) if episode_scores else 0,
    }
    
    # Print summary
    print("="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Episodes: {stats['n_episodes']}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min:  {stats['min_reward']:.2f}")
    print(f"  Max:  {stats['max_reward']:.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"\nGame Score:")
    print(f"  Mean: {stats['mean_score']:.0f}")
    print(f"  Max:  {stats['max_score']:.0f}")
    
    if record_video:
        print(f"\nVideos saved to: {video_folder}")
    
    print("="*80 + "\n")
    
    return stats


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained AeroFighters PPO agent"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_aerofighters_latest.zip",
        help="Path to trained model (default: models/ppo_aerofighters_latest.zip)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (watch agent play)"
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable rendering"
    )
    parser.set_defaults(render=True)
    
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of episodes"
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="videos/",
        help="Directory to save videos (default: videos/)"
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=10000,
        help="Max steps per video (default: 10000)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)"
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic actions"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        render=args.render,
        record_video=args.record_video,
        video_folder=args.video_folder,
        video_length=args.video_length,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
