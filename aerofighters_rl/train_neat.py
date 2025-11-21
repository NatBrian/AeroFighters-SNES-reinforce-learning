import os
import sys

import neat
import pickle
import cv2
import numpy as np
import gym
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from aerofighters_rl.env_wrapper import make_aerofighters_env
from aerofighters_rl.utils import render_game_frame


def eval_single_genome(genome_data):
    """
    Evaluate a single genome. This function is called by worker processes.
    
    Args:
        genome_data: Tuple of (genome_id, genome, config_dict, render, genome_index, total_genomes)
    
    Returns:
        Tuple of (genome_id, fitness, steps)
    """
    # Suppress warnings in worker process
    import warnings
    warnings.filterwarnings('ignore')
    
    genome_id, genome, config_dict, render, genome_index, total_genomes, genome_render_limit = genome_data
    
    # Create config from dict (needed for multiprocessing)
    import neat
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_dict)
    
    # Create the environment
    env = make_aerofighters_env(rank=genome_index, frame_skip=4, frame_stack=4, reward_shaping=True)
    
    # Create the neural network for this genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    observation = env.reset()
    
    # Fitness is the total reward accumulated
    current_fitness = 0.0
    
    done = False
    steps = 0
    max_steps = 2000  # Limit steps per episode to prevent infinite loops
    
    while not done and steps < max_steps:
        should_render = render and genome_index < genome_render_limit
        
        if should_render:
            # Use the reusable rendering function
            info_text = f"Genome {genome_index+1}/{total_genomes} (ID:{genome_id}) | Fitness: {current_fitness:.1f}"
            render_game_frame(
                env=env,
                observation=observation,
                window_name="NEAT Training - AeroFighters",
                info_text=info_text
            )

        # Flatten observation for the network
        # Taking only the last frame:
        img_input = observation[:, :, -1].flatten()
        
        # Get output from network
        output = net.activate(img_input)
        
        # Find the action with the highest activation
        action = np.argmax(output)
        
        # Debug: Print action occasionally to see what's happening
        if steps % 100 == 0 and render and genome_index < genome_render_limit:
            print(f"  Step {steps}: Action={action}, Output range=[{min(output):.3f}, {max(output):.3f}]")
        
        observation, reward, done, info = env.step(action)
        
        current_fitness += reward
        steps += 1
    
    if render and genome_index < genome_render_limit:
        cv2.destroyAllWindows()
    
    env.close()
    
    return (genome_id, current_fitness, steps)


def eval_genomes(genomes, config, render=False, genome_render_limit=float('inf'), num_workers=None, config_path=None):
    """
    Evaluate the fitness of each genome in the population.
    Uses parallel processing for faster evaluation (unless rendering is enabled).
    
    Args:
        genomes: List of (genome_id, genome) tuples
        config: NEAT configuration
        render: Whether to render the game (forces sequential evaluation)
        genome_render_limit: Only render the first N genomes (default: inf = all genomes)
        num_workers: Number of parallel workers (default: auto-detect CPU cores)
        config_path: Path to config file (needed for multiprocessing)
    """
    
    # Auto-detect number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    total_genomes = len(genomes)
    
    # If rendering, use sequential evaluation (can't render in parallel)
    if render:
        print("\n" + "="*60)
        print("RENDERING ENABLED - Using Sequential Evaluation")
        print("="*60 + "\n")
        sys.stdout.flush()
        num_workers = 1
    else:
        print("\n" + "="*60)
        print(f"PARALLEL MODE - Using {num_workers} Workers")
        print("="*60 + "\n")
        sys.stdout.flush()
    
    # Prepare genome data for workers
    # Need to pass config as a file path since Config objects can't be pickled easily
    if config_path is None:
        raise ValueError("config_path must be provided for parallel evaluation")
    
    genome_data_list = [
        (genome_id, genome, config_path, render, idx, total_genomes, genome_render_limit)
        for idx, (genome_id, genome) in enumerate(genomes)
    ]
    
    # Evaluate genomes
    if num_workers == 1:
        # Sequential evaluation
        results = [eval_single_genome(data) for data in genome_data_list]
    else:
        # Parallel evaluation
        with Pool(processes=num_workers) as pool:
            results = pool.map(eval_single_genome, genome_data_list)
    
    # Assign fitness back to genomes
    for (genome_id, fitness, steps) in results:
        # Find the genome in the original list
        for gid, genome in genomes:
            if gid == genome_id:
                genome.fitness = fitness
                print(f"Genome {genome_id}: Fitness {fitness:.2f} | Steps: {steps}")
                sys.stdout.flush()
                break

def run(config_path, args):
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population
    p = neat.Population(config)

    # Create 'neat' directory if it doesn't exist
    neat_dir = os.path.join(os.getcwd(), 'neat')
    os.makedirs(neat_dir, exist_ok=True)

    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Save checkpoints to the 'neat' directory
    # filename_prefix must be a path prefix
    checkpoint_prefix = os.path.join(neat_dir, 'neat-checkpoint-')
    p.add_reporter(neat.Checkpointer(5, filename_prefix=checkpoint_prefix))

    # Create a partial function to pass the render argument and config path
    eval_func = partial(eval_genomes, render=args.render, config_path=config_path)

    # Run for up to N generations
    winner = p.run(eval_func, args.generations)

    # Save the winner
    winner_path = os.path.join(neat_dir, 'winner.pkl')
    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)

    print(f'\nBest genome saved to {winner_path}')

def main():
    parser = argparse.ArgumentParser(description="Train NEAT agent on AeroFighters-Snes")
    
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to train (default: 50)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during training"
    )
    
    args = parser.parse_args()
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    run(config_path, args)

if __name__ == '__main__':
    main()
