# AeroFighters SNES PPO Agent (Gym Retro)

A complete reinforcement learning project that trains a Proximal Policy Optimization (PPO) agent to play the SNES game **AeroFighters** using Gym Retro. This project emphasizes **visualization** of the learning process with detailed metrics, reports, and graphs.
 
**Gym Retro Documentation**: https://retro.readthedocs.io/en/latest/getting_started.html

---

## Features

- **End-to-end training and evaluation** for AeroFighters-Snes
- **PPO with CNN policy** (Stable-Baselines3)
- **Pluggable RL algorithm architecture** - easily swap PPO for DQN, REINFORCE, or other algorithms
- **Comprehensive visualization and reporting**:
  - Real-time training metrics via TensorBoard
  - Learning curves (reward, episode length, fps)
  - Training progress plots saved as PNG
  - Episode statistics and performance reports
- **Simple reward shaping** (score delta + survival bonus)
- **Discrete action space** (11 actions) optimized for the SNES shooter game
- **"Always Shoot" strategy**: Movement actions automatically fire the weapon for optimal gameplay
- **Frame preprocessing**: grayscale, resize to 84x84, frame stacking (4 frames)
- **Vectorized environments** for faster training (8 parallel environments)
- **Video recording** of trained agent gameplay

---

## Requirements

- **Python 3.7** (tested and working)
- AeroFighters SNES ROM (legally obtained)
- Virtual environment (`.venv` recommended)

### Installation

1. **Create Python 3.7 virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install gym-retro** (pre-built wheel available for Python 3.7):
   ```bash
   pip install --default-timeout=300 gym-retro
   ```

3. **Downgrade gym** to compatible version:
   ```bash
   pip install gym==0.25.2
   ```

4. **Install remaining dependencies**:
   ```bash
   pip install stable-baselines3 shimmy>=0.2.1 rich
   pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   pip install opencv-python matplotlib tensorboard tqdm pygame pillow pandas seaborn
   ```

   Or simply:
   ```bash
   pip install -r requirements.txt
   pip install gym==0.25.2  # Downgrade after gym-retro installs gym 0.26
   ```

5. **Important**: You must legally obtain the AeroFighters SNES ROM.

---

## ROM Import Instructions

**⚠️ LEGAL NOTICE**: ROMs are **NOT included** in this repository. You must legally obtain the AeroFighters SNES ROM from your own cartridge or a legitimate source.

### Required ROM:
- **File**: `AeroFighters-Snes.sfc`
- **Expected SHA-1**: `aeedb733c7bcd5fb933eb998d9fb3c960335bb4f`

### Method 1: Using retro.import (Recommended)

1. Obtain the AeroFighters SNES ROM file

2. Verify the SHA-1 hash matches:
   ```bash
   certutil -hashfile AeroFighters-Snes.sfc SHA1  # Windows
   # or
   shasum AeroFighters-Snes.sfc  # Linux/Mac
   ```

3. Import using retro.import:
   ```bash
   python -m retro.import /path/to/rom/folder/
   ```

### Method 2: Manual Copy (If import shows "Imported 0 games")

If the automatic import doesn't work, manually copy the ROM:

1. Find the retro data directory:
   ```bash
   python -c "import retro; print(retro.data.path())"
   ```

2. Copy your ROM to the game directory:
   ```bash
   # Windows example:
   copy AeroFighters-Snes.sfc .venv\lib\site-packages\retro\data\stable\AeroFighters-Snes\rom.sfc
   
   # Linux/Mac example:
   cp AeroFighters-Snes.sfc .venv/lib/python3.7/site-packages/retro/data/stable/AeroFighters-Snes/rom.sfc
   ```

3. Verify the game works:
   ```bash
   python -c "import retro; env = retro.make('AeroFighters-Snes'); print('Success!')"
   ```

**Note**: The ROM hash MUST match `aeedb733c7bcd5fb933eb998d9fb3c960335bb4f` for the game integration to work properly.

For more details, see the [Gym Retro documentation](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms).

---

## Testing Your Setup

Before starting training, verify everything works:

```bash
# Test environment wrapper
python -m aerofighters_rl.env_wrapper

# You should see:
# ✓ Environment created!
# ✓ Observation space: Box(0, 255, (84, 84, 4), uint8)
# ✓ Action space: Discrete(11)
# ✓ Episode ran successfully with reward
```

---

## Training

### Basic Training

Start training with default settings:

```bash
# Train headless (faster, default)
python -m aerofighters_rl.train_ppo

# Train and watch the agent play (slower, forces n-envs=1)
python -m aerofighters_rl.train_ppo --render
```

### Training with Custom Parameters

```bash
python -m aerofighters_rl.train_ppo \
    --total-timesteps 1000000 \
    --n-envs 8 \
    --learning-rate 0.0001 \
    --save-freq 50000 \
    --render  # Optional: watch training
```

### What Happens During Training

- **Parallel environments**: 8 environments run simultaneously for efficient data collection (1 if rendering)
- **Model checkpoints**: Saved to `models/` directory every 50,000 timesteps by default
- **TensorBoard logs**: Real-time metrics logged to `logs/` directory
- **Training plots**: Learning curves saved as PNG in `plots/` directory
- **Progress tracking**: Episode rewards, lengths, and training statistics displayed in console

### View Training Progress

Launch TensorBoard to visualize training in real-time:

```bash
tensorboard --logdir=./logs
```

Open your browser to `http://localhost:6006` to see:
- Episode reward over time
- Episode length
- Learning rate
- Policy loss, value loss, entropy
- Frames per second (FPS)

### Training Duration

- **Short test**: 100K timesteps (~30-60 minutes on CPU, ~10-20 minutes on GPU)
- **Recommended**: 1-2M timesteps (~4-8 hours on CPU, ~1-2 hours on GPU)
- **Extended training**: 5-10M timesteps for best performance

The agent will gradually learn to:
1. Avoid obstacles
2. Shoot enemies
3. Collect power-ups
4. Survive longer and achieve higher scores

### NEAT (NeuroEvolution) Training

To train using the NEAT evolutionary algorithm:

```bash
# Basic training
python -m aerofighters_rl.train_neat

# Train with rendering (watch the agents)
python -m aerofighters_rl.train_neat --render

# Train for more generations
python -m aerofighters_rl.train_neat --generations 100
```

**Note on NEAT Features**:
Currently, the NEAT implementation is simpler than the PPO one:
- **Single Environment**: Runs one game at a time (slower than PPO's parallel training).
- **Console Logging**: Progress is printed to the terminal, not TensorBoard.
- **Checkpoints**: Saves `neat-checkpoint-N` files and a final `winner.pkl` in the `neat/` directory.
- **Visualization**: Use `--render` to watch; video recording is not yet implemented.

This will evolve a population of agents. The best genome will be saved to `neat/winner.pkl`.

---

## Evaluation

### Run Trained Agent

Evaluate the latest trained model with rendering enabled:

```bash
python -m aerofighters_rl.evaluate --model-path models/ppo_aerofighters_latest.zip --render
```

### Record Video

Record gameplay video of the trained agent:

```bash
python -m aerofighters_rl.evaluate \
    --model-path models/ppo_aerofighters_latest.zip \
    --record-video \
    --video-folder videos/ \
    --n-episodes 5
```

Videos will be saved to the `videos/` directory.

### Evaluation Options

- `--model-path`: Path to trained model (default: `models/ppo_aerofighters_latest.zip`)
- `--n-episodes`: Number of episodes to run (default: 10)
- `--render`: Enable rendering (watch the agent play)
- `--record-video`: Save video recordings
- `--video-folder`: Directory to save videos (default: `videos/`)
- `--deterministic`: Use deterministic actions (no exploration)

---

## Project Structure

```
aerofighters-rl-aerofighters-snes/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── aerofighters_rl/                    # Main package
│   ├── __init__.py                     # Package initialization
│   ├── action_mapping.py               # Discrete action space mapping
│   ├── env_wrapper.py                  # Environment creation and preprocessing
│   ├── train_ppo.py                    # Main training script
│   ├── evaluate.py                     # Evaluation and video recording
│   └── utils.py                        # Helper functions (vectorization, logging)
├── models/                             # Saved model checkpoints (created during training)
├── logs/                               # TensorBoard logs (created during training)
├── plots/                              # Training plots (created during training)
└── videos/                             # Recorded videos (created during evaluation)
```

### File Descriptions

- **`action_mapping.py`**: Defines the discrete action space (11 actions) and maps them to SNES button arrays
- **`env_wrapper.py`**: Creates the AeroFighters environment with preprocessing (grayscale, resize, frame stacking) and reward shaping
- **`train_ppo.py`**: Main training script with PPO agent, checkpoint saving, and visualization
- **`evaluate.py`**: Evaluation script with rendering and video recording capabilities
- **`utils.py`**: Utility functions for vectorized environments, logging setup, and plotting

---

## Pluggable RL Algorithm Architecture

This project is designed for **easy algorithm swapping**. The architecture separates:

1. **Environment setup** (`env_wrapper.py`) - algorithm-agnostic
2. **Action space** (`action_mapping.py`) - algorithm-agnostic
3. **Training logic** (`train_ppo.py`) - algorithm-specific

### Switching to a Different Algorithm

To use a different RL algorithm (e.g., DQN, A2C, SAC):

1. Create a new training script (e.g., `train_dqn.py`) based on `train_ppo.py`
2. Replace the PPO import and initialization:
   ```python
   # from stable_baselines3 import PPO
   from stable_baselines3 import DQN
   
   # model = PPO("CnnPolicy", env, ...)
   model = DQN("CnnPolicy", env, ...)
   ```
3. Adjust hyperparameters as needed for the new algorithm
4. Use the same environment and action space setup

All visualization, logging, and evaluation code remains the same!

---

## Hyperparameters

Default PPO hyperparameters (in `train_ppo.py`):

- **Learning rate**: 0.0001 (with linear decay)
- **n_steps**: 2048 (steps per environment before update)
- **batch_size**: 64 (minibatch size)
- **n_epochs**: 10 (optimization epochs per update)
- **gamma**: 0.99 (discount factor)
- **gae_lambda**: 0.95 (GAE parameter)
- **clip_range**: 0.2 (PPO clipping parameter)
- **ent_coef**: 0.01 (entropy coefficient for exploration)
- **vf_coef**: 0.5 (value function coefficient)
- **max_grad_norm**: 0.5 (gradient clipping)

These can be adjusted in the training script for better performance.

---

## Reward Shaping

The environment uses simple reward shaping to guide the agent:

1. **Score delta**: Reward = `(score_t - score_{t-1}) / 10.0`
2. **Survival bonus**: +0.1 per step (encourages staying alive)
3. **Death penalty**: -10.0 on game over (discourages dying)

This design is simple, interpretable, and effective for initial training.

---

## Future Improvements

- **Better reward shaping**: 
  - Enemy killed bonus
  - Power-up collection bonus
  - Position-based rewards (staying in good firing zones)
  
- **Curriculum learning**: 
  - Start with easier levels/states
  - Gradually increase difficulty
  
- **Hyperparameter tuning**: 
  - Grid search or Optuna optimization
  - Different learning rates, batch sizes, etc.
  
- **Algorithm comparison**: 
  - Train DQN, Double DQN, A2C, SAC
  - Compare sample efficiency and final performance
  
- **Advanced preprocessing**: 
  - Attention mechanisms on enemies
  - Object detection for power-ups
  
- **Multi-agent training**: 
  - Train on different aircraft/characters
  - Transfer learning between characters

- **Enhanced visualization**:
  - Attention heatmaps showing where the agent "looks"
  - Action distribution analysis
  - State-value visualization

---

## Troubleshooting

### ROM Import Shows "Imported 0 games"
This happens when your ROM's SHA-1 hash doesn't match the expected hash. Use **Method 2 (Manual Copy)** in the ROM Import section above.

### "AttributeError: module 'gym.utils.seeding' has no attribute 'hash_seed'"
You need gym 0.25.2, not 0.26.x:
```bash
pip install gym==0.25.2
```

### "TypeError: only integer scalar arrays can be converted to a scalar index"
Make sure you're using `retro.Actions.ALL` in `env_wrapper.py` (not `retro.Actions.DISCRETE`). This is already fixed in the provided code.

### Gym Deprecation Warnings
The warnings about gym being unmaintained are expected and can be ignored. The code works fine with gym 0.25.2 + gym-retro 0.8.0.

### PyTorch Download Timeout
Use increased timeout:
```bash
pip install --default-timeout=300 torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Out of Memory During Training
Reduce the number of parallel environments:
```bash
python -m aerofighters_rl.train_ppo --n-envs 4
```

---

## Acknowledgments

- [Gym Retro](https://github.com/openai/retro) for the retro game environment
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- The AeroFighters game developers

---
