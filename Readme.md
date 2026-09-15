# Flappy Bird DQN Agent

A Deep Q-Network (DQN) reinforcement learning agent that learns to play Flappy Bird using the `flappy_bird_gymnasium` environment.

## Project Structure

| File | Purpose |
|---|---|
| `agent.py` | Main training/testing loop — the `Agent` class that orchestrates everything |
| `dqn.py` | Neural network architecture (the Q-network) |
| `experience_replay.py` | Replay buffer that stores past experiences for training |
| `parameters.yaml` | Hyperparameter configurations (you can define multiple named sets) |

## How It Works

### 1. The Neural Network (`dqn.py`)
The `DQN` class is a simple feed-forward network:
- **Input**: game state (12 values by default — bird position, pipe positions, velocity, etc.)
- **Hidden layer**: 256 neurons with ReLU activation
- **Output**: 2 values (Q-values for each possible action — flap or don't flap)

Given a state, the network predicts the expected future reward for each action. The agent picks whichever action has the higher predicted value.

### 2. Experience Replay (`experience_replay.py`)
`ReplayMemory` is a FIFO buffer (`deque`) that stores `(state, action, next_state, reward, terminated)` tuples as the agent plays. Instead of learning from consecutive frames (which are highly correlated), the agent samples **random mini-batches** from this memory to train on. This stabilizes learning.

> Note: the `seed` parameter is accepted but currently unused — `random.sample` isn't seeded, so runs won't be reproducible unless you fix this yourself (e.g., `random.seed(seed)` in `__init__`).

### 3. The Agent (`agent.py`)

**Setup:**
- Loads hyperparameters from `parameters.yaml` based on the `param_set` name you pass in
- Auto-selects device: MPS (Apple Silicon) → CUDA (NVIDIA GPU) → CPU
- Creates a `runs/` directory for saving logs and model checkpoints

**Two networks (Double DQN-style setup):**
- `policy_dqn` — the network actively being trained and used to select actions
- `target_dqn` — a **frozen copy** used to compute stable target Q-values, periodically synced with `policy_dqn` every `network_sync_rate` steps

**Training loop (`run(is_training=True)`):**
1. Reset the environment, get initial state
2. For each step, choose an action via **epsilon-greedy**:
   - With probability `epsilon`: pick a random action (explore)
   - Otherwise: pick `argmax` of the policy network's Q-values (exploit)
3. Step the environment, store the transition in replay memory
4. Once the buffer has more than `mini_batch_size` samples, call `optimize()` to train
5. Decay epsilon after each episode (less exploration over time)
6. If the episode's reward beats the best so far, save the model to `runs/<param_set>.pt` and log it to `runs/<param_set>.log`
7. Sync `target_dqn` with `policy_dqn` every `network_sync_rate` steps

**Optimization step (`optimize()`):**
- Unpacks a mini-batch and stacks tensors
- Computes the **target Q-value** using the Bellman equation:

  `target_q = reward + (1 - terminated) * gamma * max(target_dqn(next_state))`

- Computes the **current Q-value** from the policy network for the actions actually taken
- Loss = MSE between current and target Q-values
- Backpropagates and updates `policy_dqn` weights via Adam optimizer

**Testing mode (`is_training=False`):**
- Loads the saved model weights from `runs/<param_set>.pt`
- Sets the network to `eval()` mode
- Runs with `render=True` so you can watch it play, always choosing the greedy (best) action — no exploration

### 4. Hyperparameters (`parameters.yaml`)
The `flappybirdv0` block defines one named parameter set:

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.001 | Learning rate for Adam optimizer |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_init` | 1.0 | Starting exploration rate (100% random) |
| `epsilon_min` | 0.05 | Minimum exploration rate |
| `epsilon_decay` | 0.9995 | Multiplicative decay applied to epsilon each episode |
| `replay_memory_size` | 100000 | Max transitions stored in replay buffer |
| `mini_batch_size` | 32 | Number of samples per training step |
| `network_sync_rate` | 10 | Steps between syncing target network with policy network |
| `reward_threshold` | 1000 | Episode ends early if reward exceeds this (safety cap) |

You can add more named sets to this file (e.g., `flappybirdv0_fast`, `flappybirdv0_tuned`) and select between them via the command line.

## Installation

```bash
pip install flappy-bird-gymnasium gymnasium torch pyyaml
```

## Usage

### Train the agent

```bash
python agent.py flappybirdv0 --train
```

This runs indefinitely (via `itertools.count()`), continuously training and saving the best-performing model to `runs/flappybirdv0.pt`. Stop it manually (Ctrl+C) once you're satisfied with performance — check `runs/flappybirdv0.log` to see reward progression over time.

### Test / watch the trained agent play

```bash
python agent.py flappybirdv0
```

Since `--train` is omitted, this loads the saved model from `runs/flappybirdv0.pt`, disables exploration, and renders the game window (`render=True`) so you can watch it play live.

> The `hyperparameters` argument (`flappybirdv0`) must match a key in `parameters.yaml` — it's used both to load hyperparameters and to locate the corresponding saved model/log files.

## Known Issues Worth Fixing

- **Testing will crash if you haven't trained yet** — `runs/flappybirdv0.pt` won't exist until a model has been saved during training.
- **`ReplayMemory`'s `seed` param is unused** — reproducibility isn't actually guaranteed despite the parameter existing.
- **No `env.close()`** — the environment is never explicitly closed (there's a commented-out reminder in the code), which could leak resources over long runs.
- **Training never stops on its own** — there's no episode limit or convergence check; it's designed to run until you manually interrupt it.
- **`self.mini_batch_size` is set twice** in `Agent.__init__` (harmless but redundant).