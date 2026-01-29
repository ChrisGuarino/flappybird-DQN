# Flappy Bird DQN

A Deep Q-Network (DQN) reinforcement learning agent that learns to play Flappy Bird, implemented in PyTorch.

[Demo Video](https://www.youtube.com/watch?v=RVMpm86equc&ab_channel=JohnnyCode)

## Overview

This project implements several DQN variants to train an agent on Gymnasium environments, with a focus on the Flappy Bird game. The agent uses experience replay and epsilon-greedy exploration to learn optimal policies.

## DQN Variants

| Variant | Description |
|---------|-------------|
| **Standard DQN** | Basic deep Q-learning with target network |
| **Double DQN** | Decouples action selection from evaluation to reduce overestimation |
| **Dueling DQN** | Separates value and advantage streams for more stable learning |

## Supported Environments

- **FlappyBird-v0** — Primary target (Double DQN + Dueling, 512 hidden nodes)
- **CartPole-v1** — Classic control benchmark
- **Acrobot-v1** — Underactuated pendulum control

Hyperparameters for each environment are defined in `hyperparameters.yml`.

## Requirements

- Python 3
- PyTorch
- Gymnasium
- flappy-bird-gymnasium
- NumPy
- Matplotlib
- PyYAML

## Usage

```bash
# Train on Flappy Bird
python agent.py --env FlappyBird-v0

# Test a trained model
python agent.py --env FlappyBird-v0 --test

# Train on CartPole
python agent.py --env CartPole-v1
```

## Project Structure

```
flappybird_dqn/
├── agent.py                # Training loop and DQN agent
├── dqn.py                  # Neural network architecture (Standard/Dueling)
├── experience_replay.py    # Replay buffer implementation
├── hyperparameters.yml     # Per-environment configuration
├── models/                 # Saved model checkpoints (.pt)
└── graphs/                 # Training progress plots
```

## Training Output

The agent periodically saves:
- Best-performing model weights to `models/`
- Reward and epsilon decay graphs to `graphs/`
- Timestamped training logs to the console
