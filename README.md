# MARL Theory of Mind (MARL_ToM)

A modular Multi-Agent Reinforcement Learning framework for implementing Theory of Mind capabilities in autonomous agents.

## Overview

This project implements a modular framework for studying Theory of Mind (ToM) in multi-agent reinforcement learning environments. The framework allows agents to model and predict the mental states of other agents, enabling more sophisticated social interactions and cooperation.

The project supports three different planner backends:
- **Classical**: MLP-based planner (PyTorch)
- **Quantum**: Parameterized Quantum Circuit planner (PennyLane)
- **Hybrid**: Fusion of classical and quantum outputs

## Project Structure

```
MARL_ToM/
├── src/                    # Main source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration classes
│   ├── environment.py     # Environment wrapper
│   ├── models.py          # Neural network models
│   ├── agent.py           # Agent implementation
│   ├── trainer.py         # Training logic
│   └── utils.py           # Utility functions
├── main.py                # Main script for running experiments
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MARL_ToM.git
cd MARL_ToM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Basic Example

```python
from src.config import CNConfig
from src.trainer import MARLTrainer
import torch

# Create configuration
cfg = CNConfig(n_agents=3, max_cycles=100, K=10, gamma=0.95)

# Create trainer with classical backend
trainer = MARLTrainer(
    cfg=cfg, 
    backend="classical", 
    device=torch.device("cpu"), 
    seed=0
)

# Train the model
avg_eval = trainer.train(episodes=50, eval_episodes=5)
print(f"Final evaluation score: {avg_eval:.2f}")
```

### Command Line Interface

Run experiments using the command line interface:

```bash
# Classical planner
python main.py --backend classical --episodes 50

# Quantum planner (requires PennyLane)
python main.py --backend quantum --episodes 50

# Hybrid planner (requires PennyLane)
python main.py --backend hybrid --episodes 50

# Custom configuration
python main.py --backend classical --n_agents 4 --max_cycles 150 --K 15
```

### Available Command Line Arguments

- `--backend`: Planner backend (`classical`, `quantum`, `hybrid`)
- `--episodes`: Number of training episodes (default: 50)
- `--eval_episodes`: Number of evaluation episodes (default: 5)
- `--n_agents`: Number of agents/landmarks (default: 3)
- `--max_cycles`: Episode length (default: 100)
- `--K`: Planner replan interval (default: 10)
- `--gamma`: Discount factor (default: 0.95)
- `--lr`: Learning rate (default: 3e-4)
- `--seed`: Random seed (default: 0)
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)
- `--log_interval`: Logging interval in episodes (default: 10)

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each component
- **Theory of Mind Models**: Various implementations of ToM reasoning
- **Multiple Planner Backends**: Classical, quantum, and hybrid planners
- **Cooperative Navigation Environment**: Based on PettingZoo MPE simple_spread_v3
- **Configurable Parameters**: Easy customization of training parameters
- **Comprehensive Logging**: Detailed training statistics and metrics

## Dependencies

### Required
- `torch>=1.9.0`: PyTorch for deep learning
- `numpy>=1.21.0`: Numerical computing
- `pettingzoo>=1.24.0`: Multi-agent environments
- `mpe>=1.0.3`: Multi-Agent Particle Environment

### Optional
- `pennylane>=0.28.0`: Quantum computing (required for quantum/hybrid backends)

### Development
- `pytest>=6.0`: Testing framework
- `black>=21.0`: Code formatting
- `flake8>=3.8`: Linting
- `mypy>=0.800`: Type checking

## Environment

The framework uses the Cooperative Navigation environment from PettingZoo MPE (`simple_spread_v3`). In this environment:

- Multiple agents must navigate to different landmarks
- Agents must avoid collisions with each other
- The goal is to minimize the total distance between agents and their assigned landmarks
- Agents can observe their own position, velocity, and relative positions of other agents and landmarks

## Theory of Mind Components

### ToMNet
- Predicts the closest landmark and goal of other agents
- Uses relative position features between agents

### Router
- Implements peer-to-peer message gating with Gumbel-Softmax
- Determines which messages to keep or drop during communication

### Planners
- **Classical**: Multi-layer perceptron for action selection
- **Quantum**: Parameterized quantum circuit with angle embedding
- **Hybrid**: Fusion of classical and quantum outputs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{marl_tom_2024,
  title={Multi-Agent Reinforcement Learning with Theory of Mind},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Contact

- Email: your.email@example.com
- GitHub Issues: [https://github.com/yourusername/MARL_ToM/issues](https://github.com/yourusername/MARL_ToM/issues)
