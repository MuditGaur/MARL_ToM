# MARL Theory of Mind: Performance Comparison Scripts

This directory contains scripts to compare the performance of different approaches (classical, quantum, hybrid) in the MARL Theory of Mind framework.

## Scripts Overview

### 1. `compare_approaches.py` - Full Comparison
Runs all three approaches (classical, quantum, hybrid) and creates comprehensive performance plots.

**Features:**
- Runs classical, quantum, and hybrid approaches
- Tracks episode rewards, training losses, and timing
- Creates 4-panel comparison plots
- Saves results to NPZ files
- Handles PennyLane availability automatically

**Usage:**
```bash
python compare_approaches.py
```

**Requirements:**
- All dependencies from `requirements.txt`
- PennyLane for quantum/hybrid approaches (optional)

### 2. `compare_classical_only.py` - Classical Only
Simplified version that runs only the classical approach for testing.

**Features:**
- Runs only classical approach
- Basic performance tracking
- Simple 2-panel plots
- Good for initial testing

**Usage:**
```bash
python compare_classical_only.py
```

**Requirements:**
- Basic dependencies (torch, numpy, matplotlib, pettingzoo)

## Configuration

Both scripts use the following default configuration:
- **Episodes**: 10 (small for quick testing)
- **Agents**: 3
- **Max cycles per episode**: 50
- **Replan interval**: 10
- **Learning rate**: 3e-4
- **Random seed**: 42 (for reproducibility)

## Expected Runtime

### Hardware: Intel i7-12700K + RTX 3080 Ti

**Classical approach only:**
- 10 episodes: ~30-60 seconds
- 50 episodes: ~2-5 minutes

**Full comparison (all three approaches):**
- 10 episodes each: ~2-8 minutes
- 50 episodes each: ~10-30 minutes

*Note: Quantum and hybrid approaches are significantly slower due to quantum circuit simulation.*

## Output

### Plots Generated

**Full Comparison (`compare_approaches.py`):**
1. **Episode Rewards Over Time** - Raw episode rewards for each approach
2. **Moving Average Reward** - Smoothed performance trends
3. **Training Loss Over Time** - Loss convergence comparison
4. **Final Performance Summary** - Bar chart comparing final metrics

**Classical Only (`compare_classical_only.py`):**
1. **Episode Rewards Over Time** - With moving average
2. **Training Loss Over Time** - Loss convergence

### Data Files

Results are automatically saved as NPZ files:
- `results_classical_10episodes.npz`
- `results_quantum_10episodes.npz`
- `results_hybrid_10episodes.npz`

These contain:
- Episode rewards
- Training losses
- ToM losses
- Communication reduction losses
- Timestamps
- Backend information

## Customization

### Modify Configuration

Edit the parameters in the `main()` function:

```python
# In compare_approaches.py or compare_classical_only.py
episodes = 10      # Number of training episodes
n_agents = 3       # Number of agents/landmarks
max_cycles = 50    # Episode length
```

### Add Custom Metrics

Extend the `PerformanceTracker` class to track additional metrics:

```python
class PerformanceTracker:
    def __init__(self, backend_name: str):
        # ... existing code ...
        self.custom_metric = []
    
    def add_episode(self, episode_reward: float, stats: dict):
        # ... existing code ...
        self.custom_metric.append(stats.get('custom_metric', 0.0))
```

## Troubleshooting

### PennyLane Not Available
If you get an error about PennyLane:
```bash
pip install pennylane
```

### Environment Import Error
If PettingZoo MPE is not available:
```bash
pip install pettingzoo mpe
```

### GPU Memory Issues
If you encounter GPU memory issues:
1. Reduce `max_cycles` (episode length)
2. Reduce `n_agents` (number of agents)
3. Use CPU instead of GPU by modifying the device selection

### Performance Issues
- Quantum approaches are inherently slower due to quantum circuit simulation
- Consider using fewer episodes for initial testing
- The classical approach is the fastest and most stable

## Example Output

```
MARL Theory of Mind: Approach Comparison
==================================================
Configuration:
  Episodes: 10
  Agents: 3
  Max cycles per episode: 50
  Replan interval: 10
Using device: cuda

============================================================
Training CLASSICAL approach
============================================================
Ep  1/10 | R= -45.23 | Avg5= -45.23 | Loss=2.456 | Time=3.2s
Ep  2/10 | R= -38.91 | Avg5= -42.07 | Loss=2.123 | Time=6.1s
...

============================================================
Training QUANTUM approach
============================================================
Ep  1/10 | R= -47.12 | Avg5= -47.12 | Loss=2.789 | Time=12.3s
...

PERFORMANCE SUMMARY
============================================================
CLASSICAL:
  Final Episode Reward: -25.34
  Average Reward: -32.45
  Best Episode: -18.67
  Total Training Time: 45.2s
  Final Loss: 1.234
```

## Next Steps

1. **Start with classical only** to verify everything works
2. **Install PennyLane** if you want to test quantum approaches
3. **Increase episodes** for more meaningful comparisons
4. **Experiment with different configurations** (agents, episode length, etc.)
5. **Analyze the saved NPZ files** for detailed analysis
