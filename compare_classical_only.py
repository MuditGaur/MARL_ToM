#!/usr/bin/env python3
"""
Simplified comparison script for MARL Theory of Mind (Classical only).

This script runs only the classical approach for testing purposes.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.config import CNConfig
from src.trainer import MARLTrainer
from src.utils import set_seed


def run_classical_test(episodes: int = 10, n_agents: int = 3, max_cycles: int = 50):
    """Run classical approach test."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    cfg = CNConfig(
        n_agents=n_agents,
        max_cycles=max_cycles,
        K=10,
        gamma=0.95
    )
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Store results
    episode_rewards = []
    training_losses = []
    timestamps = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training CLASSICAL approach")
    print(f"{'='*60}")
    
    try:
        # Create trainer
        trainer = MARLTrainer(
            cfg=cfg,
            backend="classical",
            device=device,
            seed=42,
            lr=3e-4,
            tau_cr=0.02
        )
        
        # Train with tracking
        for ep in range(1, episodes + 1):
            # Collect episode
            out = trainer.collect_episode(train=True)
            ep_rew = out["episode_reward"]
            
            # Update model
            stats = trainer.update(out["traj"])
            
            # Store results
            episode_rewards.append(ep_rew)
            training_losses.append(stats['loss_total'])
            timestamps.append(time.time() - start_time)
            
            # Log progress
            avg_last = np.mean(episode_rewards[-min(ep, 5):])
            elapsed = timestamps[-1]
            print(f"Ep {ep:2d}/{episodes} | R={ep_rew:7.2f} | Avg5={avg_last:7.2f} | "
                  f"Loss={stats['loss_total']:.3f} | Time={elapsed:.1f}s")
        
        final_avg = np.mean(episode_rewards)
        print(f"\nCLASSICAL Final Average Reward: {final_avg:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'training_losses': training_losses,
            'timestamps': timestamps
        }
        
    except Exception as e:
        print(f"ERROR: Failed to run classical approach: {e}")
        return None


def plot_results(results: dict):
    """Plot the results."""
    
    if not results:
        print("No results to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MARL Theory of Mind: Classical Approach Performance', fontsize=14, fontweight='bold')
    
    episodes = range(1, len(results['episode_rewards']) + 1)
    
    # 1. Episode Rewards
    ax1 = axes[0]
    ax1.plot(episodes, results['episode_rewards'], 
            marker='o', linewidth=2, markersize=6, 
            color='#1f77b4', label='Episode Reward')
    
    # Add moving average
    window = 3
    if len(results['episode_rewards']) >= window:
        moving_avg = [np.mean(results['episode_rewards'][max(0, i-window):i+1]) 
                     for i in range(len(results['episode_rewards']))]
        ax1.plot(episodes, moving_avg, 
                marker='s', linewidth=2, markersize=6,
                color='#ff7f0e', label=f'Moving Average (window={window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Loss
    ax2 = axes[1]
    ax2.plot(episodes, results['training_losses'], 
            marker='^', linewidth=2, markersize=6,
            color='#2ca02c', label='Training Loss')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Final Episode Reward: {results['episode_rewards'][-1]:.2f}")
    print(f"Average Reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"Best Episode: {max(results['episode_rewards']):.2f}")
    print(f"Total Training Time: {results['timestamps'][-1]:.1f}s")
    print(f"Final Loss: {results['training_losses'][-1]:.3f}")


def main():
    """Main function."""
    
    print("MARL Theory of Mind: Classical Approach Test")
    print("="*50)
    
    # Run test with small number of episodes
    episodes = 10
    n_agents = 3
    max_cycles = 50
    
    print(f"Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Agents: {n_agents}")
    print(f"  Max cycles per episode: {max_cycles}")
    print(f"  Replan interval: 10")
    
    # Run the test
    results = run_classical_test(
        episodes=episodes,
        n_agents=n_agents,
        max_cycles=max_cycles
    )
    
    # Plot results
    if results:
        plot_results(results)
        
        # Save results
        print(f"\nSaving results...")
        filename = f"results_classical_{episodes}episodes.npz"
        np.savez(filename,
                 episode_rewards=results['episode_rewards'],
                 training_losses=results['training_losses'],
                 timestamps=results['timestamps'])
        print(f"  Saved {filename}")
    else:
        print("No results to plot!")


if __name__ == "__main__":
    main()
