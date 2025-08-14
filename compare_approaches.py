#!/usr/bin/env python3
"""
Comparison script for MARL Theory of Mind approaches.

This script runs the classical, quantum, and hybrid approaches for a small number
of episodes and plots their performances for comparison.
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.config import CNConfig
from src.trainer import MARLTrainer
from src.utils import set_seed


class PerformanceTracker:
    """Track performance metrics during training."""
    
    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.episode_rewards = []
        self.training_losses = []
        self.tom_losses = []
        self.cr_losses = []
        self.timestamps = []
        self.start_time = time.time()
    
    def add_episode(self, episode_reward: float, stats: dict):
        """Add episode results to tracker."""
        self.episode_rewards.append(episode_reward)
        self.training_losses.append(stats.get('loss_total', 0.0))
        self.tom_losses.append(stats.get('loss_tom', 0.0))
        self.cr_losses.append(stats.get('loss_cr', 0.0))
        self.timestamps.append(time.time() - self.start_time)
    
    def get_avg_reward(self, window: int = None) -> float:
        """Get average reward over specified window."""
        if window is None:
            return np.mean(self.episode_rewards)
        return np.mean(self.episode_rewards[-window:])


class CustomTrainer(MARLTrainer):
    """Extended trainer with performance tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_tracker = PerformanceTracker(self.backend)
    
    def train_with_tracking(self, episodes: int = 10, log_interval: int = 1):
        """Train with performance tracking."""
        print(f"\n{'='*60}")
        print(f"Training {self.backend.upper()} approach")
        print(f"{'='*60}")
        
        # Determine log interval based on total episodes
        if episodes <= 50:
            log_interval = 1
        elif episodes <= 200:
            log_interval = 5
        elif episodes <= 500:
            log_interval = 10
        else:
            log_interval = 20
            
        print(f"Progress will be shown every {log_interval} episodes")
        print(f"{'='*60}")
        
        for ep in range(1, episodes + 1):
            # Collect episode
            out = self.collect_episode(train=True)
            ep_rew = out["episode_reward"]
            
            # Update model
            stats = self.update(out["traj"])
            
            # Track performance
            self.performance_tracker.add_episode(ep_rew, stats)
            
            # Log progress
            if ep % log_interval == 0 or ep == 1 or ep == episodes:
                avg_last = self.performance_tracker.get_avg_reward(min(ep, 10))
                elapsed = self.performance_tracker.timestamps[-1]
                progress = (ep / episodes) * 100
                print(f"Ep {ep:3d}/{episodes} ({progress:5.1f}%) | R={ep_rew:7.2f} | Avg10={avg_last:7.2f} | "
                      f"Loss={stats['loss_total']:.3f} | Time={elapsed:.1f}s")
                
                # Show best performance so far
                best_ep = max(self.performance_tracker.episode_rewards)
                print(f"    Best episode so far: {best_ep:.2f}")
        
        final_avg = self.performance_tracker.get_avg_reward()
        print(f"\n{self.backend.upper()} Final Average Reward: {final_avg:.2f}")
        return self.performance_tracker


def check_pennylane_availability():
    """Check if PennyLane is available for quantum approaches."""
    try:
        import pennylane
        return True
    except ImportError:
        return False


def run_comparison(episodes: int = 10, n_agents: int = 3, max_cycles: int = 50):
    """Run comparison of all three approaches."""
    
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
    
    # Check PennyLane availability
    has_pennylane = check_pennylane_availability()
    if not has_pennylane:
        print("WARNING: PennyLane not available. Skipping quantum and hybrid approaches.")
        print("Install with: pip install pennylane")
    
    # Define approaches to test
    approaches = ["classical"]
    if has_pennylane:
        approaches.extend(["quantum", "hybrid"])
    
    # Store results
    results = {}
    
    # Run each approach
    for backend in approaches:
        print(f"\n{'='*60}")
        print(f"Starting {backend.upper()} approach")
        print(f"{'='*60}")
        
        try:
            # Create trainer
            trainer = CustomTrainer(
                cfg=cfg,
                backend=backend,
                device=device,
                seed=42,
                lr=3e-4,
                tau_cr=0.02
            )
            
            # Train with tracking
            tracker = trainer.train_with_tracking(episodes=episodes, log_interval=1)
            results[backend] = tracker
            
        except Exception as e:
            print(f"ERROR: Failed to run {backend} approach: {e}")
            continue
    
    return results


def plot_performance_comparison(results: dict):
    """Plot performance comparison of all approaches."""
    
    if not results:
        print("No results to plot!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MARL Theory of Mind: Performance Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different approaches
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e', 'hybrid': '#2ca02c'}
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    for backend, tracker in results.items():
        episodes = range(1, len(tracker.episode_rewards) + 1)
        rewards = tracker.episode_rewards
        
        # Plot individual episode rewards
        ax1.plot(episodes, rewards, 
                marker='o', linewidth=1, markersize=3, alpha=0.6,
                color=colors.get(backend, '#000000'),
                label=f'{backend.title()}')
        
        # Add trend line (polynomial fit)
        if len(rewards) > 3:
            z = np.polyfit(episodes, rewards, 2)  # 2nd degree polynomial
            p = np.poly1d(z)
            ax1.plot(episodes, p(episodes), 
                    linewidth=3, alpha=0.8,
                    color=colors.get(backend, '#000000'),
                    linestyle='--',
                    label=f'{backend.title()} (trend)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Episode Rewards Over Time (with Trend Lines)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average Rewards (moving window)
    ax2 = axes[0, 1]
    window = 10  # Increased window for smoother average
    for backend, tracker in results.items():
        episodes = range(1, len(tracker.episode_rewards) + 1)
        if len(tracker.episode_rewards) >= window:
            moving_avg = [np.mean(tracker.episode_rewards[max(0, i-window):i+1]) 
                         for i in range(len(tracker.episode_rewards))]
            
            # Plot moving average
            ax2.plot(episodes, moving_avg, 
                    marker='s', linewidth=2, markersize=4, alpha=0.7,
                    color=colors.get(backend, '#000000'),
                    label=f'{backend.title()} (avg{window})')
            
            # Add trend line for moving average
            if len(moving_avg) > 3:
                z = np.polyfit(episodes, moving_avg, 2)
                p = np.poly1d(z)
                ax2.plot(episodes, p(episodes), 
                        linewidth=3, alpha=0.8,
                        color=colors.get(backend, '#000000'),
                        linestyle='--',
                        label=f'{backend.title()} (trend)')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title(f'Moving Average Reward (window={window}) with Trend Lines')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Losses
    ax3 = axes[1, 0]
    for backend, tracker in results.items():
        episodes = range(1, len(tracker.training_losses) + 1)
        losses = tracker.training_losses
        
        # Plot individual losses
        ax3.plot(episodes, losses, 
                marker='^', linewidth=1, markersize=3, alpha=0.6,
                color=colors.get(backend, '#000000'),
                label=f'{backend.title()}')
        
        # Add trend line for losses
        if len(losses) > 3:
            z = np.polyfit(episodes, losses, 2)
            p = np.poly1d(z)
            ax3.plot(episodes, p(episodes), 
                    linewidth=3, alpha=0.8,
                    color=colors.get(backend, '#000000'),
                    linestyle='--',
                    label=f'{backend.title()} (trend)')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Training Loss Over Time (with Trend Lines)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Performance Summary
    ax4 = axes[1, 1]
    backends = list(results.keys())
    final_avg_rewards = [results[b].get_avg_reward() for b in backends]
    final_rewards = [results[b].episode_rewards[-1] for b in backends]
    
    x = np.arange(len(backends))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, final_avg_rewards, width, 
                    label='Average Reward', alpha=0.8)
    bars2 = ax4.bar(x + width/2, final_rewards, width, 
                    label='Final Episode Reward', alpha=0.8)
    
    # Color the bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        color = colors.get(backends[i], '#000000')
        bar1.set_color(color)
        bar2.set_color(color)
    
    ax4.set_xlabel('Approach')
    ax4.set_ylabel('Reward')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([b.title() for b in backends])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'performance_comparison.png'")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    for backend, tracker in results.items():
        print(f"\n{backend.upper()}:")
        print(f"  Final Episode Reward: {tracker.episode_rewards[-1]:.2f}")
        print(f"  Average Reward: {tracker.get_avg_reward():.2f}")
        print(f"  Best Episode: {max(tracker.episode_rewards):.2f}")
        print(f"  Total Training Time: {tracker.timestamps[-1]:.1f}s")
        print(f"  Final Loss: {tracker.training_losses[-1]:.3f}")


def main():
    """Main function to run the comparison."""
    
    print("MARL Theory of Mind: Approach Comparison")
    print("="*50)
    
    # Run comparison with large number of episodes
    episodes = 500
    n_agents = 3
    max_cycles = 50
    
    print(f"Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Agents: {n_agents}")
    print(f"  Max cycles per episode: {max_cycles}")
    print(f"  Replan interval: 10")
    
    # Run the comparison
    results = run_comparison(
        episodes=episodes,
        n_agents=n_agents,
        max_cycles=max_cycles
    )
    
    # Plot results
    if results:
        plot_performance_comparison(results)
        
        # Save results
        print(f"\nSaving results...")
        for backend, tracker in results.items():
            filename = f"results_{backend}_{episodes}episodes.npz"
            np.savez(filename,
                     episode_rewards=tracker.episode_rewards,
                     training_losses=tracker.training_losses,
                     tom_losses=tracker.tom_losses,
                     cr_losses=tracker.cr_losses,
                     timestamps=tracker.timestamps,
                     backend=backend)
            print(f"  Saved {filename}")
    else:
        print("No results to plot!")


if __name__ == "__main__":
    main()
