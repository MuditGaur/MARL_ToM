#!/usr/bin/env python3
"""
Script to demonstrate trend line analysis for the MARL Theory of Mind results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_trend_analysis():
    """Demonstrate the trend line analysis approach."""
    
    print("="*80)
    print("TREND LINE ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Load the results
    results = {}
    for backend in ['classical', 'quantum', 'hybrid']:
        filename = f"results_{backend}_500episodes.npz"
        try:
            data = np.load(filename)
            results[backend] = {
                'episode_rewards': data['episode_rewards'],
                'training_losses': data['training_losses']
            }
            print(f"✓ Loaded {filename}")
        except FileNotFoundError:
            print(f"✗ {filename} not found")
            return
    
    print("\n" + "="*80)
    print("TREND ANALYSIS EXPLANATION")
    print("="*80)
    
    for backend, data in results.items():
        rewards = data['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        
        # Calculate trend line
        z = np.polyfit(episodes, rewards, 2)  # 2nd degree polynomial
        p = np.poly1d(z)
        trend_line = p(episodes)
        
        # Calculate improvement
        start_avg = np.mean(rewards[:50])  # First 50 episodes
        end_avg = np.mean(rewards[-50:])   # Last 50 episodes
        improvement = end_avg - start_avg
        
        print(f"\n{backend.upper()} APPROACH:")
        print(f"  Trend line equation: {p}")
        print(f"  Start average (episodes 1-50): {start_avg:.2f}")
        print(f"  End average (episodes 451-500): {end_avg:.2f}")
        print(f"  Overall improvement: {improvement:+.2f}")
        
        # Determine trend direction
        if z[0] > 0.001:  # Positive quadratic term
            print(f"  Trend: Accelerating improvement (convex upward)")
        elif z[0] < -0.001:  # Negative quadratic term
            print(f"  Trend: Decelerating improvement (convex downward)")
        else:
            if z[1] > 0.01:  # Positive linear term
                print(f"  Trend: Steady improvement")
            elif z[1] < -0.01:  # Negative linear term
                print(f"  Trend: Steady decline")
            else:
                print(f"  Trend: Stable performance")
    
    print("\n" + "="*80)
    print("VISUALIZATION IMPROVEMENTS")
    print("="*80)
    
    print("1. TREND LINES ADDED:")
    print("   • 2nd degree polynomial fits to show overall trends")
    print("   • Dashed lines to distinguish from raw data")
    print("   • Thicker lines (3px) for better visibility")
    
    print("\n2. RAW DATA MODIFICATIONS:")
    print("   • Reduced marker size (3px instead of 6px)")
    print("   • Lower alpha (0.6) to reduce visual noise")
    print("   • Thinner lines (1px) for raw data")
    
    print("\n3. MOVING AVERAGE IMPROVEMENTS:")
    print("   • Increased window size from 3 to 10 episodes")
    print("   • Added trend lines to moving averages")
    print("   • Better smoothing for trend visualization")
    
    print("\n4. BENEFITS:")
    print("   • Clearer visualization of learning progress")
    print("   • Easier comparison between approaches")
    print("   • Reduced visual clutter from high variance")
    print("   • Better identification of convergence patterns")
    
    # Create a simple demonstration plot
    create_demo_plot(results)

def create_demo_plot(results):
    """Create a demonstration plot showing before/after comparison."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e', 'hybrid': '#2ca02c'}
    
    # Before (original style)
    ax1.set_title('Before: Raw Data Only', fontweight='bold')
    for backend, data in results.items():
        episodes = range(1, len(data['episode_rewards']) + 1)
        ax1.plot(episodes, data['episode_rewards'], 
                marker='o', linewidth=2, markersize=6,
                color=colors.get(backend, '#000000'),
                label=f'{backend.title()}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # After (with trend lines)
    ax2.set_title('After: Raw Data + Trend Lines', fontweight='bold')
    for backend, data in results.items():
        episodes = range(1, len(data['episode_rewards']) + 1)
        rewards = data['episode_rewards']
        
        # Raw data (faded)
        ax2.plot(episodes, rewards, 
                marker='o', linewidth=1, markersize=3, alpha=0.6,
                color=colors.get(backend, '#000000'),
                label=f'{backend.title()}')
        
        # Trend line
        z = np.polyfit(episodes, rewards, 2)
        p = np.poly1d(z)
        ax2.plot(episodes, p(episodes), 
                linewidth=3, alpha=0.8,
                color=colors.get(backend, '#000000'),
                linestyle='--',
                label=f'{backend.title()} (trend)')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_analysis_demo.png', dpi=300, bbox_inches='tight')
    print(f"\nDemo plot saved as 'trend_analysis_demo.png'")
    plt.close()

if __name__ == "__main__":
    demonstrate_trend_analysis()
