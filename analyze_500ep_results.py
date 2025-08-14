#!/usr/bin/env python3
"""
Analysis script for the 500-episode comparison results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(filename):
    """Load results from npz file."""
    data = np.load(filename)
    return {
        'episode_rewards': data['episode_rewards'],
        'training_losses': data['training_losses'],
        'tom_losses': data['tom_losses'],
        'cr_losses': data['cr_losses'],
        'timestamps': data['timestamps'],
        'backend': str(data['backend'])
    }

def analyze_results():
    """Analyze the 500-episode results."""
    print("="*80)
    print("ANALYSIS OF 500-EPISODE MARL THEORY OF MIND COMPARISON")
    print("="*80)
    
    # Load all results
    results = {}
    for backend in ['classical', 'quantum', 'hybrid']:
        filename = f"results_{backend}_500episodes.npz"
        try:
            results[backend] = load_results(filename)
            print(f"✓ Loaded {filename}")
        except FileNotFoundError:
            print(f"✗ {filename} not found")
            return
    
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    for backend, data in results.items():
        rewards = data['episode_rewards']
        losses = data['training_losses']
        times = data['timestamps']
        
        print(f"\n{backend.upper()} APPROACH:")
        print(f"  Final Episode Reward: {rewards[-1]:.2f}")
        print(f"  Average Reward (all episodes): {np.mean(rewards):.2f}")
        print(f"  Best Episode: {np.max(rewards):.2f}")
        print(f"  Worst Episode: {np.min(rewards):.2f}")
        print(f"  Standard Deviation: {np.std(rewards):.2f}")
        print(f"  Final 100 Episodes Avg: {np.mean(rewards[-100:]):.2f}")
        print(f"  Final 50 Episodes Avg: {np.mean(rewards[-50:]):.2f}")
        print(f"  Total Training Time: {times[-1]:.1f}s")
        print(f"  Final Loss: {losses[-1]:.3f}")
        print(f"  Average Loss: {np.mean(losses):.3f}")
        
        # Learning progress analysis
        first_quarter = np.mean(rewards[:125])
        second_quarter = np.mean(rewards[125:250])
        third_quarter = np.mean(rewards[250:375])
        fourth_quarter = np.mean(rewards[375:])
        
        print(f"  Learning Progress:")
        print(f"    Episodes 1-125:   {first_quarter:.2f}")
        print(f"    Episodes 126-250: {second_quarter:.2f}")
        print(f"    Episodes 251-375: {third_quarter:.2f}")
        print(f"    Episodes 376-500: {fourth_quarter:.2f}")
        
        improvement = fourth_quarter - first_quarter
        print(f"    Overall Improvement: {improvement:+.2f}")
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Compare final performances
    final_rewards = [results[b]['episode_rewards'][-1] for b in results.keys()]
    avg_rewards = [np.mean(results[b]['episode_rewards']) for b in results.keys()]
    best_rewards = [np.max(results[b]['episode_rewards']) for b in results.keys()]
    training_times = [results[b]['timestamps'][-1] for b in results.keys()]
    
    backends = list(results.keys())
    
    print(f"\nFinal Episode Performance Ranking:")
    for i, (backend, reward) in enumerate(sorted(zip(backends, final_rewards), key=lambda x: x[1], reverse=True)):
        print(f"  {i+1}. {backend.title()}: {reward:.2f}")
    
    print(f"\nAverage Performance Ranking:")
    for i, (backend, avg) in enumerate(sorted(zip(backends, avg_rewards), key=lambda x: x[1], reverse=True)):
        print(f"  {i+1}. {backend.title()}: {avg:.2f}")
    
    print(f"\nBest Episode Performance Ranking:")
    for i, (backend, best) in enumerate(sorted(zip(backends, best_rewards), key=lambda x: x[1], reverse=True)):
        print(f"  {i+1}. {backend.title()}: {best:.2f}")
    
    print(f"\nTraining Time Ranking (faster is better):")
    for i, (backend, time) in enumerate(sorted(zip(backends, training_times), key=lambda x: x[1])):
        print(f"  {i+1}. {backend.title()}: {time:.1f}s")
    
    # Key insights
    print(f"\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    best_final = max(final_rewards)
    best_avg = max(avg_rewards)
    best_single = max(best_rewards)
    fastest = min(training_times)
    
    print(f"• Best final episode performance: {best_final:.2f}")
    print(f"• Best average performance: {best_avg:.2f}")
    print(f"• Best single episode: {best_single:.2f}")
    print(f"• Fastest training: {fastest:.1f}s")
    
    # Performance vs time efficiency
    print(f"\nPerformance vs Time Efficiency Analysis:")
    for backend in backends:
        data = results[backend]
        efficiency = np.mean(data['episode_rewards']) / data['timestamps'][-1]
        print(f"  {backend.title()}: {efficiency:.4f} reward/second")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if max(final_rewards) == final_rewards[backends.index('hybrid')]:
        print("• HYBRID approach shows the best final performance")
    elif max(final_rewards) == final_rewards[backends.index('quantum')]:
        print("• QUANTUM approach shows the best final performance")
    else:
        print("• CLASSICAL approach shows the best final performance")
    
    if min(training_times) == training_times[backends.index('classical')]:
        print("• CLASSICAL approach is the fastest")
    
    print("• Consider running longer training for more stable convergence")
    print("• The hybrid approach shows promise for combining classical and quantum advantages")

if __name__ == "__main__":
    analyze_results()
