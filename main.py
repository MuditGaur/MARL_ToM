#!/usr/bin/env python3
"""
Main script for running MARL Theory of Mind experiments.

This script provides a command-line interface for training and evaluating
different planner backends (classical, quantum, hybrid) in the cooperative
navigation environment.
"""

import argparse
import sys

import torch

from src.config import CNConfig
from src.trainer import MARLTrainer
from src.utils import set_seed


def main():
    """Main function for running MARL ToM experiments."""
    parser = argparse.ArgumentParser(description="MARL Theory of Mind Experiments")
    parser.add_argument("--backend", type=str, default="classical", 
                       choices=["classical", "quantum", "hybrid"],
                       help="Planner backend to use")
    parser.add_argument("--episodes", type=int, default=50, 
                       help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=5, 
                       help="Number of evaluation episodes after training")
    parser.add_argument("--n_agents", type=int, default=3, 
                       help="Number of agents/landmarks")
    parser.add_argument("--max_cycles", type=int, default=100, 
                       help="Episode length (environment steps)")
    parser.add_argument("--K", type=int, default=10, 
                       help="Planner replan interval")
    parser.add_argument("--gamma", type=float, default=0.95, 
                       help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, 
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, 
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--log_interval", type=int, default=10, 
                       help="Logging interval (episodes)")
    args = parser.parse_args()

    # Check for PennyLane if using quantum backends
    if args.backend in ("quantum", "hybrid"):
        try:
            import pennylane
        except ImportError:
            print("ERROR: --backend quantum/hybrid requires PennyLane. "
                  "Please install it with: `pip install pennylane`")
            sys.exit(1)

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Set random seed
    set_seed(args.seed)

    # Create configuration
    cfg = CNConfig(
        n_agents=args.n_agents, 
        max_cycles=args.max_cycles, 
        K=args.K, 
        gamma=args.gamma
    )

    # Create and train the model
    print(f"Starting training with backend: {args.backend}")
    print(f"Configuration: {cfg}")
    print(f"Device: {device}")
    
    trainer = MARLTrainer(
        cfg=cfg, 
        backend=args.backend, 
        device=device, 
        seed=args.seed, 
        lr=args.lr, 
        tau_cr=0.02, 
        K=args.K
    )
    
    avg_eval = trainer.train(
        episodes=args.episodes, 
        log_interval=args.log_interval, 
        eval_episodes=args.eval_episodes
    )
    
    print(f"Training completed. Backend={args.backend}, Final Eval Avg Return={avg_eval:.2f}")


if __name__ == "__main__":
    main()
