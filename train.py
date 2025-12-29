"""
Echelon Chess Engine - Training Script
Optimized for Google Colab T4 and Kaggle Free Tier GPUs

Usage:
    python train.py --iterations 100 --games_per_iter 50
    
For Colab/Kaggle:
    !python train.py --iterations 50 --games_per_iter 25 --batch_size 256
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from datetime import datetime

from model import EchelonNet, count_parameters
from selfplay import SelfPlayWorker, ReplayBuffer
from mcts import MCTS


class Trainer:
    """
    Handles the training loop for Echelon.
    Implements the AlphaZero-style training pipeline.
    """
    def __init__(self, config):
        self.config = config
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("WARNING: No GPU detected, training will be slow!")
        
        # Create model
        self.model = EchelonNet(
            in_channels=config['in_channels'],
            num_res_blocks=config['num_res_blocks'],
            num_filters=config['num_filters']
        ).to(self.device)
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config['lr_step_size'], 
            gamma=config['lr_gamma']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config['buffer_size'])
        
        # Training stats
        self.iteration = 0
        self.total_games = 0
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def self_play_iteration(self, num_games):
        """Generate self-play games and add to replay buffer."""
        self.model.eval()
        
        # Move model to CPU for self-play (more stable with MCTS)
        model_cpu = EchelonNet(
            in_channels=self.config['in_channels'],
            num_res_blocks=self.config['num_res_blocks'],
            num_filters=self.config['num_filters']
        )
        model_cpu.load_state_dict(self.model.state_dict())
        model_cpu.eval()
        
        worker = SelfPlayWorker(
            model=model_cpu,
            num_simulations=self.config['mcts_simulations'],
            temperature_threshold=self.config['temperature_threshold'],
            max_game_length=self.config['max_game_length']
        )
        
        examples = worker.generate_games(num_games, verbose=False)
        self.replay_buffer.add_examples(examples)
        self.total_games += num_games
        
        return len(examples)
    
    def train_iteration(self, num_batches):
        """Train on samples from the replay buffer."""
        self.model.train()
        
        if len(self.replay_buffer) < self.config['min_buffer_size']:
            print(f"Buffer too small ({len(self.replay_buffer)}), need {self.config['min_buffer_size']}")
            return None
        
        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        
        for batch_idx in range(num_batches):
            # Sample batch
            positions, policies, values = self.replay_buffer.sample(self.config['batch_size'])
            
            # Move to device
            positions = positions.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_values, pred_policies = self.model(positions)
            
            # Compute losses
            value_loss = nn.MSELoss()(pred_values, values)
            
            # Policy loss (cross-entropy with softmax)
            policy_loss = -torch.mean(torch.sum(policies * nn.functional.log_softmax(pred_policies, dim=1), dim=1))
            
            # Combined loss
            loss = value_loss + policy_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
        
        # Step scheduler
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        
        return {
            'total': avg_loss,
            'value': avg_value_loss,
            'policy': avg_policy_loss
        }
    
    def save_checkpoint(self, filename=None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_iter_{self.iteration}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'iteration': self.iteration,
            'total_games': self.total_games,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, filepath)
        
        print(f"Saved checkpoint: {filepath}")
        
        # Also save as 'latest.pt'
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save({
            'iteration': self.iteration,
            'total_games': self.total_games,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, latest_path)
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        print(f"Loading checkpoint: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Resumed from iteration {self.iteration}, games: {self.total_games}")
    
    def train(self, num_iterations, games_per_iter, batches_per_iter):
        """Main training loop."""
        print("\n" + "="*70)
        print("ECHELON TRAINING")
        print("="*70)
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iter}")
        print(f"Batches per iteration: {batches_per_iter}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"MCTS simulations: {self.config['mcts_simulations']}")
        print("="*70 + "\n")
        
        for i in range(num_iterations):
            self.iteration += 1
            iter_start = time.time()
            
            print(f"\n--- Iteration {self.iteration} ---")
            
            # Self-play phase
            print(f"[1/2] Generating {games_per_iter} self-play games...")
            sp_start = time.time()
            num_examples = self.self_play_iteration(games_per_iter)
            sp_time = time.time() - sp_start
            print(f"      Generated {num_examples} positions in {sp_time:.1f}s")
            print(f"      Buffer size: {len(self.replay_buffer)}")
            
            # Training phase
            print(f"[2/2] Training on {batches_per_iter} batches...")
            train_start = time.time()
            losses = self.train_iteration(batches_per_iter)
            train_time = time.time() - train_start
            
            if losses:
                print(f"      Loss: {losses['total']:.4f} (value: {losses['value']:.4f}, policy: {losses['policy']:.4f})")
                print(f"      Training time: {train_time:.1f}s")
                
                # Save best model
                if losses['total'] < self.best_loss:
                    self.best_loss = losses['total']
                    self.save_checkpoint('best.pt')
            
            iter_time = time.time() - iter_start
            print(f"      Iteration time: {iter_time:.1f}s")
            
            # Save periodic checkpoint
            if self.iteration % self.config['save_interval'] == 0:
                self.save_checkpoint()
            
            # Save buffer periodically
            if self.iteration % self.config['buffer_save_interval'] == 0:
                buffer_path = os.path.join(self.checkpoint_dir, f'buffer_iter_{self.iteration}.pkl')
                self.replay_buffer.save(buffer_path)
        
        # Final save
        self.save_checkpoint('final.pt')
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print(f"Total iterations: {self.iteration}")
        print(f"Total games: {self.total_games}")
        print(f"Best loss: {self.best_loss:.4f}")
        print("="*70)


def get_default_config():
    """Get default training configuration optimized for T4 GPU."""
    return {
        # Model architecture
        'in_channels': 13,
        'num_res_blocks': 9,      # ResNet-20 (9 blocks * 2 + 2 = 20 layers)
        'num_filters': 256,       # 256 filters as you specified
        
        # Training
        'batch_size': 256,        # Good for T4 with 256 filters
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'lr_step_size': 50,       # Decay LR every 50 iterations
        'lr_gamma': 0.5,
        
        # Replay buffer
        'buffer_size': 500000,
        'min_buffer_size': 1000,
        
        # MCTS
        'mcts_simulations': 200,  # Reduced for faster self-play
        'temperature_threshold': 30,
        'max_game_length': 512,
        
        # Checkpointing
        'checkpoint_dir': 'checkpoints',
        'save_interval': 10,
        'buffer_save_interval': 25,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Echelon Chess Engine')
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--games_per_iter', type=int, default=25, help='Self-play games per iteration')
    parser.add_argument('--batches_per_iter', type=int, default=100, help='Training batches per iteration')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--mcts_sims', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--num_filters', type=int, default=256, help='Number of filters in ResNet')
    parser.add_argument('--num_blocks', type=int, default=9, help='Number of residual blocks')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Get config
    config = get_default_config()
    config['batch_size'] = args.batch_size
    config['mcts_simulations'] = args.mcts_sims
    config['num_filters'] = args.num_filters
    config['num_res_blocks'] = args.num_blocks
    config['checkpoint_dir'] = args.checkpoint_dir
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        batches_per_iter=args.batches_per_iter
    )


if __name__ == '__main__':
    main()
