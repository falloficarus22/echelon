"""train_cpp.py

Thin wrapper around `train.py` trainer that applies defaults suitable for
the C++ backend and provides the same CLI as `train.py`.
"""
import argparse
import os

from train import Trainer, get_default_config


def main():
    parser = argparse.ArgumentParser(description='Train Echelon (C++-tuned defaults)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--games_per_iter', type=int, default=50, help='Self-play games per iteration')
    parser.add_argument('--batches_per_iter', type=int, default=100, help='Training batches per iteration')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--mcts_sims', type=int, default=400, help='MCTS simulations per move (C++ backend able to run more)')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters in ResNet')
    parser.add_argument('--num_blocks', type=int, default=5, help='Number of residual blocks')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    config = get_default_config()
    config['batch_size'] = args.batch_size
    config['mcts_simulations'] = args.mcts_sims
    config['num_filters'] = args.num_filters
    config['num_res_blocks'] = args.num_blocks
    config['checkpoint_dir'] = args.checkpoint_dir
    config['min_buffer_size'] = 50000

    # Slightly larger defaults for C++-accelerated training
    config['mcts_simulations'] = max(config.get('mcts_simulations', 100), 200)
    config['batch_size'] = max(config.get('batch_size', 256), 128)

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        batches_per_iter=args.batches_per_iter,
    )


if __name__ == '__main__':
    main()
