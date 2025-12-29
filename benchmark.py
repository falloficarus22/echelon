import torch
import argparse
from model import EchelonNet
from evaluator import Evaluator
import os

def main():
    parser = argparse.ArgumentParser(description='Benchmark Echelon Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pt', help='Path to checkpoint')
    parser.add_argument('--games', type=int, default=10, help='Number of games to play')
    parser.add_argument('--sims', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract config from checkpoint to initialize correct model architecture
    config = checkpoint.get('config', {
        'in_channels': 13,
        'num_res_blocks': 5,
        'num_filters': 128
    })
    
    print(f"Model Architecture: {config['num_res_blocks']} blocks, {config['num_filters']} filters")
    
    model = EchelonNet(
        in_channels=config['in_channels'],
        num_res_blocks=config['num_res_blocks'],
        num_filters=config['num_filters']
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Starting benchmark match: {args.games} games, {args.sims} simulations per move...")
    print(f"Baseline: Greedy Material Bot (approx. 500 Elo)")
    print("-" * 50)
    
    evaluator = Evaluator(model, device=args.device)
    elo, results = evaluator.play_match(num_games=args.games, mcts_simulations=args.sims)
    
    print("-" * 50)
    print(f"BENCHMARK RESULTS (Iteration {checkpoint.get('iteration', 'Unknown')})")
    print(f"Win Rate:  {((results['win'] + 0.5 * results['draw']) / args.games) * 100:.1f}%")
    print(f"Estimated Elo: {elo:.0f}")
    print(f"Record: {results['win']} Wins, {results['loss']} Losses, {results['draw']} Draws")
    print("-" * 50)

if __name__ == "__main__":
    main()
