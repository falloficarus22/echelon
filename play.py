import torch
from engine import BoardState
from model import EchelonNet
from mcts import MCTS
from constants import *


def print_board(board):
    """Print the chess board in a readable format"""
    print("\n  a b c d e f g h")
    for rank in range(7, -1, -1):
        print(f"{rank + 1} ", end="")
        for file in range(8):
            square = rank * 8 + file
            piece_char = '.'
            
            # Check each piece type for both colors
            for color in [WHITE, BLACK]:
                for piece_type in range(6):
                    piece_idx = piece_type + (color * 6)
                    if board.bitboards[piece_idx] & (1 << square):
                        piece_char = PIECE_SYMBOLS[piece_idx]
                        break
                if piece_char != '.':
                    break
            
            print(f"{piece_char} ", end="")
        print(f" {rank + 1}")
    print("  a b c d e f g h\n")


def square_to_index(square_str):
    """Convert algebraic notation (e.g., 'e4') to square index"""
    if len(square_str) != 2:
        return None
    
    file = ord(square_str[0].lower()) - ord('a')
    rank = int(square_str[1]) - 1
    
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return rank * 8 + file
    return None


def index_to_square(index):
    """Convert square index to algebraic notation"""
    file = chr(ord('a') + (index % 8))
    rank = str((index // 8) + 1)
    return file + rank


def get_move_from_user(board):
    """Get a move from the user in algebraic notation"""
    legal_moves = board.generate_legal_moves(board.side)
    
    while True:
        move_str = input("Your move (e.g., 'e2e4' or 'quit'): ").strip().lower()
        
        if move_str == 'quit' or move_str == 'q':
            return None
        
        if move_str == 'help' or move_str == 'h':
            print("\nLegal moves:")
            for move in legal_moves[:20]:  # Show first 20
                decoded = board.decode_move(move)
                from_sq = index_to_square(decoded['from'])
                to_sq = index_to_square(decoded['to'])
                print(f"  {from_sq}{to_sq}", end="")
            print("\n...")
            continue
        
        if len(move_str) < 4:
            print("Invalid format. Use format like 'e2e4'")
            continue
        
        from_square = square_to_index(move_str[:2])
        to_square = square_to_index(move_str[2:4])
        
        if from_square is None or to_square is None:
            print("Invalid squares. Files must be a-h, ranks must be 1-8")
            continue
        
        # Check if this move is legal
        for move in legal_moves:
            decoded = board.decode_move(move)
            if decoded['from'] == from_square and decoded['to'] == to_square:
                # Handle promotion
                if len(move_str) == 5:
                    promo_char = move_str[4].lower()
                    promo_map = {'q': MOVE_FLAG_PROMOTION_QUEEN, 
                               'r': MOVE_FLAG_PROMOTION_ROOK,
                               'b': MOVE_FLAG_PROMOTION_BISHOP, 
                               'n': MOVE_FLAG_PROMOTION_KNIGHT}
                    if promo_char in promo_map:
                        if decoded['flag'] == promo_map[promo_char]:
                            return move
                else:
                    return move
        
        print("Illegal move! Type 'help' to see legal moves.")


def play_against_engine(model_path=None, num_simulations=800, play_as_white=True):
    """
    Play a game against the engine.
    
    Args:
        model_path: Path to trained model checkpoint (None for random model)
        num_simulations: MCTS simulations for engine moves
        play_as_white: If True, human plays as White
    """
    
    # Load model
    print("Loading model...")
    model = EchelonNet(in_channels=13, num_res_blocks=5, num_filters=128)
    
    if model_path and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("Using untrained model (random play)")
    
    model.to(device)
    model.eval()
    
    # Initialize board
    board = BoardState()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board.parse_fen(start_fen)
    
    # Initialize MCTS
    mcts = MCTS(model, num_simulations=num_simulations, temperature=0.1)
    
    print("\n" + "="*70)
    print("CHESS GAME - Human vs Engine")
    print("="*70)
    print(f"You are playing as {'White' if play_as_white else 'Black'}")
    print(f"Engine using {num_simulations} MCTS simulations per move")
    print("Type 'help' to see legal moves, 'quit' to exit")
    print("="*70)
    
    move_count = 0
    
    while True:
        move_count += 1
        print_board(board)
        
        # Check for game over
        legal_moves = board.generate_legal_moves(board.side)
        if len(legal_moves) == 0:
            if board.is_in_check(board.side):
                winner = "Black" if board.side == WHITE else "White"
                print(f"Checkmate! {winner} wins!")
            else:
                print("Stalemate! Game is a draw.")
            break
        
        # Check for 50-move rule
        if board.halfmove_clock >= 100:
            print("Draw by 50-move rule!")
            break
        
        # Determine whose turn it is
        is_human_turn = (board.side == WHITE and play_as_white) or \
                       (board.side == BLACK and not play_as_white)
        
        side_name = "White" if board.side == WHITE else "Black"
        print(f"Move {move_count} - {side_name} to move")
        
        if is_human_turn:
            # Human move
            move = get_move_from_user(board)
            if move is None:
                print("Game ended by user.")
                break
        else:
            # Engine move
            print("Engine thinking...")
            import time
            start_time = time.time()
            
            best_move, move_probs = mcts.search(board)
            
            elapsed = time.time() - start_time
            
            decoded = board.decode_move(best_move)
            from_sq = index_to_square(decoded['from'])
            to_sq = index_to_square(decoded['to'])
            
            print(f"Engine plays: {from_sq}{to_sq} ({elapsed:.1f}s)")
            move = best_move
        
        # Make the move
        board.make_move(move)
        print()
    
    print("\nGame Over!")
    print("="*70)


def watch_engine_play(model_path=None, num_simulations=400):
    """
    Watch the engine play against itself.
    """
    print("Loading model...")
    model = EchelonNet(in_channels=13, num_res_blocks=5, num_filters=128)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("Using untrained model")
    
    model.eval()
    
    board = BoardState()
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board.parse_fen(start_fen)
    
    mcts = MCTS(model, num_simulations=num_simulations, temperature=0.1)
    
    print("\n" + "="*70)
    print("WATCHING ENGINE SELF-PLAY")
    print("="*70)
    
    move_count = 0
    
    while move_count < 50:  # Limit to 50 moves
        move_count += 1
        print_board(board)
        
        legal_moves = board.generate_legal_moves(board.side)
        if len(legal_moves) == 0:
            if board.is_in_check(board.side):
                winner = "Black" if board.side == WHITE else "White"
                print(f"Checkmate! {winner} wins!")
            else:
                print("Stalemate!")
            break
        
        side_name = "White" if board.side == WHITE else "Black"
        print(f"Move {move_count} - {side_name}")
        
        best_move, _ = mcts.search(board)
        
        decoded = board.decode_move(best_move)
        from_sq = index_to_square(decoded['from'])
        to_sq = index_to_square(decoded['to'])
        
        print(f"Move: {from_sq}{to_sq}\n")
        
        board.make_move(best_move)
        
        input("Press Enter for next move...")
    
    print("Game ended.")


def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    import glob
    import os
    
    # Search in both current directory and checkpoints/ folder
    checkpoints = glob.glob("checkpoint_iter_*.pt") + glob.glob("checkpoints/checkpoint_iter_*.pt")
    
    if checkpoints:
        # Extract iteration numbers and find the latest
        iterations = []
        for ckpt in checkpoints:
            try:
                # Handle paths like "checkpoints/checkpoint_iter_10.pt" correctly
                filename = os.path.basename(ckpt)
                iter_num = int(filename.split("_")[-1].replace(".pt", ""))
                iterations.append((iter_num, ckpt))
            except ValueError:
                continue
        
        if iterations:
            iterations.sort(reverse=True)
            return iterations[0][1]  # Return just the path

    if os.path.exists("checkpoints/latest.pt"):
        return "checkpoints/latest.pt"
            
    return None


if __name__ == "__main__":
    import sys
    
    print("\nEchelon Chess Engine")
    print("="*70)
    print("1. Play as White")
    print("2. Play as Black")
    print("3. Watch engine self-play")
    print("="*70)
    
    choice = input("Select option (1-3): ").strip()
    
    # Auto-discover latest checkpoint
    model_path = find_latest_checkpoint()
    
    if model_path:
        print(f"\nFound latest checkpoint: {model_path}")
    else:
        print("\nNo checkpoint found. Using untrained model (random play).")
    
    if choice == '1':
        play_against_engine(model_path, num_simulations=400, play_as_white=True)
    elif choice == '2':
        play_against_engine(model_path, num_simulations=400, play_as_white=False)
    elif choice == '3':
        watch_engine_play(model_path, num_simulations=200)
    else:
        print("Invalid choice!")