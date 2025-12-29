import sys
import os
try:
    import torch
except ImportError:
    torch = None
import numpy as np

# Add the cpp directory to the path so we can import the .so file
sys.path.append(os.path.abspath("./cpp"))

try:
    import echelon_cpp
    print("Successfully imported echelon_cpp backend!")
except ImportError as e:
    print(f"Failed to import echelon_cpp: {e}")
    sys.exit(1)

try:
    from model import EchelonNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class MockModel:
    def __init__(self, model=None):
        self.model = model
    
    def predict(self, board_tensor):
        # Return dummy values if no model
        if self.model is None:
            return 0.0, np.zeros(4672, dtype=np.float32)
            
        import torch
        # Convert numpy array to torch tensor
        # C++ sends [13, 8, 8] float array
        tensor = torch.from_numpy(board_tensor).unsqueeze(0).to("cpu")
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
        
        # policy_logits is [1, 4672], value is [1, 1]
        return value.item(), policy_logits.squeeze(0).numpy()

# 1. Initialize bitboard tables
echelon_cpp.init()

# 2. Setup model
if HAS_TORCH:
    model = EchelonNet(in_channels=13, num_res_blocks=5, num_filters=128)
    model.eval()
    mock = MockModel(model)
else:
    print("Torch not found, using dummy MockModel")
    mock = MockModel(None)

# 3. Setup board
board = echelon_cpp.BoardState()
board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# 4. Run C++ MCTS
print("Starting C++ MCTS search (100 simulations)...")
mcts = echelon_cpp.MCTS(num_simulations=20)
move_probs = mcts.search(board, mock)

print(f"MCTS completed. Found probabilities for {len(move_probs)} moves.")

# Sort moves by probability
sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 moves:")
for i, (idx, prob) in enumerate(sorted_moves[:5]):
    # We can use our python decode_index to show the move
    from move_encoder import decode_index
    # Note: we need to wrap our C++ board if we want to use python decode_index easily
    # or just print the index for now.
    print(f"  {i+1}. Move Index {idx}: {prob:.4f}")

print("\nSuccess! C++ MCTS is integrated with Python model.")
