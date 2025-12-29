import sys
import os

# Add the cpp directory to the path so we can import the .so file
sys.path.append(os.path.abspath("./cpp"))

try:
    import echelon_cpp
    print("Successfully imported echelon_cpp backend!")
except ImportError as e:
    print(f"Failed to import echelon_cpp: {e}")
    sys.exit(1)

# Initialize tables
echelon_cpp.init()

# Create a board
board = echelon_cpp.BoardState()
board.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Generate legal moves
moves = board.generate_legal_moves()
print(f"Found {len(moves)} legal moves in starting position.")

for m in moves[:5]:
    print(f"Move: {m.source} -> {m.target} (Flag: {m.flag})")

# Test evaluation
score = board.evaluate()
print(f"Initial board evaluation: {score}")

# Test make move
m = moves[0]
hist = board.make_move(m)
print(f"After move {m.source}->{m.target}, side is now changed.")

# Test unmake move
board.unmake_move(m, hist)
print("Successfully unmade move.")
