
import sys
import os
sys.path.append(os.path.abspath("./cpp"))
import echelon_cpp

print("Creating board...")
b = echelon_cpp.BoardState()
b.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

print("Generating moves...")
moves = b.generate_legal_moves()
if not moves:
    print("No moves?!")
    exit(1)

m = moves[0]
print(f"Checking move object properties for: {m}")
try:
    print(f"From: {m.from_sq}")
    print(f"To: {m.to_sq}")
    print(f"Flag: {m.flag}")
    print("Properties OK")
except AttributeError as e:
    print(f"FAILED: {e}")
    # Inspect what is available
    print(f"Dir: {dir(m)}")
