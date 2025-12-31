#!/bin/bash
set -e

echo "=== Echelon Lichess Bot Setup ==="

# 1. Install Python dependencies
echo "Installing dependencies..."
pip install torch numpy pybind11

# 2. Build C++ module
echo "Building C++ backend..."
./build_cpp.sh

# 3. Verify build
echo "Verifying build..."
python3 -c "
import sys
sys.path.insert(0, './cpp')
import echelon_cpp
echelon_cpp.init()
print('✓ C++ module working')
"

# 4. Test UCI interface
echo "Testing UCI interface..."
echo -e "uci\nisready\nquit" | python3 uci.py

echo "=== Setup Complete ==="
echo "To run: python3 uci.py"