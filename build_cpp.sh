#!/bin/bash
set -e

echo "Building Echelon C++ Backend..."

# Get Python extension suffix
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Clean old builds
rm -f cpp/*.so cpp/*.o

# Compile
cd cpp
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    attacks.cpp magic.cpp board.cpp mcts.cpp echelon_cpp_wrapper.cpp \
    -o echelon_cpp${EXT_SUFFIX}

echo "Built: echelon_cpp${EXT_SUFFIX}"
ls -lh echelon_cpp*