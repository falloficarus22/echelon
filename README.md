# Echelon Chess Engine

An AlphaZero-style chess engine with high-performance C++ core and PyTorch neural network.

## 🚀 Quick Start

### Training (Recommended - C++ Backend)
```bash
# Compile C++ engine (one-time setup)
cd cpp
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
    attacks.cpp magic.cpp board.cpp mcts.cpp echelon_cpp_wrapper.cpp \
    -o echelon_cpp$(python3-config --extension-suffix)
cd ..

# Create symlink
ln -sf cpp/echelon_cpp*.so echelon_cpp.so

# Start training
python3 train_cpp.py
```

### Play Against the Engine
```bash
python3 play.py
```

## 📁 Project Structure

### **Core Files (Use These)**
- `train_cpp.py` - **Main training script** (uses C++ backend, 60x faster)
- `model.py` - Neural network architecture (ResNet-20)
- `move_encoder.py` - Move encoding for policy head
- `play.py` - Interactive gameplay
- `evaluator.py` - Engine evaluation and benchmarking

### **C++ Backend (High Performance)**
- `cpp/board.cpp` - Chess engine core (18M NPS)
- `cpp/mcts.cpp` - Monte Carlo Tree Search
- `cpp/magic.cpp` - Magic bitboards for attack generation
- `cpp/attacks.cpp` - Piece attack tables
- `cpp/echelon_cpp_wrapper.cpp` - Python bindings

### **Legacy Python Files (Deprecated - Keep for Reference)**
- ~~`train.py`~~ - Old training script (replaced by `train_cpp.py`)
- ~~`selfplay.py`~~ - Old self-play (now in C++)
- ~~`engine.py`~~ - Pure Python engine (replaced by C++ `Board`)
- ~~`attacks.py`~~ - Python attacks (replaced by C++ version)
- ~~`magic_bitboards.py`~~ - Python magic (replaced by C++ version)

### **Notebooks**
- `train_colab.ipynb` - Google Colab training (uses C++ backend)

### **Documentation**
- `CPP_MIGRATION_SUMMARY.md` - Complete C++ migration guide
- `README.md` - This file

## 🎯 Performance

| Component | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Move Generation | 500 NPS | 18M NPS | **36,000x** |
| Perft(5) | ~5 min | 0.35s | **~850x** |
| Self-Play Game | 30-60s | 2-10s | **6-10x** |

## 🔧 Requirements

- Python 3.10+
- PyTorch
- pybind11
- g++ with C++17 support

## 📊 Training Configuration

Edit `train_cpp.py` to adjust:
- `num_simulations` - MCTS simulations per move (default: 50)
- `num_res_blocks` - Neural network depth (default: 5)
- `num_filters` - Neural network width (default: 128)
- `batch_size` - Training batch size (default: 32)

## 🎮 Usage Examples

### Train from scratch
```python
python3 train_cpp.py
```

### Play against trained model
```python
python3 play.py
# Select option 1 (Play as White) or 2 (Play as Black)
```

### Watch engine self-play
```python
python3 play.py
# Select option 3
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'echelon_cpp'"
Recompile the C++ backend and ensure the symlink is created:
```bash
cd cpp && g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
    attacks.cpp magic.cpp board.cpp mcts.cpp echelon_cpp_wrapper.cpp \
    -o echelon_cpp$(python3-config --extension-suffix)
cd .. && ln -sf cpp/echelon_cpp*.so echelon_cpp.so
```

### Slow training on CPU
Reduce simulations and model size in `train_cpp.py`:
```python
num_simulations=20  # Instead of 50
num_res_blocks=3    # Instead of 5
num_filters=64      # Instead of 128
```

## 📝 License

MIT License

## 🙏 Acknowledgments

Based on the AlphaZero algorithm by DeepMind.
