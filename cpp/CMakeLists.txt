cmake_minimum_required(VERSION 3.10)
project(echelon_cpp)

set(CMAKE_CXX_STANDARD 17)

# Directories
include_directories("-I/root/miniconda3/envs/icarus_env/include/python3.13")
include_directories("-I/root/miniconda3/envs/icarus_env/lib/python3.13/site-packages/pybind11/include")

# Add source files
set(SOURCES
    attacks.cpp
    magic.cpp
    board.cpp
    mcts.cpp
    main.cpp
)

add_executable(echelon_cpp ${SOURCES})

# Optimization flags
if(MSVC)
    target_compile_options(echelon_cpp PRIVATE /O2 /Oi /Ot)
else()
    target_compile_options(echelon_cpp PRIVATE -O3 -march=native)
endif()
