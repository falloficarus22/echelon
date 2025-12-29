cmake_minimum_required(VERSION 3.10)
project(echelon_cpp)

set(CMAKE_CXX_STANDARD 17)

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
