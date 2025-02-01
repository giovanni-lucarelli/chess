cmake_minimum_required(VERSION 3.10)

# Project name
project(ChessEngine)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/chessboard.cpp
    src/movegen.cpp
    src/game.cpp
)

# Add executable
add_executable(ChessEngine ${SOURCES})
