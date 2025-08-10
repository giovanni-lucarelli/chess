cmake_minimum_required(VERSION 3.14)

# Project name
project(Chess)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/move.cpp
    src/chessboard.cpp
    src/game.cpp
    src/mcts.cpp
    src/mcts_solver.cpp
)

# Add executable for main engine
add_executable(Chess src/main.cpp ${SOURCES})

# Compiler flags for optimization and warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Chess PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

# Google Test setup via FetchContent
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test sources
set(TEST_SOURCES
    test/test_chessboard.cpp
    test/test_move.cpp
    test/test_game.cpp
    test/test_check.cpp
)

# Create test executable
add_executable(ChessTests ${TEST_SOURCES} ${SOURCES})

# Link test executable with Google Test and required libraries
target_link_libraries(ChessTests gtest gtest_main pthread)

# Add test to CTest framework
add_test(NAME ChessTests COMMAND ChessTests)

# Custom target to run the chess 
add_custom_target(play
    COMMAND Chess
    DEPENDS Chess
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# Custom clean target
add_custom_target(clean_all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile
    COMMAND rm -rf ../build/*
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)