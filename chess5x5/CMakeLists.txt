cmake_minimum_required(VERSION 3.14)

# Project name
project(Chess5x5)

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
)

# Add executable for main engine
add_executable(Chess5x5 src/main.cpp ${SOURCES})

# Compiler flags for optimization and warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(Chess5x5 PRIVATE -Wall -Wextra -Wpedantic -O3)
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
add_executable(Chess5x5Tests ${TEST_SOURCES} ${SOURCES})

# Link test executable with Google Test and required libraries
target_link_libraries(Chess5x5Tests gtest gtest_main pthread)

# Add test to CTest framework
add_test(NAME Chess5x5Tests COMMAND Chess5x5Tests)

# Custom target to run the chess 
add_custom_target(play
    COMMAND Chess5x5
    DEPENDS Chess5x5
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# Custom clean target
add_custom_target(clean_all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile
    COMMAND rm -rf ../build/*
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)