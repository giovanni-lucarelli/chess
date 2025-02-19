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
    src/chessboard.cpp
    src/game.cpp
)

# Add executable for main engine
add_executable(ChessEngine src/main.cpp ${SOURCES})

# Google Test setup
enable_testing()
find_package(GTest REQUIRED)

# Add test sources
set(TEST_SOURCES
    test/test.cpp
)

# Create test executable
add_executable(ChessEngineTests ${TEST_SOURCES} ${SOURCES})

# Link test executable with Google Test and required libraries
target_link_libraries(ChessEngineTests GTest::GTest GTest::Main pthread)

# Add test to CTest framework
add_test(NAME ChessEngineTests COMMAND ChessEngineTests)

# Running executable
add_custom_target(play
    COMMAND ChessEngine
    DEPENDS ChessEngine
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# Testing
add_custom_target(run_tests
    COMMAND ChessEngineTests
    DEPENDS ChessEngineTests
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

# Cleaning
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile
    COMMAND rm -rf ../build/*
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)