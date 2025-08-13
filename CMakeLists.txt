cmake_minimum_required(VERSION 3.14)

project(Chess LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CHESS_SOURCES
    src/move.cpp
    src/chessboard.cpp
    src/game.cpp
    src/mcts.cpp
)


# Core library 

add_library(chess_core ${CHESS_SOURCES})
target_include_directories(chess_core
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(chess_core PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

# main cpp executable for dev
add_executable(Chess src/main.cpp)
target_link_libraries(Chess PRIVATE chess_core)

# tests
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_SOURCES
    test/test_chessboard.cpp
    test/test_move.cpp
    test/test_game.cpp
    test/test_check.cpp
)

add_executable(ChessTests ${TEST_SOURCES})
target_link_libraries(ChessTests
    PRIVATE
        chess_core
        GTest::gtest
        GTest::gtest_main
)
add_test(NAME ChessTests COMMAND ChessTests)

# Python bindings
find_package(pybind11 REQUIRED)  

pybind11_add_module(chess_py
    binding/env_binding.cpp
)
target_link_libraries(chess_py PRIVATE chess_core)
target_include_directories(chess_py PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_custom_target(play
    COMMAND Chess
    DEPENDS Chess
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Safe clean that only touches this build tree
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_BINARY_DIR}"
)
