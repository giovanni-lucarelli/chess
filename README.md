# Chess RL Project

This repository contains the final project for the **Reinforcement Learning** exam at the **University of Trieste**.

The objective is to develop a chess game environment and implement various **Reinforcement Learning (RL) algorithms** to solve simplified chess scenarios with a limited number of pieces, commonly known in chess terminology as **endgames**.

Project goals:

1. **Chess Environment Implementation** — A custom environment to simulate chess positions and moves.
2. **Reinforcement Learning Algorithms** — Design and train RL agents to play and solve specific endgame scenarios.


> **Note:** This README provides a high-level summary of the project structure, build instructions, and usage guidelines. For detailed documentation and design notes, please refer to the [`docs/`](./docs/) directory.



## Project Structure

```
chess/                              # Root directory
├── 📁 include/                     # Header files (C++ API definitions)
│   ├── bitboard.hpp               # Bitboard class for efficient board representation
│   ├── chessboard.hpp             # Chessboard class for game state management
│   ├── game.hpp                   # Game class (main game logic & interface)
│   ├── move.hpp                   # Move class for representing chess moves
│   ├── search.hpp                 # Search algorithms and evaluation functions
│   └── types.hpp                  # Common type definitions and enums
│
├── 📁 src/                        # Source files (C++ implementations)
│   ├── main.cpp                   # Entry point for standalone chess engine
│   ├── game.cpp                   # Game logic implementation
│   ├── chessboard.cpp             # Board state and move validation
│   ├── move.cpp                   # Move generation and utilities
│   ├── search.cpp                 # AI search algorithms (minimax, alpha-beta)
│   └── bindings.cpp               # Pybind11 bindings for Python integration
│
├── 📁 test/                       # Testing suite
│   └── test.cpp                   # Google Test unit tests
│
├── 📁 assets/                     # Chess piece graphics (SVG format)
│   ├── w_pawn.svg, w_rook.svg, w_knight.svg, w_bishop.svg
│   ├── w_queen.svg, w_king.svg   # White pieces
│   ├── b_pawn.svg, b_rook.svg, b_knight.svg, b_bishop.svg
│   └── b_queen.svg, b_king.svg   # Black pieces
│
├── 📁 docs/                       # Documentation
│   ├── implementation.md         # Technical implementation details
│   ├── theory.md                 # Chess engine theory and algorithms
│   └── garry-kasparov-deep-blue-ibm.jpg
│
├── 🔧 CMakeLists.txt              # Build system configuration
├── 🔧 pyproject.toml              # Python package configuration
├── 📋 requirements.txt            # Python dependencies
├── 📋 packages.txt                # System package dependencies
├── 🌐 streamlit_app.py            # Web interface for interactive gameplay
├── 📝 todo.md                     # Development roadmap and tasks
└── 📖 README.md                   # Project documentation (this file)
```

### Component Overview

**Core Engine (C++)**
- `include/` + `src/`: Core chess engine implementation
- `build/ChessEngine`: Standalone executable for command-line play
- `test/`: Unit tests for engine validation

**Python Integration**
- `src/bindings.cpp`: Exposes C++ engine to Python via pybind11
- `build/chessengine_py.so`: Compiled Python module
- `pyproject.toml`: Python packaging configuration

**User Interfaces**
- `streamlit_app.py`: Web-based GUI using Streamlit framework
- `assets/`: Visual chess piece representations

**Build & Development**
- `CMakeLists.txt`: Cross-platform build configuration
- `docs/`: Technical documentation and research materials

## Build Instructions

1. Create a build directory and navigate into it:
    ```sh
    mkdir build
    cd build
    ```

2. Configure the project using CMake:
    ```sh
    cmake ..
    ```

3. Build the project:
    ```sh
    make
    ```

## Running the Engine

- To run the chess engine executable:
    ```sh
    ./ChessEngine
    ```

- To run the tests:
    ```sh
    ctest
    ```

- Custom targets are available (e.g., `play`) as defined in the CMake file:
    ```sh
    make play
    ```

- Play using Streamlit
    ```sh
    streamlit run streamlit_app.py
    ```

## Python Bindings

The Python module is built using pybind11 and is named **chessengine_py**. To use it from Python:

1. Ensure the build directory is in your `PYTHONPATH` (or copy the module to your project directory):
    ```sh
    export PYTHONPATH=$PYTHONPATH:/path/to/build
    ```

2. Import the module in Python:
    ```python
    import chessengine_py
    game = chessengine_py.Game()
    # Use game.play(), game.is_game_over(), etc.
    ```

## Dependencies

- C++17 compiler
- CMake (>= 3.10)
- [pybind11](https://github.com/pybind/pybind11)
- [Google Test](https://github.com/google/googletest)

## Contributing

Contributions and suggestions are welcome! Please open an issue or submit a pull request.



## RL project presentation

- Describe and motivate the problem
- How did you translate the problem into a RL framework?
- How did you try to solve it? Why did you use one algorithm or another?
- Is there something else you could have done?
- Did it work? (Can you interpret the optimal policy?)
- Are the results consistent with your expectation?

**More theory, less practice**

