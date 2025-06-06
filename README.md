# Chess Engine

A C++ chess engine with Python bindings and a testing suite using Google Test. This project uses CMake as its build system and pybind11 to expose a Python module for gameplay.

## Project Structure

- **CMakeLists.txt**: Main CMake configuration; sets up build targets, tests, and Python module generation.
- **include/**: Header files.
  - `bitboard.hpp`: Contains the definition of the `Bitboard` class.
  - `chessboard.hpp`: Contains the definition of the `Chessboard` class.
  - `game.hpp`: Contains the definition of the `Game` class (main game logic).
  - `move.hpp`: Contains the definition of the `Move` class, representing a move.
  - *... other headers (e.g., types, movegen)*.
- **src/**: Source files.
  - `main.cpp`: Entry point of the chess engine.
  - `game.cpp`, `move.cpp`, `chessboard.cpp`, `search.cpp`: Implementation files.
  - `bindings.cpp`: Pybind11 bindings exposing the engine functionality to Python.
- **build/**: CMake build directory with compiled binaries, shared libraries, and test executables.
- **test/**: Tests using Google Test.
- **assets/**: SVG assets for rendering chess pieces.
- **streamlit_app.py**: A possible interface for running the chess engine as a web app.
- **todo.md**: List of upcoming features and tasks.

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

---

This README provides a basic overview of the project structure, build steps, and usage. For more detailed documentation, refer to individual source file comments and developer notes.


## RL project presentation

- Describe and motivate the problem
- How did you translate the problem into a RL framework?
- How did you try to solve it? Why did you use one algorithm or another?
- Is there something else you could have done?
- Did it work? (Can you interpret the optimal policy?)
- Are the results consistent with your expectation?

**More theory, less practice**

