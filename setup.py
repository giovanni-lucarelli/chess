from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sysconfig, platform
import pybind11

# C++ source files for the extension (no main.cpp!)
CPP_SOURCES = [
    "src/move.cpp",
    "src/chessboard.cpp",
    "src/game.cpp",
    "src/mcts/mcts.cpp",
    "binding/env_binding.cpp",
]

INCLUDE_DIRS = [
    "include",
    pybind11.get_include(),   # absolute include dirs are fine
]

def extra_compile_args():
    cc = sysconfig.get_config_var("CC") or ""
    is_windows = platform.system() == "Windows"
    if is_windows:
        return ["/std:c++17", "/O2", "/EHsc", "/DNDEBUG"]
    else:
        return ["-std=c++17", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-DNDEBUG"]

ext = Extension(
    name="chessrl.chess_py",          # import as: from chessrl import chess_py
    sources=CPP_SOURCES,              # <-- keep these RELATIVE
    include_dirs=INCLUDE_DIRS,
    language="c++",
    extra_compile_args=extra_compile_args(),
)

class BuildExt(build_ext):
    def build_extensions(self):
        super().build_extensions()

setup(
    name="chessrl",
    version="0.1.0",
    packages=find_packages(where="python", include=["chessrl", "chessrl.*"]),
    package_dir={"": "python"},
    ext_modules=[ext],
    include_package_data=True,
)
