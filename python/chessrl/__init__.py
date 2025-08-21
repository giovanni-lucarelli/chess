# Expose env and native module at package level for convenience
from .env import Env, StepResult, LichessDefender, SyzygyDefender
# chessrl.chess_py is the compiled module (imported separately when needed)
