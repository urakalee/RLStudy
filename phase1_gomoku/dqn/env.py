"""五子棋环境（参数化版，提取自 02_q_learning.ipynb Cell 2）。"""

from typing import Tuple
import numpy as np


class GomokuEnv:
    """
    board_size=5, win_count=3 → 5×5 棋盘，3连赢（快速验证）
    board_size=15, win_count=5 → 标准五子棋
    """

    def __init__(self, board_size: int = 5, win_count: int = 3):
        self.BOARD_SIZE = board_size
        self.WIN_COUNT = win_count
        self.board: np.ndarray = None
        self.current_player: int = None
        self.done: bool = None
        self.move_count: int = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.move_count = 0
        return self.board.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self.done, "Episode 已结束，请先 reset()"
        row, col = divmod(action, self.BOARD_SIZE)

        if self.board[row, col] != 0:
            return self.board.copy(), -1.0, True, {"winner": -self.current_player, "invalid": True}

        self.board[row, col] = self.current_player
        self.move_count += 1

        if self._check_win(row, col):
            self.done = True
            return self.board.copy(), 1.0, True, {"winner": self.current_player, "invalid": False}

        if self.move_count == self.BOARD_SIZE ** 2:
            self.done = True
            return self.board.copy(), 0.0, True, {"winner": 0, "invalid": False}

        self.current_player = -self.current_player
        return self.board.copy(), 0.0, False, {"winner": None, "invalid": False}

    def get_legal_actions(self) -> np.ndarray:
        return np.where(self.board.flatten() == 0)[0]

    def _check_win(self, row: int, col: int) -> bool:
        player = self.board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.WIN_COUNT:
                return True
        return False
