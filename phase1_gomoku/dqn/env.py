"""五子棋环境（参数化版，提取自 02_q_learning.ipynb Cell 2）。"""

from typing import Tuple
import numpy as np


class GomokuEnv:
    """
    board_size=5, win_count=3 → 5×5 棋盘，3连赢（快速验证）
    board_size=15, win_count=5 → 标准五子棋

    中间奖励（插件，默认关闭）：
      enable_shaping=True 时激活，奖励值可调：
      - 形成 win_count-1 连（活棋）→ +reward_near_win  (默认 +0.3)
      - 形成 win_count-2 连       → +reward_two        (默认 +0.1)
      - 堵住对手 win_count-1 连   → +reward_block       (默认 +0.2)
    enable_shaping=False（默认）→ 原始稀疏奖励，仅终局有非零 reward。
    """

    def __init__(
        self,
        board_size: int = 5,
        win_count: int = 3,
        enable_shaping: bool = False,    # 中间奖励开关（默认关闭）
        reward_near_win: float = 0.3,   # 形成 win_count-1 连
        reward_two: float = 0.1,         # 形成 win_count-2 连
        reward_block: float = 0.2,       # 堵住对手 win_count-1 连
    ):
        self.BOARD_SIZE = board_size
        self.WIN_COUNT = win_count
        self.enable_shaping = enable_shaping
        self.reward_near_win = reward_near_win
        self.reward_two = reward_two
        self.reward_block = reward_block
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

        # 赢棋：最高奖励，直接返回
        if self._check_win(row, col):
            self.done = True
            return self.board.copy(), 1.0, True, {"winner": self.current_player, "invalid": False}

        # 棋盘满：平局
        if self.move_count == self.BOARD_SIZE ** 2:
            self.done = True
            return self.board.copy(), 0.0, True, {"winner": 0, "invalid": False}

        # 中间奖励（仅 enable_shaping=True 时计算）
        reward = self._intermediate_reward(row, col) if self.enable_shaping else 0.0

        self.current_player = -self.current_player
        return self.board.copy(), reward, False, {"winner": None, "invalid": False}

    def _max_line(self, row: int, col: int, player: int) -> int:
        """统计以 (row, col) 为中心，player 在四个方向上的最大连子数。"""
        best = 1
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            best = max(best, count)
        return best

    def _intermediate_reward(self, row: int, col: int) -> float:
        """落子后（已在棋盘上）计算中间奖励。"""
        reward = 0.0
        player = self.current_player
        opponent = -player

        # 自己形成的最长连
        my_line = self._max_line(row, col, player)
        if my_line >= self.WIN_COUNT - 1:
            reward += self.reward_near_win
        elif my_line >= self.WIN_COUNT - 2 and self.WIN_COUNT >= 4:
            # win_count=3 时 win_count-2=1（单子），无意义；只在 win_count>=4 时给
            reward += self.reward_two

        # 检测是否堵住了对手的 win_count-1 连：
        # 把当前落子临时改成对手的棋，看对手在这里能形成多长的连
        self.board[row, col] = opponent
        opp_line = self._max_line(row, col, opponent)
        self.board[row, col] = player  # 还原
        if opp_line >= self.WIN_COUNT - 1:
            reward += self.reward_block

        return reward

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
