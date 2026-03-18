import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dqn.env import GomokuEnv


# ─── reset ───────────────────────────────────────────────────────────────────

def test_reset_returns_empty_board():
    env = GomokuEnv(board_size=5, win_count=3)
    board = env.reset()
    assert board.shape == (5, 5)
    assert np.all(board == 0)


def test_reset_first_player_is_black():
    env = GomokuEnv()
    env.reset()
    assert env.current_player == 1


def test_reset_clears_previous_game():
    env = GomokuEnv()
    env.reset()
    env.step(0)   # 落一子
    env.reset()
    assert np.all(env.board == 0)
    assert env.current_player == 1
    assert env.done is False


# ─── step 正常落子 ───────────────────────────────────────────────────────────

def test_step_places_piece():
    env = GomokuEnv()
    env.reset()
    board, reward, done, info = env.step(0)   # (0,0)
    assert board[0, 0] == 1       # 黑棋落在 (0,0)
    assert reward == 0.0
    assert done is False
    assert info["winner"] is None


def test_step_alternates_player():
    env = GomokuEnv()
    env.reset()
    env.step(0)   # 黑
    assert env.current_player == -1
    env.step(1)   # 白
    assert env.current_player == 1


# ─── 三连赢 ──────────────────────────────────────────────────────────────────

def _play_moves(env, moves):
    """按顺序落子，返回最后一步的 (board, reward, done, info)。"""
    result = None
    for action in moves:
        result = env.step(action)
    return result


def test_horizontal_win():
    env = GomokuEnv(board_size=5, win_count=3)
    env.reset()
    # 黑: (0,0),(0,1),(0,2)  白: (1,0),(1,1)
    _, reward, done, info = _play_moves(env, [0, 5, 1, 6, 2])
    assert done is True
    assert reward == 1.0
    assert info["winner"] == 1


def test_vertical_win():
    env = GomokuEnv(board_size=5, win_count=3)
    env.reset()
    # 黑: (0,0),(1,0),(2,0)  白: (0,1),(1,1)
    _, reward, done, info = _play_moves(env, [0, 1, 5, 6, 10])
    assert done is True
    assert info["winner"] == 1


def test_diagonal_win():
    env = GomokuEnv(board_size=5, win_count=3)
    env.reset()
    # 黑: (0,0),(1,1),(2,2)  白: (0,1),(0,2)
    _, reward, done, info = _play_moves(env, [0, 1, 6, 2, 12])
    assert done is True
    assert info["winner"] == 1


def test_white_wins():
    env = GomokuEnv(board_size=5, win_count=3)
    env.reset()
    # 黑: (0,0),(0,1)  白: (1,0),(1,1),(1,2)
    _, reward, done, info = _play_moves(env, [0, 5, 1, 6, 3, 7])
    assert done is True
    assert info["winner"] == -1


# ─── 平局 ────────────────────────────────────────────────────────────────────

def test_draw_on_full_board():
    """3×3 棋盘，2连赢——填满且无人赢。"""
    env = GomokuEnv(board_size=3, win_count=4)   # win_count>棋盘，不可能赢
    env.reset()
    for action in range(9):
        board, reward, done, info = env.step(action)
    assert done is True
    assert reward == 0.0
    assert info["winner"] == 0


# ─── 非法落子 ────────────────────────────────────────────────────────────────

def test_illegal_move_on_occupied_cell():
    env = GomokuEnv()
    env.reset()
    env.step(0)   # 黑落 (0,0)
    _, reward, done, info = env.step(0)   # 白尝试落同一格
    assert done is True
    assert reward == -1.0
    assert info["invalid"] is True
    assert info["winner"] == 1   # 黑赢（白犯规）


# ─── get_legal_actions ───────────────────────────────────────────────────────

def test_legal_actions_full_at_start():
    env = GomokuEnv(board_size=5)
    env.reset()
    assert len(env.get_legal_actions()) == 25


def test_legal_actions_decreases_after_move():
    env = GomokuEnv(board_size=5)
    env.reset()
    env.step(0)
    assert len(env.get_legal_actions()) == 24


def test_legal_actions_excludes_occupied():
    env = GomokuEnv(board_size=5)
    env.reset()
    env.step(7)
    assert 7 not in env.get_legal_actions()


# ─── done 后不能再 step ──────────────────────────────────────────────────────

def test_cannot_step_after_done():
    env = GomokuEnv(board_size=5, win_count=3)
    env.reset()
    _play_moves(env, [0, 5, 1, 6, 2])   # 黑三连赢
    with pytest.raises(AssertionError):
        env.step(10)
