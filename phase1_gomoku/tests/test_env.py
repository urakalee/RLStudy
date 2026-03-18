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


# ─── 中间奖励 ────────────────────────────────────────────────────────────────

def test_near_win_reward_on_two_in_a_row():
    """win_count=3：形成两连（win_count-1）应得 reward_near_win。"""
    env = GomokuEnv(board_size=5, win_count=3, enable_shaping=True, reward_near_win=0.3)
    env.reset()
    env.step(0)   # 黑 (0,0)，单子，无奖励
    env.step(10)  # 白 (2,0)，远离，无奖励
    _, reward, done, _ = env.step(1)   # 黑 (0,1)，形成两连
    assert done is False
    assert reward == pytest.approx(0.3)


def test_no_near_win_reward_for_single_piece():
    """落单子时，中间奖励应为 0。"""
    env = GomokuEnv(board_size=5, win_count=3, enable_shaping=True, reward_near_win=0.3)
    env.reset()
    _, reward, done, _ = env.step(12)  # 黑 (2,2)，空棋盘第一手
    assert done is False
    assert reward == pytest.approx(0.0)


def test_block_reward_when_blocking_opponent_two():
    """白方堵住黑棋两连，应得 reward_block。"""
    env = GomokuEnv(board_size=5, win_count=3, enable_shaping=True, reward_near_win=0.3, reward_block=0.2)
    env.reset()
    env.step(0)   # 黑 (0,0)
    env.step(10)  # 白 (2,0)，远离
    env.step(1)   # 黑 (0,1)，黑已两连
    # 白落 (0,2)：堵住黑两连，同时白自身无连 → block 奖励
    _, reward, done, _ = env.step(2)
    assert done is False
    assert reward == pytest.approx(0.2)


def test_block_plus_near_win_reward():
    """白落子既堵住对手两连、又形成自己两连，奖励叠加。"""
    env = GomokuEnv(board_size=5, win_count=3, enable_shaping=True, reward_near_win=0.3, reward_block=0.2)
    env.reset()
    # 黑: (0,0),(0,1) 两连；白: (1,0) 单子
    env.step(0)   # 黑
    env.step(5)   # 白 (1,0)
    env.step(1)   # 黑 (0,1)，黑两连
    # 白落 (1,1)：白形成两连(reward_near_win) + 堵黑(reward_block)
    _, reward, done, _ = env.step(6)
    assert done is False
    assert reward == pytest.approx(0.3 + 0.2)


def test_win_reward_not_affected_by_intermediate():
    """赢棋时 reward 始终是 1.0，不叠加中间奖励。"""
    env = GomokuEnv(board_size=5, win_count=3, enable_shaping=True, reward_near_win=0.3)
    env.reset()
    _, reward, done, info = _play_moves(env, [0, 5, 1, 6, 2])
    assert reward == pytest.approx(1.0)
    assert done is True


def test_intermediate_reward_disabled_when_zero():
    """enable_shaping=False（默认）时，非终局步骤奖励始终为 0。"""
    env = GomokuEnv(board_size=5, win_count=3)  # 默认 enable_shaping=False
    env.reset()
    env.step(0)
    env.step(10)
    _, reward, _, _ = env.step(1)   # 黑两连，但奖励关闭
    assert reward == pytest.approx(0.0)


def test_intermediate_reward_board_not_corrupted():
    """_intermediate_reward 内部临时修改棋盘后必须还原，不影响后续 step。"""
    env = GomokuEnv(board_size=5, win_count=3, enable_shaping=True)
    env.reset()
    env.step(0)   # 黑 (0,0)
    env.step(10)  # 白 (2,0)
    env.step(1)   # 黑 (0,1)，触发中间奖励计算
    # 此时 (0,1) 必须是黑棋，而不是被临时修改后没还原的对手棋
    assert env.board[0, 1] == 1
