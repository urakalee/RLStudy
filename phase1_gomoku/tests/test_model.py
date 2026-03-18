import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dqn.model import QNetwork, board_to_tensor


# ─── board_to_tensor ────────────────────────────────────────────────────────

def test_board_to_tensor_shape():
    board = np.zeros((5, 5), dtype=np.int8)
    t = board_to_tensor(board, current_player=1)
    assert t.shape == (1, 2, 5, 5)


def test_board_to_tensor_my_pieces_on_ch0():
    board = np.zeros((5, 5), dtype=np.int8)
    board[1, 2] = 1   # 黑棋
    t = board_to_tensor(board, current_player=1)
    # ch0 在 (1,2) 应为 1.0，其余为 0
    assert t[0, 0, 1, 2].item() == 1.0
    assert t[0, 0].sum().item() == 1.0


def test_board_to_tensor_opponent_on_ch1():
    board = np.zeros((5, 5), dtype=np.int8)
    board[3, 4] = -1   # 白棋
    t = board_to_tensor(board, current_player=1)
    # ch1 在 (3,4) 应为 1.0
    assert t[0, 1, 3, 4].item() == 1.0
    assert t[0, 1].sum().item() == 1.0


def test_board_to_tensor_perspective_swap():
    """白方视角时，ch0/ch1 应对调。"""
    board = np.zeros((5, 5), dtype=np.int8)
    board[0, 0] = 1    # 黑棋
    board[1, 1] = -1   # 白棋
    t_black = board_to_tensor(board, current_player=1)
    t_white = board_to_tensor(board, current_player=-1)
    # 黑方视角 ch0 有黑棋
    assert t_black[0, 0, 0, 0].item() == 1.0
    # 白方视角 ch0 有白棋
    assert t_white[0, 0, 1, 1].item() == 1.0
    # 白方视角 ch1 有黑棋
    assert t_white[0, 1, 0, 0].item() == 1.0


def test_board_to_tensor_dtype():
    board = np.zeros((5, 5), dtype=np.int8)
    t = board_to_tensor(board, current_player=1)
    assert t.dtype == torch.float32


def test_board_to_tensor_empty_board_is_all_zeros():
    board = np.zeros((5, 5), dtype=np.int8)
    t = board_to_tensor(board, current_player=1)
    assert t.sum().item() == 0.0


# ─── QNetwork ────────────────────────────────────────────────────────────────

def test_qnetwork_output_shape_5x5():
    net = QNetwork(board_size=5)
    x = torch.zeros(1, 2, 5, 5)
    out = net(x)
    assert out.shape == (1, 25)


def test_qnetwork_output_shape_batch():
    net = QNetwork(board_size=5)
    x = torch.zeros(64, 2, 5, 5)
    out = net(x)
    assert out.shape == (64, 25)


def test_qnetwork_output_shape_3x3():
    net = QNetwork(board_size=3)
    x = torch.zeros(1, 2, 3, 3)
    out = net(x)
    assert out.shape == (1, 9)


def test_qnetwork_output_has_no_nan():
    net = QNetwork(board_size=5)
    x = torch.randn(4, 2, 5, 5)
    out = net(x)
    assert not torch.isnan(out).any()


def test_qnetwork_output_can_be_negative():
    """Q 值不应被 ReLU 截断，允许负数。"""
    net = QNetwork(board_size=5)
    torch.manual_seed(42)
    x = torch.randn(16, 2, 5, 5)
    out = net(x)
    assert (out < 0).any(), "输出全为正数，怀疑末层意外加了 ReLU"


def test_qnetwork_parameter_count_5x5():
    net = QNetwork(board_size=5)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params == 435_385
