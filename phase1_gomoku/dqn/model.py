"""Q 网络 + 棋盘编码工具（提取自 02_q_learning.ipynb Cell 3）。"""

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


class QNetwork(nn.Module):
    """
    输入: (batch, 2, N, N) 双通道棋盘（ch0=我方, ch1=对方）
    输出: (batch, N*N)  每个位置的 Q 值
    """

    def __init__(self, board_size: int = 5):
        super().__init__()
        n = board_size
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * n * n, 256), nn.ReLU(),
            nn.Linear(256, n * n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(start_dim=1)
        return self.fc(h)


def board_to_tensor(board: np.ndarray, current_player: int) -> torch.Tensor:
    """棋盘 → (1, 2, N, N) tensor，当前玩家视角。"""
    ch0 = (board == current_player).astype(np.float32)
    ch1 = (board == -current_player).astype(np.float32)
    return torch.tensor(np.stack([ch0, ch1]), dtype=torch.float32).unsqueeze(0)
