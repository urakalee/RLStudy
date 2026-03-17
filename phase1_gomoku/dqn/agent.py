"""DQN Agent（提取自 02_q_learning.ipynb Cell 4-5）。"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import DEVICE, QNetwork, board_to_tensor


class ReplayBuffer:
    """固定容量 FIFO，超出时丢弃最旧数据。"""

    def __init__(self, capacity: int = 20_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, player):
        self.buffer.append((state, action, reward, next_state, done, player))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(
        self,
        board_size: int = 5,
        gamma: float = 0.95,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 5000,
        batch_size: int = 64,
        buffer_capacity: int = 20_000,
        target_update_freq: int = 200,
    ):
        self.board_size = board_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.online_net = QNetwork(board_size).to(DEVICE)
        self.target_net = QNetwork(board_size).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.steps = 0

    @property
    def epsilon(self) -> float:
        progress = min(self.steps / self.eps_decay, 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    def select_action(self, board: np.ndarray, current_player: int, legal: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return int(np.random.choice(legal))
        with torch.no_grad():
            q = self.online_net(board_to_tensor(board, current_player).to(DEVICE))
            q = q.squeeze(0).cpu().numpy()
        return int(legal[np.argmax(q[legal])])

    def greedy_action(self, board: np.ndarray, player: int, legal: np.ndarray) -> int:
        """纯贪心（ε=0），用于评估和对战。"""
        with torch.no_grad():
            q = self.online_net(board_to_tensor(board, player).to(DEVICE)).squeeze(0).cpu().numpy()
        return int(legal[np.argmax(q[legal])])

    def q_values(self, board: np.ndarray, player: int) -> np.ndarray:
        """返回当前局面所有位置的 Q 值，shape=(N*N,)。"""
        with torch.no_grad():
            q = self.online_net(board_to_tensor(board, player).to(DEVICE)).squeeze(0).cpu().numpy()
        return q

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, players = zip(*batch)

        s_t = torch.cat([board_to_tensor(s, p) for s, p in zip(states, players)]).to(DEVICE)
        s_next_t = torch.cat([board_to_tensor(ns, -p) for ns, p in zip(next_states, players)]).to(DEVICE)

        a_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        r_t = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        d_t = torch.tensor(dones, dtype=torch.float32).to(DEVICE)

        q_pred = self.online_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(s_next_t).max(dim=1).values
            # 零和博弈：对手收益高 = 我方收益低，用减号
            q_target = r_t - self.gamma * q_next * (1 - d_t)

        loss = nn.functional.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
