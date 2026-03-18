import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dqn.agent import DQNAgent, ReplayBuffer
from dqn.env import GomokuEnv


# ─── ReplayBuffer ────────────────────────────────────────────────────────────

def _dummy_transition(action=0, reward=0.0, done=False):
    board = np.zeros((5, 5), dtype=np.int8)
    return board, action, reward, board, done, 1


def test_replay_buffer_len():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    buf.push(*_dummy_transition())
    assert len(buf) == 1


def test_replay_buffer_evicts_oldest():
    buf = ReplayBuffer(capacity=3)
    for i in range(4):
        buf.push(*_dummy_transition(action=i))
    assert len(buf) == 3
    # action=0 的那条应被淘汰，剩下 1,2,3
    actions = [t[1] for t in buf.buffer]
    assert 0 not in actions
    assert 3 in actions


def test_replay_buffer_sample_size():
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.push(*_dummy_transition())
    batch = buf.sample(8)
    assert len(batch) == 8


def test_replay_buffer_sample_raises_if_too_few():
    buf = ReplayBuffer(capacity=100)
    buf.push(*_dummy_transition())
    with pytest.raises(ValueError):
        buf.sample(10)


# ─── DQNAgent.epsilon ────────────────────────────────────────────────────────

def test_epsilon_starts_at_one():
    agent = DQNAgent(eps_start=1.0, eps_end=0.05, eps_decay=1000)
    assert agent.epsilon == pytest.approx(1.0)


def test_epsilon_decays_linearly():
    agent = DQNAgent(eps_start=1.0, eps_end=0.0, eps_decay=100)
    agent.steps = 50
    assert agent.epsilon == pytest.approx(0.5, abs=1e-6)


def test_epsilon_clamps_at_end():
    agent = DQNAgent(eps_start=1.0, eps_end=0.05, eps_decay=100)
    agent.steps = 9999
    assert agent.epsilon == pytest.approx(0.05)


# ─── select_action ───────────────────────────────────────────────────────────

def test_select_action_always_legal():
    """不管 ε 多大，select_action 返回的动作必须在 legal 列表内。"""
    agent = DQNAgent(board_size=5)
    env = GomokuEnv(board_size=5)
    board = env.reset()
    for _ in range(50):
        legal = env.get_legal_actions()
        action = agent.select_action(board, 1, legal)
        assert action in legal


def test_greedy_action_is_legal():
    agent = DQNAgent(board_size=5)
    env = GomokuEnv(board_size=5)
    board = env.reset()
    legal = env.get_legal_actions()
    action = agent.greedy_action(board, 1, legal)
    assert action in legal


# ─── train_step ──────────────────────────────────────────────────────────────

def test_train_step_returns_zero_when_buffer_empty():
    agent = DQNAgent(board_size=5, batch_size=64)
    assert agent.train_step() == 0.0


def test_train_step_returns_zero_when_buffer_insufficient():
    agent = DQNAgent(board_size=5, batch_size=64)
    for _ in range(63):
        agent.buffer.push(*_dummy_transition())
    assert agent.train_step() == 0.0


def test_train_step_returns_positive_loss_when_buffer_ready():
    agent = DQNAgent(board_size=5, batch_size=8, buffer_capacity=100)
    env = GomokuEnv(board_size=5)
    state = env.reset()
    # 填满 buffer（至少 8 条）
    for _ in range(20):
        legal = env.get_legal_actions()
        action = int(np.random.choice(legal))
        player = env.current_player
        next_state, reward, done, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done, player)
        if done:
            state = env.reset()
        else:
            state = next_state
    loss = agent.train_step()
    assert loss > 0.0


def test_train_step_increments_steps():
    agent = DQNAgent(board_size=5, batch_size=8, buffer_capacity=100)
    # 填满 buffer
    for _ in range(20):
        board = np.zeros((5, 5), dtype=np.int8)
        agent.buffer.push(board, 0, 0.0, board, False, 1)
    before = agent.steps
    agent.train_step()
    assert agent.steps == before + 1


def test_target_net_syncs_at_freq():
    """steps 达到 target_update_freq 的倍数时，target_net 应与 online_net 同步。"""
    agent = DQNAgent(board_size=5, batch_size=8, buffer_capacity=100, target_update_freq=10)
    board = np.zeros((5, 5), dtype=np.int8)
    for _ in range(100):
        agent.buffer.push(board, 0, 0.0, board, False, 1)

    # 先把 online_net 和 target_net 权重拉开
    for p in agent.online_net.parameters():
        p.data.fill_(0.5)
    for p in agent.target_net.parameters():
        p.data.fill_(0.0)

    # 跑到 steps=10，触发同步
    agent.steps = 9
    agent.train_step()   # steps 变为 10 → 触发同步

    online_params = [p.data.clone() for p in agent.online_net.parameters()]
    target_params = [p.data.clone() for p in agent.target_net.parameters()]
    for o, t in zip(online_params, target_params):
        assert (o == t).all(), "target_net 未同步"


# ─── q_values ────────────────────────────────────────────────────────────────

def test_q_values_shape():
    agent = DQNAgent(board_size=5)
    board = np.zeros((5, 5), dtype=np.int8)
    q = agent.q_values(board, player=1)
    assert q.shape == (25,)


def test_q_values_no_nan():
    agent = DQNAgent(board_size=5)
    board = np.zeros((5, 5), dtype=np.int8)
    q = agent.q_values(board, player=1)
    assert not np.isnan(q).any()
