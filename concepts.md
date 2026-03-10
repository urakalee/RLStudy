# RL 核心概念速读

读完目标：能用自己的话解释每个概念，并在五子棋代码里找到对应位置。

---

## 1. 强化学习是什么

监督学习：给样本+标签，学映射。
强化学习：给环境+奖励信号，学行为。

核心区别：**没有正确答案，只有结果的好坏。**

```
Agent（智能体）
  ↓ 选动作 action
Environment（环境）
  ↓ 返回新状态 state + 奖励 reward
Agent 根据 reward 调整策略
  ↓ 循环
```

---

## 2. MDP：马尔可夫决策过程

RL 问题的数学框架，五个要素：

| 符号 | 名称 | 五子棋对应 |
|---|---|---|
| S | 状态空间（State） | 棋盘上所有可能的局面 |
| A | 动作空间（Action） | 所有合法落子位置 |
| R | 奖励函数（Reward） | 赢+1 / 输-1 / 平局0 / 中间步0 |
| P | 转移概率（Transition） | 落子后棋盘如何变化（确定性） |
| γ | 折扣因子（Discount） | 未来奖励的权重，γ∈[0,1] |

**马尔可夫性**：下一个状态只取决于当前状态+动作，与历史无关。
→ 五子棋天然满足：当前棋盘已包含所有信息。

**折扣因子 γ 的直觉**：
- γ=0：只看眼前一步的奖励
- γ=1：未来奖励和当前一样重要
- γ=0.99：轻微偏好更快获得奖励（常用值）

---

## 3. Policy：策略

**Policy π(a|s)**：在状态 s 下，选择动作 a 的概率分布。

```python
# 随机策略（探索用）
def random_policy(state):
    legal_moves = get_legal_moves(state)
    return random.choice(legal_moves)

# 贪心策略（利用用）
def greedy_policy(state, q_table):
    legal_moves = get_legal_moves(state)
    return max(legal_moves, key=lambda a: q_table[state][a])
```

策略就是 Agent 的"大脑"，RL 的目标就是找到最优策略 π*。

---

## 4. Value Function：价值函数

**问题**：当前局面好不好？

### 状态价值 V(s)
从状态 s 出发，按策略 π 执行，期望累计奖励是多少。

```
V(s) = E[r₀ + γr₁ + γ²r₂ + ... | s₀=s]
```

五子棋直觉：
- V(我已四连) ≈ 0.9（快赢了）
- V(对方已四连) ≈ -0.9（快输了）
- V(开局空棋盘) ≈ 0（胜负未定）

### 动作价值 Q(s, a)
在状态 s 执行动作 a 之后，期望累计奖励是多少。

```
Q(s, a) = E[r₀ + γr₁ + γ²r₂ + ... | s₀=s, a₀=a]
```

Q-Learning（表格版）就是维护这张 Q 表；DQN 则用神经网络来拟合这个函数——原理相同，只是换了存储方式。

**V 和 Q 的关系**：
```
V(s) = max_a Q(s, a)      # 最优策略下
Q(s, a) = R(s,a) + γV(s') # Bellman 方程
```

---

## 5. Bellman 方程：价值的递推关系

价值函数满足递推关系，这是 RL 算法的数学基础：

```
Q(s, a) = R(s, a) + γ · max_{a'} Q(s', a')
          ↑当前奖励      ↑下一步最优价值（折扣后）
```

直觉：**一步棋的价值 = 这步棋的即时收益 + 落子后局面的未来价值**

---

## 6. Exploration vs Exploitation：探索与利用

**困境**：
- 只利用（Exploitation）：一直用目前最好的策略 → 可能错过更好的走法
- 只探索（Exploration）：随机乱走 → 永远学不到好策略

**ε-greedy 解法**（最简单）：
```python
if random() < epsilon:
    action = random_action()   # 探索
else:
    action = best_known_action()  # 利用

# 训练初期 epsilon=1.0（全探索）
# 随训练进行逐渐减小到 0.05（主要利用）
```

---

## 7. On-policy vs Off-policy

| | On-policy | Off-policy |
|---|---|---|
| 含义 | 用当前策略采集数据来训练当前策略 | 用旧数据/其他策略的数据训练 |
| 代表算法 | PPO, SARSA | Q-Learning, DQN |
| 五子棋应用 | self-play 当前模型 | 存储历史对局回放 |

---

## 8. 主流算法一句话总结

| 算法 | 思路 | 适合场景 |
|---|---|---|
| Q-Learning | 维护 Q 表，TD 更新 | 小状态空间（入门用） |
| DQN | 用神经网络近似 Q 函数 | 较大状态空间 |
| Policy Gradient | 直接优化策略参数 | 连续动作空间 |
| PPO | Policy Gradient + 裁剪防止更新过大 | 工业界主流 |
| GRPO | PPO 的简化版，无需 Critic 网络 | LLM 对齐（阶段二用） |

---

## 9. 概念对照表：五子棋 vs 阶段二 Tool-Use

| RL 概念 | 五子棋 | Tool-Use LLM |
|---|---|---|
| State | 棋盘局面（15×15矩阵） | 对话上下文（token序列） |
| Action | 落子位置（0~224） | 生成的 token（词表大小） |
| Reward | 赢/输/平 | 答案对错 / 格式是否正确 |
| Policy | 落子概率分布 | 语言模型（下一个token的概率） |
| Episode | 一局棋（终局结束） | 一次问答（生成完整回复） |
| Self-play | 和自己下棋 | 模型生成多个回答互相比较 |

---

## 检查自测

读完后能回答这些问题，说明概念已掌握：

1. 为什么五子棋满足马尔可夫性？
2. γ=0 和 γ=1 训练出的棋手行为有什么不同？
3. Q(s,a) 和 V(s) 哪个信息量更大？为什么？
4. 为什么纯贪心策略（ε=0）在训练初期是有害的？
5. GRPO 为什么比 PPO 简单？（提示：少了什么网络？）
