# 当前进度快照

> 供下次对话继续使用，新对话开始时先读这个文件。

## 已完成

- [x] 确定学习路线（见 `learning_roadmap.md`）
- [x] 阅读 RL 核心概念（见 `concepts.md`）
- [x] 安装 tmux
- [x] 创建 conda 环境 `rl_study`（Python 3.11）
- [x] 安装 jupyterlab / torch / numpy / matplotlib
- [x] JupyterLab 可通过浏览器访问（`http://192.168.31.163:8888`）
- [x] **阶段一**：`phase1_gomoku/01_gomoku_env.ipynb` 完成
  - GomokuEnv 类（reset/step/get_legal_actions/_check_win）
  - 四方向五连检测验证
  - 随机策略完整 episode + transitions 收集
  - matplotlib 棋盘可视化
  - 200 局统计胜率分布

## 下一步

- [x] **阶段一**：`phase1_gomoku/02_q_learning.ipynb` 完成并运行
  - DQN Agent（QNetwork CNN + ReplayBuffer + Target Network）
  - 5×5 棋盘（3连赢）自我对弈，训练 3000 局
  - ε-greedy 探索（1.0 → 0.05 线性衰减）
  - 训练曲线可视化（胜率 / Loss / ε 衰减）
  - DQN vs 随机策略评估
  - Q 值热力图可视化（可解释性）

- [ ] **阶段二**：`phase2_tooluse/` RLVR Tool-Use
  - 模型：Qwen2.5-1.5B 或 3B
  - 算法：GRPO
  - 任务：训练小模型调用计算器/代码执行工具

## RL 概念已建立直觉（来自 01_gomoku_env）

| 概念 | 在五子棋中的体现 |
|---|---|
| State | board（15×15 int8 array） |
| Action | 0~224 整数（row*15+col） |
| Reward | +1赢 / 0未结束/平 / -1非法动作 |
| Episode | 一局棋，done=True 结束 |
| Transition | (s, a, r, s', done) —— 训练数据基本单元 |
| Policy | action = f(state)，当前是 random_policy |

## 项目结构

```
rl_study/
├── README.md
├── learning_roadmap.md   # 详细学习路线（10小时规划）
├── concepts.md           # RL 核心概念速读
├── setup_log.md          # 环境搭建记录
├── progress.md           # 本文件：当前进度
├── phase1_gomoku/
│   ├── 01_gomoku_env.ipynb    # 已完成：五子棋环境 + 随机对战
│   └── 02_q_learning.ipynb   # 待做：Q-Learning Agent
└── phase2_tooluse/       # 阶段二：RLVR Tool-Use
    ├── rewards/
    └── tools/
```

## 环境信息

| 项目 | 值 |
|---|---|
| conda 环境 | `rl_study` |
| Jupyter 启动命令 | 见 `setup_log.md` |
| 远程机器 IP | `192.168.31.163` |
| 工作目录 | `/Users/liqiang/claudelab/rl_study` |
