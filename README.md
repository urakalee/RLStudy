# RL Study

通过两个小项目学习强化学习，总时间约 10 小时。

## 目录结构

```
rl_study/
├── README.md
├── learning_roadmap.md           # 详细学习路线规划
├── concepts.md                   # RL 核心概念速读
├── setup_log.md                  # 环境搭建记录
├── progress.md                   # 当前进度
├── tools/
│   └── fix_notebook_source.py
├── phase1_gomoku/                # 阶段一：五子棋 RL（Hour 1-4）
│   ├── 01_gomoku_env.ipynb       # 五子棋环境 + 随机对战
│   ├── 02_q_learning.ipynb       # DQN 训练
│   ├── dqn/                      # DQN 模块
│   └── checkpoints/              # 训练检查点
└── phase2_tooluse/               # 阶段二：RLVR Tool-Use（Hour 5-10）
    ├── 01_setup_verify.ipynb     # 模型推理验证
    ├── 02_reward_design.ipynb    # Reward 函数设计
    ├── 03_grpo_training.ipynb    # GRPO 训练（核心）
    └── 04_eval_analysis.ipynb    # 评估对比
```

## 两阶段目标

| 阶段 | 时间 | 目标 |
|---|---|---|
| 五子棋 | 3-4h | 建立 RL 核心概念：MDP/Policy/Value/Reward |
| RLVR Tool-Use | 6-8h | 用 GRPO 训练 Qwen2.5 学会调用工具 |

详见 [learning_roadmap.md](./learning_roadmap.md)
