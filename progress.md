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
- [x] **阶段一**：`phase1_gomoku/02_q_learning.ipynb` 完成
  - DQN Agent（QNetwork CNN + ReplayBuffer + Target Network）
  - 5×5 棋盘（3连赢）自我对弈，训练完成
  - ε-greedy 探索（1.0 → 0.05 线性衰减）
  - 训练曲线可视化（胜率 / Loss / ε 衰减）
  - DQN vs 随机策略评估
  - Q 值热力图可视化（可解释性）

## 下一步

- [x] **阶段二**：`phase2_tooluse/01_setup_verify.ipynb` 完成
  - Qwen3.5-2B 从 ModelScope 本地加载（1.88B 参数，mps:0）
  - MPS 推理正常，单题约 3-15s
  - GSM8K 7473 训练 / 1319 测试，本地 jsonl 加载
  - Baseline（未训练）：3 题全错，模型把示例文字当表达式，`<think>` 标签缺失
  - **关键观察**：格式不稳定（tool_call 填示例而非真实表达式）→ RLVR 要解决的核心问题

- [ ] **阶段二**：`phase2_tooluse/02_reward_design.ipynb`
  - 格式奖励：是否正确调用了工具
  - 结果奖励：最终答案是否正确
  - 参考 DeepSeek-R1 的 reward shaping 思路

- [ ] **阶段二**：`phase2_tooluse/03_grpo_training.ipynb`
  - 使用 TRL GRPOTrainer
  - 数据集：GSM8K 子集
  - 记录训练曲线：格式正确率 / 答案正确率

- [ ] **阶段二**：`phase2_tooluse/04_eval_analysis.ipynb`
  - 对比 baseline vs RLVR 效果
  - Case study 分析

## RL 概念已建立直觉（来自 phase1）

| 概念 | 在五子棋中的体现 |
|---|---|
| State | board（15×15 int8 array） |
| Action | 0~224 整数（row*15+col） |
| Reward | +1赢 / 0未结束/平 / -1非法动作 |
| Episode | 一局棋，done=True 结束 |
| Transition | (s, a, r, s', done) —— 训练数据基本单元 |
| Policy | action = f(state)，当前是 random_policy |
| Q-function | Q(s,a)：在状态 s 执行动作 a 的期望累计奖励 |
| DQN | 用 CNN 近似 Q 函数，ReplayBuffer 打破相关性 |
| Target Network | 稳定训练目标，每 N 步同步一次 |

## 环境信息

| 项目 | 值 |
|---|---|
| conda 环境 | `rl_study` |
| Jupyter 启动命令 | 见 `setup_log.md` |
| 工作目录 | `/Users/liqiang/claudelab/rl_study` |
