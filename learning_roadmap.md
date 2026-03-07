# RL 学习路线详细规划

## 约束条件
- 总时间：10小时
- 硬件：M1 Ultra 64G（主力）/ M4 16G（备用）
- 起点：无 RL 基础
- 终点兴趣：tool use

---

## 阶段一：RL 核心概念（3-4小时）

### 目标
不是做出好棋手，是让 RL 训练循环在脑子里成型。

### 任务：极简五子棋 RL

**Hour 1：环境建模**
- 棋盘状态表示（15x15 numpy array）
- 合法动作空间
- 终局判断（五连子/平局）
- 奖励设计：赢+1，输-1，平0

**Hour 2：最简策略训练**
- 用 Q-Learning 或 policy gradient 跑通训练循环
- Self-play：模型和自己对战
- 理解 exploration（epsilon-greedy）

**Hour 3-4：理解 & 调试**
- 观察训练曲线（reward 是否在上升）
- 理解过拟合/欠拟合在 RL 中的表现
- 阅读 Spinning Up 对应章节加深理解

### 核心概念清单（完成后应能解释）
- [ ] Markov Decision Process (MDP)
- [ ] Policy vs Value function
- [ ] On-policy vs Off-policy
- [ ] Discount factor (gamma)
- [ ] Exploration vs Exploitation

---

## 阶段二：RLVR Tool-Use（6-8小时）

### 目标
训练 Qwen2.5 小模型学会在回答问题时正确调用工具。

### 任务设计

**工具集（从简到繁）**
1. 计算器（四则运算）- 最简单，reward 完全可验证
2. Python 代码执行器 - 稍复杂，但更有趣
3. （可选）搜索工具 - 需要额外 reward 设计

**数据集**
- GSM8K（小学数学题）：答案可精确验证，是 RLVR 标准 benchmark
- 或自制简单计算题

**算法：GRPO**
```
对同一问题采样 G 个回答
→ 用验证器打分（对/错）
→ 计算组内相对优势
→ 更新策略（无需 critic 网络）
```

**模型选择**
- 首选：Qwen2.5-1.5B-Instruct（M4 16G 也能跑）
- 升级：Qwen2.5-3B-Instruct（M1 Ultra 跑更快）

### 工程路线

**Hour 1：环境搭建**
- 安装 TRL + transformers + torch (MPS backend)
- 验证 Qwen2.5 在本机推理正常
- 定义工具调用格式（JSON schema / XML tags）

**Hour 2-3：Reward 函数设计**
- 格式奖励：是否正确调用了工具
- 结果奖励：最终答案是否正确
- 参考 DeepSeek-R1 的 reward shaping 思路

**Hour 4-6：GRPO 训练循环**
- 参考 TRL 的 GRPOTrainer
- 或参考 open-r1/Logic-RL 的实现
- 记录训练曲线：格式正确率 / 答案正确率

**Hour 7-8：评估 & 分析**
- 对比 SFT baseline vs RLVR
- 分析模型学到了什么（case study）
- 尝试调整 reward 权重观察效果

---

## 参考资源

### 必读
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - 第1-3章
- [TRL 文档 GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)

### 参考代码
- [open-r1](https://github.com/huggingface/open-r1) - HuggingFace 官方 R1 复现
- [Logic-RL](https://github.com/Unakar/Logic-RL) - 小模型 RLVR 参考
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) - 极简 RLVR 实现

### 论文（选读）
- DeepSeek-R1 技术报告 - GRPO 算法来源
- RLVR 相关综述

---

## 里程碑检查点

| 时间点 | 检查项 |
|---|---|
| Hour 4 结束 | 能手写 RL 训练循环，解释 MDP 五要素 |
| Hour 6 结束 | Qwen 能在本机推理 + 工具调用格式跑通 |
| Hour 8 结束 | GRPO 训练曲线显示答案正确率有提升 |
| Hour 10 结束 | 能对比分析 baseline vs RLVR 效果差异 |
