# 当前进度快照

> 供下次对话继续使用，新对话开始时先读这个文件。

## 已完成

- [x] 确定学习路线（见 `learning_roadmap.md`）
- [x] 阅读 RL 核心概念（见 `concepts.md`）
- [x] 安装 tmux
- [x] 创建 conda 环境 `rl_study`（Python 3.11）
- [x] 安装 jupyterlab / torch / numpy / matplotlib
- [x] JupyterLab 可通过浏览器访问（`http://192.168.31.163:8888`）

## 下一步

- [ ] **阶段一**：在 JupyterLab 里创建第一个 notebook，实现五子棋环境
  - 文件建议放在：`phase1_gomoku/01_gomoku_env.ipynb`
  - 目标：实现棋盘状态、合法动作、终局判断、奖励函数

## 项目结构

```
rl_study/
├── README.md
├── learning_roadmap.md   # 详细学习路线（10小时规划）
├── concepts.md           # RL 核心概念速读
├── setup_log.md          # 环境搭建记录
├── progress.md           # 本文件：当前进度
├── phase1_gomoku/        # 阶段一：五子棋
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
