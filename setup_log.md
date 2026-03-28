# 环境搭建记录

## 机器配置

| 项目 | 说明 |
|---|---|
| 远程机器 | Mac Mini M4 16G（代码运行在这里） |
| 连接方式 | 局域网 SSH（偶尔不稳定） |

---

## 安装步骤

### 1. 安装 tmux

```bash
brew install tmux
```

tmux 解决的核心问题：SSH 断线后，远程进程不会死。

### 2. 创建 conda 环境

```bash
conda create -n rl_study python=3.11 -y
```

选 Python 3.11 的原因：torch/ML 生态对 3.11 支持最成熟。

### 3. 安装依赖

```bash
# 注意：必须用绝对路径，避免 pip 装到其他虚拟环境
/opt/homebrew/Caskroom/miniconda/base/envs/rl_study/bin/python -m pip install jupyterlab numpy matplotlib torch
```

**踩坑记录**：直接用 `pip install` 或 `conda run -n rl_study pip install` 时，
pip 被系统另一个 poetry 虚拟环境劫持，包装到了错误的地方。
解决方案：用 conda 环境的绝对路径 Python 执行 `python -m pip install`。

### 4. 启动 Jupyter（每次使用前）

```bash
# 第一步：新建 tmux session（首次）或重连（断线后）
tmux new -s rl          # 首次
tmux attach -t rl       # 断线重连

# 第二步：在 tmux 里启动 Jupyter
conda activate rl_study
/opt/homebrew/Caskroom/miniconda/base/envs/rl_study/bin/jupyter lab \
  --no-browser \
  --ip=0.0.0.0 \
  --port=8888 \
  --notebook-dir=/Users/liqiang/claudelab/rl_study
```

### 5. 本地浏览器访问

Jupyter 启动后终端会打印带 token 的 URL，格式如下：
```
http://127.0.0.1:8888/lab?token=xxxxxx...
```

将 `127.0.0.1` 替换为远程机器局域网 IP：
```
http://xxx.x.x.x:8888/lab?token=xxxxxx...
```

---

## tmux 常用操作

| 操作 | 命令 |
|---|---|
| 挂起 session（保持后台运行） | `Ctrl+B` 然后按 `D` |
| 重新连接 session | `tmux attach -t rl` |
| 查看所有 session | `tmux ls` |
| kill session | `tmux kill-session -t rl` |

---

## 已安装包版本

| 包 | 版本 | 用途 |
|---|---|---|
| Python | 3.11.15 | |
| jupyterlab | 4.5.5 | |
| torch | 2.10.0 | MPS 可用 |
| numpy | 2.4.2 | |
| matplotlib | 3.10.8 | |
| transformers | 5.3.0 | phase2 模型加载 |
| trl | 0.29.1 | phase2 GRPO 训练 |
| datasets | 4.8.3 | phase2 数据集 |
| accelerate | 1.13.0 | trl 依赖 |
| peft | 0.18.1 | phase2 LoRA 训练 |
