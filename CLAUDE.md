# RL Study — AI 工作规范

## Notebook 编辑规则

**每次用 `NotebookEdit` 修改 `.ipynb` 文件后，提交前必须执行：**

```bash
python tools/fix_notebook_source.py <修改的notebook路径>
```

原因：`NotebookEdit` 工具会把 cell 的 `source` 字段写成单个字符串，
而 Jupyter 规范是字符串数组（每行一个元素）。
数组格式的 git diff 可按行对比，字符串格式会把整个 cell 显示为一次替换，完全不可读。

示例：
```bash
# 修改了 phase1_gomoku/02_q_learning.ipynb 后
python tools/fix_notebook_source.py phase1_gomoku/02_q_learning.ipynb
git add phase1_gomoku/02_q_learning.ipynb
git commit ...
```

## 运行环境

项目使用 conda 环境 `rl_study`（Python 3.11）。

```bash
conda activate rl_study
```

## 运行测试

```bash
conda activate rl_study
cd phase1_gomoku
python -m pytest tests/ -v
```

依赖（已安装在 rl_study 环境）：pytest

## Notebook Cell 沟通约定

每个代码 cell 第一行用注释写编号，格式为 `# X-Y`，对应章节标题编号。

示例：
- `# 1-1 导入依赖 & 工具函数` → 第 1 章第 1 个代码 cell
- `# 6-1 配置 callback + 初始化 trainer` → 第 6 章第 1 个代码 cell

**沟通时直接说"第 X-Y cell"，在 Jupyter 里找第一行注释匹配即可。**

新增或修改 notebook 时，所有代码 cell 必须在第一行加上 `# X-Y 简要描述` 的编号注释。

## 调试原则

**遇到与预期不一致的现象，先看日志/加 print，不要靠猜。**

- Notebook 的 print 输出只在 cell 输出区域显示，不会出现在启动终端
- 训练场景可用 `trainer.state.log_history` 查看已收集的历史指标
- 其他场景直接在 cell 里加 print

**耗时操作的打断原则：**

如果当前有耗时操作（如训练）正在运行，评估是否值得打断：
- 刚启动不久 / 发现配置明显有误 → 果断打断，修好再重跑，不要浪费时间等它跑完
- 已跑大半 / 问题是非关键的 → 等跑完再修
- 不确定是否有问题 → 打断，用少量步数（如 max_steps=10）验证配置正确后再正式跑
