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
