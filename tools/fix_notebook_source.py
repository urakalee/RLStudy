#!/usr/bin/env python3
"""
把 Jupyter notebook 中 markdown cell 的 source 字段
从字符串格式（"line1\nline2\n..."）统一转回数组格式（["line1\n", "line2\n", ...]）。

用法：
    python tools/fix_notebook_source.py phase1_gomoku/02_q_learning.ipynb
    python tools/fix_notebook_source.py phase1_gomoku/02_q_learning.ipynb --dry-run
    python tools/fix_notebook_source.py phase1_gomoku/*.ipynb
"""

import json
import sys
from pathlib import Path


def str_to_source_array(text: str) -> list[str]:
    """把多行字符串拆成 Jupyter source 数组，每行保留末尾 \\n，最后一行不加。"""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def fix_notebook(path: Path, dry_run: bool = False) -> int:
    """返回修改的 cell 数量。"""
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0

    for cell in nb.get("cells", []):
        src = cell.get("source")
        if isinstance(src, str):
            cell["source"] = str_to_source_array(src)
            changed += 1

    if changed and not dry_run:
        path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    return changed


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    dry_run = "--dry-run" in args
    paths = [Path(a) for a in args if not a.startswith("--")]

    for p in paths:
        if not p.exists():
            print(f"[跳过] 文件不存在: {p}")
            continue
        n = fix_notebook(p, dry_run=dry_run)
        tag = "[dry-run]" if dry_run else "[已修复]"
        print(f"{tag} {p}  ({n} 个 cell 需要转换)")


if __name__ == "__main__":
    main()
