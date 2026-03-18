#!/usr/bin/env python3
"""
eval_cases.py — 对比多个训练结果在预设 case 上的 Q 值表现

用法：
  python tools/eval_cases.py                        # 评估所有 checkpoints
  python tools/eval_cases.py --runs 20260318_112303 20260318_114157
  python tools/eval_cases.py --ckpt-dir /path/to/checkpoints
  python tools/eval_cases.py --case all             # 运行所有 case（默认）
  python tools/eval_cases.py --case winning_move    # 只跑某个 case
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── 路径设置 ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
PHASE1_DIR  = REPO_ROOT / "phase1_gomoku"

sys.path.insert(0, str(PHASE1_DIR))

from dqn.model import QNetwork, board_to_tensor, DEVICE  # noqa: E402

# ── 预设 Case 定义 ────────────────────────────────────────────────────────────

def make_board(black: list[tuple], white: list[tuple], size: int = 5) -> np.ndarray:
    b = np.zeros((size, size), dtype=np.int8)
    for r, c in black:
        b[r, c] = 1
    for r, c in white:
        b[r, c] = -1
    return b


# Case 格式：
#   name        - 显示名称
#   desc        - 局面描述
#   board       - numpy 棋盘
#   player      - 当前落子方（1=黑, -1=白）
#   target      - 期望最优动作 (row, col)
#   target_desc - 为什么这步是最优
CASES = [
    {
        "name": "winning_move",
        "desc": "黑间隔两连 (2,1)(2,3) + 白两连 (0,0)(0,1)，黑下 (2,2) 横向三连赢",
        "board": make_board(black=[(2,1),(2,3)], white=[(0,0),(0,1)]),
        "player": 1,
        "target": (2, 2),
        "target_desc": "三连必赢",
    },
    {
        "name": "block_opponent",
        "desc": "白两连 (1,0)(1,1)，黑需堵住 (1,2) 防止白三连赢",
        "board": make_board(black=[(0,0)], white=[(1,0),(1,1)]),
        "player": 1,
        "target": (1, 2),
        "target_desc": "封白制胜位",
    },
    {
        "name": "take_win",
        "desc": "黑两连 (0,0)(0,1)，黑应直接下 (0,2) 三连赢",
        "board": make_board(black=[(0,0),(0,1)], white=[(2,2)]),
        "player": 1,
        "target": (0, 2),
        "target_desc": "直接三连赢",
    },
]

CASE_MAP = {c["name"]: c for c in CASES}

# ── 核心评估逻辑 ──────────────────────────────────────────────────────────────

def load_net(ckpt_dir: Path, board_size: int = 5) -> Optional[QNetwork]:
    weights = ckpt_dir / "online_net.pt"
    if not weights.exists():
        return None
    net = QNetwork(board_size=board_size).to(DEVICE)
    net.load_state_dict(torch.load(weights, map_location=DEVICE))
    net.eval()
    return net


def eval_case(net: QNetwork, case: dict) -> dict:
    board  = case["board"]
    player = case["player"]
    tr, tc = case["target"]
    target_action = tr * board.shape[0] + tc

    t = board_to_tensor(board, player).to(DEVICE)
    with torch.no_grad():
        q = net(t).squeeze().cpu().numpy()

    occupied = board.flatten() != 0
    legal_q  = [(a, float(q[a])) for a in range(len(q)) if not occupied[a]]
    legal_q.sort(key=lambda x: -x[1])

    best_action, best_q = legal_q[0]
    target_q = float(q[target_action])
    best_r, best_c = divmod(best_action, board.shape[0])

    return {
        "target_q":    target_q,
        "best_action": (best_r, best_c),
        "best_q":      best_q,
        "correct":     best_action == target_action,
        "rank":        next(i for i, (a, _) in enumerate(legal_q) if a == target_action) + 1,
    }


def load_meta(ckpt_dir: Path) -> dict:
    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


# ── 显示工具 ──────────────────────────────────────────────────────────────────

def render_board(board: np.ndarray) -> str:
    symbols = {0: ".", 1: "X", -1: "O"}
    lines = []
    n = board.shape[0]
    header = "  " + " ".join(str(c) for c in range(n))
    lines.append(header)
    for r in range(n):
        row = f"{r} " + " ".join(symbols[board[r, c]] for c in range(n))
        lines.append(row)
    return "\n".join(lines)


def mark_target(board: np.ndarray, target: tuple, best: tuple) -> str:
    """在棋盘上标注目标位和实际最优位。"""
    symbols = {0: ".", 1: "X", -1: "O"}
    n = board.shape[0]
    tr, tc = target
    br, bc = best
    lines = []
    header = "  " + " ".join(str(c) for c in range(n))
    lines.append(header)
    for r in range(n):
        row_chars = []
        for c in range(n):
            if board[r, c] != 0:
                row_chars.append(symbols[board[r, c]])
            elif (r, c) == target == best:
                row_chars.append("★")   # 目标即最优
            elif (r, c) == target:
                row_chars.append("T")   # 目标位（期望）
            elif (r, c) == best:
                row_chars.append("B")   # 实际最优（错误）
            else:
                row_chars.append(".")
        lines.append(f"{r} " + " ".join(row_chars))
    return "\n".join(lines)


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="对比多个训练结果在预设 case 上的表现")
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="指定 run_id 列表，不填则评估所有 checkpoints"
    )
    parser.add_argument(
        "--ckpt-dir", default=None,
        help=f"checkpoint 根目录（默认 {PHASE1_DIR}/checkpoints）"
    )
    parser.add_argument(
        "--case", default="all",
        choices=["all"] + list(CASE_MAP.keys()),
        help="要运行的 case（默认 all）"
    )
    parser.add_argument(
        "--board", action="store_true",
        help="显示每个 case 的棋盘图"
    )
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_dir) if args.ckpt_dir else PHASE1_DIR / "checkpoints"
    if not ckpt_root.exists():
        print(f"错误：找不到 checkpoint 目录 {ckpt_root}")
        sys.exit(1)

    # 选择 runs
    if args.runs:
        run_dirs = [ckpt_root / r for r in args.runs]
    else:
        run_dirs = sorted(ckpt_root.iterdir())
    run_dirs = [d for d in run_dirs if d.is_dir() and (d / "online_net.pt").exists()]

    if not run_dirs:
        print("没有找到任何有效的 checkpoint")
        sys.exit(1)

    # 选择 cases
    cases = CASES if args.case == "all" else [CASE_MAP[args.case]]

    # ── 打印 case 说明 ──
    print("=" * 70)
    print("预设 Case 说明")
    print("=" * 70)
    for i, case in enumerate(cases, 1):
        print(f"\n[Case {i}] {case['name']}")
        print(f"  {case['desc']}")
        print(f"  期望最优：{case['target']}（{case['target_desc']}）")
        if args.board:
            for line in render_board(case["board"]).split("\n"):
                print(f"  {line}")
    print()

    # ── 逐 case 输出汇总表 ──
    for case in cases:
        print("=" * 70)
        print(f"Case: {case['name']}  |  期望落点: {case['target']}")
        print("=" * 70)
        print(f"{'run_id':<22} {'ep':>6} {'Q(target)':>10} {'最优动作':>10} {'最优Q':>8}  {'结果':>4}  {'排名':>4}")
        print("-" * 70)

        for run_dir in run_dirs:
            meta = load_meta(run_dir)
            episodes = meta.get("config", {}).get("episodes", "?")
            net = load_net(run_dir)
            if net is None:
                print(f"{run_dir.name:<22}  (无法加载权重)")
                continue

            r = eval_case(net, case)
            ok = "✓" if r["correct"] else "✗"
            best_str = f"({r['best_action'][0]},{r['best_action'][1]})"
            print(
                f"{run_dir.name:<22} {str(episodes):>6} "
                f"{r['target_q']:>10.4f} {best_str:>10} "
                f"{r['best_q']:>8.4f}  {ok:>4}  #{r['rank']:<3}"
            )

        print()

    # ── 综合通过率 ──
    if len(cases) > 1:
        print("=" * 70)
        print("综合通过率（所有 case）")
        print("=" * 70)
        print(f"{'run_id':<22} {'ep':>6}  {'通过/总数':>10}  {'通过率':>8}")
        print("-" * 70)
        for run_dir in run_dirs:
            meta    = load_meta(run_dir)
            episodes = meta.get("config", {}).get("episodes", "?")
            net = load_net(run_dir)
            if net is None:
                continue
            passed = sum(eval_case(net, c)["correct"] for c in cases)
            total  = len(cases)
            pct    = passed / total * 100
            bar    = "█" * passed + "░" * (total - passed)
            print(f"{run_dir.name:<22} {str(episodes):>6}  {passed}/{total} {bar}  {pct:>6.1f}%")
        print()


if __name__ == "__main__":
    main()
