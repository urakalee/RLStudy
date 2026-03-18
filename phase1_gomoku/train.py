#!/usr/bin/env python3
"""
DQN 五子棋训练程序

用法：
  python train.py train                        # 训练（20000 局，保存到 checkpoints/）
  python train.py train --episodes 5000        # 自定义局数
  python train.py train --resume <run_id>      # 接着某次训练继续跑
  python train.py list                         # 列出所有保存的训练结果
  python train.py eval                         # 评估最新权重（vs 随机）
  python train.py eval --run <run_id>          # 评估指定权重
  python train.py plot                         # 生成训练曲线 HTML（最新）
  python train.py plot --run <run_id>          # 指定训练结果
  python train.py heatmap                      # 交互式 Q 值热力图
  python train.py heatmap --run <run_id>
  python train.py play                         # 终端人机对战（你执黑）
  python train.py play --run <run_id>
  python train.py play --you white             # 你执白
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# 把 dqn 包所在目录加入路径，支持从任意 cwd 调用
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from dqn import DEVICE, DQNAgent, GomokuEnv

# ─── 路径约定 ────────────────────────────────────────────────────────────────

CKPT_ROOT = _HERE / "checkpoints"

# ─── 权重管理 ────────────────────────────────────────────────────────────────

def _run_dir(run_id: str) -> Path:
    return CKPT_ROOT / run_id


def _latest_run_id() -> str | None:
    if not CKPT_ROOT.exists():
        return None
    runs = sorted(CKPT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir()]
    return runs[0].name if runs else None


def save_checkpoint(agent: DQNAgent, run_id: str, config: dict, log: dict):
    d = _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(agent.online_net.state_dict(), d / "online_net.pt")
    torch.save(agent.target_net.state_dict(), d / "target_net.pt")
    torch.save(agent.optimizer.state_dict(), d / "optimizer.pt")
    meta = {
        "run_id": run_id,
        "steps": agent.steps,
        "config": config,
        "saved_at": datetime.now().isoformat(),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (d / "log.json").write_text(json.dumps(log, ensure_ascii=False))
    print(f"[保存] {d}")


def load_checkpoint(run_id: str | None = None) -> tuple[DQNAgent, dict, dict]:
    """加载权重，返回 (agent, config, log)。"""
    if run_id is None:
        run_id = _latest_run_id()
        if run_id is None:
            sys.exit("找不到任何已保存的训练结果，请先运行 train。")
        print(f"[加载] 最新结果: {run_id}")
    else:
        print(f"[加载] {run_id}")

    d = _run_dir(run_id)
    if not d.exists():
        sys.exit(f"找不到 {d}，请用 'list' 查看可用的 run_id。")

    meta = json.loads((d / "meta.json").read_text())
    log = json.loads((d / "log.json").read_text())
    cfg = meta["config"]

    agent = DQNAgent(
        board_size=cfg["board_size"],
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        eps_start=cfg.get("eps_end", 0.05),   # 恢复时从 eps_end 继续
        eps_end=cfg["eps_end"],
        eps_decay=cfg["eps_decay"],
        batch_size=cfg["batch_size"],
        buffer_capacity=cfg["buffer_capacity"],
        target_update_freq=cfg["target_update_freq"],
    )
    agent.online_net.load_state_dict(torch.load(d / "online_net.pt", map_location=DEVICE, weights_only=True))
    agent.target_net.load_state_dict(torch.load(d / "target_net.pt", map_location=DEVICE, weights_only=True))
    agent.optimizer.load_state_dict(torch.load(d / "optimizer.pt", map_location=DEVICE, weights_only=True))
    agent.steps = meta["steps"]
    return agent, cfg, log


# ─── 训练 ────────────────────────────────────────────────────────────────────

def cmd_train(args):
    config = {
        "board_size":        args.board_size,
        "win_count":         args.win_count,
        "episodes":          args.episodes,
        "gamma":             0.95,
        "lr":                1e-3,
        "eps_start":         1.0,
        "eps_end":           0.05,
        "eps_decay":         5000,
        "batch_size":        64,
        "buffer_capacity":   20_000,
        "target_update_freq":200,
        "log_interval":      args.log_interval,
    }

    if args.resume:
        agent, saved_cfg, log = load_checkpoint(args.resume)
        config.update(saved_cfg)
        run_id = args.resume
        print(f"接着第 {agent.steps} 步继续训练，目标再跑 {args.episodes} 局")
    else:
        agent = DQNAgent(
            board_size=config["board_size"],
            gamma=config["gamma"],
            lr=config["lr"],
            eps_end=config["eps_end"],
            eps_decay=config["eps_decay"],
            batch_size=config["batch_size"],
            buffer_capacity=config["buffer_capacity"],
            target_update_freq=config["target_update_freq"],
        )
        log = {"win_history": [], "loss_history": [], "eps_history": []}
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = GomokuEnv(board_size=config["board_size"], win_count=config["win_count"])
    n = config["episodes"]
    interval = config["log_interval"]

    print(f"\n▶ 开始训练  run_id={run_id}  设备={DEVICE}")
    print(f"  棋盘={config['board_size']}×{config['board_size']}  连赢={config['win_count']}  局数={n}\n")

    t0 = time.time()
    recent_wins = []

    for ep in range(n):
        state = env.reset()
        done = False

        while not done:
            player = env.current_player
            legal = env.get_legal_actions()
            action = agent.select_action(state, player, legal)
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done, player)
            loss = agent.train_step()
            if loss > 0:
                log["loss_history"].append(loss)
            state = next_state

        log["win_history"].append(info["winner"])
        log["eps_history"].append(agent.epsilon)
        recent_wins.append(info["winner"])

        if (ep + 1) % interval == 0:
            bw = sum(1 for w in recent_wins if w == 1)
            ww = sum(1 for w in recent_wins if w == -1)
            dr = sum(1 for w in recent_wins if w == 0)
            avg_loss = float(np.mean(log["loss_history"][-interval:])) if log["loss_history"] else 0.0
            elapsed = time.time() - t0
            eps_per_s = (ep + 1) / elapsed
            eta = (n - ep - 1) / eps_per_s

            # ASCII 进度条（50格）
            done_frac = (ep + 1) / n
            bar_done = int(done_frac * 40)
            bar = "█" * bar_done + "░" * (40 - bar_done)
            pct = done_frac * 100

            print(
                f"[{bar}] {pct:5.1f}%  "
                f"Ep {ep+1:>5}/{n}  "
                f"ε={agent.epsilon:.3f}  "
                f"loss={avg_loss:.4f}  "
                f"黑={bw} 白={ww} 平={dr}  "
                f"ETA {eta:.0f}s"
            )
            recent_wins = []

    elapsed = time.time() - t0
    print(f"\n训练完成，耗时 {elapsed:.1f}s")
    save_checkpoint(agent, run_id, config, log)
    print(f"\n提示：运行以下命令查看结果：")
    print(f"  python train.py plot --run {run_id}")
    print(f"  python train.py eval --run {run_id}")


# ─── list ────────────────────────────────────────────────────────────────────

def cmd_list(_args):
    if not CKPT_ROOT.exists() or not any(CKPT_ROOT.iterdir()):
        print("暂无保存的训练结果。")
        return

    runs = sorted(CKPT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir()]
    print(f"{'run_id':<25}  {'局数':>6}  {'步数':>8}  {'棋盘':>6}  保存时间")
    print("-" * 70)
    for r in runs:
        meta_file = r / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        cfg = meta["config"]
        ep = len(json.loads((r / "log.json").read_text()).get("win_history", []))
        board_str = f"{cfg['board_size']}×{cfg['board_size']}"
        saved_at = meta.get("saved_at", "")[:16].replace("T", " ")
        print(f"{r.name:<25}  {ep:>6}  {meta['steps']:>8}  {board_str:>6}  {saved_at}")


# ─── eval ────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    agent, cfg, _ = load_checkpoint(args.run)
    env = GomokuEnv(board_size=cfg["board_size"], win_count=cfg["win_count"])
    n = args.games

    print(f"\n▶ 评估（vs 随机，各 {n} 局）  run_id={args.run or _latest_run_id()}\n")

    for dqn_player, label in [(1, "先手(黑)"), (-1, "后手(白)")]:
        wins = draws = losses = 0
        for _ in range(n):
            state = env.reset()
            done = False
            while not done:
                legal = env.get_legal_actions()
                if env.current_player == dqn_player:
                    action = agent.greedy_action(state, env.current_player, legal)
                else:
                    action = int(np.random.choice(legal))
                state, _, done, info = env.step(action)
            w = info["winner"]
            if w == dqn_player:
                wins += 1
            elif w == 0:
                draws += 1
            else:
                losses += 1
        print(f"  DQN {label} vs 随机 {n}局: 赢={wins}({wins/n*100:.1f}%)  平={draws}  输={losses}")

    print("\n（随机基准：综合胜率约 50%）")


# ─── plot（HTML）─────────────────────────────────────────────────────────────

def cmd_plot(args):
    run_id = args.run or _latest_run_id()
    if run_id is None:
        sys.exit("找不到训练结果。")

    _, cfg, log = load_checkpoint(run_id)

    win_history = log.get("win_history", [])
    loss_history = log.get("loss_history", [])
    eps_history = log.get("eps_history", [])

    def smooth(data, w=200):
        if len(data) < w:
            return list(range(len(data))), data
        kernel = np.ones(w) / w
        smoothed = np.convolve(data, kernel, mode="valid").tolist()
        xs = list(range(w // 2, w // 2 + len(smoothed)))
        return xs, smoothed

    # 胜率序列
    black_wins = [1 if w == 1 else 0 for w in win_history]
    white_wins = [1 if w == -1 else 0 for w in win_history]

    bx, by = smooth(black_wins)
    wx, wy = smooth(white_wins)
    lx, ly = smooth(loss_history, w=min(200, max(1, len(loss_history) // 10)))

    n_ep = len(win_history)
    board_str = f"{cfg['board_size']}×{cfg['board_size']}"

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>训练曲线 — {run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: sans-serif; background:#f5f5f5; padding:20px; }}
  h2   {{ color:#333; }}
  .meta {{ color:#666; font-size:14px; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .card {{ background:#fff; border-radius:8px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  .full {{ grid-column:1/-1; }}
  canvas{{ max-height:300px; }}
</style>
</head>
<body>
<h2>训练曲线</h2>
<div class="meta">
  run_id: <b>{run_id}</b> &nbsp;|&nbsp;
  棋盘: {board_str} &nbsp;|&nbsp;
  总局数: {n_ep} &nbsp;|&nbsp;
  总步数: {cfg.get('episodes','?')}
</div>
<div class="grid">

  <div class="card">
    <h3>胜率（滑动平均 200 局）</h3>
    <canvas id="winChart"></canvas>
  </div>

  <div class="card">
    <h3>训练 Loss（滑动平均）</h3>
    <canvas id="lossChart"></canvas>
  </div>

  <div class="card full">
    <h3>ε 探索率衰减</h3>
    <canvas id="epsChart"></canvas>
  </div>

</div>
<script>
const winCtx  = document.getElementById('winChart').getContext('2d');
const lossCtx = document.getElementById('lossChart').getContext('2d');
const epsCtx  = document.getElementById('epsChart').getContext('2d');

new Chart(winCtx, {{
  type: 'line',
  data: {{
    labels: {bx},
    datasets: [
      {{ label:'黑方胜率', data:{by}, borderColor:'#222', backgroundColor:'rgba(0,0,0,.05)', pointRadius:0, tension:.3 }},
      {{ label:'白方胜率', data:{wy}, borderColor:'#999', backgroundColor:'rgba(0,0,0,.02)', pointRadius:0, tension:.3, borderDash:[4,4] }},
      {{ label:'50% 基准', data: Array({len(bx)}).fill(0.5), borderColor:'#e44', pointRadius:0, borderDash:[2,4] }},
    ]
  }},
  options: {{ scales: {{ y: {{ min:0, max:1 }} }}, plugins: {{ legend: {{ position:'top' }} }} }}
}});

new Chart(lossCtx, {{
  type: 'line',
  data: {{
    labels: {lx},
    datasets: [{{ label:'Loss', data:{ly}, borderColor:'#3a7bd5', backgroundColor:'rgba(58,123,213,.05)', pointRadius:0, tension:.3 }}]
  }},
  options: {{ plugins: {{ legend: {{ display:false }} }} }}
}});

new Chart(epsCtx, {{
  type: 'line',
  data: {{
    labels: {list(range(len(eps_history)))},
    datasets: [{{ label:'ε', data:{eps_history}, borderColor:'#e8901a', backgroundColor:'rgba(232,144,26,.05)', pointRadius:0, tension:.1 }}]
  }},
  options: {{ scales: {{ y: {{ min:0, max:1.05 }} }}, plugins: {{ legend: {{ display:false }} }} }}
}});
</script>
</body>
</html>"""

    out_path = _run_dir(run_id) / "curves.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[生成] {out_path}")
    print(f"用浏览器打开：file://{out_path.resolve()}")


# ─── heatmap（交互式 + HTML）────────────────────────────────────────────────

def _render_board_ascii(board: np.ndarray) -> str:
    n = board.shape[0]
    sym = {0: "·", 1: "X", -1: "O"}
    header = "    " + "  ".join(f"{c}" for c in range(n))
    lines = [header]
    for r in range(n):
        row = "  ".join(sym[int(board[r, c])] for c in range(n))
        lines.append(f"{r:>2}  {row}")
    return "\n".join(lines)


def _q_heatmap_html(board: np.ndarray, q_vals: np.ndarray, player: int, run_id: str, idx: int) -> Path:
    n = board.shape[0]
    q_map = q_vals.reshape(n, n).tolist()
    board_list = board.tolist()

    out_dir = _run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{idx:03d}.html"

    q_flat = [q_vals[r * n + c] for r in range(n) for c in range(n)]
    q_min, q_max = float(min(q_flat)), float(max(q_flat))
    player_str = "黑(X)" if player == 1 else "白(O)"

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Q 值热力图</title>
<style>
  body {{ font-family:sans-serif; background:#f5f5f5; padding:20px; }}
  h2 {{ color:#333; }}
  .meta {{ color:#666; font-size:13px; margin-bottom:16px; }}
  .wrap {{ display:flex; gap:32px; align-items:flex-start; }}
  table {{ border-collapse:collapse; }}
  td {{
    width:52px; height:52px; text-align:center; vertical-align:middle;
    font-size:12px; font-weight:bold; border:1px solid #ccc;
    position:relative;
  }}
  .piece {{ font-size:22px; display:block; }}
  .qval  {{ font-size:10px; color:#333; }}
  .legend {{ margin-top:16px; font-size:13px; color:#555; }}
</style>
</head>
<body>
<h2>Q 值热力图</h2>
<div class="meta">run_id: <b>{run_id}</b> &nbsp;|&nbsp; 当前视角: <b>{player_str}</b> &nbsp;|&nbsp; Q 值范围: [{q_min:.2f}, {q_max:.2f}]</div>
<div class="wrap">
<div>
<table>
"""
    # 表头
    html += "<tr><td></td>" + "".join(f"<td><b>{c}</b></td>" for c in range(n)) + "</tr>\n"

    for r in range(n):
        html += f"<tr><td><b>{r}</b></td>"
        for c in range(n):
            q = q_map[r][c]
            piece = board_list[r][c]
            # 颜色插值：红(负) → 白(0) → 绿(正)
            norm = (q - q_min) / (q_max - q_min + 1e-9)
            if norm < 0.5:
                t = norm * 2
                R, G, B = int(220 + (255-220)*t), int(50 + (255-50)*t), int(50 + (255-50)*t)
            else:
                t = (norm - 0.5) * 2
                R, G, B = int(255 - (255-50)*t), 255, int(255 - (255-50)*t)
            bg = f"rgb({R},{G},{B})"
            piece_sym = {"1": "⚫", "-1": "⚪", "0": ""}.get(str(piece), "")
            html += f'<td style="background:{bg}"><span class="piece">{piece_sym}</span><span class="qval">{q:.2f}</span></td>'
        html += "</tr>\n"

    html += f"""</table>
</div>
<div class="legend">
  <b>图例</b><br>
  ⚫ = 黑棋(X) &nbsp; ⚪ = 白棋(O)<br><br>
  <b>颜色含义</b><br>
  <span style="background:rgb(50,255,50);padding:2px 8px">■</span> 高 Q 值（倾向落子）<br>
  <span style="background:rgb(255,255,255);padding:2px 8px;border:1px solid #ccc">■</span> 中性<br>
  <span style="background:rgb(220,50,50);padding:2px 8px">■</span> 低 Q 值（不倾向落子）
</div>
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


def cmd_heatmap(args):
    run_id = args.run or _latest_run_id()
    agent, cfg, _ = load_checkpoint(args.run)
    n = cfg["board_size"]
    board = np.zeros((n, n), dtype=np.int8)
    player = 1  # 默认黑方视角
    heatmap_idx = 0

    print(f"\n▶ Q 值热力图  run_id={run_id}  棋盘={n}×{n}")
    print("命令：b <行> <列>  落黑棋 | w <行> <列>  落白棋")
    print("      show         生成热力图 HTML  | clear  清空棋盘")
    print("      perspective black/white  切换视角 | q  退出\n")

    while True:
        print(_render_board_ascii(board))
        print(f"当前视角: {'黑(X)' if player == 1 else '白(O)'}")
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "q":
            break

        elif cmd == "clear":
            board = np.zeros((n, n), dtype=np.int8)
            print("棋盘已清空。")

        elif cmd in ("b", "w"):
            if len(parts) != 3:
                print("格式：b <行> <列>  或  w <行> <列>")
                continue
            try:
                r, c = int(parts[1]), int(parts[2])
            except ValueError:
                print("行列必须是整数。")
                continue
            if not (0 <= r < n and 0 <= c < n):
                print(f"坐标超出范围（0~{n-1}）。")
                continue
            board[r, c] = 1 if cmd == "b" else -1

        elif cmd == "perspective":
            if len(parts) < 2 or parts[1] not in ("black", "white"):
                print("用法：perspective black  或  perspective white")
                continue
            player = 1 if parts[1] == "black" else -1
            print(f"视角切换为 {'黑(X)' if player == 1 else '白(O)'}。")

        elif cmd == "show":
            q_vals = agent.q_values(board, player)
            out = _q_heatmap_html(board, q_vals, player, run_id, heatmap_idx)
            heatmap_idx += 1
            print(f"[生成] {out}")
            print(f"用浏览器打开：file://{out.resolve()}")

        else:
            print("未知命令。")


# ─── play ────────────────────────────────────────────────────────────────────

def cmd_play(args):
    agent, cfg, _ = load_checkpoint(args.run)
    n = cfg["board_size"]
    env = GomokuEnv(board_size=n, win_count=cfg["win_count"])

    human_player = 1 if args.you == "black" else -1
    ai_player = -human_player
    human_str = "黑(X)" if human_player == 1 else "白(O)"
    ai_str = "黑(X)" if ai_player == 1 else "白(O)"

    print(f"\n▶ 人机对战  你={human_str}  AI={ai_str}  棋盘={n}×{n}")
    print("输入格式：<行> <列>（如 2 3）  |  q 退出\n")

    state = env.reset()

    while not env.done:
        print(_render_board_ascii(env.board))

        if env.current_player == human_player:
            while True:
                try:
                    line = input(f"\n你的落子 [{human_str}] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n退出。")
                    return
                if line.lower() == "q":
                    print("退出。")
                    return
                parts = line.split()
                if len(parts) != 2:
                    print("请输入 行 列，如：2 3")
                    continue
                try:
                    r, c = int(parts[0]), int(parts[1])
                except ValueError:
                    print("行列必须是整数。")
                    continue
                if not (0 <= r < n and 0 <= c < n):
                    print(f"坐标超出范围（0~{n-1}）。")
                    continue
                action = r * n + c
                if env.board[r, c] != 0:
                    print("该位置已有棋子，请重新选择。")
                    continue
                break
        else:
            legal = env.get_legal_actions()
            action = agent.greedy_action(state, env.current_player, legal)
            r, c = divmod(action, n)
            print(f"\nAI [{ai_str}] 落子: ({r}, {c})")

        state, _, done, info = env.step(action)

    print("\n" + _render_board_ascii(env.board))
    winner = info["winner"]
    if winner == human_player:
        print("\n🎉 你赢了！")
    elif winner == ai_player:
        print("\n🤖 AI 赢了。")
    else:
        print("\n平局。")


# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DQN 五子棋训练程序", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="训练 DQN Agent")
    p_train.add_argument("--episodes",     type=int, default=20000)
    p_train.add_argument("--board-size",   type=int, default=5, dest="board_size")
    p_train.add_argument("--win-count",    type=int, default=3, dest="win_count")
    p_train.add_argument("--log-interval", type=int, default=500, dest="log_interval")
    p_train.add_argument("--resume",       type=str, default=None, help="接着某次 run_id 继续训练")

    # list
    sub.add_parser("list", help="列出所有保存的训练结果")

    # eval
    p_eval = sub.add_parser("eval", help="评估（vs 随机对战）")
    p_eval.add_argument("--run",   type=str, default=None)
    p_eval.add_argument("--games", type=int, default=500)

    # plot
    p_plot = sub.add_parser("plot", help="生成训练曲线 HTML")
    p_plot.add_argument("--run", type=str, default=None)

    # heatmap
    p_heat = sub.add_parser("heatmap", help="交互式 Q 值热力图")
    p_heat.add_argument("--run", type=str, default=None)

    # play
    p_play = sub.add_parser("play", help="终端人机对战")
    p_play.add_argument("--run", type=str, default=None)
    p_play.add_argument("--you", choices=["black", "white"], default="black", help="你执黑还是白（默认黑）")

    args = parser.parse_args()
    {"train": cmd_train, "list": cmd_list, "eval": cmd_eval, "plot": cmd_plot, "heatmap": cmd_heatmap, "play": cmd_play}[args.cmd](args)


if __name__ == "__main__":
    main()
