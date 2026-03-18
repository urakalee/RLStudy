"""
测试 train.py 各子命令的集成行为。
使用临时目录隔离 checkpoints，不污染真实数据。
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

import train as train_mod


# ─── fixture：把 CKPT_ROOT 重定向到临时目录 ──────────────────────────────────

@pytest.fixture()
def tmp_ckpt(tmp_path, monkeypatch):
    """每个测试用独立的临时 checkpoints 目录。"""
    monkeypatch.setattr(train_mod, "CKPT_ROOT", tmp_path / "checkpoints")
    return tmp_path / "checkpoints"


# ─── train ───────────────────────────────────────────────────────────────────

def _make_train_args(**kwargs):
    defaults = dict(
        episodes=30,
        board_size=5,
        win_count=3,
        log_interval=10,
        resume=None,
    )
    defaults.update(kwargs)

    class Args:
        pass

    a = Args()
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


def test_train_creates_checkpoint_files(tmp_ckpt):
    train_mod.cmd_train(_make_train_args())
    runs = list(tmp_ckpt.iterdir())
    assert len(runs) == 1, "应创建恰好一个 run 目录"
    run_dir = runs[0]
    for fname in ("online_net.pt", "target_net.pt", "optimizer.pt", "meta.json", "log.json"):
        assert (run_dir / fname).exists(), f"缺少文件: {fname}"


def test_train_meta_contains_expected_keys(tmp_ckpt):
    train_mod.cmd_train(_make_train_args(episodes=20))
    run_dir = next(tmp_ckpt.iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    for key in ("run_id", "steps", "config", "saved_at"):
        assert key in meta


def test_train_log_win_history_length(tmp_ckpt):
    n = 20
    train_mod.cmd_train(_make_train_args(episodes=n))
    run_dir = next(tmp_ckpt.iterdir())
    log = json.loads((run_dir / "log.json").read_text())
    assert len(log["win_history"]) == n


def test_train_resume_continues_from_checkpoint(tmp_ckpt):
    """先训练 20 局保存，再 resume 20 局，最终 win_history 共 40 条。"""
    train_mod.cmd_train(_make_train_args(episodes=20))
    run_id = next(tmp_ckpt.iterdir()).name

    train_mod.cmd_train(_make_train_args(episodes=20, resume=run_id))

    run_dir = tmp_ckpt / run_id
    log = json.loads((run_dir / "log.json").read_text())
    assert len(log["win_history"]) == 40


# ─── list ────────────────────────────────────────────────────────────────────

def test_list_empty(tmp_ckpt, capsys):
    class Args:
        pass
    train_mod.cmd_list(Args())
    out = capsys.readouterr().out
    assert "暂无" in out


def test_list_shows_run(tmp_ckpt, capsys):
    train_mod.cmd_train(_make_train_args(episodes=10))

    class Args:
        pass
    train_mod.cmd_list(Args())
    out = capsys.readouterr().out
    assert "5×5" in out


# ─── eval ────────────────────────────────────────────────────────────────────

def _make_eval_args(run=None, games=20):
    class Args:
        pass
    a = Args()
    a.run = run
    a.games = games
    return a


def test_eval_runs_without_error(tmp_ckpt, capsys):
    train_mod.cmd_train(_make_train_args(episodes=30))
    train_mod.cmd_eval(_make_eval_args(games=10))
    out = capsys.readouterr().out
    assert "先手" in out
    assert "后手" in out


# ─── plot ────────────────────────────────────────────────────────────────────

def _make_plot_args(run=None):
    class Args:
        pass
    a = Args()
    a.run = run
    return a


def test_plot_generates_html(tmp_ckpt):
    train_mod.cmd_train(_make_train_args(episodes=20))
    train_mod.cmd_plot(_make_plot_args())
    run_dir = next(tmp_ckpt.iterdir())
    html_file = run_dir / "curves.html"
    assert html_file.exists()
    content = html_file.read_text(encoding="utf-8")
    assert "<canvas" in content
    assert "Chart" in content


# ─── 加载不存在的 run_id 应退出 ──────────────────────────────────────────────

def test_load_nonexistent_run_id_exits(tmp_ckpt):
    with pytest.raises(SystemExit):
        train_mod.load_checkpoint("nonexistent_run_id")


def test_load_with_no_runs_exits(tmp_ckpt):
    with pytest.raises(SystemExit):
        train_mod.load_checkpoint(None)
