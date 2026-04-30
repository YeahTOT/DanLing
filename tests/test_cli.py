"""测试 CLI 基本可用性。"""

import json

from typer.testing import CliRunner

from danling.cli import app

runner = CliRunner()


def test_version() -> None:
    """danling --version 应正常输出并返回 0。"""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "danling v" in result.stdout


def test_help() -> None:
    """danling --help 应展示 status/watch/doctor 命令。"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "status" in result.stdout
    assert "watch" in result.stdout
    assert "doctor" in result.stdout


def test_doctor() -> None:
    """danling doctor 应返回 0。"""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_status() -> None:
    """danling status PATH 应返回 0。"""
    result = runner.invoke(app, ["status", "tests/fixtures/ultralytics_run", "--source", "csv"])
    assert result.exit_code == 0


def test_statusline() -> None:
    """danling statusline PATH 应返回单行状态。"""
    result = runner.invoke(app, ["statusline", "tests/fixtures/ultralytics_run", "--source", "csv"])
    assert result.exit_code == 0
    assert "DanLing" in result.stdout


def test_state_json() -> None:
    """danling state PATH --json 应输出聚合状态。"""
    result = runner.invoke(
        app,
        [
            "state",
            "tests/fixtures/ultralytics_run",
            "--source",
            "csv",
            "--json",
            "--no-save",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "pet_mood" in data
    assert "name" in data["realm"]
    assert "level" not in data


def test_config_tui_missing_textual_has_clear_message(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "textual" else object()
    )

    result = runner.invoke(app, ["config", "tui"])

    assert result.exit_code == 1
    assert "danling[tui]" in result.stdout


def test_doctor_path_json() -> None:
    """danling doctor PATH --json 应输出诊断列表。"""
    result = runner.invoke(
        app, ["doctor", "tests/fixtures/ultralytics_run", "--source", "csv", "--json"]
    )
    assert result.exit_code == 0
    assert result.stdout.strip().startswith("[")


def test_simulate_cli_writes_results_csv(tmp_path) -> None:
    """danling simulate 应把源 CSV 回放到目标 run 目录。"""
    output_dir = tmp_path / "run"

    result = runner.invoke(
        app,
        [
            "simulate",
            "tests/fixtures/ultralytics_run/results.csv",
            str(output_dir),
            "--interval",
            "0",
            "--max-rows",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "results.csv").is_file()
    assert "2/5" in result.stdout
