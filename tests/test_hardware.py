"""测试硬件监控抽象层。"""

from __future__ import annotations

import subprocess

from typer.testing import CliRunner

from danling.cli import app
from danling.hardware import read_all_hardware
from danling.hardware.nvidia import NvidiaSMIReader

runner = CliRunner()


def test_nvidia_reader_parses_multiple_gpus(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="0, RTX 4090, 78, 21400, 24564, 70, 310\n1, RTX 3090, 5, 1000, 24576, 40, 80\n",
        stderr="",
    )
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)

    hardware = NvidiaSMIReader().read()

    assert len(hardware) == 2
    assert hardware[0].device_type == "nvidia"
    assert hardware[0].device_id == "0"
    assert hardware[0].util_percent == 78
    assert hardware[0].memory_used_mb == 21400


def test_nvidia_reader_timeout_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3)

    monkeypatch.setattr("subprocess.run", raise_timeout)

    assert NvidiaSMIReader().read() == []


def test_read_all_hardware_no_commands(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)

    assert read_all_hardware() == []


def test_hardware_cli_json(monkeypatch) -> None:
    monkeypatch.setattr("danling.cli.read_all_hardware", lambda: [])

    result = runner.invoke(app, ["hardware", "--json"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "[]"
