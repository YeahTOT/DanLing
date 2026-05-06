"""测试 CLI 基本可用性。"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from typer.testing import CliRunner

from danling.cli import app
from danling.models import DanLingState, HardwareSnapshot, MetricSnapshot, PetMood
from danling.remote import RemoteProfile

runner = CliRunner()


@dataclass(frozen=True)
class FakeScalarEvent:
    step: int
    value: float
    wall_time: float


def install_fake_tensorboard(
    monkeypatch,
    events_by_file: dict[str, dict[str, list[FakeScalarEvent]]],
):
    class FakeEventAccumulator:
        def __init__(self, path: str) -> None:
            self.path = path

        def Reload(self) -> None:
            return None

        def Tags(self) -> dict[str, list[str]]:
            return {"scalars": list(events_by_file.get(self.path, {}))}

        def Scalars(self, tag: str) -> list[FakeScalarEvent]:
            return events_by_file[self.path][tag]

    tensorboard = ModuleType("tensorboard")
    backend = ModuleType("tensorboard.backend")
    event_processing = ModuleType("tensorboard.backend.event_processing")
    event_accumulator = ModuleType("tensorboard.backend.event_processing.event_accumulator")
    event_accumulator.EventAccumulator = FakeEventAccumulator

    monkeypatch.setitem(sys.modules, "tensorboard", tensorboard)
    monkeypatch.setitem(sys.modules, "tensorboard.backend", backend)
    monkeypatch.setitem(sys.modules, "tensorboard.backend.event_processing", event_processing)
    monkeypatch.setitem(
        sys.modules,
        "tensorboard.backend.event_processing.event_accumulator",
        event_accumulator,
    )


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


def test_remote_status_uses_saved_remote_profile(monkeypatch, tmp_path) -> None:
    created: dict[str, object] = {}
    profile = RemoteProfile(
        name="lab",
        host="trainbox",
        remote_path="/runs/exp",
        identity=tmp_path / "id_ed25519",
        port=2222,
    )

    class FakeRemoteMonitor:
        def __init__(
            self,
            remote_profile: RemoteProfile,
            config_path: Path | None,
            *,
            ssh_options: tuple[str, ...],
            remote_hardware: bool,
            sync_timeout: float,
        ) -> None:
            created.update(
                {
                    "profile": remote_profile,
                    "config_path": config_path,
                    "ssh_options": ssh_options,
                    "remote_hardware": remote_hardware,
                    "sync_timeout": sync_timeout,
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read_state(self) -> DanLingState:
            return DanLingState(
                pet_mood=PetMood.happy,
                metric=MetricSnapshot(epoch=8, train_loss=1.2, score=0.7, score_name="mAP50"),
                hardware=[HardwareSnapshot(device_type="nvidia", util_percent=66)],
            )

    monkeypatch.setattr("danling.cli.RemoteTrainingMonitor", FakeRemoteMonitor, raising=False)
    monkeypatch.setattr("danling.cli.load_remote_profile", lambda name: profile)

    result = runner.invoke(
        app,
        [
            "remote",
            "lab",
            "status",
            "--ssh-option",
            "StrictHostKeyChecking=no",
            "--sync-timeout",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert "mAP50" in result.stdout
    assert created == {
        "profile": profile,
        "config_path": None,
        "ssh_options": ("StrictHostKeyChecking=no",),
        "remote_hardware": True,
        "sync_timeout": 4.0,
    }


def test_remote_status_without_profile_shows_setup_hint(monkeypatch) -> None:
    def raise_missing(name):
        raise KeyError(name)

    monkeypatch.setattr("danling.cli.load_remote_profile", raise_missing)

    result = runner.invoke(app, ["remote", "missing", "status"])

    assert result.exit_code == 1
    assert "danling remote setup missing" in result.stderr


def test_remote_hardware_renders_saved_profile_gpu(monkeypatch, tmp_path) -> None:
    profile = RemoteProfile(
        name="lab",
        host="trainbox",
        remote_path="/runs/exp",
        identity=tmp_path / "id_ed25519",
    )
    created: dict[str, object] = {}

    def fake_read_hardware(options):
        created["options"] = options
        return [
            HardwareSnapshot(
                device_type="nvidia",
                device_id="0",
                name="RTX 4090",
                util_percent=76,
                memory_used_mb=21_400,
                memory_total_mb=24_000,
                temperature_c=62,
                power_w=310,
            )
        ]

    monkeypatch.setattr("danling.cli.load_remote_profile", lambda name: profile)
    monkeypatch.setattr("danling.cli.read_remote_nvidia_hardware", fake_read_hardware)

    result = runner.invoke(app, ["remote", "lab", "hardware"])

    assert result.exit_code == 0
    assert "RTX 4090" in result.stdout
    assert "76%" in result.stdout
    assert "310W" in result.stdout
    assert created["options"].host == "trainbox"


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


def test_simulate_cli_writes_results_csv_from_tensorboard(tmp_path, monkeypatch) -> None:
    """danling simulate 应能把 TensorBoard scalar 历史回放为 results.csv。"""
    train_event = tmp_path / "tb" / "train" / "events.out.tfevents.train"
    val_event = tmp_path / "tb" / "val" / "events.out.tfevents.val"
    output_dir = tmp_path / "run"
    train_event.parent.mkdir(parents=True)
    val_event.parent.mkdir(parents=True)
    train_event.write_text("", encoding="utf-8")
    val_event.write_text("", encoding="utf-8")
    install_fake_tensorboard(
        monkeypatch,
        {
            str(train_event): {
                "Loss/total": [
                    FakeScalarEvent(step=1, value=1.2, wall_time=10.0),
                    FakeScalarEvent(step=2, value=1.0, wall_time=20.0),
                ],
                "LearningRate/group0": [
                    FakeScalarEvent(step=2, value=0.01, wall_time=20.0),
                ],
            },
            str(val_event): {
                "Loss/total": [
                    FakeScalarEvent(step=1, value=1.5, wall_time=11.0),
                    FakeScalarEvent(step=2, value=1.3, wall_time=21.0),
                ],
                "mAP50": [
                    FakeScalarEvent(step=1, value=0.6, wall_time=11.0),
                    FakeScalarEvent(step=2, value=0.7, wall_time=21.0),
                ],
            },
        },
    )

    result = runner.invoke(
        app,
        [
            "simulate",
            str(tmp_path / "tb"),
            str(output_dir),
            "--source",
            "tensorboard",
            "--interval",
            "0",
            "--max-rows",
            "2",
        ],
    )

    assert result.exit_code == 0
    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[0]["time"] == "11.0"
    assert rows[0]["train/loss"] == "1.2"
    assert rows[0]["val/loss"] == "1.5"
    assert rows[0]["metrics/mAP50(B)"] == "0.6"
    assert rows[1]["epoch"] == "2"
    assert rows[1]["time"] == "21.0"
    assert rows[1]["train/loss"] == "1.0"
    assert rows[1]["val/loss"] == "1.3"
    assert rows[1]["metrics/mAP50(B)"] == "0.7"
    assert rows[1]["lr"] == "0.01"
    assert rows[1]["train/Loss/total"] == "1.0"
    assert rows[1]["val/Loss/total"] == "1.3"
    assert rows[1]["val/mAP50"] == "0.7"
    assert "2/2" in result.stdout
