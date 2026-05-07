"""Tests for SSH-key-backed remote CSV training visualization."""

from __future__ import annotations

import subprocess

import pytest

from danling.models import HardwareSnapshot, MetricSnapshot
from danling.remote import (
    RemoteKeyRequired,
    RemoteOptions,
    RemoteProfile,
    RemoteProfileStore,
    RemoteTrainingMonitor,
    build_ssh_command,
    load_remote_profile,
    read_remote_csv_to_cache,
    read_remote_history_to_cache,
    read_remote_nvidia_hardware,
    save_remote_profile,
)


def test_build_ssh_command_requires_identity_key() -> None:
    with pytest.raises(RemoteKeyRequired, match="SSH key is required"):
        build_ssh_command(RemoteOptions(host="trainbox"), "true")


def test_build_ssh_command_forces_batchmode_with_identity(tmp_path) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    options = RemoteOptions(
        host="trainbox",
        port=2222,
        identity=identity,
        ssh_options=("StrictHostKeyChecking=no",),
    )

    command = build_ssh_command(options, "echo ok")

    assert command == [
        "ssh",
        "-p",
        "2222",
        "-i",
        str(identity),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "trainbox",
        "echo ok",
    ]


def test_remote_profile_store_round_trips_profiles(tmp_path) -> None:
    store = RemoteProfileStore(tmp_path / "profiles.json")
    profile = RemoteProfile(
        name="lab",
        host="tot@127.0.0.1",
        remote_path="/runs/exp",
        identity=tmp_path / "id_ed25519",
        port=2222,
        source="csv",
    )

    save_remote_profile(profile, store=store)

    loaded = load_remote_profile("lab", store=store)
    assert loaded == profile


def test_read_remote_csv_to_cache_reads_results_csv_over_ssh(tmp_path, monkeypatch) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "__DANLING_CSV_PATH__/remote/results.csv\n"
            "__DANLING_CSV_MTIME__1700000000\n"
            "__DANLING_REMOTE_NOW__1700000010\n"
            "epoch,metrics/mAP50(B)\n"
            "1,0.5\n",
            "",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    local_path = read_remote_csv_to_cache(
        RemoteOptions("trainbox", identity=identity),
        "/remote",
        tmp_path / "cache",
    )

    assert local_path == tmp_path / "cache"
    assert (local_path / "results.csv").read_text(encoding="utf-8") == (
        "epoch,metrics/mAP50(B)\n1,0.5\n"
    )
    assert int((local_path / "results.csv").stat().st_mtime) == 1_700_000_000
    assert calls[0][0] == "ssh"
    assert "-i" in calls[0]
    assert "BatchMode=yes" in calls[0]


def test_read_remote_history_to_cache_reads_ultralytics_log_over_ssh(
    tmp_path, monkeypatch
) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            "__DANLING_LOG_PATH__/remote/sot.log\n"
            "__DANLING_LOG_MTIME__1700000000\n"
            "__DANLING_REMOTE_NOW__1700000010\n"
            "Epoch GPU_mem box_loss cls_loss dfl_loss Instances Size\n"
            "8/100 21.5G 1.475 1.309 0.006581 196 960: 40% 1484/3709\n",
            "",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    local_path, remote_now = read_remote_history_to_cache(
        RemoteOptions("trainbox", identity=identity),
        "/remote/sot.log",
        "ultralytics-log",
        tmp_path / "cache",
    )

    assert local_path == tmp_path / "cache" / "sot.log"
    assert "8/100" in local_path.read_text(encoding="utf-8")
    assert int(local_path.stat().st_mtime) == 1_700_000_000
    assert remote_now == 1_700_000_010


def test_read_remote_history_to_cache_expands_remote_home_path(
    tmp_path, monkeypatch
) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    seen: dict[str, str] = {}

    def fake_run(command, **kwargs):
        seen["remote_command"] = command[-1]
        return subprocess.CompletedProcess(
            command,
            0,
            "__DANLING_LOG_PATH__/home/train/logs/sot.log\n"
            "__DANLING_LOG_MTIME__1700000000\n"
            "__DANLING_REMOTE_NOW__1700000010\n"
            "Epoch GPU_mem box_loss cls_loss dfl_loss Instances Size\n"
            "9/100 9.43G 1.494 1.288 0.00624 76 960: 1% 110/7417\n",
            "",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    local_path, _remote_now = read_remote_history_to_cache(
        RemoteOptions("trainbox", identity=identity),
        "~/logs/sot.log",
        "ultralytics-log",
        tmp_path / "cache",
    )

    assert local_path == tmp_path / "cache" / "sot.log"
    assert 'p="$HOME/${p#~/}"' in seen["remote_command"]


def test_read_remote_nvidia_hardware_uses_key_and_batchmode(tmp_path, monkeypatch) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            "0, RTX 4090, 78, 21400, 24564, 70, 310\n",
            "",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    hardware = read_remote_nvidia_hardware(RemoteOptions("trainbox", identity=identity))

    assert len(hardware) == 1
    assert hardware[0].device_type == "nvidia"
    assert hardware[0].device_id == "0"
    assert hardware[0].name == "RTX 4090"
    assert "BatchMode=yes" in seen["command"]


def test_read_remote_nvidia_hardware_uses_path_fallback_probe(tmp_path, monkeypatch) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)

    read_remote_nvidia_hardware(RemoteOptions("trainbox", identity=identity))

    remote_command = seen["command"][-1]
    assert "command -v nvidia-smi" in remote_command
    assert "/usr/bin/nvidia-smi" in remote_command
    assert "/usr/local/cuda/bin/nvidia-smi" in remote_command
    assert "--query-gpu=index,name,utilization.gpu" in remote_command


def test_read_remote_nvidia_hardware_nonzero_returns_empty(tmp_path, monkeypatch) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    completed = subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="missing")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)

    assert read_remote_nvidia_hardware(RemoteOptions("trainbox", identity=identity)) == []


def test_remote_training_monitor_builds_state_from_csv_over_ssh(tmp_path, monkeypatch) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    states: dict[str, object] = {}

    def fake_read_history_to_cache(options, remote_path, source, cache_dir):
        states["cache"] = (options.host, options.identity, remote_path, source, cache_dir)
        return tmp_path / "mirror", 200.0

    def fake_read_history(path, source):
        states["read_history"] = (path, source)
        return [MetricSnapshot(epoch=3, train_loss=1.0, score=0.7, score_name="mAP50")]

    def fake_hardware(options):
        states["hardware_host"] = options.host
        return [HardwareSnapshot(device_type="nvidia", util_percent=40)]

    monkeypatch.setattr("danling.remote.read_remote_history_to_cache", fake_read_history_to_cache)
    monkeypatch.setattr("danling.remote.read_history", fake_read_history)
    monkeypatch.setattr("danling.remote.read_remote_nvidia_hardware", fake_hardware)

    profile = RemoteProfile(
        name="lab",
        host="trainbox",
        remote_path="/runs/exp",
        identity=identity,
    )
    monitor = RemoteTrainingMonitor.from_profile(profile, config_path=None, cache_dir=tmp_path)

    state = monitor.read_state()

    assert state.metric is not None
    assert state.metric.epoch == 3
    assert state.hardware[0].device_type == "nvidia"
    assert states["cache"] == ("trainbox", identity, "/runs/exp", "csv", tmp_path)
    assert states["read_history"] == (str(tmp_path / "mirror"), "csv")
    assert states["hardware_host"] == "trainbox"


def test_remote_training_monitor_uses_profile_source_and_remote_clock(
    tmp_path, monkeypatch
) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("private", encoding="utf-8")
    log_path = tmp_path / "mirror" / "sot.log"
    log_path.parent.mkdir()
    log_path.write_text("log", encoding="utf-8")
    import os

    os.utime(log_path, (1_700_000_000, 1_700_000_000))
    seen: dict[str, object] = {}

    def fake_read_history_to_cache(options, remote_path, source, cache_dir):
        seen["cache"] = (remote_path, source, cache_dir)
        return log_path, 1_700_000_010.0

    def fake_read_history(path, source):
        seen["read_history"] = (path, source)
        return [
            MetricSnapshot(
                source="ultralytics-log",
                source_path=str(log_path),
                epoch=8,
                timestamp=1_700_000_000,
                train_loss=1.0,
            )
        ]

    monkeypatch.setattr("danling.remote.read_remote_history_to_cache", fake_read_history_to_cache)
    monkeypatch.setattr("danling.remote.read_history", fake_read_history)
    monkeypatch.setattr("danling.remote.read_remote_nvidia_hardware", lambda options: [])

    profile = RemoteProfile(
        name="lab",
        host="trainbox",
        remote_path="/remote/sot.log",
        identity=identity,
        source="ultralytics-log",
    )
    monitor = RemoteTrainingMonitor.from_profile(profile, config_path=None, cache_dir=tmp_path)

    state = monitor.read_state()

    assert state.metric is not None
    assert state.metric.epoch == 8
    assert state.pet_mood.value != "sleeping"
    assert seen["cache"] == ("/remote/sot.log", "ultralytics-log", tmp_path)
    assert seen["read_history"] == (str(log_path), "ultralytics-log")
