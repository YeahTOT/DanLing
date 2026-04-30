"""Tests for remote SSH-key setup helpers."""

from __future__ import annotations

import subprocess

import pytest

from danling.remote import RemoteProfile, RemoteProfileStore, load_remote_profile
from danling.renderers.remote_setup_tui import (
    RemoteSetupValues,
    check_remote_gpu,
    check_ssh_key_connection,
    generate_ssh_keypair,
    save_remote_setup_values,
    validate_remote_setup_values,
)


def test_validate_remote_setup_requires_key_path() -> None:
    errors = validate_remote_setup_values(
        RemoteSetupValues(
            name="lab",
            host="trainbox",
            remote_path="/runs/exp",
            identity="",
        )
    )

    assert "SSH 私钥路径不能为空" in errors


def test_generate_ssh_keypair_runs_ssh_keygen(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)

    generate_ssh_keypair(tmp_path / "danling_remote_ed25519")

    assert calls == [
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(tmp_path / "danling_remote_ed25519"),
        ]
    ]


def test_check_ssh_key_connection_uses_batchmode(tmp_path, monkeypatch) -> None:
    key = tmp_path / "id_ed25519"
    key.write_text("private", encoding="utf-8")
    seen: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)

    ok, message = check_ssh_key_connection("trainbox", key, port=2222, timeout=3)

    assert ok is True
    assert "连接成功" in message
    assert seen["command"] == [
        "ssh",
        "-p",
        "2222",
        "-i",
        str(key),
        "-o",
        "BatchMode=yes",
        "trainbox",
        "true",
    ]


def test_check_remote_gpu_reports_gpu_summary(tmp_path, monkeypatch) -> None:
    key = tmp_path / "id_ed25519"
    key.write_text("private", encoding="utf-8")

    def fake_read_hardware(options):
        assert options.host == "trainbox"
        assert options.identity == key
        return [
            type(
                "GPU",
                (),
                {
                    "device_id": "0",
                    "name": "RTX 4090",
                    "util_percent": 78.0,
                    "memory_used_mb": 21_400.0,
                    "memory_total_mb": 24_564.0,
                },
            )()
        ]

    monkeypatch.setattr(
        "danling.renderers.remote_setup_tui.read_remote_nvidia_hardware",
        fake_read_hardware,
    )

    ok, message = check_remote_gpu("trainbox", key, port=None, timeout=3)

    assert ok is True
    assert "GPU:0 RTX 4090" in message
    assert "78%" in message
    assert "20.9/24.0G" in message


def test_save_remote_setup_values_persists_profile(tmp_path) -> None:
    key = tmp_path / "id_ed25519"
    key.write_text("private", encoding="utf-8")
    store = RemoteProfileStore(tmp_path / "profiles.json")

    profile = save_remote_setup_values(
        RemoteSetupValues(
            name="lab",
            host="trainbox",
            port="2222",
            remote_path="/runs/exp",
            identity=str(key),
            sync_timeout="5",
        ),
        store=store,
    )

    assert profile == RemoteProfile(
        name="lab",
        host="trainbox",
        remote_path="/runs/exp",
        identity=key,
        port=2222,
        source="csv",
        sync_timeout=5.0,
    )
    assert load_remote_profile("lab", store=store) == profile


def test_save_remote_setup_values_rejects_missing_key(tmp_path) -> None:
    store = RemoteProfileStore(tmp_path / "profiles.json")

    with pytest.raises(ValueError, match="SSH 私钥文件不存在"):
        save_remote_setup_values(
            RemoteSetupValues(
                name="lab",
                host="trainbox",
                remote_path="/runs/exp",
                identity=str(tmp_path / "missing"),
            ),
            store=store,
        )
