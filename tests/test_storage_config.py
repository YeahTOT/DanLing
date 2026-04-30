"""测试配置加载和 JSON 持久化。"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from danling.cli import app
from danling.config import DanLingConfig, config_to_yaml_text, load_config
from danling.storage.json_store import JsonStateStore

runner = CliRunner()


def test_load_config_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "danling.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pet_name: 小丹灵",
                "loss_window: 7",
                "watch_interval: 0.5",
                "baseline_score: 0.5",
                "sota_score: 0.9",
                "realm_thresholds:",
                "  炼器期: 0.50",
                "  筑基期: 0.60",
                "  结丹期: 0.70",
                "  元婴期: 0.80",
                "  化神期: 0.90",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.pet_name == "小丹灵"
    assert config.loss_window == 7
    assert config.watch_interval == 0.5
    assert config.baseline_score == 0.5
    assert config.sota_score == 0.9
    assert config.realm_thresholds == {
        "炼器期": 0.50,
        "筑基期": 0.60,
        "结丹期": 0.70,
        "元婴期": 0.80,
        "化神期": 0.90,
    }


def test_config_to_yaml_text_serializes_realm_calibration() -> None:
    text = config_to_yaml_text(
        DanLingConfig(
            pet_name="小丹灵",
            primary_score="precision",
            baseline_score=0.5,
            sota_score=0.9,
            realm_thresholds={
                "炼器期": 0.50,
                "筑基期": 0.60,
                "结丹期": 0.70,
                "元婴期": 0.80,
                "化神期": 0.90,
            },
        )
    )

    assert "pet_name: 小丹灵" in text
    assert "primary_score: precision" in text
    assert "baseline_score: 0.5" in text
    assert "sota_score: 0.9" in text
    assert "realm_thresholds:" in text
    assert "  化神期: 0.9" in text


def test_json_state_store_load_save_update(tmp_path) -> None:
    store = JsonStateStore(tmp_path / "state.json")

    assert store.load() == {}

    store.save({"level": 2})
    assert store.load()["level"] == 2

    store.update_run("abc", {"best_score": 0.9})
    assert store.get_run("abc")["best_score"] == 0.9


def test_init_creates_config_file(tmp_path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "baseline_score:" in Path("danling.yaml").read_text(encoding="utf-8")
        data = json.loads(runner.invoke(app, ["config", "show", "--json"]).stdout)
        assert data["pet_name"] == "DanLing"
        assert data["baseline_score"] is None


def test_state_store_does_not_persist_derived_level(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = runner.invoke(
        app,
        [
            "state",
            "tests/fixtures/ultralytics_run",
            "--source",
            "csv",
            "--no-hardware",
        ],
    )

    assert result.exit_code == 0
    state_file = tmp_path / ".danling" / "state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "level" not in data


def test_reset_yes_removes_state_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    state_file = tmp_path / ".danling" / "state.json"
    state_file.parent.mkdir()
    state_file.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["reset", "--yes"])

    assert result.exit_code == 0
    assert not state_file.exists()
