"""Tests for the desktop pet sidecar surface."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from typer.testing import CliRunner

from danling.api.app import create_app
from danling.cli import app as cli_app


def call_route(app, path: str, method: str = "GET", **kwargs):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            result = route.endpoint(**kwargs)
            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result
    raise AssertionError(f"route not found: {method} {path}")


def test_desktop_state_returns_idle_configured_pet_when_no_path(tmp_path: Path) -> None:
    config_path = tmp_path / "danling.yaml"
    config_path.write_text("pet_name: XiaoDan\nsource: auto\ndata_path:\n", encoding="utf-8")

    app = create_app(config_path=config_path, no_hardware=True)

    payload = call_route(app, "/api/state")

    assert payload["pet_name"] == "XiaoDan"
    assert payload["pet_mood"] == "idle"
    assert payload["desktop"]["animation"] == "running"
    assert payload["desktop"]["status_label"] == "等待配置训练日志"


def test_desktop_config_api_saves_full_config_tui_form(tmp_path: Path) -> None:
    config_path = tmp_path / "danling.yaml"
    app = create_app(config_path=config_path, no_hardware=True)

    payload = call_route(
        app,
        "/api/config",
        method="PUT",
        values={
            "pet_name": "DanLing",
            "primary_score": "mAP50",
            "primary_loss": "train_loss",
            "baseline_score": "0.5",
            "sota_score": "0.9",
            "watch_interval": "1.5",
            "loss_window": "7",
            "no_improve_patience": "12",
            "stale_seconds": "600",
            "high_memory_ratio": "0.92",
            "hot_util_percent": "88",
            "source": "tensorboard",
            "data_path": "logs/tensorboard",
            "realm_threshold_炼器期": "0.50",
            "realm_threshold_筑基期": "0.62",
            "realm_threshold_结丹期": "0.74",
            "realm_threshold_元婴期": "0.84",
            "realm_threshold_化神期": "0.90",
        },
    )

    assert payload["config"]["source"] == "tensorboard"
    saved = config_path.read_text(encoding="utf-8")
    assert "data_path: logs/tensorboard" in saved
    assert "炼器期: 0.5" in saved


def test_desktop_config_suggest_reuses_config_tui_auto_generation(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "epoch,metrics/mAP50(B)",
                "1,0.10",
                "2,0.20",
                "3,0.30",
                "4,0.50",
                "5,0.40",
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(config_path=tmp_path / "danling.yaml", no_hardware=True)

    payload = call_route(
        app,
        "/api/config/suggest",
        method="POST",
        payload={"path": str(csv_path), "metric": "mAP50", "source": "csv"},
    )

    assert payload["suggested"] == {
        "primary_score": "mAP50",
        "baseline_score": "0.1",
        "sota_score": "0.5",
    }


def test_desktop_api_cli_exposes_sidecar_entrypoint() -> None:
    result = CliRunner().invoke(cli_app, ["api", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
