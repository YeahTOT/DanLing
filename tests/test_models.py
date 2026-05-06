"""测试核心模型与配置默认值。"""

from __future__ import annotations

import json

from danling.config import DanLingConfig
from danling.models import (
    DanLingState,
    FurnaceState,
    HardwareSnapshot,
    MetricSnapshot,
    PetMood,
    Severity,
)


def test_hardware_memory_ratio() -> None:
    snapshot = HardwareSnapshot(memory_used_mb=9_000, memory_total_mb=10_000)

    assert snapshot.memory_ratio == 0.9


def test_state_to_dict_serializes_enums() -> None:
    state = DanLingState(
        pet_mood=PetMood.happy,
        furnace_state=FurnaceState.burning,
        metric=MetricSnapshot(epoch=3, score=0.8, score_name="mAP50"),
    )

    data = state.to_dict()

    assert data["pet_mood"] == "happy"
    assert data["furnace_state"] == "burning"
    assert json.loads(json.dumps(data, ensure_ascii=False))["metric"]["epoch"] == 3


def test_config_defaults_cover_engine_fields() -> None:
    config = DanLingConfig()

    assert config.pet_name == "DanLing"
    assert config.loss_window == 5
    assert config.no_improve_patience == 10
    assert config.stale_seconds == 3600
    assert config.high_memory_ratio == 0.90
    assert config.hot_util_percent == 90.0
    assert config.watch_interval == 2.0
    assert config.baseline_score is None
    assert config.sota_score is None
    assert config.realm_thresholds is None
    assert Severity.warning.value == "warning"
