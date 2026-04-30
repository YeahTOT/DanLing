"""测试趋势分析和状态引擎。"""

from __future__ import annotations

import math

import pytest

from danling.config import DanLingConfig
from danling.engine.state import build_state
from danling.engine.trends import analyze_trends
from danling.models import FurnaceState, HardwareSnapshot, MetricSnapshot, PetMood


def metric(
    train_loss: float | None = None,
    score: float | None = None,
    timestamp: float | None = None,
    epoch: int | None = None,
) -> MetricSnapshot:
    return MetricSnapshot(
        source="test",
        run_path="/tmp/run",
        epoch=epoch,
        timestamp=timestamp,
        train_loss=train_loss,
        score=score,
        score_name="mAP50" if score is not None else None,
    )


def test_analyze_trends_loss_slope_and_new_best() -> None:
    history = [
        metric(train_loss=3.0, score=0.50, timestamp=100, epoch=1),
        metric(train_loss=2.0, score=0.55, timestamp=110, epoch=2),
        metric(train_loss=1.0, score=0.60, timestamp=120, epoch=3),
    ]

    trend = analyze_trends(history, loss_window=3, now=130)

    assert trend.current_loss == 1.0
    assert trend.previous_loss == 2.0
    assert trend.loss_slope < 0
    assert trend.is_new_best is True
    assert trend.no_improve_count == 0
    assert trend.stale_seconds == 10


def test_build_state_loss_down_is_happy() -> None:
    state = build_state(
        [metric(train_loss=3.0), metric(train_loss=2.0), metric(train_loss=1.0)],
        hardware=[],
        config=DanLingConfig(),
    )

    assert state.pet_mood == PetMood.happy


def test_build_state_new_best_is_evolving() -> None:
    state = build_state(
        [metric(train_loss=1.0, score=0.5), metric(train_loss=1.0, score=0.8)],
        hardware=[],
        config=DanLingConfig(),
    )

    assert state.pet_mood == PetMood.evolving
    assert any(event.severity.value == "info" for event in state.events)


def test_build_state_derives_realm_from_equal_baseline_sota_ranges() -> None:
    state = build_state(
        [metric(train_loss=1.0, score=0.71)],
        hardware=[],
        config=DanLingConfig(baseline_score=0.50, sota_score=0.90),
    )

    assert state.realm.name == "结丹期"
    assert state.realm.rank == 3
    assert state.realm.score_floor == pytest.approx(0.70)
    assert state.realm.score_ceiling == pytest.approx(0.80)
    assert state.realm.progress == pytest.approx(0.10)


def test_build_state_uses_manual_realm_thresholds_when_configured() -> None:
    state = build_state(
        [metric(train_loss=1.0, score=0.84)],
        hardware=[],
        config=DanLingConfig(
            realm_thresholds={
                "炼器期": 0.40,
                "筑基期": 0.55,
                "结丹期": 0.70,
                "元婴期": 0.82,
                "化神期": 0.93,
            }
        ),
    )

    assert state.realm.name == "元婴期"
    assert state.realm.rank == 4
    assert state.realm.score_floor == pytest.approx(0.82)
    assert state.realm.score_ceiling == pytest.approx(0.93)


def test_build_state_sota_or_better_is_huashen_realm() -> None:
    state = build_state(
        [metric(train_loss=1.0, score=0.95)],
        hardware=[],
        config=DanLingConfig(baseline_score=0.50, sota_score=0.90),
    )

    assert state.realm.name == "化神期"
    assert state.realm.rank == 5
    assert state.realm.score_floor == pytest.approx(0.90)
    assert state.realm.score_ceiling is None
    assert state.realm.progress == pytest.approx(1.0)


def test_build_state_uncalibrated_realm_falls_back_to_liqi() -> None:
    state = build_state(
        [metric(train_loss=1.0, score=0.80)],
        hardware=[],
        config=DanLingConfig(),
    )

    assert state.realm.name == "炼器期"
    assert state.realm.rank == 1
    assert state.realm.score_floor is None
    assert state.realm.score_ceiling is None
    assert state.realm.progress is None


def test_build_state_uses_configured_primary_score_for_realm() -> None:
    snapshot = metric(train_loss=1.0, score=0.62)
    snapshot.raw["precision"] = 0.92

    state = build_state(
        [snapshot],
        hardware=[],
        config=DanLingConfig(
            primary_score="precision",
            baseline_score=0.50,
            sota_score=0.90,
        ),
    )

    assert state.metric is not None
    assert state.metric.score == pytest.approx(0.92)
    assert state.metric.score_name == "precision"
    assert state.realm.name == "化神期"


def test_build_state_loss_explosion_is_sick() -> None:
    state = build_state(
        [metric(train_loss=1.0), metric(train_loss=2.5)],
        hardware=[],
        config=DanLingConfig(),
    )

    assert state.pet_mood == PetMood.sick


def test_build_state_nan_is_failed() -> None:
    state = build_state(
        [metric(train_loss=1.0), metric(train_loss=math.nan)],
        hardware=[],
        config=DanLingConfig(),
    )

    assert state.pet_mood == PetMood.failed


def test_build_state_no_improve_is_anxious() -> None:
    state = build_state(
        [
            metric(train_loss=1.0, score=0.80),
            metric(train_loss=1.0, score=0.79),
            metric(train_loss=1.0, score=0.78),
        ],
        hardware=[],
        config=DanLingConfig(no_improve_patience=2),
    )

    assert state.pet_mood == PetMood.anxious


def test_build_state_stale_is_sleeping() -> None:
    state = build_state(
        [metric(train_loss=1.0, timestamp=100)],
        hardware=[],
        config=DanLingConfig(stale_seconds=10),
        now=200,
    )

    assert state.pet_mood == PetMood.sleeping


def test_build_state_hardware_furnace_rules() -> None:
    smoking = build_state(
        [metric(train_loss=1.0)],
        hardware=[HardwareSnapshot(util_percent=80, memory_used_mb=91, memory_total_mb=100)],
        config=DanLingConfig(high_memory_ratio=0.9),
    )
    exploded = build_state(
        [metric(train_loss=1.0)],
        hardware=[HardwareSnapshot(util_percent=10, memory_used_mb=99, memory_total_mb=100)],
        config=DanLingConfig(),
    )

    assert smoking.furnace_state == FurnaceState.smoking
    assert exploded.furnace_state == FurnaceState.exploded
