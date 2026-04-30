"""DanLing 状态推理引擎。"""

from __future__ import annotations

from dataclasses import replace

from danling.config import DanLingConfig
from danling.engine.diagnostics import diagnose
from danling.engine.realms import infer_realm
from danling.engine.trends import analyze_trends
from danling.models import (
    DanLingEvent,
    DanLingState,
    FurnaceState,
    HardwareSnapshot,
    MetricSnapshot,
    PetMood,
    Severity,
)


def build_state(
    history: list[MetricSnapshot],
    hardware: list[HardwareSnapshot] | None,
    config: DanLingConfig,
    persisted: dict | None = None,
    now: float | None = None,
) -> DanLingState:
    """根据指标历史、硬件和持久化数据构造聚合状态。"""
    persisted = persisted or {}
    hardware = hardware or []
    history = _apply_primary_score(history, config)
    metric = history[-1] if history else None
    trend = analyze_trends(history, config.loss_window, now=now)

    events: list[DanLingEvent] = []
    mood = _pet_mood(trend, config)
    furnace = _furnace_state(hardware, config)
    realm = infer_realm(trend.current_score, config)

    persisted_best = persisted.get("best_score")
    if trend.is_new_best and trend.current_score is not None:
        if not isinstance(persisted_best, (int, float)) or trend.current_score > float(
            persisted_best
        ):
            events.append(
                DanLingEvent(
                    Severity.info,
                    "新的最佳指标",
                    f"{metric.score_name if metric else 'score'} 达到 {trend.current_score:.4f}",
                    metric.timestamp if metric else None,
                )
            )

    if mood == PetMood.sick:
        events.append(DanLingEvent(Severity.error, "loss 爆炸", "当前 loss 明显高于上一个快照。"))
    if mood == PetMood.anxious:
        events.append(DanLingEvent(Severity.warning, "指标停滞", "训练指标长时间没有提升。"))
    if mood == PetMood.sleeping:
        events.append(DanLingEvent(Severity.warning, "日志休眠", "训练日志长时间未更新。"))

    for diagnostic in diagnose(history, hardware, config, now=now):
        events.append(
            DanLingEvent(
                severity=diagnostic.severity,
                title=diagnostic.title,
                message=diagnostic.message,
                timestamp=metric.timestamp if metric else None,
            )
        )

    return DanLingState(
        pet_name=config.pet_name or persisted.get("pet_name", "DanLing"),
        pet_mood=mood,
        furnace_state=furnace,
        realm=realm,
        metric=metric,
        hardware=hardware,
        events=events,
    )


def _pet_mood(trend, config: DanLingConfig) -> PetMood:
    if trend.current_loss is None and trend.current_score is None:
        return PetMood.idle
    if trend.has_nan_or_inf:
        return PetMood.failed
    if trend.stale_seconds is not None and trend.stale_seconds > config.stale_seconds:
        return PetMood.sleeping
    if (
        trend.current_loss is not None
        and trend.previous_loss is not None
        and trend.current_loss > trend.previous_loss * 2.0
    ):
        return PetMood.sick
    if trend.is_new_best:
        return PetMood.evolving
    if trend.no_improve_count >= config.no_improve_patience and trend.current_score is not None:
        return PetMood.anxious
    if trend.loss_slope is not None and trend.loss_slope < 0:
        return PetMood.happy
    return PetMood.normal


def _apply_primary_score(
    history: list[MetricSnapshot], config: DanLingConfig
) -> list[MetricSnapshot]:
    if not config.primary_score:
        return history

    selected: list[MetricSnapshot] = []
    for snapshot in history:
        score = _metric_number(snapshot, config.primary_score)
        if score is None:
            selected.append(snapshot)
        else:
            selected.append(replace(snapshot, score=score, score_name=config.primary_score))
    return selected


def _metric_number(snapshot: MetricSnapshot, name: str) -> float | None:
    raw_value = snapshot.raw.get(name)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    attr_name = {
        "mAP50": "map50",
        "mAP50-95": "map5095",
        "map50": "map50",
        "map5095": "map5095",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
        "score": "score",
    }.get(name, name)
    value = getattr(snapshot, attr_name, None)
    return float(value) if isinstance(value, (int, float)) else None


def _furnace_state(hardware: list[HardwareSnapshot], config: DanLingConfig) -> FurnaceState:
    if not hardware:
        return FurnaceState.unknown
    if any((device.memory_ratio or 0) >= 0.98 for device in hardware):
        return FurnaceState.exploded
    if any((device.memory_ratio or 0) >= config.high_memory_ratio for device in hardware):
        return FurnaceState.smoking
    if any((device.util_percent or 0) >= config.hot_util_percent for device in hardware):
        return FurnaceState.hot
    if any((device.util_percent or 0) >= 50 for device in hardware):
        return FurnaceState.burning
    if any((device.util_percent or 0) > 0 for device in hardware):
        return FurnaceState.warm
    return FurnaceState.cold
