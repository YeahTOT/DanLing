"""训练趋势分析。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from danling.models import MetricSnapshot, _dataclass_to_dict


@dataclass
class TrendSnapshot:
    current_loss: float | None = None
    previous_loss: float | None = None
    loss_delta: float | None = None
    loss_slope: float | None = None
    current_score: float | None = None
    best_score: float | None = None
    score_delta: float | None = None
    is_new_best: bool = False
    no_improve_count: int = 0
    has_nan_or_inf: bool = False
    stale_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def analyze_trends(
    history: list[MetricSnapshot],
    loss_window: int = 5,
    score_epsilon: float = 1e-6,
    now: float | None = None,
) -> TrendSnapshot:
    """根据指标历史生成趋势快照。"""
    if not history:
        return TrendSnapshot()

    latest = history[-1]
    current_loss = _loss_value(latest)
    previous_loss = _previous_loss(history[:-1])
    loss_delta = (
        current_loss - previous_loss
        if current_loss is not None and previous_loss is not None
        else None
    )
    loss_slope = _loss_slope(history, max(loss_window, 2))

    scores = [_score_value(snapshot) for snapshot in history]
    current_score = scores[-1]
    finite_scores = [score for score in scores if _is_finite_number(score)]
    best_score = max(finite_scores) if finite_scores else None
    previous_best = max(finite_scores[:-1]) if len(finite_scores) > 1 else None
    is_new_best = False
    if current_score is not None and _is_finite_number(current_score):
        is_new_best = previous_best is None or current_score > previous_best + score_epsilon

    score_delta = None
    previous_score = next((score for score in reversed(scores[:-1]) if score is not None), None)
    if current_score is not None and previous_score is not None:
        score_delta = current_score - previous_score

    return TrendSnapshot(
        current_loss=current_loss,
        previous_loss=previous_loss,
        loss_delta=loss_delta,
        loss_slope=loss_slope,
        current_score=current_score,
        best_score=best_score,
        score_delta=score_delta,
        is_new_best=is_new_best,
        no_improve_count=_no_improve_count(scores, score_epsilon),
        has_nan_or_inf=_has_nan_or_inf(history),
        stale_seconds=_stale_seconds(latest, now),
    )


def _loss_value(snapshot: MetricSnapshot) -> float | None:
    return snapshot.train_loss if snapshot.train_loss is not None else snapshot.val_loss


def _score_value(snapshot: MetricSnapshot) -> float | None:
    return snapshot.score


def _previous_loss(history: list[MetricSnapshot]) -> float | None:
    for snapshot in reversed(history):
        value = _loss_value(snapshot)
        if value is not None:
            return value
    return None


def _loss_slope(history: list[MetricSnapshot], window: int) -> float | None:
    losses = [
        value
        for value in (_loss_value(snapshot) for snapshot in history)
        if _is_finite_number(value)
    ]
    recent = losses[-window:]
    if len(recent) < 2:
        return None
    return (recent[-1] - recent[0]) / (len(recent) - 1)


def _no_improve_count(scores: list[float | None], epsilon: float) -> int:
    best = -math.inf
    last_best_index: int | None = None
    for index, score in enumerate(scores):
        if score is None or not _is_finite_number(score):
            continue
        if score > best + epsilon:
            best = score
            last_best_index = index
    if last_best_index is None:
        return 0
    return len(scores) - 1 - last_best_index


def _has_nan_or_inf(history: list[MetricSnapshot]) -> bool:
    for snapshot in history:
        for value in [
            snapshot.train_loss,
            snapshot.val_loss,
            snapshot.score,
            snapshot.map50,
            snapshot.map5095,
        ]:
            if value is not None and not math.isfinite(value):
                return True
    return False


def _stale_seconds(snapshot: MetricSnapshot, now: float | None) -> float | None:
    if snapshot.timestamp is None:
        return None
    current = time.time() if now is None else now
    return max(0.0, current - snapshot.timestamp)


def _is_finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
