"""丹灵修炼境界推理。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from danling.config import DanLingConfig
from danling.models import CultivationRealm

REALM_NAMES = ["炼器期", "筑基期", "结丹期", "元婴期", "化神期"]


def infer_realm(score: float | None, config: DanLingConfig) -> CultivationRealm:
    """根据当前主指标和配置推导丹灵境界。"""
    thresholds = _realm_thresholds(config)
    if score is None or not math.isfinite(score) or thresholds is None:
        return CultivationRealm()

    realm_index = 0
    for index, (_name, floor) in enumerate(thresholds):
        if score >= floor:
            realm_index = index
        else:
            break

    name, floor = thresholds[realm_index]
    ceiling = thresholds[realm_index + 1][1] if realm_index + 1 < len(thresholds) else None
    return CultivationRealm(
        name=name,
        rank=realm_index + 1,
        score_floor=floor,
        score_ceiling=ceiling,
        progress=_realm_progress(score, floor, ceiling),
    )


def _realm_thresholds(config: DanLingConfig) -> list[tuple[str, float]] | None:
    manual = _manual_thresholds(config.realm_thresholds)
    if manual is not None:
        return manual

    baseline = _finite_float(config.baseline_score)
    sota = _finite_float(config.sota_score)
    if baseline is None or sota is None or sota <= baseline:
        return None

    step = (sota - baseline) / (len(REALM_NAMES) - 1)
    return [(name, baseline + step * index) for index, name in enumerate(REALM_NAMES)]


def _manual_thresholds(
    value: Mapping[str, object] | Sequence[object] | None,
) -> list[tuple[str, float]] | None:
    if isinstance(value, Mapping):
        thresholds = [
            (name, _finite_float(value.get(name)))
            for name in REALM_NAMES
            if name in value
        ]
    elif isinstance(value, Sequence) and not isinstance(value, str):
        thresholds = [
            (name, _finite_float(item))
            for name, item in zip(REALM_NAMES, value, strict=False)
        ]
    else:
        return None

    if len(thresholds) != len(REALM_NAMES):
        return None
    normalized = [(name, floor) for name, floor in thresholds if floor is not None]
    if len(normalized) != len(REALM_NAMES):
        return None
    threshold_pairs = zip(normalized, normalized[1:], strict=False)
    if any(current >= next_floor for (_, current), (_, next_floor) in threshold_pairs):
        return None
    return normalized


def _realm_progress(score: float, floor: float, ceiling: float | None) -> float | None:
    if ceiling is None:
        return 1.0 if score >= floor else 0.0
    if ceiling <= floor:
        return None
    return max(0.0, min(1.0, (score - floor) / (ceiling - floor)))


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
