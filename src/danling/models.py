"""核心数据模型。

定义 DanLing 中的领域类型：指标快照、硬件快照、宠物状态、事件、枚举。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any


class PetMood(str, Enum):
    """宠物情绪枚举。"""

    idle = "idle"
    happy = "happy"
    normal = "normal"
    anxious = "anxious"
    sick = "sick"
    sleeping = "sleeping"
    evolving = "evolving"
    failed = "failed"


class FurnaceState(str, Enum):
    """炼丹炉状态枚举。"""

    cold = "cold"
    warm = "warm"
    burning = "burning"
    hot = "hot"
    smoking = "smoking"
    exploded = "exploded"
    unknown = "unknown"


class Severity(str, Enum):
    """事件严重程度枚举。"""

    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


def _dataclass_to_dict(obj: Any) -> Any:
    """递归序列化 dataclass 和 Enum 为 dict/基本类型。"""
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


@dataclass
class CultivationRealm:
    """丹灵修炼境界。"""

    name: str = "炼器期"
    rank: int = 1
    score_floor: float | None = None
    score_ceiling: float | None = None
    progress: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass
class MetricSnapshot:
    """单次训练步骤/epoch 的指标快照。"""

    source: str = ""
    run_path: str | None = None
    source_path: str | None = None
    step: int | None = None
    epoch: int | None = None
    timestamp: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    score: float | None = None
    score_name: str | None = None
    map50: float | None = None
    map5095: float | None = None
    precision: float | None = None
    recall: float | None = None
    accuracy: float | None = None
    lr: float | None = None
    raw: dict[str, float | int | str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """递归序列化为 dict，Enum 转字符串。"""
        return _dataclass_to_dict(self)


@dataclass
class HardwareSnapshot:
    """单个硬件设备的状态快照。"""

    device_type: str = "unknown"
    device_id: str | None = None
    name: str | None = None
    util_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    timestamp: float | None = None
    raw: dict[str, float | int | str | None] = field(default_factory=dict)

    @property
    def memory_ratio(self) -> float | None:
        """显存使用比例。"""
        if (
            self.memory_used_mb is not None
            and self.memory_total_mb is not None
            and self.memory_total_mb > 0
        ):
            return self.memory_used_mb / self.memory_total_mb
        return None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass
class DanLingEvent:
    """单个状态事件。"""

    severity: Severity = Severity.info
    title: str = ""
    message: str = ""
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass
class DanLingState:
    """宠物聚合状态。"""

    pet_name: str = "DanLing"
    pet_mood: PetMood = PetMood.idle
    furnace_state: FurnaceState = FurnaceState.unknown
    realm: CultivationRealm = field(default_factory=CultivationRealm)
    metric: MetricSnapshot | None = None
    hardware: list[HardwareSnapshot] = field(default_factory=list)
    events: list[DanLingEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)
