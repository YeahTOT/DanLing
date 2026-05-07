"""TensorBoard event file Reader。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from danling.models import MetricSnapshot
from danling.readers.base import BaseReader

_EVENT_GLOB = "events.out.tfevents.*"
_SCOPE_PREFIXES = {"train", "val", "valid", "validation", "eval", "test"}


class TensorBoardReader(BaseReader):
    @property
    def name(self) -> str:
        return "tensorboard"

    @staticmethod
    def detect(path: str) -> bool:
        p = Path(path)
        if p.is_file():
            return p.name.startswith("events.out.tfevents")
        if p.is_dir():
            return any(p.rglob(_EVENT_GLOB))
        return False

    def read_latest(self, path: str) -> MetricSnapshot | None:
        history = self.read_history(path)
        return history[-1] if history else None

    def read_history(self, path: str, limit: int | None = None) -> list[MetricSnapshot]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard reader requires: pip install danling[tensorboard]"
            ) from exc

        by_step: dict[int, dict[str, float]] = defaultdict(dict)
        timestamps: dict[int, float] = {}
        root = Path(path)
        sources = _event_sources(path)

        for source in sources:
            accumulator = EventAccumulator(str(source))
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
            scope = _scope_for_source(root, source)

            for tag in tags:
                scoped_tag = _scoped_tag(tag, scope)
                for event in accumulator.Scalars(tag):
                    step = int(event.step)
                    by_step[step][scoped_tag] = float(event.value)
                    wall_time = float(event.wall_time)
                    previous = timestamps.get(step)
                    timestamps[step] = wall_time if previous is None else max(previous, wall_time)

        run_path = _run_path(root)

        history = [
            _snapshot_for_step(step, raw, timestamps.get(step), str(run_path), str(root))
            for step, raw in by_step.items()
        ]
        history.sort(key=lambda item: item.step if item.step is not None else -1)
        return history[-limit:] if limit is not None else history


def _event_sources(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.name.startswith("events.out.tfevents") else []
    if p.is_dir():
        return sorted(p.rglob(_EVENT_GLOB))
    return []


def _run_path(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _scope_for_source(root: Path, source: Path) -> str | None:
    if root.is_file():
        return source.parent.name if source.parent.name in _SCOPE_PREFIXES else None

    try:
        relative_parent = source.parent.relative_to(root)
    except ValueError:
        return None
    if not relative_parent.parts:
        return root.name if root.name in _SCOPE_PREFIXES else None
    first_part = relative_parent.parts[0]
    return first_part if first_part in _SCOPE_PREFIXES else None


def _scoped_tag(tag: str, scope: str | None) -> str:
    if scope is None:
        return tag
    first_segment = tag.split("/", 1)[0].strip().lower()
    if first_segment in _SCOPE_PREFIXES:
        return tag
    return f"{scope}/{tag}"


def _snapshot_for_step(
    step: int, raw: dict[str, float], timestamp: float | None, source: str, source_path: str = ""
) -> MetricSnapshot:
    normalized_raw = _normalize_raw(raw)

    train_loss = _first(
        normalized_raw,
        [
            "train/loss",
            "train/loss/total",
            "train/total_loss",
            "loss/train",
            "loss",
            "loss/total",
        ],
    )
    if train_loss is None:
        train_loss = _sum_existing(
            normalized_raw,
            [
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "train/loss/box",
                "train/loss/cls",
                "train/loss/dfl",
            ],
        )

    val_loss = _first(
        normalized_raw,
        [
            "val/loss",
            "val/loss/total",
            "valid/loss",
            "validation/loss",
            "eval/loss",
            "loss/val",
            "loss/validation",
        ],
    )
    if val_loss is None:
        val_loss = _sum_existing(
            normalized_raw,
            [
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "val/loss/box",
                "val/loss/cls",
                "val/loss/dfl",
            ],
        )

    map50 = _first(
        normalized_raw,
        [
            "metrics/map50(b)",
            "metrics/map50",
            "map50",
            "val/map50",
            "val/metrics/map50",
            "val/metrics/map50(b)",
            "validation/map50",
            "validation/metrics/map50",
        ],
    )
    map5095 = _first(
        normalized_raw,
        [
            "metrics/map50-95(b)",
            "metrics/map50-95",
            "map50-95",
            "val/map50-95",
            "val/metrics/map50-95",
            "val/metrics/map50-95(b)",
            "validation/map50-95",
            "validation/metrics/map50-95",
        ],
    )
    precision = _first(
        normalized_raw,
        [
            "metrics/precision(b)",
            "metrics/precision",
            "precision",
            "val/precision",
            "val/metrics/precision",
            "val/metrics/precision(b)",
        ],
    )
    recall = _first(
        normalized_raw,
        [
            "metrics/recall(b)",
            "metrics/recall",
            "recall",
            "val/recall",
            "val/metrics/recall",
            "val/metrics/recall(b)",
        ],
    )
    accuracy = _first(
        normalized_raw,
        [
            "accuracy",
            "acc",
            "accuracy/top1",
            "val/accuracy",
            "val/acc",
            "validation/accuracy",
        ],
    )
    lr = _first(
        normalized_raw,
        [
            "lr",
            "learning_rate",
            "learningrate",
            "lr/pg0",
            "train/lr",
            "train/learning_rate",
            "train/learningrate",
            "train/learningrate/group0",
            "learningrate/group0",
        ],
    )
    score, score_name = _score(map50, map5095, accuracy)

    return MetricSnapshot(
        source="tensorboard",
        run_path=str(Path(source).parent if Path(source).is_file() else source),
        source_path=source_path or None,
        step=step,
        epoch=step,
        timestamp=timestamp,
        train_loss=train_loss,
        val_loss=val_loss,
        score=score,
        score_name=score_name,
        map50=map50,
        map5095=map5095,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        lr=lr,
        raw=raw,
    )


def _normalize_raw(raw: dict[str, float]) -> dict[str, float]:
    return {_normalize_key(key): float(value) for key, value in raw.items()}


def _normalize_key(key: str) -> str:
    return "/".join(part.strip().lower() for part in key.replace("\\", "/").split("/"))


def _first(raw: dict[str, float], keys: list[str]) -> float | None:
    for key in keys:
        normalized_key = _normalize_key(key)
        if normalized_key in raw:
            return float(raw[normalized_key])
    return None


def _sum_existing(raw: dict[str, float], keys: list[str]) -> float | None:
    values = [float(raw[_normalize_key(key)]) for key in keys if _normalize_key(key) in raw]
    return sum(values) if values else None


def _score(
    map50: float | None, map5095: float | None, accuracy: float | None
) -> tuple[float | None, str | None]:
    if map50 is not None:
        return map50, "mAP50"
    if map5095 is not None:
        return map5095, "mAP50-95"
    if accuracy is not None:
        return accuracy, "accuracy"
    return None, None
