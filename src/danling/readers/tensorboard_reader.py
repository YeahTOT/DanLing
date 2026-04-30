"""TensorBoard event file Reader。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from danling.models import MetricSnapshot
from danling.readers.base import BaseReader


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
            return any(p.glob("events.out.tfevents.*"))
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

        source = _event_source(path)
        accumulator = EventAccumulator(str(source))
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        by_step: dict[int, dict[str, float]] = defaultdict(dict)
        timestamps: dict[int, float] = {}

        for tag in tags:
            for event in accumulator.Scalars(tag):
                by_step[int(event.step)][tag] = float(event.value)
                timestamps[int(event.step)] = float(event.wall_time)

        history = [
            _snapshot_for_step(step, raw, timestamps.get(step), str(source))
            for step, raw in by_step.items()
        ]
        history.sort(key=lambda item: item.step if item.step is not None else -1)
        return history[-limit:] if limit is not None else history


def _event_source(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    return p


def _snapshot_for_step(
    step: int, raw: dict[str, float], timestamp: float | None, source: str
) -> MetricSnapshot:
    train_loss = _first(raw, ["train/loss", "loss/train", "loss"])
    if train_loss is None:
        train_loss = _sum_existing(raw, ["train/box_loss", "train/cls_loss", "train/dfl_loss"])

    val_loss = _first(raw, ["val/loss", "loss/val"])
    if val_loss is None:
        val_loss = _sum_existing(raw, ["val/box_loss", "val/cls_loss", "val/dfl_loss"])

    map50 = _first(raw, ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
    map5095 = _first(raw, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"])
    accuracy = _first(raw, ["accuracy", "acc", "accuracy/top1", "val/accuracy"])
    lr = _first(raw, ["lr", "learning_rate", "lr/pg0"])
    score, score_name = _score(map50, map5095, accuracy)

    return MetricSnapshot(
        source="tensorboard",
        run_path=str(Path(source).parent if Path(source).is_file() else source),
        step=step,
        epoch=step,
        timestamp=timestamp,
        train_loss=train_loss,
        val_loss=val_loss,
        score=score,
        score_name=score_name,
        map50=map50,
        map5095=map5095,
        accuracy=accuracy,
        lr=lr,
        raw=raw,
    )


def _first(raw: dict[str, float], keys: list[str]) -> float | None:
    for key in keys:
        if key in raw:
            return float(raw[key])
    return None


def _sum_existing(raw: dict[str, float], keys: list[str]) -> float | None:
    values = [float(raw[key]) for key in keys if key in raw]
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
