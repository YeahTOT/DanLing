"""CSV 训练日志回放仿真器。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from danling.models import MetricSnapshot
from danling.readers.registry import read_history
from danling.readers.ultralytics_csv import UltralyticsCSVReader

_RESULT_FIELDS = [
    "epoch",
    "time",
    "train/loss",
    "val/loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "accuracy",
    "lr",
]


@dataclass(frozen=True)
class PlaybackProgress:
    """单次回放进度。"""

    source_path: Path
    target_path: Path
    rows_written: int
    total_rows: int


def playback_results_csv(
    source_path: str | Path,
    output_path: str | Path,
    interval: float = 2.0,
    max_rows: int | None = None,
    overwrite: bool = True,
):
    """将源 `results.csv` 按行回放到目标 `results.csv`。

    `output_path` 可以是目录，也可以是具体 CSV 文件路径。默认每写一行
    sleep `interval` 秒；测试或 dry-run 可传 `interval=0`。
    """
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"source CSV not found: {source}")

    target = _resolve_target_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows:
        raise ValueError(f"source CSV is empty: {source}")

    header, data_rows = rows[0], [row for row in rows[1:] if row]
    total_rows = len(data_rows)
    rows_to_write = data_rows if max_rows is None else data_rows[:max_rows]
    mode = "w" if overwrite or not target.exists() else "a"
    should_write_header = mode == "w" or target.stat().st_size == 0

    with target.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if should_write_header:
            writer.writerow(header)
        for index, row in enumerate(rows_to_write, start=1):
            writer.writerow(row)
            file.flush()
            yield PlaybackProgress(
                source_path=source,
                target_path=target,
                rows_written=index,
                total_rows=total_rows,
            )
            if interval > 0 and index < len(rows_to_write):
                time.sleep(interval)


def playback_training_log(
    source_path: str | Path,
    output_path: str | Path,
    source: str = "csv",
    interval: float = 2.0,
    max_rows: int | None = None,
    overwrite: bool = True,
):
    """按行回放训练日志。

    CSV 源保持原文件表头和行内容；TensorBoard 等 Reader 源会先标准化为
    DanLing 可继续用 CSV Reader 读取的 `results.csv`。
    """
    normalized_source = "csv" if source == "ultralytics" else source
    if normalized_source == "csv" or (
        normalized_source == "auto" and UltralyticsCSVReader.detect(str(source_path))
    ):
        yield from playback_results_csv(
            source_path,
            output_path,
            interval=interval,
            max_rows=max_rows,
            overwrite=overwrite,
        )
        return

    history = read_history(str(source_path), source=normalized_source, limit=None)
    yield from playback_metric_snapshots(
        history,
        output_path,
        source_path=Path(source_path),
        interval=interval,
        max_rows=max_rows,
        overwrite=overwrite,
    )


def playback_metric_snapshots(
    history: list[MetricSnapshot],
    output_path: str | Path,
    source_path: Path,
    interval: float = 2.0,
    max_rows: int | None = None,
    overwrite: bool = True,
):
    """将标准化指标快照回放为 `results.csv`。"""
    target = _resolve_target_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    total_rows = len(history)
    rows_to_write = history if max_rows is None else history[:max_rows]
    extra_fields = _extra_raw_fields(rows_to_write)
    header = [*_RESULT_FIELDS, *extra_fields]
    mode = "w" if overwrite or not target.exists() else "a"
    should_write_header = mode == "w" or target.stat().st_size == 0

    with target.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if should_write_header:
            writer.writeheader()
        for index, metric in enumerate(rows_to_write, start=1):
            writer.writerow(_snapshot_row(metric, extra_fields))
            file.flush()
            yield PlaybackProgress(
                source_path=source_path,
                target_path=target,
                rows_written=index,
                total_rows=total_rows,
            )
            if interval > 0 and index < len(rows_to_write):
                time.sleep(interval)


def _resolve_target_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path if path.suffix == ".csv" else path / "results.csv"


def _extra_raw_fields(history: list[MetricSnapshot]) -> list[str]:
    fields = set(_RESULT_FIELDS)
    for metric in history:
        fields.update(str(key) for key in metric.raw)
    return sorted(fields - set(_RESULT_FIELDS))


def _snapshot_row(metric: MetricSnapshot, extra_fields: list[str]) -> dict[str, str]:
    row = {
        "epoch": _format_value(metric.epoch if metric.epoch is not None else metric.step),
        "time": _format_value(metric.timestamp),
        "train/loss": _format_value(metric.train_loss),
        "val/loss": _format_value(metric.val_loss),
        "metrics/precision(B)": _format_value(metric.precision),
        "metrics/recall(B)": _format_value(metric.recall),
        "metrics/mAP50(B)": _format_value(metric.map50),
        "metrics/mAP50-95(B)": _format_value(metric.map5095),
        "accuracy": _format_value(metric.accuracy),
        "lr": _format_value(metric.lr),
    }
    for field in extra_fields:
        row[field] = _format_value(metric.raw.get(field))
    return row


def _format_value(value) -> str:
    return "" if value is None else str(value)
