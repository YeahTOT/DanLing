"""CSV 训练日志回放仿真器。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path


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


def _resolve_target_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path if path.suffix == ".csv" else path / "results.csv"
