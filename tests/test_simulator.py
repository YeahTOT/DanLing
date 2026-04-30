"""测试 CSV 回放仿真器。"""

from __future__ import annotations

import csv
from pathlib import Path

from danling.simulator import playback_results_csv


def _write_source(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "epoch,time,train/box_loss,metrics/mAP50(B)",
                "1,1.0,1.2,0.5",
                "2,2.0,1.1,0.6",
                "3,3.0,1.0,0.7",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_playback_results_csv_writes_header_and_limited_rows(tmp_path) -> None:
    source = tmp_path / "source.csv"
    output_dir = tmp_path / "run"
    _write_source(source)

    progress = list(playback_results_csv(source, output_dir, interval=0, max_rows=2))

    target = output_dir / "results.csv"
    with target.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(progress) == 2
    assert progress[-1].rows_written == 2
    assert progress[-1].target_path == target
    assert rows == [
        ["epoch", "time", "train/box_loss", "metrics/mAP50(B)"],
        ["1", "1.0", "1.2", "0.5"],
        ["2", "2.0", "1.1", "0.6"],
    ]


def test_playback_results_csv_appends_when_requested(tmp_path) -> None:
    source = tmp_path / "source.csv"
    output_dir = tmp_path / "run"
    _write_source(source)

    list(playback_results_csv(source, output_dir, interval=0, max_rows=1))
    list(playback_results_csv(source, output_dir, interval=0, max_rows=1, overwrite=False))

    with (output_dir / "results.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert rows == [
        ["epoch", "time", "train/box_loss", "metrics/mAP50(B)"],
        ["1", "1.0", "1.2", "0.5"],
        ["1", "1.0", "1.2", "0.5"],
    ]
