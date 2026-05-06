"""测试 Reader 注册系统。"""

from __future__ import annotations

from pathlib import Path

import pytest

from danling.readers.registry import ReaderRegistry, read_history

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ultralytics_run"


def test_registry_auto_detects_csv_fixture() -> None:
    registry = ReaderRegistry.default()

    reader = registry.detect(str(FIXTURE_DIR))

    assert reader is not None
    assert reader.name == "csv"


def test_read_history_source_csv_and_auto() -> None:
    assert len(read_history(str(FIXTURE_DIR), source="csv")) == 5
    assert len(read_history(str(FIXTURE_DIR), source="auto")) == 5


def test_read_history_source_ultralytics_log_alias(tmp_path) -> None:
    log_path = tmp_path / "train.txt"
    log_path.write_text(
        "\n".join(
            [
                "Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size",
                "1/10 1.0G 1.0 2.0 3.0 4 640: 100% 1/1",
            ]
        ),
        encoding="utf-8",
    )

    history = read_history(str(log_path), source="log")

    assert len(history) == 1
    assert history[0].source == "ultralytics-log"


def test_unknown_source_has_friendly_error() -> None:
    with pytest.raises(ValueError, match="Unsupported source"):
        read_history(str(FIXTURE_DIR), source="unknown")


def test_auto_without_matching_reader_has_friendly_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="No supported training log"):
        read_history(str(tmp_path), source="auto")
