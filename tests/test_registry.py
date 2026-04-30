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


def test_unknown_source_has_friendly_error() -> None:
    with pytest.raises(ValueError, match="Unsupported source"):
        read_history(str(FIXTURE_DIR), source="unknown")


def test_auto_without_matching_reader_has_friendly_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="No supported training log"):
        read_history(str(tmp_path), source="auto")
