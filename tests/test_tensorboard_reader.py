"""测试 TensorBoard Reader 的可选依赖边界。"""

from __future__ import annotations

import builtins

import pytest

from danling.readers.tensorboard_reader import TensorBoardReader


def test_tensorboard_detects_event_file(tmp_path) -> None:
    event_file = tmp_path / "events.out.tfevents.test"
    event_file.write_text("", encoding="utf-8")

    reader = TensorBoardReader()

    assert reader.detect(str(tmp_path)) is True
    assert reader.detect(str(event_file)) is True


def test_tensorboard_missing_dependency_has_clear_error(tmp_path, monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tensorboard"):
            raise ImportError("no tensorboard")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"danling\[tensorboard\]"):
        TensorBoardReader().read_history(str(tmp_path))
