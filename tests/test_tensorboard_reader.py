"""测试 TensorBoard Reader 的可选依赖边界。"""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

from danling.readers.tensorboard_reader import TensorBoardReader


@dataclass(frozen=True)
class FakeScalarEvent:
    step: int
    value: float
    wall_time: float


def install_fake_tensorboard(
    monkeypatch,
    events_by_file: dict[str, dict[str, list[FakeScalarEvent]]],
):
    class FakeEventAccumulator:
        def __init__(self, path: str) -> None:
            self.path = path

        def Reload(self) -> None:
            return None

        def Tags(self) -> dict[str, list[str]]:
            return {"scalars": list(events_by_file.get(self.path, {}))}

        def Scalars(self, tag: str) -> list[FakeScalarEvent]:
            return events_by_file[self.path][tag]

    tensorboard = ModuleType("tensorboard")
    backend = ModuleType("tensorboard.backend")
    event_processing = ModuleType("tensorboard.backend.event_processing")
    event_accumulator = ModuleType("tensorboard.backend.event_processing.event_accumulator")
    event_accumulator.EventAccumulator = FakeEventAccumulator

    monkeypatch.setitem(sys.modules, "tensorboard", tensorboard)
    monkeypatch.setitem(sys.modules, "tensorboard.backend", backend)
    monkeypatch.setitem(sys.modules, "tensorboard.backend.event_processing", event_processing)
    monkeypatch.setitem(
        sys.modules,
        "tensorboard.backend.event_processing.event_accumulator",
        event_accumulator,
    )


def test_tensorboard_detects_event_file(tmp_path) -> None:
    event_file = tmp_path / "events.out.tfevents.test"
    event_file.write_text("", encoding="utf-8")

    reader = TensorBoardReader()

    assert reader.detect(str(tmp_path)) is True
    assert reader.detect(str(event_file)) is True


def test_tensorboard_detects_nested_logdir(tmp_path) -> None:
    event_file = tmp_path / "train" / "events.out.tfevents.test"
    event_file.parent.mkdir()
    event_file.write_text("", encoding="utf-8")

    assert TensorBoardReader.detect(str(tmp_path)) is True


def test_tensorboard_reads_nested_train_val_scalars(tmp_path, monkeypatch) -> None:
    train_event = tmp_path / "train" / "events.out.tfevents.train"
    val_event = tmp_path / "val" / "events.out.tfevents.val"
    train_event.parent.mkdir()
    val_event.parent.mkdir()
    train_event.write_text("", encoding="utf-8")
    val_event.write_text("", encoding="utf-8")
    install_fake_tensorboard(
        monkeypatch,
        {
            str(train_event): {
                "Loss/total": [
                    FakeScalarEvent(step=1, value=1.2, wall_time=10.0),
                    FakeScalarEvent(step=2, value=1.0, wall_time=20.0),
                ],
                "LearningRate/group0": [
                    FakeScalarEvent(step=2, value=0.01, wall_time=20.0),
                ],
            },
            str(val_event): {
                "Loss/total": [
                    FakeScalarEvent(step=1, value=1.5, wall_time=11.0),
                    FakeScalarEvent(step=2, value=1.3, wall_time=21.0),
                ],
                "mAP50": [
                    FakeScalarEvent(step=2, value=0.7, wall_time=21.0),
                ],
            },
        },
    )

    history = TensorBoardReader().read_history(str(tmp_path))

    assert len(history) == 2
    latest = history[-1]
    assert latest.source == "tensorboard"
    assert latest.run_path == str(tmp_path)
    assert latest.step == 2
    assert latest.epoch == 2
    assert latest.train_loss == pytest.approx(1.0)
    assert latest.val_loss == pytest.approx(1.3)
    assert latest.map50 == pytest.approx(0.7)
    assert latest.score == pytest.approx(0.7)
    assert latest.score_name == "mAP50"
    assert latest.lr == pytest.approx(0.01)
    assert latest.raw["train/Loss/total"] == pytest.approx(1.0)
    assert latest.raw["val/Loss/total"] == pytest.approx(1.3)


def test_tensorboard_uses_direct_train_val_dir_as_scope(tmp_path, monkeypatch) -> None:
    val_event = tmp_path / "val" / "events.out.tfevents.val"
    val_event.parent.mkdir()
    val_event.write_text("", encoding="utf-8")
    install_fake_tensorboard(
        monkeypatch,
        {
            str(val_event): {
                "Loss/total": [
                    FakeScalarEvent(step=1, value=1.5, wall_time=11.0),
                ],
            },
        },
    )

    history = TensorBoardReader().read_history(str(val_event.parent))

    assert history[0].train_loss is None
    assert history[0].val_loss == pytest.approx(1.5)
    assert history[0].raw["val/Loss/total"] == pytest.approx(1.5)


def test_tensorboard_missing_dependency_has_clear_error(tmp_path, monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tensorboard"):
            raise ImportError("no tensorboard")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"danling\[tensorboard\]"):
        TensorBoardReader().read_history(str(tmp_path))
