"""Reader 注册和自动检测。"""

from __future__ import annotations

from collections.abc import Iterable

from danling.models import MetricSnapshot
from danling.readers.base import BaseReader
from danling.readers.tensorboard_reader import TensorBoardReader
from danling.readers.ultralytics_csv import UltralyticsCSVReader
from danling.readers.ultralytics_log import UltralyticsLogReader


class ReaderRegistry:
    def __init__(self, readers: Iterable[BaseReader] | None = None) -> None:
        self._readers: list[BaseReader] = []
        for reader in readers or []:
            self.register(reader)

    @classmethod
    def default(cls) -> ReaderRegistry:
        registry = cls()
        registry.register(UltralyticsCSVReader())
        registry.register(UltralyticsLogReader())
        registry.register(TensorBoardReader())
        return registry

    def register(self, reader_cls_or_instance) -> None:
        reader = (
            reader_cls_or_instance()
            if isinstance(reader_cls_or_instance, type)
            else reader_cls_or_instance
        )
        self._readers.append(reader)

    def get_reader(self, name: str) -> BaseReader | None:
        normalized = _normalize_source(name)
        for reader in self._readers:
            if reader.name == normalized:
                return reader
        return None

    def detect(self, path: str) -> BaseReader | None:
        for reader in self._readers:
            try:
                if reader.detect(path):
                    return reader
            except Exception:
                continue
        return None

    def read_history(
        self, path: str, source: str = "auto", limit: int | None = None
    ) -> list[MetricSnapshot]:
        reader = self.detect(path) if source == "auto" else self.get_reader(source)
        if reader is None:
            if source == "auto":
                raise ValueError(
                    "No supported training log found. "
                    "Expected results.csv, Ultralytics log/txt, or TensorBoard event files."
                )
            raise ValueError(f"Unsupported source: {source}")
        return reader.read_history(path, limit=limit)


def read_history(path: str, source: str = "auto", limit: int | None = None) -> list[MetricSnapshot]:
    return ReaderRegistry.default().read_history(path, source=source, limit=limit)


def read_latest(path: str, source: str = "auto") -> MetricSnapshot | None:
    history = read_history(path, source=source)
    return history[-1] if history else None


def _normalize_source(source: str) -> str:
    if source == "ultralytics":
        return "csv"
    if source in {"log", "txt", "ultralytics_log", "ultralytics-log"}:
        return "ultralytics-log"
    return source
