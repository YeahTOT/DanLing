"""Reader 抽象基类。

定义所有数据源 Reader 的统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from danling.models import MetricSnapshot


class BaseReader(ABC):
    """数据源 Reader 抽象基类。

    所有 Reader 实现都必须继承此类并实现其抽象方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Reader 名称，例如 "csv", "tensorboard"。"""
        ...

    @staticmethod
    @abstractmethod
    def detect(path: str) -> bool:
        """检测给定路径是否可被此 Reader 读取。"""
        ...

    @abstractmethod
    def read_latest(self, path: str) -> MetricSnapshot | None:
        """读取最新一条指标快照。"""
        ...

    @abstractmethod
    def read_history(self, path: str, limit: int | None = None) -> list[MetricSnapshot]:
        """读取历史指标快照列表。"""
        ...
