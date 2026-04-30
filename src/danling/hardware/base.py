"""硬件 Reader 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from danling.models import HardwareSnapshot


class BaseHardwareReader(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Reader 名称。"""
        ...

    @abstractmethod
    def available(self) -> bool:
        """当前系统是否可读取该硬件。"""
        ...

    @abstractmethod
    def read(self) -> list[HardwareSnapshot]:
        """读取硬件快照。"""
        ...
