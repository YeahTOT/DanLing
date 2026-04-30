"""Reader 层 — 数据源读取抽象。

Reader 只负责读取和标准化指标，不负责判断宠物情绪。
"""

from __future__ import annotations

from .base import BaseReader

__all__ = ["BaseReader"]
