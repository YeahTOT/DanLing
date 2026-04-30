"""硬件读取入口。"""

from __future__ import annotations

from danling.hardware.ascend import AscendNPUSMIReader
from danling.hardware.base import BaseHardwareReader
from danling.hardware.nvidia import NvidiaSMIReader
from danling.models import HardwareSnapshot


def get_hardware_readers() -> list[BaseHardwareReader]:
    return [NvidiaSMIReader(), AscendNPUSMIReader()]


def read_all_hardware() -> list[HardwareSnapshot]:
    snapshots: list[HardwareSnapshot] = []
    for reader in get_hardware_readers():
        try:
            if reader.available():
                snapshots.extend(reader.read())
        except Exception:
            continue
    return snapshots
