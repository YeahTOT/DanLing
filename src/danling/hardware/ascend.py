"""Ascend `npu-smi` 硬件读取。"""

from __future__ import annotations

import re
import shutil
import subprocess
import time

from danling.hardware.base import BaseHardwareReader
from danling.models import HardwareSnapshot


class AscendNPUSMIReader(BaseHardwareReader):
    @property
    def name(self) -> str:
        return "ascend"

    def available(self) -> bool:
        return shutil.which("npu-smi") is not None

    def read(self) -> list[HardwareSnapshot]:
        if not self.available():
            return []
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return []
        if result.returncode != 0:
            return []
        return _parse_output(result.stdout)


def _parse_output(output: str) -> list[HardwareSnapshot]:
    snapshots: list[HardwareSnapshot] = []
    for line in output.splitlines():
        numbers = re.findall(r"\d+(?:\.\d+)?", line)
        if "NPU" not in line.upper() or not numbers:
            continue
        snapshots.append(
            HardwareSnapshot(
                device_type="ascend",
                device_id=numbers[0],
                name="Ascend NPU",
                timestamp=time.time(),
                raw={"line": line.strip()},
            )
        )
    return snapshots
