"""NVIDIA `nvidia-smi` 硬件读取。"""

from __future__ import annotations

import shutil
import subprocess
import time

from danling.hardware.base import BaseHardwareReader
from danling.models import HardwareSnapshot


class NvidiaSMIReader(BaseHardwareReader):
    @property
    def name(self) -> str:
        return "nvidia"

    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def read(self) -> list[HardwareSnapshot]:
        if not self.available():
            return []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return []

        if result.returncode != 0:
            return []
        return [_parse_line(line) for line in result.stdout.splitlines() if line.strip()]


def _parse_line(line: str) -> HardwareSnapshot:
    parts = [part.strip() for part in line.split(",")]
    while len(parts) < 7:
        parts.append("")
    index, name, util, used, total, temp, power = parts[:7]
    return HardwareSnapshot(
        device_type="nvidia",
        device_id=index or None,
        name=name or None,
        util_percent=_float_or_none(util),
        memory_used_mb=_float_or_none(used),
        memory_total_mb=_float_or_none(total),
        temperature_c=_float_or_none(temp),
        power_w=_float_or_none(power),
        timestamp=time.time(),
        raw={
            "index": index,
            "name": name,
            "utilization.gpu": util,
            "memory.used": used,
            "memory.total": total,
            "temperature.gpu": temp,
            "power.draw": power,
        },
    )


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
