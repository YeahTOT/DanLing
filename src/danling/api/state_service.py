"""State aggregation service used by the desktop sidecar."""

from __future__ import annotations

from pathlib import Path

from danling.config import DanLingConfig, load_config
from danling.engine.state import build_state
from danling.hardware import read_all_hardware
from danling.models import DanLingEvent, DanLingState, Severity
from danling.readers.registry import read_history
from danling.storage.json_store import JsonStateStore


class DanLingStateService:
    """Read config/logs/hardware and build a desktop-friendly state."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        path: Path | None = None,
        source: str | None = None,
        no_hardware: bool = False,
    ) -> None:
        self.config_path = config_path
        self.path = path
        self.source = source
        self.no_hardware = no_hardware

    def load_config(self) -> DanLingConfig:
        if self.config_path is not None and not self.config_path.is_file():
            return DanLingConfig()
        return load_config(str(self.config_path) if self.config_path is not None else None)

    def config_file(self) -> Path:
        return self.config_path or Path.cwd() / "danling.yaml"

    def resolve_path_and_source(
        self, config: DanLingConfig | None = None
    ) -> tuple[Path | None, str]:
        config = config or self.load_config()
        path = self.path
        if path is None and config.data_path:
            path = Path(config.data_path).expanduser()

        source = self.source or config.source or "auto"
        return path, source

    def read_state(self) -> tuple[DanLingState, bool]:
        config = self.load_config()
        path, source = self.resolve_path_and_source(config)
        if path is None:
            return DanLingState(pet_name=config.pet_name), False

        try:
            history = read_history(str(path), source=source)
            hardware = [] if self.no_hardware else read_all_hardware()
            persisted = JsonStateStore().load()
            return build_state(history, hardware, config, persisted=persisted), True
        except Exception as exc:
            return (
                DanLingState(
                    pet_name=config.pet_name,
                    events=[
                        DanLingEvent(
                            severity=Severity.warning,
                            title="日志读取失败",
                            message=str(exc),
                        )
                    ],
                ),
                True,
            )
