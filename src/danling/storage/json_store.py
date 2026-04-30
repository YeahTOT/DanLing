"""JSON 状态持久化。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonStateStore:
    """原子写入的本地状态存储。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path.home() / ".danling" / "state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, self.path)

    def update_run(self, run_id: str, data: dict[str, Any]) -> None:
        state = self.load()
        runs = state.setdefault("runs", {})
        current = runs.get(run_id, {})
        current.update(data)
        runs[run_id] = current
        self.save(state)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.load().get("runs", {}).get(run_id, {})
