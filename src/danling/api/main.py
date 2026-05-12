"""Uvicorn entrypoint for the DanLing desktop sidecar."""

from __future__ import annotations

from pathlib import Path

from danling.api.app import create_app


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: Path | None = None,
    path: Path | None = None,
    source: str | None = None,
    no_hardware: bool = False,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised through CLI environments.
        raise RuntimeError("Desktop API requires: pip install danling[web]") from exc

    uvicorn.run(
        create_app(
            config_path=config_path,
            path=path,
            source=source,
            no_hardware=no_hardware,
        ),
        host=host,
        port=port,
    )
