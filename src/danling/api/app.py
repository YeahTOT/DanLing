"""FastAPI app for the DanLing desktop pet sidecar."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from danling.api.schemas import state_to_desktop_payload
from danling.api.state_service import DanLingStateService
from danling.renderers.config_tui import (
    config_to_form_values,
    save_config_form,
    suggest_config_from_results,
)


def create_app(
    *,
    config_path: Path | None = None,
    path: Path | None = None,
    source: str | None = None,
    no_hardware: bool = False,
):
    """Create the FastAPI app lazily so CLI users need `danling[web]` only for API mode."""
    try:
        from fastapi import Body, FastAPI, WebSocket
    except ImportError as exc:  # pragma: no cover - exercised through CLI environments.
        raise RuntimeError("Desktop API requires: pip install danling[web]") from exc

    app = FastAPI(title="DanLing Desktop Pet API", version="1")
    request_body = Body(...)
    service = DanLingStateService(
        config_path=config_path,
        path=path,
        source=source,
        no_hardware=no_hardware,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        state, has_path = service.read_state()
        return state_to_desktop_payload(state, has_path=has_path)

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        config_path = service.config_file()
        config = service.load_config()
        return {
            "path": str(config_path),
            "config": config.to_dict(),
            "values": config_to_form_values(config),
        }

    @app.put("/api/config")
    def put_config(values: dict[str, str] = request_body) -> dict[str, Any]:
        config_path = service.config_file()
        config = save_config_form(config_path, values)
        return {
            "path": str(config_path),
            "config": config.to_dict(),
            "values": config_to_form_values(config),
        }

    @app.post("/api/config/suggest")
    def suggest_config(payload: dict[str, str] = request_body) -> dict[str, dict[str, str]]:
        suggested = suggest_config_from_results(
            payload.get("path", ""),
            payload.get("metric", ""),
            source=payload.get("source", "csv"),
        )
        return {"suggested": suggested}

    @app.websocket("/ws/state")
    async def websocket_state(websocket: WebSocket) -> None:
        await websocket.accept()
        config = service.load_config()
        interval = max(config.watch_interval, 0.5)
        while True:
            state, has_path = service.read_state()
            await websocket.send_json(state_to_desktop_payload(state, has_path=has_path))
            await asyncio.sleep(interval)

    return app


def default_app():
    """ASGI factory for `uvicorn danling.api.app:default_app --factory`."""
    return create_app()
