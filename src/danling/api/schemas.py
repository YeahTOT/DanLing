"""Small serialization helpers for the desktop API."""

from __future__ import annotations

from typing import Any

from danling.models import DanLingState

ACTIVE_ANIMATION = "running"


def state_to_desktop_payload(state: DanLingState, *, has_path: bool) -> dict[str, Any]:
    """Return the state JSON plus desktop-only display hints."""
    payload = state.to_dict()
    payload["desktop"] = {
        "animation": animation_for_state(payload.get("pet_mood")),
        "status_label": status_label_for_state(state, has_path=has_path),
    }
    return payload


def animation_for_state(pet_mood: object) -> str:
    """Map domain moods to the first desktop pet animation set."""
    mood = str(pet_mood or "")
    if mood in {"failed", "sick"}:
        return "failed"
    if mood in {"sleeping", "idle"}:
        return ACTIVE_ANIMATION
    if mood in {"anxious"}:
        return "waiting"
    if mood in {"evolving", "happy", "normal"}:
        return ACTIVE_ANIMATION
    return ACTIVE_ANIMATION


def status_label_for_state(state: DanLingState, *, has_path: bool) -> str:
    """Return a compact Chinese status label for the pet bubble."""
    if not has_path:
        return "等待配置训练日志"
    if state.events:
        return state.events[0].title or state.events[0].message
    if state.metric is None:
        return "等待训练指标"
    metric = state.metric
    if metric.score is not None:
        score_name = metric.score_name or "score"
        return f"{score_name}: {metric.score:.4f}"
    if metric.train_loss is not None:
        return f"loss: {metric.train_loss:.4f}"
    return state.pet_mood.value
