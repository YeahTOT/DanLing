"""单行状态渲染。"""

from __future__ import annotations

from danling.models import DanLingState


def render_statusline(state: DanLingState) -> str:
    """渲染适合 shell prompt/status line 使用的单行状态。"""
    mood_icon = _mood_icon(state.pet_mood.value)
    parts = [f"{mood_icon} {state.pet_name} {state.realm.name} {state.pet_mood.value}"]

    metric = state.metric
    if metric is not None:
        if metric.train_loss is not None:
            parts.append(f"loss {metric.train_loss:.3f}")
        if metric.score is not None:
            parts.append(f"{metric.score_name or 'score'} {metric.score:.3f}")
        if metric.epoch is not None:
            parts.append(f"epoch {metric.epoch}")
    else:
        parts.append("metric unknown")

    if state.hardware:
        device = state.hardware[0]
        hw_bits = []
        if device.util_percent is not None:
            hw_bits.append(f"{device.util_percent:.0f}%")
        if device.memory_used_mb is not None and device.memory_total_mb is not None:
            hw_bits.append(
                f"mem {device.memory_used_mb / 1024:.1f}/{device.memory_total_mb / 1024:.1f}G"
            )
        furnace_icon = _furnace_icon(state.furnace_state.value)
        hardware_text = " ".join(hw_bits)
        parts.append(f"{furnace_icon} {device.device_type.upper()} {hardware_text}")
    else:
        parts.append("hw unknown")

    return " | ".join(parts)


def _mood_icon(mood: str) -> str:
    return {
        "happy": "🐉😄",
        "evolving": "🐉✨",
        "anxious": "🐉😟",
        "sick": "🐉🤒",
        "sleeping": "🐉💤",
        "failed": "🐉💀",
        "idle": "🐉",
    }.get(mood, "🐉")


def _furnace_icon(state: str) -> str:
    return {
        "burning": "🔥",
        "hot": "🔥",
        "smoking": "💨",
        "exploded": "💥",
        "warm": "♨",
        "cold": "·",
    }.get(state, "?")
