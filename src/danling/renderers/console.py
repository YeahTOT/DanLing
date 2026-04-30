"""Rich 终端渲染器。"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from danling.models import DanLingEvent, DanLingState, HardwareSnapshot, MetricSnapshot
from danling.renderers.statusline import render_statusline


def render_state_panel(state: DanLingState):
    """渲染完整状态面板。"""
    hardware = _hardware_table(state.hardware)
    group = Group(render_metric_table(state.metric), hardware, render_events(state.events))
    return Panel(
        group, title=render_statusline(state), border_style=_border_style(state.pet_mood.value)
    )


def render_events(events: list[DanLingEvent]):
    table = Table(title="最近事件")
    table.add_column("级别", style="cyan")
    table.add_column("标题", style="yellow")
    table.add_column("说明")
    if not events:
        table.add_row("info", "暂无事件", "-")
    for event in events[-5:]:
        table.add_row(event.severity.value, event.title, event.message)
    return table


def render_metric_table(metric: MetricSnapshot | None):
    table = Table(title="训练指标")
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    if metric is None:
        table.add_row("metric", "unknown")
        return table

    rows = [
        ("epoch", metric.epoch),
        ("train_loss", _fmt(metric.train_loss)),
        ("val_loss", _fmt(metric.val_loss)),
        (metric.score_name or "score", _fmt(metric.score)),
        ("mAP50", _fmt(metric.map50)),
        ("mAP50-95", _fmt(metric.map5095)),
        ("lr", _fmt(metric.lr)),
    ]
    for key, value in rows:
        table.add_row(str(key), "-" if value is None else str(value))
    return table


def _hardware_table(hardware: list[HardwareSnapshot]):
    table = Table(title="炼丹炉")
    table.add_column("设备", style="cyan")
    table.add_column("利用率")
    table.add_column("显存")
    table.add_column("温度")
    if not hardware:
        table.add_row("unknown", "-", "-", "-")
        return table
    for device in hardware:
        memory = "-"
        if device.memory_used_mb is not None and device.memory_total_mb is not None:
            memory = f"{device.memory_used_mb / 1024:.1f}/{device.memory_total_mb / 1024:.1f}G"
        table.add_row(
            f"{device.device_type}:{device.device_id or '-'} {device.name or ''}".strip(),
            f"{device.util_percent:.0f}%" if device.util_percent is not None else "-",
            memory,
            f"{device.temperature_c:.0f}C" if device.temperature_c is not None else "-",
        )
    return table


def _fmt(value: float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _border_style(mood: str) -> str:
    return {
        "failed": "red",
        "sick": "red",
        "anxious": "yellow",
        "evolving": "magenta",
        "happy": "green",
    }.get(mood, "cyan")
