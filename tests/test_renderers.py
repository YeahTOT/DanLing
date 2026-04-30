"""测试 Rich 渲染器和 statusline。"""

from __future__ import annotations

from rich.console import Console

from danling.models import (
    CultivationRealm,
    DanLingEvent,
    DanLingState,
    HardwareSnapshot,
    MetricSnapshot,
    Severity,
)
from danling.renderers.console import render_events, render_metric_table, render_state_panel
from danling.renderers.statusline import render_statusline


def test_statusline_handles_missing_metric_and_hardware() -> None:
    line = render_statusline(DanLingState())

    assert line
    assert "hw unknown" in line


def test_statusline_includes_metric_and_hardware() -> None:
    state = DanLingState(
        realm=CultivationRealm(name="结丹期", rank=3),
        metric=MetricSnapshot(train_loss=1.234, score=0.672, score_name="mAP50"),
        hardware=[
            HardwareSnapshot(
                device_type="nvidia",
                util_percent=78,
                memory_used_mb=21_400,
                memory_total_mb=24_000,
            )
        ],
    )

    line = render_statusline(state)

    assert "结丹期" in line
    assert "Lv." not in line
    assert "1.234" in line
    assert "mAP50" in line
    assert "78%" in line


def test_rich_renderables_can_render() -> None:
    state = DanLingState(
        metric=MetricSnapshot(epoch=5, train_loss=1.0),
        events=[DanLingEvent(severity=Severity.warning, title="Test", message="hello")],
    )
    console = Console(record=True, width=120)

    console.print(render_state_panel(state))
    console.print(render_metric_table(state.metric))
    console.print(render_events(state.events))

    output = console.export_text()
    assert "DanLing" in output
    assert "epoch" in output
    assert "Test" in output
