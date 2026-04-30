"""测试 Textual TUI 可选依赖边界。"""

from __future__ import annotations

from pathlib import Path

from rich.cells import cell_len
from typer.testing import CliRunner

from danling.cli import app
from danling.models import (
    CultivationRealm,
    DanLingEvent,
    DanLingState,
    FurnaceState,
    HardwareSnapshot,
    MetricSnapshot,
    PetMood,
    Severity,
)
from danling.renderers.textual_app import (
    FURNACE_FRAMES,
    create_app,
    furnace_animation_frame,
    pet_animation_frame,
    render_tui_events,
    render_tui_hardware,
    render_tui_metrics,
    render_tui_pet,
    render_tui_text,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ultralytics_run"
runner = CliRunner()


def test_tui_missing_textual_has_clear_message(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "textual" else object()
    )

    result = runner.invoke(app, ["tui", str(FIXTURE_DIR), "--source", "csv"])

    assert result.exit_code == 1
    assert "danling[tui]" in result.stdout


def test_render_tui_text_includes_state_summary() -> None:
    state = DanLingState(
        pet_mood=PetMood.happy,
        furnace_state=FurnaceState.burning,
        realm=CultivationRealm(name="筑基期", rank=2),
        metric=MetricSnapshot(epoch=3, train_loss=1.23, score=0.78, score_name="mAP50"),
        hardware=[
            HardwareSnapshot(
                device_type="nvidia",
                device_id="0",
                util_percent=76,
                memory_used_mb=21_400,
                memory_total_mb=24_000,
                temperature_c=62,
            )
        ],
        events=[
            DanLingEvent(
                severity=Severity.info,
                title="指标创新高",
                message="mAP50 创新高，丹灵升级！",
            )
        ],
    )

    text = render_tui_text(state, path=FIXTURE_DIR, source="csv")

    assert "DanLing TUI" in text
    assert "happy" in text
    assert "境界: 筑基期" in text
    assert "epoch: 3" in text
    assert "mAP50: 0.7800" in text
    assert "GPU" in text
    assert "指标创新高" in text


def test_pet_animation_frame_changes_with_frame_index() -> None:
    frame_a = pet_animation_frame(PetMood.happy, 0)
    frame_b = pet_animation_frame(PetMood.happy, 1)

    assert frame_a != frame_b
    assert "DanLing" in frame_a


def test_pet_animation_frame_differs_by_mood() -> None:
    happy = pet_animation_frame(PetMood.happy, 0)
    failed = pet_animation_frame(PetMood.failed, 0)

    assert happy != failed
    assert "happy" in happy
    assert "failed" in failed


def test_furnace_animation_frame_changes_by_state() -> None:
    burning = furnace_animation_frame(FurnaceState.burning, 0)
    smoking = furnace_animation_frame(FurnaceState.smoking, 0)
    exploded = furnace_animation_frame(FurnaceState.exploded, 0)

    assert burning != smoking
    assert smoking != exploded
    assert "furnace" in burning
    assert "^^" in burning
    assert "~~~" in smoking


def test_unknown_furnace_frame_is_offline_furnace_not_question_placeholder() -> None:
    frame = furnace_animation_frame(FurnaceState.unknown, 0)

    assert "offline" in frame
    assert "???" not in frame
    assert "^^" in frame


def test_furnace_art_uses_ascii_outline_with_only_flame_icon_non_ascii() -> None:
    for frames in FURNACE_FRAMES.values():
        for frame in frames:
            assert all(char.isascii() or char == "🔥" for char in frame)


def test_intact_furnace_frames_keep_bottom_outline_stable() -> None:
    intact_states = set(FURNACE_FRAMES) - {"exploded"}

    for state in intact_states:
        for frame in FURNACE_FRAMES[state]:
            lines = frame.strip().splitlines()

            assert lines[-1] == " |_____________|"


def test_intact_furnace_frames_keep_top_aligned_with_body_center() -> None:
    intact_states = set(FURNACE_FRAMES) - {"exploded"}

    for state in intact_states:
        for frame in FURNACE_FRAMES[state]:
            lines = frame.strip().splitlines()
            body_center = _line_span_center(lines[-1])
            lid_index = next(
                index
                for index, line in enumerate(lines)
                if "____" in line and not line.strip().startswith("|")
            )

            assert _line_span_center(lines[lid_index]) == body_center
            assert _line_span_center(lines[lid_index + 1]) == body_center
            assert _line_span_center(lines[lid_index + 2]) == body_center
            assert lines[lid_index + 1].index("V") == body_center


def test_active_furnace_frames_use_flames_at_the_hearth_bottom() -> None:
    active_states = {"warm", "burning", "hot", "smoking"}

    for state in active_states:
        for frame in FURNACE_FRAMES[state]:
            hearth_bottom = frame.strip().splitlines()[-2]

            assert "🔥" in hearth_bottom
            assert "_" not in hearth_bottom


def test_furnace_frames_keep_hearth_bottom_display_width_stable() -> None:
    intact_states = set(FURNACE_FRAMES) - {"exploded"}

    for state in intact_states:
        for frame in FURNACE_FRAMES[state]:
            lines = frame.strip().splitlines()

            assert cell_len(lines[-2]) == cell_len(lines[-1])


def test_tui_panel_helpers_include_metrics_hardware_and_events() -> None:
    state = DanLingState(
        pet_mood=PetMood.evolving,
        furnace_state=FurnaceState.hot,
        realm=CultivationRealm(name="元婴期", rank=4),
        metric=MetricSnapshot(epoch=156, train_loss=1.284, score=0.631, score_name="mAP50"),
        hardware=[
            HardwareSnapshot(
                device_type="nvidia",
                util_percent=76,
                memory_used_mb=21_400,
                memory_total_mb=24_000,
            )
        ],
        events=[DanLingEvent(Severity.info, "升级", "表现优异")],
    )

    assert "156" in render_tui_metrics(state)
    assert "元婴期" in render_tui_metrics(state)
    assert "GPU" in render_tui_hardware(state)
    assert "升级" in render_tui_events(state)


def test_tui_pet_and_metrics_include_realm_progress_and_range() -> None:
    state = DanLingState(
        pet_mood=PetMood.evolving,
        furnace_state=FurnaceState.hot,
        realm=CultivationRealm(
            name="筑基期",
            rank=2,
            score_floor=0.60,
            score_ceiling=0.65,
            progress=0.10,
        ),
        metric=MetricSnapshot(epoch=4, score=0.605, score_name="mAP50"),
    )

    assert "境界进度: 10%" in render_tui_pet(state)
    assert "境界区间: 0.6000 - 0.6500" in render_tui_metrics(state)


def test_tui_metrics_render_top_realm_range_and_unconfigured_fallback() -> None:
    top_state = DanLingState(
        realm=CultivationRealm(name="化神期", score_floor=0.80, progress=1.0)
    )
    empty_state = DanLingState(realm=CultivationRealm(name="炼器期"))

    assert "境界区间: >= 0.8000" in render_tui_metrics(top_state)
    assert "境界进度: 未配置" in render_tui_pet(empty_state)
    assert "境界区间: 未配置" in render_tui_metrics(empty_state)


def test_create_app_instantiates_when_textual_is_available() -> None:
    try:
        import textual  # noqa: F401
    except ImportError:
        return

    app = create_app(FIXTURE_DIR, source="csv", interval=0.5, no_hardware=True)

    assert app is not None


def _line_span_center(line: str) -> float:
    indexes = [index for index, char in enumerate(line) if char != " "]
    return (indexes[0] + indexes[-1]) / 2
