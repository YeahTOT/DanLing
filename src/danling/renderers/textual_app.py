"""可选 Textual 全屏 TUI。"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from danling.config import load_config
from danling.engine.state import build_state
from danling.hardware import read_all_hardware
from danling.models import DanLingState, FurnaceState, PetMood
from danling.readers.registry import read_history

ANIMATION_INTERVAL = 0.25

PET_FRAMES: dict[str, list[str]] = {
    "idle": [
        r"""
DanLing idle
   /\_/\
  ( o.o )  ...
  /|   |\
   /___\
""",
        r"""
DanLing idle
   /\_/\
  ( -.- )  ...
  /|   |\
   /___\
""",
    ],
    "happy": [
        r"""
DanLing happy
    /\_/\
  >( ^.^ )<  训练顺利
   /|~~~|\
    /___\
""",
        r"""
DanLing happy
   \/\_/\/
  >( ^o^ )<  尾巴摇摇
   /|~~~|\
    /___\
""",
    ],
    "normal": [
        r"""
DanLing normal
    /\_/\
   ( o_o )  观察中
   /|   |\
    /___\
""",
        r"""
DanLing normal
    /\_/\
   ( o_o )  盯着炉火
  _/|   |\_
    /___\
""",
    ],
    "anxious": [
        r"""
DanLing anxious
    /\_/\
   ( >_< )  指标卡住
   /|~~~|\
    /___\
""",
        r"""
DanLing anxious
    /\_/\
   ( ;_; )  需要关注
  _/|~~~|\_
    /___\
""",
    ],
    "sick": [
        r"""
DanLing sick
    /\_/\
   ( x_x )  loss 异常
   /|~~~|\
    /___\
""",
        r"""
DanLing sick
    /\_/\
   ( @.@ )  头晕
  _/|~~~|\_
    /___\
""",
    ],
    "sleeping": [
        r"""
DanLing sleeping
    /\_/\
   ( -.- )  zZ
   /|___|\
""",
        r"""
DanLing sleeping
    /\_/\
   ( -.- )  zZz
   /|___|\
""",
    ],
    "evolving": [
        r"""
DanLing evolving
  ✨ /\_/\ ✨
  >( ^.^ )<  升级中
   /|~~~|\
    /___\
""",
        r"""
DanLing evolving
 ✨ \/\_/\/ ✨
  >( ^O^ )<  光芒闪烁
   /|~~~|\
    /___\
""",
    ],
    "failed": [
        r"""
DanLing failed
    /\_/\
   ( X_X )  训练失败
   /|...|\
""",
        r"""
DanLing failed
    /\_/\
   ( x_x )  请检查日志
   /|...|\
""",
    ],
}

FURNACE_FRAMES: dict[str, list[str]] = {
    "cold": [
        r"""
furnace cold
        |
    ____|____
   /    V    \
  /  ALCHEMY  \
 |   [____]    |
 |_____________|
""",
        r"""
furnace cold
        |
    ____|____
   /    V    \
  /  ALCHEMY  \
 |   [ .. ]    |
 |_____________|
""",
    ],
    "warm": [
        r"""
furnace warm
        |
    ____|____
   /    V    \
  /  ALCHEMY  \
 |    /^^\     |
 |    🔥🔥     |
 |_____________|
""",
        r"""
furnace warm
        |
    ____|____
   /    V    \
  /  ALCHEMY  \
 |    \^^/     |
 |    🔥🔥     |
 |_____________|
""",
    ],
    "burning": [
        r"""
furnace burning
       ) (
      (^^^)
    ____|____
   /    V    \
  /  ALCHEMY  \
 |   /^^^^\    |
 |   🔥🔥🔥    |
 |_____________|
""",
        r"""
furnace burning
      (   )
       )^(
    ____|____
   /    V    \
  /  ALCHEMY  \
 |   \^^^^/    |
 |   🔥🔥🔥    |
 |_____________|
""",
    ],
    "hot": [
        r"""
furnace hot
    ~~~) (~~~
     (^^^^^)
    ____|____
   /    V    \
  /  ALCHEMY  \
 |  /^^^^^^\   |
 |   🔥🔥🔥    |
 |_____________|
""",
        r"""
furnace hot
    ~~~( )~~~
     (^^^^^)
    ____|____
   /    V    \
  /  ALCHEMY  \
 |  \^^^^^^/   |
 |   🔥🔥🔥    |
 |_____________|
""",
    ],
    "smoking": [
        r"""
furnace smoking
    ~~~   ~~~
       ) (
      (^^^)
    ____|____
   /    V    \
  /  ALCHEMY  \
 |   /^^^^\    |
 |   🔥🔥🔥    |
 |_____________|
""",
        r"""
furnace smoking
   ~~~     ~~~
      (   )
       )^(
    ____|____
   /    V    \
  /  ALCHEMY  \
 |   \^^^^/    |
 |   🔥🔥🔥    |
 |_____________|
""",
    ],
    "exploded": [
        r"""
furnace exploded
      BOOM!
    ** FIRE **
    ___    ___
   /   \__/   \
  /  BROKEN    \
 |__  \__/  ___|
""",
        r"""
furnace exploded
     BOOM!!
   *** FIRE ***
    ___    ___
   /  _\__/_  \
  /  BROKEN    \
 |___      ____|
""",
    ],
    "unknown": [
        r"""
furnace offline
        |
    ____|____
   /    V    \
  /  OFFLINE  \
 |    /^^\     |
 |    \__/     |
 |_____________|
""",
        r"""
furnace offline
        |
    ____|____
   /    V    \
  /  OFFLINE  \
 |    \^^/     |
 |    /__\     |
 |_____________|
""",
    ],
}


def textual_available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def require_textual() -> None:
    if not textual_available():
        raise RuntimeError("Textual UI requires: pip install danling[tui]")


def create_app(
    path: Path,
    source: str = "auto",
    interval: float = 2.0,
    state_provider: Callable[[], DanLingState] | None = None,
    display_label: str | None = None,
    **kwargs,
):
    """创建 Textual App；未安装 Textual 时给出友好错误。"""
    require_textual()

    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Static

    config_path = kwargs.get("config")
    no_hardware = bool(kwargs.get("no_hardware", False))
    path_label = display_label or str(path)

    class DanLingTUIApp(App):
        BINDINGS = [("q", "quit", "退出"), ("r", "refresh", "刷新"), ("h", "help", "帮助")]
        CSS = """
        Screen {
            background: #061016;
            color: #d6fff8;
        }

        #dashboard {
            height: 1fr;
            padding: 1;
        }

        #left-pane {
            width: 42%;
            min-width: 34;
        }

        #right-pane {
            width: 58%;
            min-width: 48;
        }

        .panel {
            border: round #16d9c5;
            padding: 1 2;
            margin: 0 1 1 0;
            background: #0b1720;
        }

        #events-panel {
            height: 9;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self.frame_index = 0
            self.state: DanLingState | None = None
            self.error_message: str | None = None
            self.show_help = False

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="dashboard"):
                with Vertical(id="left-pane"):
                    yield Static("DanLing TUI loading...", id="pet-panel", classes="panel")
                    yield Static("", id="furnace-panel", classes="panel")
                with Vertical(id="right-pane"):
                    yield Static("", id="metrics-panel", classes="panel")
                    yield Static("", id="hardware-panel", classes="panel")
                    yield Static("", id="events-panel", classes="panel")
            yield Footer()

        def on_mount(self) -> None:
            self.refresh_state()
            self.set_interval(ANIMATION_INTERVAL, self.animate)
            self.set_interval(interval, self.refresh_state)

        def refresh_state(self) -> None:
            try:
                self.state = _read_tui_state(
                    path,
                    source,
                    config_path,
                    no_hardware,
                    state_provider=state_provider,
                )
                self.error_message = None
            except Exception as exc:
                self.state = None
                self.error_message = str(exc)
            self.update_panels()

        def animate(self) -> None:
            self.frame_index += 1
            self.update_panels()

        def update_panels(self) -> None:
            pet_panel = self.query_one("#pet-panel", Static)
            furnace_panel = self.query_one("#furnace-panel", Static)
            metrics_panel = self.query_one("#metrics-panel", Static)
            hardware_panel = self.query_one("#hardware-panel", Static)
            events_panel = self.query_one("#events-panel", Static)

            if self.show_help:
                pet_panel.update("DanLing Help\n\nq 退出\nr 立即刷新\nh 切换帮助")
                furnace_panel.update("动画刷新: 0.25s\n数据刷新: --interval")
                metrics_panel.update(
                    f"path: {path_label}\nsource: {source}\nconfig: --config danling.yaml"
                )
                hardware_panel.update(
                    "配置: danling config tui\n"
                    "监控: danling tui logs/run --config danling.yaml --no-hardware"
                )
                events_panel.update("再次按 h 返回监控界面")
                return

            if self.error_message is not None or self.state is None:
                pet_panel.update(f"DanLing TUI\npath: {path_label}\nsource: {source}")
                furnace_panel.update("furnace unknown")
                metrics_panel.update(f"error: {self.error_message or 'loading'}")
                hardware_panel.update("hardware: unknown")
                events_panel.update("events: waiting for data")
                return

            pet_panel.update(render_tui_pet(self.state, self.frame_index))
            furnace_frame = furnace_animation_frame(
                self.state.furnace_state, self.frame_index
            )
            furnace_panel.update(furnace_frame)
            metrics_panel.update(
                render_tui_metrics(self.state, source=source, path=path_label)
            )
            hardware_panel.update(render_tui_hardware(self.state))
            events_panel.update(render_tui_events(self.state, self.frame_index))

        def action_refresh(self) -> None:
            self.refresh_state()

        def action_help(self) -> None:
            self.show_help = not self.show_help
            self.update_panels()

    return DanLingTUIApp()


def pet_animation_frame(mood: PetMood | str, frame_index: int) -> str:
    """返回宠物动画帧。"""
    mood_value = _enum_value(mood)
    frames = PET_FRAMES.get(mood_value, PET_FRAMES["normal"])
    return frames[frame_index % len(frames)].strip()


def furnace_animation_frame(state: FurnaceState | str, frame_index: int) -> str:
    """返回炼丹炉动画帧。"""
    state_value = _enum_value(state)
    frames = FURNACE_FRAMES.get(state_value, FURNACE_FRAMES["unknown"])
    return frames[frame_index % len(frames)].strip()


def render_tui_text(state: DanLingState, path: Path, source: str, frame_index: int = 0) -> str:
    """把 DanLingState 渲染成 Textual Static 可显示的文本。"""
    lines = [
        "DanLing TUI",
        f"path: {path}",
        f"source: {source}",
        f"updated: {time.strftime('%H:%M:%S')}",
        "",
        render_tui_pet(state, frame_index),
        "",
        furnace_animation_frame(state.furnace_state, frame_index),
        "",
        render_tui_metrics(state, source=source, path=str(path)),
        "",
        render_tui_hardware(state),
        "",
        render_tui_events(state, frame_index),
    ]
    return "\n".join(lines)


def render_tui_metrics(
    state: DanLingState,
    *,
    source: str | None = None,
    path: str | None = None,
) -> str:
    """渲染训练指标区域。"""
    metric = state.metric
    lines = [
        "训练概览",
        f"pet: {state.pet_name} {state.realm.name} {state.pet_mood.value}",
        f"furnace: {state.furnace_state.value}",
        f"境界区间: {_fmt_realm_range(state.realm)}",
    ]
    if source:
        lines.insert(1, f"日志模式: {source}")
    if path:
        lines.insert(2 if source else 1, f"日志路径: {path}")
    if metric is None:
        lines.append("metric: unknown")
    else:
        lines.extend(
            [
                f"epoch: {metric.epoch if metric.epoch is not None else '-'}",
                f"train_loss: {_fmt(metric.train_loss)}",
                f"val_loss: {_fmt(metric.val_loss)}",
                f"{metric.score_name or 'score'}: {_fmt(metric.score)}",
                f"mAP50-95: {_fmt(metric.map5095)}",
                f"lr: {_fmt(metric.lr)}",
            ]
        )
    return "\n".join(lines)


def render_tui_pet(state: DanLingState, frame_index: int = 0) -> str:
    """渲染宠物文字图案和当前境界。"""
    return "\n".join(
        [
            pet_animation_frame(state.pet_mood, frame_index),
            "",
            f"境界: {state.realm.name}",
            f"境界进度: {_fmt_realm_progress(state.realm)}",
        ]
    )


def render_tui_hardware(state: DanLingState) -> str:
    """渲染硬件监控区域。"""
    lines = ["硬件监控"]
    if state.hardware:
        for device in state.hardware:
            device_name = "GPU" if device.device_type == "nvidia" else device.device_type.upper()
            lines.append(
                f"- {device_name}:{device.device_id or '-'} "
                f"util={_fmt(device.util_percent)}% mem={_fmt_memory(device)}"
            )
            if device.temperature_c is not None:
                lines.append(f"  temp={_fmt(device.temperature_c)}C")
    else:
        lines.append("hardware: unknown")
    return "\n".join(lines)


def render_tui_events(state: DanLingState, frame_index: int = 0) -> str:
    """渲染事件/诊断区域。"""
    pulse = "✨" if frame_index % 2 == 0 else "·"
    lines = [f"最近事件 {pulse}"]
    if state.events:
        for event in state.events[-5:]:
            lines.append(f"- [{event.severity.value}] {event.title}: {event.message}")
    else:
        lines.append("- 暂无事件")
    return "\n".join(lines)


def _read_tui_state(
    path: Path,
    source: str,
    config_path: Path | None,
    no_hardware: bool,
    state_provider: Callable[[], DanLingState] | None = None,
) -> DanLingState:
    if state_provider is not None:
        return state_provider()
    config = load_config(str(config_path) if config_path is not None else None)
    history = read_history(str(path), source=source)
    hardware = [] if no_hardware else read_all_hardware()
    return build_state(history, hardware, config)


def _read_tui_text(
    path: Path,
    source: str,
    config_path: Path | None,
    no_hardware: bool,
) -> str:
    try:
        state = _read_tui_state(path, source, config_path, no_hardware)
        return render_tui_text(state, path=path, source=source)
    except Exception as exc:
        return f"DanLing TUI\npath: {path}\nsource: {source}\nerror: {exc}"


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _fmt_memory(device) -> str:
    if device.memory_used_mb is None or device.memory_total_mb is None:
        return "-"
    return f"{device.memory_used_mb / 1024:.1f}/{device.memory_total_mb / 1024:.1f}G"


def _fmt_realm_progress(realm) -> str:
    progress = realm.progress
    if progress is None:
        if realm.score_floor is not None:
            return "等待指标"
        return "未配置"
    return f"{max(0, min(100, round(progress * 100)))}%"


def _fmt_realm_range(realm) -> str:
    if realm.score_floor is None:
        return "未配置"
    if realm.score_ceiling is None:
        return f">= {realm.score_floor:.4f}"
    return f"{realm.score_floor:.4f} - {realm.score_ceiling:.4f}"


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)
