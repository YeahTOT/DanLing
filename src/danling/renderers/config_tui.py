"""配置 TUI 及其可测试的表单逻辑。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from danling.config import (
    REALM_THRESHOLD_NAMES,
    DanLingConfig,
    config_to_yaml_text,
    load_config,
)
from danling.models import MetricSnapshot
from danling.readers.registry import read_history

REALM_SEGMENT_COUNT = len(REALM_THRESHOLD_NAMES)
TEXT_FIELD_IDS = ["pet_name", "primary_score", "primary_loss"]
FLOAT_FIELD_IDS = [
    "baseline_score",
    "sota_score",
    "watch_interval",
    "high_memory_ratio",
    "hot_util_percent",
]
INT_FIELD_IDS = ["loss_window", "no_improve_patience", "stale_seconds"]
CONFIG_FIELD_IDS = TEXT_FIELD_IDS + FLOAT_FIELD_IDS + INT_FIELD_IDS
REALM_FIELD_PREFIX = "realm_threshold_"
AUTO_RESULTS_PATH_ID = "auto-results-path"
AUTO_METRIC_ID = "auto-metric"
AUTO_SOURCE_MENU_ID = "auto-source-menu"
AUTO_FORM_ID = "auto-form"
AUTO_PREVIEW_WRAP_ID = "auto-preview-wrap"
AUTO_PATH_LABEL_ID = "auto-path-label"
AUTO_HELP_ID = "auto-help"
AUTO_SOURCE_LABELS = {
    "csv": "Ultralytics results.csv",
    "tensorboard": "TensorBoard event/logdir",
}


def realm_field_id(name: str) -> str:
    """返回境界阈值表单字段 ID。"""
    return f"{REALM_FIELD_PREFIX}{name}"


CONFIG_FORM_ROWS = [
    ("pet_name", "丹灵名称"),
    ("primary_score", "主指标（如 mAP50）"),
    ("baseline_score", "Baseline 精度（如 0.60）"),
    ("sota_score", "SOTA 精度（如 0.80）"),
    (realm_field_id("炼器期"), "炼器期阈值（可留空）"),
    (realm_field_id("筑基期"), "筑基期阈值（可留空）"),
    (realm_field_id("结丹期"), "结丹期阈值（可留空）"),
    (realm_field_id("元婴期"), "元婴期阈值（可留空）"),
    (realm_field_id("化神期"), "化神期阈值（可留空）"),
    ("watch_interval", "刷新间隔秒"),
    ("loss_window", "loss 窗口"),
    ("no_improve_patience", "无提升耐心"),
    ("stale_seconds", "日志休眠秒数"),
]


def config_to_form_values(config: DanLingConfig) -> dict[str, str]:
    """把配置展开为 Textual Input 使用的字符串字段。"""
    values = {
        field_id: _format_form_value(getattr(config, field_id))
        for field_id in CONFIG_FIELD_IDS
    }
    thresholds = _realm_threshold_mapping(config.realm_thresholds)
    for name in REALM_THRESHOLD_NAMES:
        values[realm_field_id(name)] = _format_form_value(thresholds.get(name))
    return values


def config_from_form_values(
    values: Mapping[str, str], base: DanLingConfig | None = None
) -> DanLingConfig:
    """把表单字符串转换为配置。"""
    base = base or DanLingConfig()
    merged = config_to_form_values(base)
    merged.update(values)

    thresholds = _thresholds_from_form_values(merged)
    return replace(
        base,
        pet_name=_text_value(merged["pet_name"], base.pet_name) or "DanLing",
        primary_score=_optional_text_value(merged["primary_score"]),
        primary_loss=_optional_text_value(merged["primary_loss"]),
        loss_window=_int_value("loss_window", merged["loss_window"], base.loss_window),
        no_improve_patience=_int_value(
            "no_improve_patience",
            merged["no_improve_patience"],
            base.no_improve_patience,
        ),
        stale_seconds=_int_value("stale_seconds", merged["stale_seconds"], base.stale_seconds),
        high_memory_ratio=_float_value(
            "high_memory_ratio",
            merged["high_memory_ratio"],
            base.high_memory_ratio,
        ),
        hot_util_percent=_float_value(
            "hot_util_percent",
            merged["hot_util_percent"],
            base.hot_util_percent,
        ),
        watch_interval=_float_value(
            "watch_interval",
            merged["watch_interval"],
            base.watch_interval,
        ),
        baseline_score=_optional_float_value("baseline_score", merged["baseline_score"]),
        sota_score=_optional_float_value("sota_score", merged["sota_score"]),
        realm_thresholds=thresholds,
    )


def validate_config_form(values: Mapping[str, str]) -> list[str]:
    """校验配置表单，返回中文错误列表。"""
    errors: list[str] = []

    for field in FLOAT_FIELD_IDS:
        if _has_value(values, field) and _parse_float(values[field]) is None:
            errors.append(f"{field} 必须是数字")

    for field in INT_FIELD_IDS:
        if _has_value(values, field) and _parse_int(values[field]) is None:
            errors.append(f"{field} 必须是整数")

    baseline = _parse_float(values.get("baseline_score", ""))
    sota = _parse_float(values.get("sota_score", ""))
    has_baseline = _has_value(values, "baseline_score")
    has_sota = _has_value(values, "sota_score")
    if has_baseline != has_sota:
        errors.append("baseline_score 和 sota_score 需要同时填写")
    elif baseline is not None and sota is not None and baseline >= sota:
        errors.append("baseline_score 必须小于 sota_score")

    threshold_values = [
        values.get(realm_field_id(name), "").strip()
        for name in REALM_THRESHOLD_NAMES
    ]
    filled_thresholds = [value for value in threshold_values if value]
    if 0 < len(filled_thresholds) < REALM_SEGMENT_COUNT:
        errors.append("境界阈值需要全部留空，或五个全部填写")
    elif len(filled_thresholds) == REALM_SEGMENT_COUNT:
        parsed_thresholds = [_parse_float(value) for value in threshold_values]
        for name, parsed_value in zip(
            REALM_THRESHOLD_NAMES,
            parsed_thresholds,
            strict=True,
        ):
            if parsed_value is None:
                errors.append(f"{realm_field_id(name)} 必须是数字")
        if all(value is not None for value in parsed_thresholds):
            numeric_thresholds = [value for value in parsed_thresholds if value is not None]
            threshold_pairs = zip(numeric_thresholds, numeric_thresholds[1:], strict=False)
            if any(current >= next_value for current, next_value in threshold_pairs):
                errors.append("境界阈值必须严格递增")

    return errors


def render_realm_preview(values: Mapping[str, str]) -> str:
    """渲染配置 TUI 右侧境界分段预览。"""
    errors = validate_config_form(values)
    if errors:
        return "\n".join(["境界预览", "输入错误:", *[f"- {error}" for error in errors]])

    manual_thresholds = _manual_threshold_preview(values)
    if manual_thresholds is not None:
        return _render_threshold_ranges(manual_thresholds)

    baseline = _parse_float(values.get("baseline_score", ""))
    sota = _parse_float(values.get("sota_score", ""))
    if baseline is None or sota is None:
        return "\n".join(
            [
                "境界预览",
                "境界未启用",
                "填写 baseline_score 和 sota_score 后自动等分。",
            ]
        )

    step = (sota - baseline) / (REALM_SEGMENT_COUNT - 1)
    thresholds = [
        (name, baseline + step * index)
        for index, name in enumerate(REALM_THRESHOLD_NAMES)
    ]
    return _render_threshold_ranges(thresholds)


def suggest_config_from_results(
    path: str | Path, metric_name: str, source: str = "csv"
) -> dict[str, str]:
    """从历史训练日志生成主指标、baseline 和 SOTA 建议。"""
    result_path = str(path).strip()
    if not result_path:
        raise ValueError("历史路径不能为空")
    metric = metric_name.strip()
    if not metric:
        raise ValueError("关心指标不能为空")

    normalized_source = "csv" if source == "ultralytics" else source
    if normalized_source not in {"csv", "tensorboard", "auto"}:
        raise ValueError(f"不支持的数据源: {source}")

    try:
        history = read_history(result_path, source=normalized_source)
    except RuntimeError:
        raise
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not history:
        source_label = AUTO_SOURCE_LABELS.get(normalized_source, normalized_source)
        raise ValueError(f"未读取到 {source_label} 历史结果")

    values = [
        value
        for snapshot in history
        if (value := _metric_value(snapshot, metric)) is not None
    ]
    if not values:
        raise ValueError(f"未找到指标: {metric}")

    baseline_index = max(0, math.ceil(len(values) * 0.2) - 1)
    baseline = values[baseline_index]
    sota = max(values)
    if baseline >= sota:
        raise ValueError("无法生成 baseline/SOTA: 20%结果必须小于最佳结果")

    return {
        "primary_score": metric,
        "baseline_score": str(baseline),
        "sota_score": str(sota),
    }


def save_config_form(path: Path, values: Mapping[str, str]) -> DanLingConfig:
    """把表单值保存为配置文件。"""
    errors = validate_config_form(values)
    if errors:
        raise ValueError("; ".join(errors))
    base = load_config(str(path)) if path.is_file() else DanLingConfig()
    config = config_from_form_values(values, base=base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_to_yaml_text(config), encoding="utf-8")
    return config


def textual_available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def require_textual() -> None:
    if not textual_available():
        raise RuntimeError("Textual config UI requires: pip install danling[tui]")


def create_config_app(config_path: Path | None = None):
    """创建配置编辑 Textual App。"""
    require_textual()

    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, Input, Label, Static

    target_path = config_path or Path.cwd() / "danling.yaml"
    initial_config = load_config(str(target_path)) if target_path.is_file() else DanLingConfig()
    initial_values = config_to_form_values(initial_config)
    widget_ids = {
        field_id: f"config-field-{index}"
        for index, (field_id, _label) in enumerate(CONFIG_FORM_ROWS)
    }

    class DanLingConfigApp(App):
        BINDINGS = [
            ("1", "manual", "手动配置"),
            ("2", "auto", "自动生成"),
            ("escape", "home", "主页"),
            ("ctrl+g", "generate", "生成"),
            ("ctrl+s", "save", "保存"),
            ("q", "quit", "退出"),
        ]
        CSS = """
        Screen {
            background: #061016;
            color: #d6fff8;
        }

        #config-home, #config-manual, #config-auto {
            height: 1fr;
        }

        #config-home {
            padding: 2 4;
        }

        #config-home-panel {
            border: round #16d9c5;
            padding: 2 3;
            background: #0b1720;
            height: auto;
        }

        #auto-source-menu {
            padding: 2 4;
            height: 1fr;
        }

        #auto-source-panel {
            border: round #16d9c5;
            padding: 2 3;
            background: #0b1720;
            height: auto;
        }

        #auto-form {
            height: 1fr;
        }

        #config-left, #auto-left {
            width: 54%;
            padding: 1 2;
        }

        #config-right, #auto-right {
            width: 46%;
            padding: 1 2;
        }

        #config-preview, #auto-preview {
            border: round #16d9c5;
            padding: 1 2;
            background: #0b1720;
            height: 1fr;
        }

        Input {
            margin-bottom: 1;
        }

        #config-message {
            padding: 0 2 1 2;
            color: #16d9c5;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self.mode = "home"
            self.auto_source = "csv"

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(
                f"config: {target_path} | 1 手动配置 | 2 自动生成 | q 退出",
                id="config-message",
            )
            with Vertical(id="config-home"):
                yield Static(
                    "\n".join(
                        [
                            "DanLing 配置主页",
                            "",
                            "1. 手动配置",
                            "   编辑 pet_name、主指标、baseline/SOTA 和境界阈值。",
                            "",
                            "2. 自动生成",
                            "   先选择 Ultralytics results.csv 或 TensorBoard，",
                            "   再输入历史路径和关心指标，",
                            "   自动使用 20% 位置结果作为 baseline，历史最佳作为 SOTA。",
                            "",
                            "按 1 或 2 进入子菜单；按 Esc 回到主页。",
                        ]
                    ),
                    id="config-home-panel",
                )
            with Horizontal(id="config-manual"):
                with VerticalScroll(id="config-left"):
                    for field_id, label in CONFIG_FORM_ROWS:
                        yield Label(label)
                        yield Input(
                            value=initial_values.get(field_id, ""),
                            id=widget_ids[field_id],
                        )
                with Vertical(id="config-right"):
                    yield Static(render_realm_preview(initial_values), id="config-preview")
            with Vertical(id="config-auto"):
                with Vertical(id=AUTO_SOURCE_MENU_ID):
                    yield Static(
                        "\n".join(
                            [
                                "自动生成数据源",
                                "",
                                "1. Ultralytics results.csv",
                                "   读取训练目录或 results.csv 文件。",
                                "",
                                "2. TensorBoard event/logdir",
                                "   读取 events.out.tfevents.* 文件或日志目录。",
                                "",
                                "按 1 或 2 选择数据源；按 Esc 回到主页。",
                            ]
                        ),
                        id="auto-source-panel",
                    )
                with Horizontal(id=AUTO_FORM_ID):
                    with VerticalScroll(id="auto-left"):
                        yield Label("历史路径", id=AUTO_PATH_LABEL_ID)
                        yield Input(
                            value="",
                            placeholder="例如 logs/run/results.csv",
                            id=AUTO_RESULTS_PATH_ID,
                        )
                        yield Label("关心指标")
                        yield Input(
                            value=initial_values.get("primary_score", "") or "mAP50",
                            placeholder="例如 mAP50 / mAP50-95 / precision",
                            id=AUTO_METRIC_ID,
                        )
                        yield Static("按 Ctrl+G 自动生成，按 Ctrl+S 保存。", id=AUTO_HELP_ID)
                    with Vertical(id=AUTO_PREVIEW_WRAP_ID):
                        yield Static(
                            "自动生成预览\n\n选择数据源后输入历史路径和关心指标，按 Ctrl+G。",
                            id="auto-preview",
                        )
            yield Footer()

        def on_mount(self) -> None:
            self._show_screen("home")

        def on_input_changed(self, _event) -> None:
            if self.mode == "manual":
                self._update_preview()

        def action_save(self) -> None:
            values = self._form_values()
            message = self.query_one("#config-message", Static)
            try:
                save_config_form(target_path, values)
            except ValueError as exc:
                message.update(f"输入错误: {exc}")
                self._update_preview()
            else:
                message.update(f"已保存: {target_path}")
                self._update_preview()

        def action_home(self) -> None:
            self._show_screen("home")

        def action_manual(self) -> None:
            if self.mode == "auto-menu":
                self._select_auto_source("csv")
                return
            self._show_screen("manual")

        def action_auto(self) -> None:
            if self.mode == "auto-menu":
                self._select_auto_source("tensorboard")
                return
            self._show_screen("auto-menu")

        def action_generate(self) -> None:
            if self.mode != "auto-form":
                return
            message = self.query_one("#config-message", Static)
            auto_preview = self.query_one("#auto-preview", Static)
            path_value = self.query_one(f"#{AUTO_RESULTS_PATH_ID}", Input).value.strip()
            metric_value = self.query_one(f"#{AUTO_METRIC_ID}", Input).value.strip()
            try:
                suggested = suggest_config_from_results(
                    path_value,
                    metric_value,
                    source=self.auto_source,
                )
            except (RuntimeError, ValueError) as exc:
                message.update(f"自动生成失败: {exc}")
                auto_preview.update(f"自动生成预览\n\n输入错误: {exc}")
                return

            for field_id, value in suggested.items():
                self.query_one(f"#{widget_ids[field_id]}", Input).value = value
            for name in REALM_THRESHOLD_NAMES:
                self.query_one(f"#{widget_ids[realm_field_id(name)]}", Input).value = ""

            preview = render_realm_preview(self._form_values())
            message.update("已生成 baseline/SOTA，可按 Ctrl+S 保存，或按 1 手动微调")
            auto_preview.update(
                "\n".join(
                    [
                        "自动生成预览",
                        f"数据源: {AUTO_SOURCE_LABELS[self.auto_source]}",
                        f"历史路径: {path_value}",
                        f"关心指标: {suggested['primary_score']}",
                        f"baseline_score: {suggested['baseline_score']}",
                        f"sota_score: {suggested['sota_score']}",
                        "",
                        preview,
                    ]
                )
            )

        def _select_auto_source(self, source: str) -> None:
            self.auto_source = source
            source_label = AUTO_SOURCE_LABELS[source]
            path_label = self.query_one(f"#{AUTO_PATH_LABEL_ID}", Label)
            path_input = self.query_one(f"#{AUTO_RESULTS_PATH_ID}", Input)
            help_text = self.query_one(f"#{AUTO_HELP_ID}", Static)
            preview = self.query_one("#auto-preview", Static)

            if source == "tensorboard":
                path_label.update("TensorBoard event/logdir 路径")
                path_input.placeholder = "例如 logs/tensorboard 或 logs/tensorboard/train"
                help_text.update("按 Ctrl+G 自动生成；需要安装 danling[tensorboard]。")
            else:
                path_label.update("Ultralytics results.csv 路径")
                path_input.placeholder = "例如 logs/run/results.csv 或 runs/detect/train"
                help_text.update("按 Ctrl+G 自动生成，按 Ctrl+S 保存。")

            preview.update(
                "\n".join(
                    [
                        "自动生成预览",
                        f"数据源: {source_label}",
                        "",
                        "输入历史路径和关心指标后，按 Ctrl+G。",
                    ]
                )
            )
            self._show_screen("auto-form")

        def _form_values(self) -> dict[str, str]:
            return {
                field_id: self.query_one(f"#{widget_ids[field_id]}", Input).value
                for field_id, _label in CONFIG_FORM_ROWS
            }

        def _update_preview(self) -> None:
            preview = self.query_one("#config-preview", Static)
            preview.update(render_realm_preview(self._form_values()))

        def _show_screen(self, mode: str) -> None:
            self.mode = mode
            self.query_one("#config-home").display = mode == "home"
            self.query_one("#config-manual").display = mode == "manual"
            self.query_one("#config-auto").display = mode in {"auto-menu", "auto-form"}
            self.query_one(f"#{AUTO_SOURCE_MENU_ID}").display = mode == "auto-menu"
            self.query_one(f"#{AUTO_FORM_ID}").display = mode == "auto-form"
            message = self.query_one("#config-message", Static)
            if mode == "home":
                message.update(f"config: {target_path} | 1 手动配置 | 2 自动生成 | q 退出")
            elif mode == "manual":
                message.update(
                    f"config: {target_path} | Ctrl+S 保存 | Esc 主页 | q 退出 | 手动配置"
                )
                self._update_preview()
            elif mode == "auto-menu":
                message.update(
                    f"config: {target_path} | 1 Ultralytics results.csv | "
                    "2 TensorBoard | Esc 主页"
                )
            else:
                message.update(
                    f"config: {target_path} | {AUTO_SOURCE_LABELS[self.auto_source]} | "
                    "Ctrl+G 生成 | Ctrl+S 保存 | Esc 主页"
                )

    return DanLingConfigApp()


def _realm_threshold_mapping(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return {
            name: float(threshold)
            for name, threshold in value.items()
            if isinstance(threshold, (int, float))
        }
    if isinstance(value, list):
        return {
            name: float(threshold)
            for name, threshold in zip(REALM_THRESHOLD_NAMES, value, strict=False)
            if isinstance(threshold, (int, float))
        }
    return {}


def _thresholds_from_form_values(values: Mapping[str, str]) -> dict[str, float] | None:
    thresholds: dict[str, float] = {}
    for name in REALM_THRESHOLD_NAMES:
        value = values.get(realm_field_id(name), "").strip()
        if not value:
            continue
        thresholds[name] = _optional_float_value(realm_field_id(name), value) or 0.0
    return thresholds if len(thresholds) == len(REALM_THRESHOLD_NAMES) else None


def _manual_threshold_preview(values: Mapping[str, str]) -> list[tuple[str, float]] | None:
    threshold_values = [
        values.get(realm_field_id(name), "").strip()
        for name in REALM_THRESHOLD_NAMES
    ]
    if not all(threshold_values):
        return None
    return [
        (name, _parse_float(value) or 0.0)
        for name, value in zip(REALM_THRESHOLD_NAMES, threshold_values, strict=True)
    ]


def _render_threshold_ranges(thresholds: list[tuple[str, float]]) -> str:
    lines = ["境界预览"]
    for index, (name, floor) in enumerate(thresholds):
        if index + 1 < len(thresholds):
            lines.append(f"{name}: {floor:.4f} - {thresholds[index + 1][1]:.4f}")
        else:
            lines.append(f"{name}: >= {floor:.4f}")
    return "\n".join(lines)


def _metric_value(snapshot: MetricSnapshot, metric_name: str) -> float | None:
    raw_value = snapshot.raw.get(metric_name)
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        return value if math.isfinite(value) else None

    attr_name = {
        "mAP50": "map50",
        "mAP50-95": "map5095",
        "map50": "map50",
        "map5095": "map5095",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
        "score": "score",
    }.get(metric_name, metric_name)
    value = getattr(snapshot, attr_name, None)
    if isinstance(value, (int, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return None


def _has_value(values: Mapping[str, str], field: str) -> bool:
    return bool(values.get(field, "").strip())


def _parse_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _format_form_value(value: object) -> str:
    return "" if value is None else str(value)


def _text_value(value: str, default: str) -> str:
    stripped = value.strip()
    return stripped if stripped else default


def _optional_text_value(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _int_value(field: str, value: str, default: int) -> int:
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return int(stripped)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _float_value(field: str, value: str, default: float) -> float:
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number") from exc


def _optional_float_value(field: str, value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number") from exc
