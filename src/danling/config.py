"""配置系统模块。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

REALM_THRESHOLD_NAMES = ["炼器期", "筑基期", "结丹期", "元婴期", "化神期"]


@dataclass
class DanLingConfig:
    """DanLing 全局配置。"""

    pet_name: str = "DanLing"
    primary_score: str | None = None
    primary_loss: str | None = None
    loss_window: int = 5
    no_improve_patience: int = 10
    stale_seconds: int = 3600
    high_memory_ratio: float = 0.90
    hot_util_percent: float = 90.0
    watch_interval: float = 2.0
    baseline_score: float | None = None
    sota_score: float | None = None
    realm_thresholds: dict[str, float] | list[float] | None = None
    source: str = "auto"
    data_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | None = None) -> DanLingConfig:
    """加载 `danling.yaml` 配置；缺失时返回默认配置。"""
    config_path = _resolve_config_path(path)
    if config_path is None:
        return DanLingConfig()

    data = _load_yaml_like(config_path)
    if not isinstance(data, dict):
        return DanLingConfig()

    allowed = {field.name for field in fields(DanLingConfig)}
    values = {key: value for key, value in data.items() if key in allowed}
    return DanLingConfig(**values)


def default_config_text() -> str:
    """返回示例配置文件内容。"""
    return "\n".join(
        [
            "pet_name: DanLing",
            "primary_score:",
            "primary_loss:",
            "loss_window: 5",
            "no_improve_patience: 10",
            "stale_seconds: 3600",
            "high_memory_ratio: 0.90",
            "hot_util_percent: 90.0",
            "watch_interval: 2.0",
            "baseline_score:",
            "sota_score:",
            "realm_thresholds:",
            "#   炼器期: 0.50",
            "#   筑基期: 0.60",
            "#   结丹期: 0.70",
            "#   元婴期: 0.80",
            "#   化神期: 0.90",
            "source: auto",
            "data_path:",
            "",
        ]
    )


def config_to_yaml_text(config: DanLingConfig) -> str:
    """把配置序列化为稳定的轻量 YAML 文本。"""
    data = config.to_dict()
    lines: list[str] = []
    for field in fields(DanLingConfig):
        value = data[field.name]
        if field.name == "realm_thresholds":
            lines.extend(_realm_threshold_lines(value))
        else:
            lines.append(f"{field.name}: {_format_config_scalar(value)}")
    lines.append("")
    return "\n".join(lines)


def _realm_threshold_lines(value: object) -> list[str]:
    if not value:
        return ["realm_thresholds:"]

    lines = ["realm_thresholds:"]
    if isinstance(value, dict):
        for name in REALM_THRESHOLD_NAMES:
            if name in value:
                lines.append(f"  {name}: {_format_config_scalar(value[name])}")
        return lines

    if isinstance(value, list):
        for name, threshold in zip(REALM_THRESHOLD_NAMES, value, strict=False):
            lines.append(f"  {name}: {_format_config_scalar(threshold)}")
        return lines

    return ["realm_thresholds:"]


def _format_config_scalar(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _resolve_config_path(path: str | None) -> Path | None:
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return p

    cwd_config = Path.cwd() / "danling.yaml"
    return cwd_config if cwd_config.is_file() else None


def _load_yaml_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return _parse_simple_yaml(text)

    loaded = yaml.safe_load(text)
    return loaded or {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """解析简单 key: value 配置，作为 PyYAML 缺失时的轻量降级。"""
    result: dict[str, Any] = {}
    current_mapping_key: str | None = None
    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip() or ":" not in line_without_comment:
            continue
        stripped = line_without_comment.strip()
        is_nested = line_without_comment[:1].isspace()
        if is_nested and current_mapping_key is not None:
            key, value = stripped.split(":", 1)
            mapping = result.get(current_mapping_key)
            if not isinstance(mapping, dict):
                mapping = {}
                result[current_mapping_key] = mapping
            mapping[key.strip()] = _parse_scalar(value.strip())
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        result[key] = _parse_scalar(value.strip())
        current_mapping_key = key if value.strip() == "" else None
    return result


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
