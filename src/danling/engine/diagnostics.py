"""训练异常诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from danling.config import DanLingConfig
from danling.engine.trends import analyze_trends
from danling.models import HardwareSnapshot, MetricSnapshot, Severity, _dataclass_to_dict


@dataclass
class Diagnostic:
    severity: Severity
    code: str
    title: str
    message: str
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def diagnose(
    history: list[MetricSnapshot],
    hardware: list[HardwareSnapshot],
    config: DanLingConfig,
    now: float | None = None,
) -> list[Diagnostic]:
    """识别常见训练异常并给出建议。"""
    if not history:
        return []

    diagnostics: list[Diagnostic] = []
    trend = analyze_trends(history, config.loss_window, now=now)
    latest = history[-1]

    if trend.has_nan_or_inf:
        diagnostics.append(
            Diagnostic(
                Severity.critical,
                "NAN_LOSS",
                "指标出现 NaN/inf",
                "训练指标中出现非有限数值。",
                "检查学习率、数据、混合精度设置和梯度爆炸风险。",
            )
        )

    if _is_loss_explosion(trend.current_loss, trend.previous_loss):
        diagnostics.append(
            Diagnostic(
                Severity.error,
                "LOSS_EXPLOSION",
                "loss 快速上升",
                "当前 train_loss 已超过上一个有效 loss 的 2 倍。",
                "降低学习率，检查数据增强和 label 是否异常。",
            )
        )

    if trend.no_improve_count >= config.no_improve_patience and trend.current_score is not None:
        diagnostics.append(
            Diagnostic(
                Severity.warning,
                "NO_IMPROVEMENT",
                "指标长时间无提升",
                f"score 已连续 {trend.no_improve_count} 个快照未创新高。",
                "检查学习率调度、数据质量和过拟合情况。",
            )
        )

    if trend.stale_seconds is not None and trend.stale_seconds > config.stale_seconds:
        diagnostics.append(
            Diagnostic(
                Severity.warning,
                "STALE_LOG",
                "训练日志长时间未更新",
                f"最新日志距今 {trend.stale_seconds:.0f} 秒。",
                "检查训练进程是否仍在运行。",
            )
        )

    for device in hardware:
        ratio = device.memory_ratio
        if ratio is not None and ratio >= 0.98:
            diagnostics.append(
                Diagnostic(
                    Severity.critical,
                    "MEMORY_CRITICAL",
                    "显存接近耗尽",
                    f"{device.name or device.device_type} 显存使用率 {ratio:.0%}。",
                    "立即检查 OOM 风险，降低 batch size 或 imgsz。",
                )
            )
        elif ratio is not None and ratio >= config.high_memory_ratio:
            diagnostics.append(
                Diagnostic(
                    Severity.warning,
                    "HIGH_MEMORY",
                    "显存使用偏高",
                    f"{device.name or device.device_type} 显存使用率 {ratio:.0%}。",
                    "降低 batch size、imgsz、num_workers，或开启梯度累积。",
                )
            )

        if (
            _is_log_active(trend, config)
            and device.util_percent is not None
            and device.util_percent < 20
        ):
            diagnostics.append(
                Diagnostic(
                    Severity.warning,
                    "LOW_UTILIZATION",
                    "硬件利用率偏低",
                    f"{device.name or device.device_type} 利用率 {device.util_percent:.0f}%。",
                    "检查 dataloader、CPU 瓶颈、数据路径和 batch size。",
                )
            )

    if _checkpoint_missing(latest):
        diagnostics.append(
            Diagnostic(
                Severity.warning,
                "CHECKPOINT_MISSING",
                "未发现最新 checkpoint",
                "训练目录中缺少 weights/last.pt。",
                "检查保存配置或训练目录是否正确。",
            )
        )

    return diagnostics


def _is_loss_explosion(current: float | None, previous: float | None) -> bool:
    return current is not None and previous is not None and current > previous * 2.0


def _is_log_active(trend, config: DanLingConfig) -> bool:
    return trend.stale_seconds is None or trend.stale_seconds <= config.stale_seconds


def _checkpoint_missing(snapshot: MetricSnapshot) -> bool:
    if snapshot.run_path is None or snapshot.epoch is None or snapshot.epoch <= 1:
        return False
    run_path = Path(snapshot.run_path)
    if not run_path.exists():
        return False
    return not (run_path / "weights" / "last.pt").is_file()
