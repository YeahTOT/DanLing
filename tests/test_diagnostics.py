"""测试异常诊断系统。"""

from __future__ import annotations

import math

from danling.config import DanLingConfig
from danling.engine.diagnostics import diagnose
from danling.models import HardwareSnapshot, MetricSnapshot, Severity


def metric(
    train_loss: float | None = None,
    score: float | None = None,
    timestamp: float | None = None,
    run_path: str | None = None,
    epoch: int | None = None,
) -> MetricSnapshot:
    return MetricSnapshot(
        source="test",
        run_path=run_path,
        epoch=epoch,
        timestamp=timestamp,
        train_loss=train_loss,
        score=score,
        score_name="mAP50" if score is not None else None,
    )


def codes(
    history: list[MetricSnapshot], hardware: list[HardwareSnapshot] | None = None
) -> set[str]:
    return {diag.code for diag in diagnose(history, hardware or [], DanLingConfig(), now=200)}


def test_nan_loss_diagnostic() -> None:
    diagnostics = diagnose([metric(train_loss=math.nan)], [], DanLingConfig())

    assert diagnostics[0].code == "NAN_LOSS"
    assert diagnostics[0].severity == Severity.critical


def test_loss_explosion_diagnostic() -> None:
    assert "LOSS_EXPLOSION" in codes([metric(train_loss=1.0), metric(train_loss=3.0)])


def test_no_improvement_diagnostic() -> None:
    diagnostics = diagnose(
        [metric(score=0.8), metric(score=0.7), metric(score=0.6)],
        [],
        DanLingConfig(no_improve_patience=2),
    )

    assert "NO_IMPROVEMENT" in {diag.code for diag in diagnostics}


def test_stale_log_diagnostic() -> None:
    assert "STALE_LOG" in codes([metric(train_loss=1.0, timestamp=-4000)])


def test_memory_and_low_utilization_diagnostics() -> None:
    diagnostics = diagnose(
        [metric(train_loss=1.0, timestamp=199)],
        [HardwareSnapshot(util_percent=10, memory_used_mb=99, memory_total_mb=100)],
        DanLingConfig(high_memory_ratio=0.9),
        now=200,
    )
    found = {diag.code for diag in diagnostics}

    assert "MEMORY_CRITICAL" in found
    assert "LOW_UTILIZATION" in found


def test_checkpoint_missing_diagnostic(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()

    found = codes([metric(run_path=str(run), epoch=2)])

    assert "CHECKPOINT_MISSING" in found
