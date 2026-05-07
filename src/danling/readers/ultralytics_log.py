"""Ultralytics console log Reader.

Parse the text output produced by Ultralytics training when stdout/stderr is
redirected to a ``.log`` or ``.txt`` file.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import replace
from pathlib import Path

from danling.models import MetricSnapshot

from .base import BaseReader

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAIN_RE = re.compile(
    r"^\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"(?P<gpu_mem>\S+)\s+"
    r"(?P<box_loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s+"
    r"(?P<cls_loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s+"
    r"(?P<dfl_loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s+"
    r"(?P<instances>\d+)\s+(?P<size>\d+)\s*:",
    re.IGNORECASE,
)
_VAL_RE = re.compile(
    r"^\s*all\s+"
    r"(?P<images>\d+)\s+"
    r"(?P<instances>\d+)\s+"
    r"(?P<precision>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s+"
    r"(?P<recall>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s+"
    r"(?P<map50>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s+"
    r"(?P<map5095>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*$",
    re.IGNORECASE,
)
_LOG_SUFFIXES = {".log", ".txt"}
_COMMON_LOG_NAMES = (
    "train.log",
    "training.log",
    "output.log",
    "stdout.log",
    "nohup.out",
)


class UltralyticsLogReader(BaseReader):
    """Reader for Ultralytics official console training output."""

    @property
    def name(self) -> str:
        return "ultralytics-log"

    @staticmethod
    def detect(path: str) -> bool:
        log_path = UltralyticsLogReader._resolve_path(path)
        if log_path is None:
            return False
        return _looks_like_ultralytics_log(log_path)

    def read_latest(self, path: str) -> MetricSnapshot | None:
        history = self.read_history(path)
        return history[-1] if history else None

    def read_history(self, path: str, limit: int | None = None) -> list[MetricSnapshot]:
        log_path = self._resolve_path(path)
        if log_path is None or not log_path.is_file():
            return []

        file_mtime = log_path.stat().st_mtime
        history: list[MetricSnapshot] = []
        current: MetricSnapshot | None = None
        for line in _iter_clean_lines(log_path):
            train_match = _TRAIN_RE.match(line)
            if train_match:
                current = _train_match_to_snapshot(
                    train_match,
                    source_path=str(log_path),
                    file_mtime=file_mtime,
                )
                history.append(current)
                continue

            val_match = _VAL_RE.match(line)
            if val_match and current is not None:
                current = _with_validation_metrics(current, val_match)
                history[-1] = current

        if limit is not None:
            history = history[-limit:]
        return history

    @staticmethod
    def _resolve_path(path: str) -> Path | None:
        p = Path(path).expanduser()
        if p.is_file() and _is_log_file(p):
            return p
        if not p.is_dir():
            return None

        for name in _COMMON_LOG_NAMES:
            candidate = p / name
            if candidate.is_file():
                return candidate

        candidates = [candidate for candidate in p.iterdir() if candidate.is_file()]
        log_candidates = [candidate for candidate in candidates if _is_log_file(candidate)]
        if not log_candidates:
            return None
        return max(log_candidates, key=lambda candidate: candidate.stat().st_mtime)


def _train_match_to_snapshot(
    match: re.Match[str], source_path: str, file_mtime: float
) -> MetricSnapshot:
    box_loss = _float(match.group("box_loss"))
    cls_loss = _float(match.group("cls_loss"))
    dfl_loss = _float(match.group("dfl_loss"))
    raw: dict[str, float | int | str | None] = {
        "epoch": int(match.group("epoch")),
        "total_epochs": int(match.group("total")),
        "GPU_mem": match.group("gpu_mem"),
        "train/box_loss": box_loss,
        "train/cls_loss": cls_loss,
        "train/dfl_loss": dfl_loss,
        "instances": int(match.group("instances")),
        "imgsz": int(match.group("size")),
    }
    return MetricSnapshot(
        source="ultralytics-log",
        run_path=os.path.dirname(source_path),
        source_path=source_path,
        epoch=int(match.group("epoch")),
        timestamp=file_mtime,
        train_loss=box_loss + cls_loss + dfl_loss,
        raw=raw,
    )


def _with_validation_metrics(
    snapshot: MetricSnapshot, match: re.Match[str]
) -> MetricSnapshot:
    precision = _float(match.group("precision"))
    recall = _float(match.group("recall"))
    map50 = _float(match.group("map50"))
    map5095 = _float(match.group("map5095"))
    raw = dict(snapshot.raw)
    raw.update(
        {
            "val/images": int(match.group("images")),
            "val/instances": int(match.group("instances")),
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map5095,
        }
    )
    return replace(
        snapshot,
        score=map50,
        score_name="mAP50",
        map50=map50,
        map5095=map5095,
        precision=precision,
        recall=recall,
        raw=raw,
    )


def _looks_like_ultralytics_log(path: Path) -> bool:
    saw_epoch_header = False
    saw_train_line = False
    try:
        for index, line in enumerate(_iter_clean_lines(path)):
            saw_epoch_header = saw_epoch_header or "Epoch" in line and "GPU_mem" in line
            saw_train_line = saw_train_line or _TRAIN_RE.match(line) is not None
            if saw_train_line and (saw_epoch_header or "mAP50" in line):
                return True
            if index > 200:
                break
    except OSError:
        return False
    return saw_train_line


def _iter_clean_lines(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in re.split(r"[\r\n]+", text):
        cleaned = _ANSI_RE.sub("", line).strip()
        if cleaned:
            yield cleaned


def _is_log_file(path: Path) -> bool:
    return path.name == "nohup.out" or path.suffix.lower() in _LOG_SUFFIXES


def _float(value: str) -> float:
    number = float(value)
    return number if math.isfinite(number) else 0.0
