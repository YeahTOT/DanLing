"""Ultralytics YOLO results.csv Reader。

读取 Ultralytics 训练输出的 results.csv 文件，将每行转换为 MetricSnapshot。
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from danling.models import MetricSnapshot

from .base import BaseReader

logger = logging.getLogger(__name__)

# 用于识别 train_loss 的候选列名（按优先级）
_TRAIN_LOSS_KEY = "train/loss"
_TRAIN_LOSS_CANDIDATES = [_TRAIN_LOSS_KEY, "loss/train", "loss"]
_TRAIN_BOX_KEY = "train/box_loss"
_TRAIN_CLS_KEY = "train/cls_loss"
_TRAIN_DFL_KEY = "train/dfl_loss"

# 用于识别 val_loss 的候选列名
_VAL_LOSS_KEY = "val/loss"
_VAL_LOSS_CANDIDATES = [_VAL_LOSS_KEY, "loss/val"]
_VAL_BOX_KEY = "val/box_loss"
_VAL_CLS_KEY = "val/cls_loss"
_VAL_DFL_KEY = "val/dfl_loss"

# mAP50 候选名
_MAP50_CANDIDATES = ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"]
# mAP50-95 候选名
_MAP5095_CANDIDATES = ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"]
# precision 候选名
_PRECISION_CANDIDATES = ["metrics/precision(B)", "metrics/precision", "precision"]
# recall 候选名
_RECALL_CANDIDATES = ["metrics/recall(B)", "metrics/recall", "recall"]
# lr 候选名
_LR_CANDIDATES = ["lr/pg0", "lr", "learning_rate"]


def _find_key(header: list[str], candidates: list[str]) -> str | None:
    """在表头中按优先级查找第一个匹配的列名。"""
    for cand in candidates:
        if cand in header:
            return cand
    return None


def _parse_number(val: str) -> float | int:
    """将字符串解析为数字，优先 int，否则 float。"""
    try:
        f = float(val)
    except (ValueError, TypeError):
        return 0.0
    if f == int(f) and "." not in val and "e" not in val.lower():
        return int(f)
    return f


class UltralyticsCSVReader(BaseReader):
    """Ultralytics YOLO results.csv Reader。"""

    @property
    def name(self) -> str:
        return "csv"

    @staticmethod
    def detect(path: str) -> bool:
        p = Path(path)
        if p.is_dir():
            return (p / "results.csv").is_file()
        if p.is_file():
            return p.name == "results.csv" or p.suffix == ".csv"
        return False

    def read_latest(self, path: str) -> MetricSnapshot | None:
        history = self.read_history(path)
        return history[-1] if history else None

    def read_history(self, path: str, limit: int | None = None) -> list[MetricSnapshot]:
        filepath = self._resolve_path(path)
        if filepath is None or not filepath.is_file():
            return []

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []

            raw_header = [h.strip() for h in reader.fieldnames]
            file_mtime = filepath.stat().st_mtime
            result: list[MetricSnapshot] = []

            for row in reader:
                raw_row = {k.strip(): v.strip() for k, v in row.items()}
                metric = self._row_to_metric(raw_header, raw_row, str(filepath), file_mtime)
                if metric is not None:
                    result.append(metric)

            if limit is not None:
                result = result[-limit:]

        return result

    @staticmethod
    def _resolve_path(path: str) -> Path | None:
        p = Path(path)
        if p.is_dir():
            csv_path = p / "results.csv"
            return csv_path if csv_path.is_file() else None
        if p.is_file():
            return p
        return None

    @classmethod
    def _row_to_metric(
        cls,
        header: list[str],
        row: dict[str, str],
        source_path: str,
        file_mtime: float | None = None,
    ) -> MetricSnapshot | None:
        raw: dict[str, float | int | str | None] = {}

        # 按表头顺序解析所有数值字段
        for key in header:
            val = row.get(key, "")
            if val == "" or val is None:
                raw[key] = None
            else:
                try:
                    raw[key] = _parse_number(val)
                except (ValueError, TypeError):
                    raw[key] = val

        # 提取 epoch
        epoch = None
        if "epoch" in header:
            ep_val = raw.get("epoch")
            epoch = int(ep_val) if isinstance(ep_val, (int, float)) else None

        # Ultralytics 的 time 是训练耗时，不是 wall-clock timestamp。
        # 用 results.csv 的 mtime 判断日志是否还在更新。
        timestamp = file_mtime

        # 提取 train_loss
        train_loss = None
        train_loss_key = _find_key(header, _TRAIN_LOSS_CANDIDATES)
        if train_loss_key is not None:
            val = raw.get(train_loss_key)
            train_loss = float(val) if isinstance(val, (int, float)) else None
        else:
            # 求和 train/box_loss + train/cls_loss + train/dfl_loss
            components = []
            for comp_key in [_TRAIN_BOX_KEY, _TRAIN_CLS_KEY, _TRAIN_DFL_KEY]:
                if comp_key in header:
                    val = raw.get(comp_key)
                    if isinstance(val, (int, float)):
                        components.append(float(val))
            if components:
                train_loss = sum(components)

        # 提取 val_loss
        val_loss = None
        val_loss_key = _find_key(header, _VAL_LOSS_CANDIDATES)
        if val_loss_key is not None:
            v = raw.get(val_loss_key)
            val_loss = float(v) if isinstance(v, (int, float)) else None
        else:
            components = []
            for comp_key in [_VAL_BOX_KEY, _VAL_CLS_KEY, _VAL_DFL_KEY]:
                if comp_key in header:
                    v = raw.get(comp_key)
                    if isinstance(v, (int, float)):
                        components.append(float(v))
            if components:
                val_loss = sum(components)

        # 提取 map50
        map50 = None
        map50_key = _find_key(header, _MAP50_CANDIDATES)
        if map50_key is not None:
            v = raw.get(map50_key)
            map50 = float(v) if isinstance(v, (int, float)) else None

        # 提取 map5095
        map5095 = None
        map5095_key = _find_key(header, _MAP5095_CANDIDATES)
        if map5095_key is not None:
            v = raw.get(map5095_key)
            map5095 = float(v) if isinstance(v, (int, float)) else None

        # 提取 precision
        precision = None
        prec_key = _find_key(header, _PRECISION_CANDIDATES)
        if prec_key is not None:
            v = raw.get(prec_key)
            precision = float(v) if isinstance(v, (int, float)) else None

        # 提取 recall
        recall = None
        rec_key = _find_key(header, _RECALL_CANDIDATES)
        if rec_key is not None:
            v = raw.get(rec_key)
            recall = float(v) if isinstance(v, (int, float)) else None

        # 提取 accuracy
        accuracy = None
        if "accuracy" in header:
            v = raw.get("accuracy")
            accuracy = float(v) if isinstance(v, (int, float)) else None

        # 提取 lr
        lr = None
        lr_key = _find_key(header, _LR_CANDIDATES)
        if lr_key is not None:
            v = raw.get(lr_key)
            lr = float(v) if isinstance(v, (int, float)) else None

        # 计算 score
        score = None
        score_name = None
        if map50 is not None:
            score = map50
            score_name = "mAP50"
        elif map5095 is not None:
            score = map5095
            score_name = "mAP50-95"
        elif accuracy is not None:
            score = accuracy
            score_name = "accuracy"

        return MetricSnapshot(
            source="csv",
            run_path=os.path.dirname(source_path),
            epoch=epoch,
            timestamp=timestamp,
            train_loss=train_loss,
            val_loss=val_loss,
            score=score,
            score_name=score_name,
            map50=map50,
            map5095=map5095,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            lr=lr,
            raw=raw,
        )
