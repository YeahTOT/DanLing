"""测试 UltralyticsCSVReader 和 inspect CLI 命令。"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from danling.cli import app
from danling.readers.ultralytics_csv import UltralyticsCSVReader
from danling.readers.ultralytics_log import UltralyticsLogReader

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ultralytics_run"
runner = CliRunner()


class TestUltralyticsCSVReader:
    """Reader 单元测试。"""

    def test_detect_directory(self) -> None:
        """检测目录下存在 results.csv 时应返回 True。"""
        reader = UltralyticsCSVReader()
        assert reader.detect(str(FIXTURE_DIR)) is True

    def test_detect_nonexistent(self) -> None:
        """检测不存在的路径应返回 False。"""
        reader = UltralyticsCSVReader()
        assert reader.detect("/nonexistent/path") is False

    def test_read_history_count(self) -> None:
        """读取历史应返回正确行数（5 行数据）。"""
        reader = UltralyticsCSVReader()
        history = reader.read_history(str(FIXTURE_DIR))
        assert len(history) == 5

    def test_read_history_limit(self) -> None:
        """limit 参数应限制返回数量。"""
        reader = UltralyticsCSVReader()
        history = reader.read_history(str(FIXTURE_DIR), limit=2)
        assert len(history) == 2

    def test_read_latest_metric_mapping(self) -> None:
        """最新一行指标映射应正确。"""
        reader = UltralyticsCSVReader()
        metric = reader.read_latest(str(FIXTURE_DIR))
        assert metric is not None
        # 最后一行 epoch=5
        assert metric.epoch == 5
        assert metric.source == "csv"

        # train_loss = 0.82 + 0.32 + 0.001 = 1.141
        assert metric.train_loss is not None
        assert abs(metric.train_loss - 1.141) < 0.001

        # val_loss = 0.92 + 0.48 + 0.002 = 1.402
        assert metric.val_loss is not None
        assert abs(metric.val_loss - 1.402) < 0.001

        # map50 = 0.78
        assert metric.map50 == 0.78
        assert metric.map5095 == 0.60

        # score 应为 map50
        assert metric.score == 0.78
        assert metric.score_name == "mAP50"

        # lr = 0.006
        assert metric.lr == 0.006

    def test_train_loss_sum(self) -> None:
        """train_loss 应由 box+cls+dfl 分量求和（因为无 train/loss 列）。"""
        reader = UltralyticsCSVReader()
        history = reader.read_history(str(FIXTURE_DIR))
        # 第一行：box=1.20, cls=0.50, dfl=0.002 → sum=1.702
        first = history[0]
        assert first.train_loss is not None
        assert abs(first.train_loss - 1.702) < 0.001

    def test_timestamp_uses_results_csv_mtime_and_preserves_raw_time(self, tmp_path) -> None:
        """Ultralytics time 是训练耗时，timestamp 应使用文件 mtime。"""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        csv_path = run_dir / "results.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "epoch,time,train/box_loss,metrics/mAP50(B)",
                    "1,999.0,1.2,0.5",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        mtime = 1_700_000_000
        csv_path.touch()
        import os

        os.utime(csv_path, (mtime, mtime))

        metric = UltralyticsCSVReader().read_latest(str(run_dir))

        assert metric is not None
        assert metric.timestamp == mtime
        assert metric.raw["time"] == 999.0

    def test_to_dict_serializable(self) -> None:
        """to_dict() 结果应能被 json.dumps 序列化。"""
        reader = UltralyticsCSVReader()
        metric = reader.read_latest(str(FIXTURE_DIR))
        assert metric is not None
        d = metric.to_dict()
        s = json.dumps(d, ensure_ascii=False)
        assert len(s) > 0
        parsed = json.loads(s)
        assert parsed["epoch"] == 5
        assert parsed["score_name"] == "mAP50"


class TestInspectCLI:
    """inspect 命令集成测试。"""

    def test_inspect_json_cli(self) -> None:
        """danling inspect --json 应输出合法 JSON 并包含关键字段。"""
        result = runner.invoke(app, ["inspect", str(FIXTURE_DIR), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["epoch"] == 5
        assert data["score_name"] == "mAP50"
        assert data["train_loss"] is not None
        assert data["map50"] == 0.78

    def test_inspect_table_cli(self) -> None:
        """danling inspect 默认模式（非 JSON）应正常返回。"""
        result = runner.invoke(app, ["inspect", str(FIXTURE_DIR), "--no-json"])
        assert result.exit_code == 0
        assert "epoch" in result.stdout.lower() or "5" in result.stdout

    def test_inspect_bad_path(self) -> None:
        """inspect 无 results.csv 的路径应返回错误。"""
        result = runner.invoke(app, ["inspect", "/tmp/nonexistent", "--json"])
        assert result.exit_code != 0


class TestUltralyticsLogReader:
    """Ultralytics 官方控制台日志 Reader 单元测试。"""

    def test_reads_official_console_log_output(self, tmp_path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text(
            "\n".join(
                [
                    "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size",
                    "      1/100      22.7G      1.411      1.365   0.007033"
                    "          7        960: 100% 3709/3709 1.1s/it 1:09:38",
                    "                 Class     Images  Instances      Box(P"
                    "          R      mAP50  mAP50-95): 100% 636/636 1.9it/s 5:31",
                    "                   all      40647      81710      0.859"
                    "      0.657      0.718      0.532",
                    "",
                    "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size",
                    "      2/100      22.4G      1.451      1.335   0.006962"
                    "         29        960: 100% 3709/3709 1.0s/it 1:04:47",
                    "                 Class     Images  Instances      Box(P"
                    "          R      mAP50  mAP50-95): 100% 636/636 1.6it/s 6:35",
                    "                   all      40647      81710      0.868"
                    "      0.657      0.719      0.529",
                ]
            ),
            encoding="utf-8",
        )

        history = UltralyticsLogReader().read_history(str(log_path))

        assert len(history) == 2
        latest = history[-1]
        assert latest.source == "ultralytics-log"
        assert latest.epoch == 2
        assert latest.raw["total_epochs"] == 100
        assert latest.raw["GPU_mem"] == "22.4G"
        assert latest.train_loss is not None
        assert abs(latest.train_loss - 2.792962) < 0.000001
        assert latest.precision == 0.868
        assert latest.recall == 0.657
        assert latest.map50 == 0.719
        assert latest.map5095 == 0.529
        assert latest.score == 0.719
        assert latest.score_name == "mAP50"

    def test_keeps_in_progress_epoch_without_validation_row(self, tmp_path) -> None:
        log_path = tmp_path / "nohup.out"
        log_path.write_text(
            "\n".join(
                [
                    "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size",
                    "      5/100      22.4G      1.524      1.415   0.006991"
                    "        176        960: 83% 3067/3709 1.0it/s 50:20<10:16",
                ]
            ),
            encoding="utf-8",
        )

        latest = UltralyticsLogReader().read_latest(str(log_path))

        assert latest is not None
        assert latest.epoch == 5
        assert latest.train_loss is not None
        assert latest.map50 is None

    def test_detects_log_inside_directory(self, tmp_path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text(
            "\n".join(
                [
                    "Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size",
                    "1/10 1.0G 1.0 2.0 3.0 4 640: 100% 1/1",
                ]
            ),
            encoding="utf-8",
        )

        assert UltralyticsLogReader.detect(str(tmp_path)) is True
