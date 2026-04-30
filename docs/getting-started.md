# Getting Started

DanLing 是一个训练状态解释 CLI。最小闭环是：读取训练日志、配置主指标和境界阈值、查看状态、再进入 watch/TUI。

## 1. 先读一个示例 run

```bash
danling inspect examples/ultralytics_run --json
danling status examples/ultralytics_run --no-hardware
danling statusline examples/ultralytics_run --no-hardware
```

如果你的训练目录里包含 `results.csv`，`--source auto` 会自动识别。遇到不支持的数据源时，DanLing 会提示期望的文件类型，而不是输出难懂 traceback。

## 2. 配置境界

```bash
danling init
danling config tui
```

在配置 TUI 中先填这三项：

```text
主指标: mAP50
Baseline 精度: 0.60
SOTA 精度: 0.80
```

五个境界阈值可以留空。保存时按 `Ctrl+S`，退出按 `q`。

保存后确认配置能被读回：

```bash
danling config show --config danling.yaml
danling status examples/ultralytics_run --config danling.yaml --no-hardware
```

## 3. 看一个“训练中”的目录

用一个终端回放 CSV，另一个终端观察 DanLing：

```bash
danling simulate logs/ultralytics/results.csv logs/run --interval 2
```

```bash
danling tui logs/run --source csv --config danling.yaml --no-hardware
```

没有安装 Textual 时，用 Rich watch：

```bash
danling watch logs/run --source csv --config danling.yaml --no-hardware
```

## 常见情况

**看不到硬件状态？**
没有 GPU/NPU、`nvidia-smi` / `npu-smi` 不存在、命令超时或解析失败时，DanLing 会把硬件降级为 unknown。先加 `--no-hardware` 跑通训练指标即可。

**没装 Textual？**
`danling tui` 和 `danling config tui` 需要：

```bash
pip install "danling[tui]"
```

不安装也可以使用 `danling status`、`danling watch` 和 `danling config show`。

**为什么一直是炼器期？**
通常是没有配置 `baseline_score` / `sota_score`，或者当前主指标还没有达到下一境界阈值。YOLO `mAP50` 小数格式可先用 `baseline_score: 0.60`、`sota_score: 0.80`。
