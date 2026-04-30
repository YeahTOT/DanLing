# Commands

DanLing 的命令按“读取、解释、展示、配置、诊断”划分。多数命令都可以配合 `--config`、`--source`、`--json` 和 `--no-hardware` 使用。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `danling --version` | 显示当前版本号 |
| `danling inspect PATH` | 读取训练日志并输出最新指标快照 |
| `danling state PATH` | 输出 DanLing 聚合状态，可配合 `--json` |
| `danling status PATH` | 渲染一次 Rich 状态面板 |
| `danling statusline PATH` | 输出适合 shell prompt 或状态栏的一行状态 |
| `danling watch PATH` | 循环刷新 Rich 状态面板 |
| `danling tui PATH` | 启动可选 Textual 全屏监控 TUI |
| `danling simulate [SRC] [OUT]` | 按行回放 CSV/TensorBoard 训练日志到目标 `results.csv` |
| `danling hardware` | 显示当前 GPU/NPU 硬件状态 |
| `danling doctor [PATH]` | 无 PATH 时检查环境，有 PATH 时输出训练诊断 |
| `danling init` | 在当前目录创建 `danling.yaml` |
| `danling config show` | 显示当前配置 |
| `danling config tui` | 启动 Textual 配置界面编辑 `danling.yaml` |
| `danling reset --yes` | 删除 `~/.danling/state.json` |

## 常用参数

- `--source auto/csv/tensorboard`：选择数据源。监控/检查命令默认 `auto`，`simulate` 默认 `csv`。
- `--config PATH`：指定配置文件。
- `--json`：输出 JSON。
- `--no-hardware`：跳过硬件读取。
- `--no-save`：不更新本地状态文件。
- `--debug`：显示 traceback，便于开发排查。

## 示例

```bash
danling inspect examples/ultralytics_run --json
danling status examples/ultralytics_run --config danling.yaml --no-hardware
danling watch logs/run --source csv --config danling.yaml --no-hardware
danling doctor logs/run --source csv --config danling.yaml --no-hardware
```
