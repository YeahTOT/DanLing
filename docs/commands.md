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
| `danling remote setup PROFILE` | 打开 TUI 配置远程 SSH 密钥和日志目录 |
| `danling remote hardware PROFILE` | 通过 SSH 密钥读取远程 NVIDIA GPU 状态 |
| `danling remote status PROFILE` | 通过 SSH 密钥读取远程训练日志并渲染一次状态面板 |
| `danling remote watch PROFILE` | 通过 SSH 密钥循环刷新远程训练状态面板 |
| `danling remote tui PROFILE` | 通过 SSH 密钥启动远程训练 Textual 全屏监控 TUI |
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

## 远程配置

`danling remote` 强制使用 SSH 密钥。先运行：

```bash
danling remote setup trainbox
```

配置 TUI 会辅助填写 `user@host`、远程 `results.csv` 所在目录或文件、SSH 私钥路径，并提示执行 `ssh-copy-id -i <key>.pub user@host`。连接测试使用 `ssh -i <key> -o BatchMode=yes user@host true`，通过后保存到 `~/.danling/remote_profiles.json`。

配置页也可以按 `Ctrl+H` 测试远程 NVIDIA GPU。监控和 `remote hardware` 会在远程端查找 `nvidia-smi`、`/usr/bin/nvidia-smi`、`/usr/local/cuda/bin/nvidia-smi`，读取利用率、显存、温度和功耗。

监控阶段只传配置名称：

```bash
danling remote status trainbox --config danling.yaml
danling remote hardware trainbox
danling remote tui trainbox --config danling.yaml
```

- `--ssh-option TEXT`：传给 `ssh -o` 的附加选项，可重复；`BatchMode=yes` 会被强制启用。
- `--remote-hardware/--no-remote-hardware`：是否读取远程 NVIDIA GPU 状态，默认读取。
- `--sync-timeout FLOAT`：单次 SSH 超时秒数，默认 `10`。

当前远程监控只支持 Ultralytics `results.csv`。TensorBoard 仍可本地读取，但远程 TUI 不再做 event 文件镜像。

## 示例

```bash
danling inspect examples/ultralytics_run --json
danling status examples/ultralytics_run --config danling.yaml --no-hardware
danling watch logs/run --source csv --config danling.yaml --no-hardware
danling doctor logs/run --source csv --config danling.yaml --no-hardware
danling remote setup trainbox
danling remote hardware trainbox
danling remote tui trainbox --config danling.yaml
```
