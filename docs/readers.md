# Readers

Reader 只负责读取和标准化训练指标，不负责判断宠物情绪。

## CSV

`UltralyticsCSVReader` 支持目录或文件：

```bash
danling inspect runs/detect/train --source csv --json
```

它会宽松处理表头空格、缺失列和常见 YOLO 指标名。

## Ultralytics Log/TXT

`UltralyticsLogReader` 支持读取官方训练控制台输出保存成的 `.log`、`.txt` 或 `nohup.out`：

```bash
python train.py > train.log 2>&1
danling inspect train.log --source ultralytics-log --json
danling inspect runs/detect/train --source auto --json
```

它会解析 `Epoch GPU_mem box_loss cls_loss dfl_loss ...` 训练行，以及随后 `all ... Box(P R mAP50 mAP50-95)` 的验证汇总行。训练中的最后一个 epoch 即使还没有验证汇总，也会保留 loss 快照，方便读取后台进程正在写入的日志。

## TensorBoard

`TensorBoardReader` 是可选能力：

```bash
pip install "danling[tensorboard]"
danling inspect logs/tensorboard --source tensorboard --json
danling inspect logs/tensorboard/train --source tensorboard --json
```

它会递归查找 `events.out.tfevents.*`，并用 `train/`、`val/` 这样的子目录名给裸 scalar tag 补充上下文。

未安装依赖时，只有 TensorBoard 读取会提示安装 extra，其他功能不受影响。

## Registry

`ReaderRegistry` 支持 `auto`、`csv`、`ultralytics-log`、`tensorboard`。`auto` 按 CSV、Ultralytics log/txt、TensorBoard 的顺序检测。

## Remote SSH

远程可视化不新增专用 Reader。`danling remote status/watch/tui` 会通过 SSH 密钥执行远程命令，把远程 `results.csv` 内容拉到本地临时文件，然后继续使用 CSV Reader：

- 远程路径是文件时读取该文件。
- 远程路径是目录时读取目录下的 `results.csv`。
- SSH 命令强制使用 `-i <key> -o BatchMode=yes`，不会等待密码输入。
- 远程文件 mtime 会写回本地临时 `results.csv`，用于日志休眠判断。

示例：

```bash
danling remote setup trainbox
danling remote hardware trainbox
danling remote status trainbox --config danling.yaml
danling remote watch trainbox --config danling.yaml
```

当前远程监控支持 Ultralytics `results.csv` 和官方控制台 `.log/.txt`。TensorBoard 远程 event 文件镜像不在这个入口中处理。
