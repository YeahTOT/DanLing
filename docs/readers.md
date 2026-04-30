# Readers

Reader 只负责读取和标准化训练指标，不负责判断宠物情绪。

## CSV

`UltralyticsCSVReader` 支持目录或文件：

```bash
danling inspect runs/detect/train --source csv --json
```

它会宽松处理表头空格、缺失列和常见 YOLO 指标名。

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

`ReaderRegistry` 支持 `auto`、`csv`、`tensorboard`。`auto` 按 CSV、TensorBoard 的顺序检测。
