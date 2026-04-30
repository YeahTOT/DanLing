# DanLing（丹灵）

**当前版本：v1.0.0**

训练跑起来以后，终端里经常只剩一串 loss、mAP、显存和利用率数字。DanLing 想做的事情很简单：把这些训练指标翻译成一眼能看懂的状态。

它会读取训练日志和硬件状态，判断训练是在变好、停滞、日志停更、loss 异常，还是可能遇到显存风险；再用宠物情绪、炼丹炉状态、修炼境界和诊断建议展示出来。你不用从一堆数字里猜训练是否健康，DanLing 会先帮你做一次“翻译”。

DanLing 不替代 TensorBoard、W&B、MLflow、Aim 或 ClearML。它更像一个轻量的**训练状态解释层**和**终端展示层**：只读训练输出，不接管训练任务，适合放在 SSH 终端、训练脚本旁边、shell statusline、Claude Code status line 或本地演示流程中使用。

## 适合谁

- 经常在终端里盯训练日志，希望更快判断训练状态的人
- 使用 Ultralytics `results.csv` 或 TensorBoard event 做本地训练记录的人
- 需要在 SSH、远程服务器或轻量环境里查看训练进度的人
- 想把训练过程做成更直观、更容易演示的 CLI/TUI 工具链的人

## 核心功能

- **读取训练日志**：支持 Ultralytics `results.csv`，也可以安装可选依赖读取 TensorBoard event
- **理解训练状态**：根据 loss、score、日志更新时间等信息判断训练是否正常、停滞、异常或正在变好
- **展示修炼境界**：用 baseline/SOTA 或手动阈值，把当前指标映射到炼器期、筑基期、结丹期、元婴期、化神期
- **给出诊断建议**：提示 NaN/inf、loss 爆炸、指标不提升、日志停更、显存过高、checkpoint 缺失等常见问题
- **读取硬件状态**：支持 NVIDIA GPU 和 Ascend NPU 的基础信息读取，失败时自动降级，不影响日志分析
- **多种终端展示方式**：提供一次性状态面板、循环刷新、全屏 TUI、配置 TUI 和适合状态栏使用的单行输出
- **回放训练日志**：可以把历史 CSV 或 TensorBoard scalar 按行回放成新的 `results.csv`，方便演示和调试

## 项目特色

- **轻量只读**：DanLing 只读取训练日志和硬件信息，不会启动、停止或修改你的训练任务
- **终端友好**：不需要 Web 服务也能看状态，适合远程服务器和命令行工作流
- **表达直观**：保留真实指标，同时用状态、境界和诊断把数字变得更容易理解
- **可逐步使用**：只装基础包就能读 CSV；需要 TUI、TensorBoard、YAML 时再安装对应 extra
- **失败不打断流程**：没有 GPU/NPU、硬件命令不可用或可选依赖缺失时，会尽量降级而不是直接崩掉

## 界面预览

### 全屏监控 TUI

![DanLing 全屏监控 TUI](assert/main.png)

### 配置 TUI

![DanLing 配置 TUI](assert/config.png)

## 许可与使用边界

本项目源代码公开，但当前采用 [CC BY-NC-ND 4.0](LICENSE)（署名-非商业性使用-禁止演绎 4.0 国际）许可，**不允许商业使用**，也**不允许对外分发修改后的版本**。

您可以：

- 在非商业场景中查看、复制、学习和分享原始版本
- 在本地为学习、研究或个人使用修改代码
- 通过 Issue 或 Pull Request 向主仓库贡献改进

您不可以：

- 将 DanLing 集成到商业产品、付费服务或企业盈利性项目中
- 将修改后的 DanLing 版本重新发布、打包分发或作为衍生产品提供
- 在未获书面授权前，将本软件用于任何商业目的

商业授权、企业内部使用或二次分发需求，需要联系作者并获得单独书面许可。这里的“源代码公开”不等同于 MIT/Apache/BSD 这类可自由商用许可证。

## 当前版本能力

v1.0.0 是 DanLing 的第一个可用版本，重点是把本地 CLI/TUI 的基础闭环跑通：能读真实训练输出，能推理训练健康度，能配置修炼境界，也能在终端里持续展示当前状态。

### 功能清单

| 能力 | v1.0.0 状态 | 用来做什么 |
| --- | --- | --- |
| 命令行入口 | 可用 | 提供 `inspect`、`state`、`status`、`statusline`、`watch`、`doctor`、`hardware`、`simulate`、`config`、`tui` 等命令 |
| Ultralytics CSV 读取 | 可用 | 读取训练目录或 `results.csv`，识别 loss、mAP、precision、recall、accuracy、lr 等指标 |
| TensorBoard 读取 | 可选可用 | 安装 `danling[tensorboard]` 后，读取 `events.out.tfevents.*` 中的 scalar |
| 训练状态推理 | 可用 | 根据指标趋势判断宠物情绪、最佳指标、日志休眠、loss 异常和训练状态 |
| 修炼境界 | 可用 | 根据 `baseline_score` / `sota_score` 自动等分，也支持五个手动阈值 |
| 诊断建议 | 初版可用 | 覆盖 NaN/inf、loss 爆炸、指标停滞、日志停更、显存风险、低利用率、checkpoint 缺失等场景 |
| 硬件读取 | 初版可用 | 支持 NVIDIA `nvidia-smi` 和 Ascend `npu-smi info` 的保守读取；失败时自动降级 |
| Rich 终端面板 | 可用 | 一次性展示状态，或循环刷新训练状态 |
| statusline | 可用 | 输出适合 shell prompt、状态栏或 Claude Code status line 集成的单行状态 |
| Textual TUI | 可选可用 | 安装 `danling[tui]` 后，使用全屏监控 TUI 和配置 TUI |
| 配置管理 | 可用 | 支持 `danling init`、`danling config show`、`danling config tui` |
| 训练日志回放 | 可用 | `danling simulate` 可按行回放 CSV 或 TensorBoard scalar，方便演示训练过程 |

### 当前限制

- 还不能直接读取 W&B、MLflow、Aim、ClearML 等实验平台
- 还没有 Web Dashboard；v1.0.0 重点是 CLI/TUI
- Ascend NPU 读取目前是保守文本解析，指标完整度低于 NVIDIA
- Textual TUI 和 TensorBoard 读取是可选能力，需要额外安装依赖
- DanLing 只读训练日志，不负责启动、停止或修改训练任务

### 版本维护约定

- v1.x 会尽量保持 CLI 命令、配置字段和核心数据模型兼容
- 破坏性变更应进入后续大版本，并在 README 或发布说明中说明迁移方式
- 新 Reader、新硬件后端、新渲染端应保持“读取、推理、渲染”边界清晰
- 当前授权策略保持非商业使用；商业授权不随版本升级自动开放

## 安装

基础安装：

```bash
pip install danling
```

常用可选能力：

```bash
pip install "danling[tui]"          # Textual 全屏 TUI 和配置 TUI
pip install "danling[tensorboard]"  # TensorBoard event reader
pip install "danling[yaml]"         # PyYAML 配置解析
```

本地源码开发：

```bash
pip install -e ".[dev,tui,yaml,tensorboard]"
```

项目要求 Python 3.10 或更新版本。

## 快速启动

### 1. 初始化配置

```bash
danling init
```

这会在当前目录生成 `danling.yaml`。推荐先用配置 TUI 编辑：

```bash
danling config tui
```

最小有效配置示例：

```yaml
primary_score: mAP50
baseline_score: 0.60
sota_score: 0.80
realm_thresholds:
```

`realm_thresholds` 可以先留空。DanLing 会把 baseline 到 SOTA 自动等分为五个境界。

### 2. 检查示例训练输出

```bash
danling inspect examples/ultralytics_run --json
danling state examples/ultralytics_run --config danling.yaml --no-hardware --json
danling status examples/ultralytics_run --config danling.yaml --no-hardware
danling statusline examples/ultralytics_run --config danling.yaml --no-hardware
danling doctor examples/ultralytics_run --config danling.yaml --no-hardware
```

如果当前机器没有 GPU/NPU，或硬件命令不可用，先加 `--no-hardware` 跑通日志读取、境界和状态显示。

### 3. 持续监控训练目录

Rich watch：

```bash
danling watch logs/run --source csv --config danling.yaml --no-hardware
```

Textual 全屏 TUI：

```bash
danling tui logs/run --source csv --config danling.yaml --no-hardware
```

没有安装 Textual 时，先使用 `danling watch`；需要全屏 TUI 再安装：

```bash
pip install "danling[tui]"
```

### 4. 仿真训练日志增长

只想快速生成一个可读取的 `logs/run/results.csv`：

```bash
danling simulate logs/ultralytics/results.csv logs/run --interval 0 --max-rows 20
```

TensorBoard event 也可以先回放成同样的 `results.csv`：

```bash
danling simulate logs/tensorboard logs/run --source tensorboard --interval 0 --max-rows 20
```

`logs/run` 会自动生成，最终文件是 `logs/run/results.csv`。默认会覆盖目标文件；如果需要继续追加，使用 `--append`。回放后的目录继续用 `--source csv` 监控。

终端 A：

```bash
danling simulate logs/ultralytics/results.csv logs/run --interval 2
```

终端 B：

```bash
danling tui logs/run --source csv --config danling.yaml --no-hardware
```

也可以用 Rich watch：

```bash
danling watch logs/run --source csv --config danling.yaml --no-hardware
```

## 命令说明

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

常用参数：

- `--source auto/csv/tensorboard`：选择数据源，监控/检查命令默认 `auto`，`simulate` 默认 `csv`
- `--config PATH`：指定配置文件
- `--json`：输出 JSON
- `--no-hardware`：跳过硬件读取
- `--no-save`：不更新本地状态文件
- `--debug`：显示 traceback，便于开发排查

## 支持的数据源

### Ultralytics `results.csv`

支持传入训练目录或 CSV 文件：

```bash
danling inspect runs/detect/train --source csv --json
danling inspect runs/detect/train/results.csv --source csv --json
```

当前会识别：

- epoch
- train loss、val loss，包含 Ultralytics 常见的 box/cls/dfl loss 求和
- `mAP50`、`mAP50-95`
- precision、recall、accuracy
- learning rate
- 原始列会保留在 `raw` 字段中

### TensorBoard event

需要安装：

```bash
pip install "danling[tensorboard]"
```

使用示例：

```bash
danling inspect logs/tensorboard --source tensorboard --json
danling inspect logs/tensorboard/train --source tensorboard --json
danling inspect logs/tensorboard/train/events.out.tfevents.xxx --source tensorboard --json
```

目录会递归查找 `events.out.tfevents.*`。常见的 `train/`、`val/` 子目录会作为指标上下文，用来区分裸的 `Loss/total` 等 scalar。

### 自动识别

```bash
danling status runs/detect/train --source auto
```

`auto` 会优先通过已注册 Reader 判断目录中是否存在 `results.csv` 或 TensorBoard event 文件。

## 配置

默认配置由 `danling init` 生成：

```yaml
pet_name: DanLing
primary_score:
primary_loss:
loss_window: 5
no_improve_patience: 10
stale_seconds: 300
high_memory_ratio: 0.90
hot_util_percent: 90.0
watch_interval: 2.0
baseline_score:
sota_score:
realm_thresholds:
#   炼器期: 0.50
#   筑基期: 0.60
#   结丹期: 0.70
#   元婴期: 0.80
#   化神期: 0.90
```

YOLO `results.csv` 常见配置：

```yaml
primary_score: mAP50
baseline_score: 0.60
sota_score: 0.80
```

如果日志里的指标是 `0.623` 这种小数，baseline/SOTA 也填 `0.60/0.80`；只有日志本身是 `62.3` 这种百分制数字时才填 `60/80`。

## 配置 TUI

`danling config tui` 提供两个入口：

- `1 手动配置`：编辑主指标、baseline/SOTA、境界阈值和刷新/诊断参数
- `2 自动生成`：先选择 Ultralytics `results.csv` 或 TensorBoard event/logdir，再输入历史路径和关心指标，自动生成 `primary_score`、`baseline_score` 和 `sota_score`

配置 TUI 的保存规则：

- 数值字段必须是数字
- 整数字段必须是整数
- `baseline_score` 和 `sota_score` 需要同时填写
- `baseline_score` 必须小于 `sota_score`
- 五个境界阈值只能全部留空，或五个全部填写
- 手动境界阈值必须严格递增

五个境界阈值留空时，DanLing 会根据 baseline/SOTA 自动等分，并在右侧预览炼器期、筑基期、结丹期、元婴期、化神期的分段。

## 状态推理规则

### 宠物情绪

| 情绪 | 触发条件 |
| --- | --- |
| `idle` | 当前没有 loss 和 score |
| `failed` | 指标出现 NaN/inf |
| `sleeping` | 最新日志更新时间超过 `stale_seconds` |
| `sick` | 当前 loss 超过上一个有效 loss 的 2 倍 |
| `evolving` | 当前 score 创新高 |
| `anxious` | score 连续 `no_improve_patience` 个快照未提升 |
| `happy` | loss 窗口斜率下降 |
| `normal` | 其他正常状态 |

### 修炼境界

- 默认使用当前主指标 `score`
- 设置 `primary_score` 后，优先使用该指标
- 配置 `baseline_score` 和 `sota_score` 后，自动等分为五个境界
- 配置完整 `realm_thresholds` 后，按手动阈值划分境界
- 达到或超过最高阈值后进入 `化神期`

### 炼丹炉状态

| 炼丹炉状态 | 触发条件 |
| --- | --- |
| `unknown` | 无硬件信息 |
| `cold` | 有硬件但利用率为 0 |
| `warm` | 利用率大于 0 |
| `burning` | 利用率达到 50% |
| `hot` | 利用率达到 `hot_util_percent` |
| `smoking` | 显存比例达到 `high_memory_ratio` |
| `exploded` | 显存比例达到 98% |

## 诊断能力

`danling doctor PATH` 会输出训练诊断：

```bash
danling doctor runs/detect/train --source auto --config danling.yaml
danling doctor runs/detect/train --source auto --config danling.yaml --json
```

v1.0.0 覆盖的诊断项包括：

- `NAN_LOSS`：指标出现 NaN/inf
- `LOSS_EXPLOSION`：loss 快速上升
- `NO_IMPROVEMENT`：指标长时间无提升
- `STALE_LOG`：训练日志长时间未更新
- `MEMORY_CRITICAL`：显存接近耗尽
- `HIGH_MEMORY`：显存使用偏高
- `LOW_UTILIZATION`：训练活跃但硬件利用率偏低
- `CHECKPOINT_MISSING`：训练目录缺少 `weights/last.pt`

## 硬件支持

NVIDIA GPU：

```bash
danling hardware --json
```

DanLing 会调用：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
```

Ascend NPU：

```bash
npu-smi info
```

硬件命令不存在、返回失败或解析失败时，DanLing 会返回空硬件列表，不影响日志读取和状态推理。需要完全跳过硬件读取时使用：

```bash
danling status runs/detect/train --no-hardware
```

## 本地状态

DanLing 只读训练日志，不会修改训练输出。默认会把跨运行状态写入：

```text
~/.danling/state.json
```

该文件用于记录累计更新次数、历史最佳 score、历史最佳 loss、失败次数和各训练目录的最近状态。

跳过写入：

```bash
danling status runs/detect/train --no-save
```

清空状态：

```bash
danling reset --yes
```

## 开发与维护

项目使用 src-layout：

```text
src/danling/
tests/
```

推荐开发检查：

```bash
pip install -e ".[dev,tui,yaml,tensorboard]"
pytest
ruff check src tests
python -m build --no-isolation
```

模块边界：

- `readers`：只读数据源并输出 `MetricSnapshot`
- `engine`：只做状态推理、趋势分析和诊断
- `hardware`：只读取硬件快照
- `renderers`：只负责终端/TUI 展示
- `storage`：只负责本地状态读写
- `cli.py`：负责命令编排、参数处理和错误出口

测试覆盖当前主要模块：

- CLI
- 配置加载与配置 TUI 表单逻辑
- CSV / TensorBoard Reader
- 状态推理、境界和诊断
- NVIDIA / Ascend 硬件解析
- Rich/statusline/Textual 渲染
- CSV 回放仿真
- 本地状态存储

## 贡献说明

欢迎提交问题、建议和 Pull Request，但需要遵守当前许可边界：

- 贡献内容一旦合并，将随本项目以 CC BY-NC-ND 4.0 许可发布
- Fork 后的本地修改可以用于非商业学习和研究
- 未经授权，不得对外分发修改版或商业化使用
- 新功能应优先补充测试和文档
- 新数据源 Reader 应保持可选依赖，不应影响 CSV 基础能力

建议贡献方向：

- 新训练平台 Reader
- 更完整的硬件后端
- 更清晰的诊断规则
- TUI 可用性优化
- 文档、示例和测试补充

## 常见问题

**DanLing 是开源软件吗？**

DanLing 源代码公开，允许非商业场景查看、学习和分享原始版本。但它当前不是 MIT/Apache/BSD 这类可自由商用的宽松许可证项目；商业使用和修改版分发都需要额外授权。

**没有 TensorBoard 依赖会影响 CSV 吗？**

不会。CSV Reader 是基础能力；TensorBoard 是可选依赖，只有显式读取 event files 时才需要。

**没有 GPU/NPU 会失败吗？**

不会。硬件读取失败会降级为空硬件列表。也可以使用 `--no-hardware` 跳过硬件读取。

**DanLing 会改训练日志吗？**

不会。DanLing 只读训练日志；持久化状态只写入 `~/.danling/state.json`。

**能替代 TensorBoard / W&B / MLflow 吗？**

不能，也不打算替代。DanLing 关注“当前训练状态是否健康”和“终端里如何快速看懂”。

## 许可证

Copyright (c) 2025 贾涛

本软件采用 [CC BY-NC-ND 4.0](LICENSE)（署名-非商业性使用-禁止演绎 4.0 国际）许可。

您可以在非商业场景中分享、复制原始版本，但**不得用于商业目的**，也**不得分发修改后的版本**。商业使用、企业内部盈利性使用、二次分发或特殊授权需求，需另行获得作者书面授权。
