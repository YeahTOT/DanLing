# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DanLing（丹灵）是一个面向深度学习训练过程的 Python CLI/TUI 工具，将 loss/mAP/GPU-NPU 指标解读为宠物情绪、炼丹炉状态和修炼境界。定位为轻量**状态解释层**和**终端呈现层**，不替代 TensorBoard/W&B。

## 常用命令

```bash
# 开发安装（含所有可选依赖）
pip install -e ".[dev,tui,yaml,tensorboard]"

# 测试
pytest                                  # 全部测试
pytest tests/test_engine.py -q         # 单个测试文件

# 代码检查
ruff check src tests

# 构建
python -m build --no-isolation
```

## 核心架构：读取 → 推理 → 渲染

```
readers/          →  engine/              →  renderers/
(数据读取)           (状态推理+诊断)           (终端展示)
     ↓                   ↓                      ↓
 MetricSnapshot    →  DanLingState       →  Rich面板/statusline/TUI
                         ↑
                    hardware/              storage/
                   (硬件快照)            (本地状态持久化)
```

所有模块通过 `cli.py` 编排，cli.py 负责参数解析和错误出口，不包含业务逻辑。

**模块边界（不可逾越）：**
- `readers/` — 只读数据源，输出 `MetricSnapshot` 列表
- `engine/` — 只做状态推理（`state.py`）、趋势分析（`trends.py`）、境界推断（`realms.py`）、诊断（`diagnostics.py`）
- `hardware/` — 只读硬件快照，失败时优雅降级为空列表
- `renderers/` — 只负责终端/TUI 展示，不修改状态
- `storage/` — 只负责 `~/.danling/state.json` 读写

## 核心数据流

1. `readers/registry.py::read_history()` 根据 `--source` (auto/csv/tensorboard) 选择 Reader，返回 `list[MetricSnapshot]`
2. `cli.py::_build_state_for_path()` 调用 `engine/state.py::build_state()`，传入 history + hardware + config + persisted
3. `build_state()` 内部：`_apply_primary_score()` → `analyze_trends()` → `_pet_mood()` + `_furnace_state()` + `infer_realm()` + `diagnose()` → 聚合为 `DanLingState`
4. `DanLingState` 送到 renderer（Rich panel / statusline / Textual TUI）

## 关键设计决策

- **Reader 注册机制**：`ReaderRegistry` 维护已注册 Reader 列表，`source=auto` 时逐个调用 `detect()`，第一个匹配的胜出。新增数据源只需实现 `BaseReader` 三个抽象方法并注册。
- **可选依赖**：TensorBoard（`tensorboard`）、TUI（`textual`）、YAML（`pyyaml`）都是可选 extras。CSV Reader 和基础 CLI 是核心，不依赖任何可选包。代码中 `importlib.util.find_spec()` 检查可用性。
- **配置降级**：`config.py` 自带轻量 YAML 解析器（`_parse_simple_yaml`），PyYAML 缺失时仍可工作。
- **硬件降级**：`nvidia-smi` / `npu-smi` 不可用时返回空硬件列表，不影响日志读取和状态推理。
- **`primary_score` 映射**：配置中的指标名通过 `_metric_number()` 映射到 `MetricSnapshot` 的具名字段（mAP50→map50, mAP50-95→map5095 等），未匹配时查 `raw` 字典。
- **`score` vs `loss`**：`MetricSnapshot.score` 是"越高越好"的主指标（如 mAP），`train_loss`/`val_loss` 是"越低越好"的指标。`primary_score` 配置指定用哪个 raw 字段作为 score。
