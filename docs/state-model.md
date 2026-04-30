# State Model

DanLing 的状态模型只做一件事：把训练快照和硬件快照解释成更容易读懂的终端状态。它不会启动、停止或修改训练任务。

## 宠物情绪

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

## 修炼境界

- 默认使用当前主指标 `score`。
- 设置 `primary_score` 后，优先使用该指标。
- 配置 `baseline_score` 和 `sota_score` 后，自动等分为五个境界。
- 配置完整 `realm_thresholds` 后，按手动阈值划分境界。
- 达到或超过最高阈值后进入 `化神期`。

## 炼丹炉状态

| 炼丹炉状态 | 触发条件 |
| --- | --- |
| `unknown` | 无硬件信息 |
| `cold` | 有硬件但利用率为 0 |
| `warm` | 利用率大于 0 |
| `burning` | 利用率达到 50% |
| `hot` | 利用率达到 `hot_util_percent` |
| `smoking` | 显存比例达到 `high_memory_ratio` |
| `exploded` | 显存比例达到 98% |

## 本地状态

DanLing 会把跨运行状态写入：

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
