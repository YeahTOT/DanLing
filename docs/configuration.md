# Configuration

DanLing 会在当前目录查找 `danling.yaml`。也可以显式传入：

```bash
danling status runs/train --config danling.yaml
danling config show --config danling.yaml
danling config tui --config danling.yaml
```

字段：

- `pet_name`
- `primary_score`
- `primary_loss`
- `loss_window`
- `no_improve_patience`
- `stale_seconds`
- `high_memory_ratio`
- `hot_util_percent`
- `watch_interval`
- `baseline_score`
- `sota_score`
- `realm_thresholds`

## 境界配置

YOLO `results.csv` 推荐先用 `mAP50` 做主指标：

```yaml
primary_score: mAP50
baseline_score: 0.60
sota_score: 0.80
realm_thresholds:
```

`realm_thresholds` 可以留空。只配置 `baseline_score` 和 `sota_score` 时，DanLing 会自动等分：

```text
炼器期 0.60
筑基期 0.65
结丹期 0.70
元婴期 0.75
化神期 0.80
```

如果你更关注严格指标，可以改用 `mAP50-95`：

```yaml
primary_score: mAP50-95
baseline_score: 0.44
sota_score: 0.60
realm_thresholds:
```

如果你在调 precision/recall，也可以用 `precision`：

```yaml
primary_score: precision
baseline_score: 0.70
sota_score: 0.90
realm_thresholds:
```

注意指标尺度要和日志一致：如果日志里的值是 `0.623`，配置也填 `0.60/0.80`；只有日志本身是 `62.3` 这种百分制数字时才填 `60/80`。

## 配置 TUI 预览

`danling config tui` 会先进入配置主页，提供两个子菜单：

- `1 手动配置`：显示“左表单 + 右预览”。左侧编辑 `danling.yaml` 字段，右侧实时展示炼器期、筑基期、结丹期、元婴期、化神期的分段。
- `2 自动生成`：输入历史 Ultralytics `results.csv` 路径和关心指标，按 `Ctrl+G` 自动生成 `primary_score`、`baseline_score` 和 `sota_score`。

- 五个 `realm_thresholds` 全留空时，右侧按 `baseline_score` 到 `sota_score` 自动等分。
- 五个 `realm_thresholds` 全部填写时，右侧按手动阈值预览。
- baseline/SOTA 和五个境界阈值都为空时可以保存，右侧提示“境界未启用”。

自动生成规则：目前只支持 Ultralytics `results.csv`；`baseline_score` 取关心指标在历史顺序 20% 位置的结果，`sota_score` 取该指标历史最好结果。指标可以填 `mAP50`、`mAP50-95`、`precision`、`recall`，也可以填 CSV 原始列名。

保存前会阻止这些输入：

- 数值字段不是数字，或整数配置不是整数。
- 只填 `baseline_score` 或只填 `sota_score`。
- `baseline_score >= sota_score`。
- 境界阈值只填了一部分。
- 五个境界阈值没有严格递增。

校验失败时 TUI 显示中文错误，不写入配置文件。

手动境界阈值示例：

```yaml
primary_score: precision
baseline_score: 0.50
sota_score: 0.90
realm_thresholds:
  炼器期: 0.50
  筑基期: 0.60
  结丹期: 0.70
  元婴期: 0.80
  化神期: 0.90
```

只有想手动控制每个境界的起点时才需要填写 `realm_thresholds`。达到或超过 `sota_score` 或 `化神期` 阈值时进入 `化神期`。

本地养成状态保存在 `~/.danling/state.json`，可用 `danling reset --yes` 清理。
