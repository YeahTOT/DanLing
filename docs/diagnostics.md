# Diagnostics

DanLing 会识别以下训练风险：

- `NAN_LOSS`：指标出现 NaN/inf
- `LOSS_EXPLOSION`：当前 loss 超过上一有效 loss 的 2 倍
- `NO_IMPROVEMENT`：score 连续多个快照未创新高
- `STALE_LOG`：日志长时间未更新
- `HIGH_MEMORY`：显存超过配置阈值
- `MEMORY_CRITICAL`：显存达到 98% 以上
- `LOW_UTILIZATION`：日志仍更新但硬件利用率偏低
- `CHECKPOINT_MISSING`：训练目录缺少 `weights/last.pt`

使用：

```bash
danling doctor runs/detect/train --json
```

