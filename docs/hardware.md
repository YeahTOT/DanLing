# Hardware

DanLing 的硬件层只负责读取设备快照。

## NVIDIA

使用 `nvidia-smi` query CSV 输出，读取：

- device id
- name
- util percent
- memory used/total
- temperature
- power draw

`danling remote hardware/status/watch/tui` 会通过已配置的 SSH 密钥在远程服务器执行同一组 `nvidia-smi` query，并复用本地解析逻辑。远程端会依次尝试 `nvidia-smi`、`/usr/bin/nvidia-smi`、`/usr/local/cuda/bin/nvidia-smi`，避免非交互 SSH 环境 PATH 太短导致读不到 GPU。

## Ascend

使用 `npu-smi info` 的保守解析。不同版本输出差异很大，解析不到时会返回空列表或 raw-only 信息。

## 降级

命令不存在、超时或输出异常时，硬件读取返回 `[]`，不会影响 `status`、`state`、`watch`。
