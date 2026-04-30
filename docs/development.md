# Development

本项目使用 src-layout：

```text
src/danling/
tests/
```

推荐开发命令：

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
python -m build
```

设计边界：

- Reader 只读数据源并输出 `MetricSnapshot`
- Engine 只推理状态和诊断
- Hardware 只读取硬件快照
- Renderer 只显示状态
- CLI 只负责 orchestration 和错误处理

