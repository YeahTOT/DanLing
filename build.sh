UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
UV_CACHE_DIR=/tmp/uv-cache uv run python -m build --no-isolation
