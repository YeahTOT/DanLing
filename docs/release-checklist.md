# Release Checklist

- [x] `pytest`
- [x] `ruff check src tests`
- [x] `python -m build --no-isolation`
- [x] `pip install dist/*.whl` in a temporary venv
- [x] `danling --help`
- [x] `danling status examples/ultralytics_run`
- [x] `danling inspect examples/ultralytics_run --json`
- [x] `danling hardware --json`
- [ ] Verify `danling[tensorboard]`
- [ ] Verify `danling[tui]`
- [x] Confirm version in `pyproject.toml` and `src/danling/__init__.py`

Note: isolated `python -m build` requires a Python installation with `ensurepip`
available. In this environment the isolated build failed because `python3.10-venv`
is missing, so release verification used `--no-isolation` with `hatchling`
installed in the active venv.
