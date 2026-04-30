# DanLing v1.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete DanLing's roadmap through v1.0.0 stable release preparation.

**Architecture:** Keep readers, engine, hardware, renderers, storage, and CLI orchestration separated. Heavy integrations such as TensorBoard, Textual, and YAML remain optional extras with clear runtime messages.

**Tech Stack:** Python 3.10+, Typer, Rich, pytest, optional PyYAML, optional tensorboard, optional textual.

---

### Task 1: Complete Core Configuration And State Engine

**Files:**
- Modify: `src/danling/config.py`
- Create: `src/danling/engine/trends.py`
- Create: `src/danling/engine/diagnostics.py`
- Create: `src/danling/engine/state.py`
- Test: `tests/test_models.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_diagnostics.py`

**Steps:**
1. Write failing tests for full config defaults, trend analysis, state mood/furnace rules, and diagnostics.
2. Run focused tests and verify they fail because modules/fields are missing.
3. Implement minimal config, trends, diagnostics, and state logic.
4. Run focused tests, then full pytest.

### Task 2: Add Terminal Rendering And CLI Orchestration

**Files:**
- Modify: `src/danling/cli.py`
- Create: `src/danling/renderers/console.py`
- Create: `src/danling/renderers/statusline.py`
- Test: `tests/test_renderers.py`
- Modify: `tests/test_cli.py`

**Steps:**
1. Write failing tests for `status`, `statusline`, `state`, and renderer behavior without metrics/hardware.
2. Implement renderer functions and shared CLI state-loading helpers.
3. Keep `watch` based on `rich.live.Live` and test only non-loop command surfaces.
4. Run CLI and renderer tests.

### Task 3: Add Hardware Monitoring, Config Loading, And Persistence

**Files:**
- Create: `src/danling/hardware/base.py`
- Create: `src/danling/hardware/nvidia.py`
- Create: `src/danling/hardware/ascend.py`
- Create: `src/danling/hardware/__init__.py`
- Modify: `src/danling/config.py`
- Create: `src/danling/storage/json_store.py`
- Test: `tests/test_hardware.py`
- Test: `tests/test_storage_config.py`

**Steps:**
1. Write failing tests with monkeypatched `shutil.which`, `subprocess.run`, and `HOME`.
2. Implement tolerant hardware readers, YAML config loading, init/reset/config commands, and atomic JSON persistence.
3. Wire hardware and `--no-hardware`, `--config`, `--no-save` into status/state/watch.
4. Run focused tests and full pytest.

### Task 4: Add TensorBoard Reader And Reader Registry

**Files:**
- Create: `src/danling/readers/tensorboard_reader.py`
- Create: `src/danling/readers/registry.py`
- Modify: `src/danling/cli.py`
- Test: `tests/test_tensorboard_reader.py`
- Test: `tests/test_registry.py`

**Steps:**
1. Write failing tests for registry auto/csv detection and TensorBoard optional dependency behavior.
2. Implement registry and TensorBoard scalar mapping.
3. Add `--source auto|csv|tensorboard` to path-based CLI commands.
4. Run registry, reader, and CLI tests.

### Task 5: Add Optional Textual TUI Shell

**Files:**
- Create: `src/danling/renderers/textual_app.py`
- Modify: `src/danling/cli.py`
- Test: `tests/test_textual_tui.py`

**Steps:**
1. Write failing test for friendly message when Textual is missing.
2. Implement optional import and lightweight app class.
3. Run focused tests.

### Task 6: v1.0 Documentation, Examples, And Release Checks

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `src/danling/__init__.py`
- Create: `docs/getting-started.md`
- Create: `docs/configuration.md`
- Create: `docs/readers.md`
- Create: `docs/hardware.md`
- Create: `docs/diagnostics.md`
- Create: `docs/development.md`
- Create: `docs/release-checklist.md`
- Create: `examples/ultralytics_run/results.csv`
- Create: `examples/danling.yaml`

**Steps:**
1. Add release docs and examples.
2. Set version to `1.0.0`.
3. Run `pytest`, `ruff check src tests`, `python -m build` if the build module is available, and CLI smoke checks.
4. Record verification results in `progress.md`.

