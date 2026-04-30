# Config TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a terminal TUI command for editing DanLing configuration files.

**Architecture:** Keep config serialization in `danling.config`, pure form conversion helpers in a new `danling.renderers.config_tui` module, and Typer wiring in `danling.cli`. The Textual UI is optional and only imported inside the command/app factory.

**Tech Stack:** Python dataclasses, Typer, optional Textual widgets, pytest.

---

### Task 1: Config Serialization Helpers

**Files:**
- Modify: `src/danling/config.py`
- Test: `tests/test_storage_config.py`

**Step 1: Write the failing test**

Assert that a config with baseline, SOTA, and realm thresholds serializes to stable YAML text.

**Step 2: Run test**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_storage_config.py -q`
Expected: FAIL because the serializer does not exist.

**Step 3: Implement serializer**

Add `config_to_yaml_text(config: DanLingConfig) -> str`.

**Step 4: Run test**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_storage_config.py -q`
Expected: PASS.

### Task 2: Pure Config TUI Form Logic

**Files:**
- Create: `src/danling/renderers/config_tui.py`
- Test: `tests/test_config_tui.py`

**Step 1: Write failing tests**

Test form values convert to `DanLingConfig`, blank values become `None`, and config files can be saved.

**Step 2: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_config_tui.py -q`
Expected: FAIL because the module does not exist.

**Step 3: Implement pure helpers**

Add form field constants, value parsing, `config_to_form_values`, `config_from_form_values`, and `save_config_form`.

**Step 4: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_config_tui.py -q`
Expected: PASS.

### Task 3: CLI And Textual App Wiring

**Files:**
- Modify: `src/danling/cli.py`
- Modify: `src/danling/renderers/config_tui.py`
- Modify: `README.md`
- Modify: `docs/configuration.md`
- Test: `tests/test_cli.py`
- Test: `tests/test_config_tui.py`

**Step 1: Write failing tests**

Assert `danling config tui` shows a friendly missing dependency message and app creation works when Textual is installed.

**Step 2: Run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli.py tests/test_config_tui.py -q`
Expected: FAIL until the command and app factory exist.

**Step 3: Implement wiring and docs**

Add `config tui` command, create a small Textual form app with `Ctrl+S` save and `q` quit, and document usage.

**Step 4: Full verification**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
Run: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests`
Expected: PASS.
