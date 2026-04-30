# Cultivation Realms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace numeric DanLing levels with score-calibrated cultivation realms.

**Architecture:** Add a focused realm inference module under `danling.engine`, expose a `CultivationRealm` model on `DanLingState`, and keep rendering code as thin formatting over that model. Config owns calibration inputs; state inference owns threshold normalization.

**Tech Stack:** Python dataclasses, Typer CLI, Rich renderers, optional Textual renderer, pytest.

---

### Task 1: Realm Model And Inference

**Files:**
- Create: `src/danling/engine/realms.py`
- Modify: `src/danling/models.py`
- Modify: `src/danling/config.py`
- Test: `tests/test_engine.py`

**Step 1: Write failing tests**

Add tests for default equal thresholds, manual thresholds, SOTA overflow, and missing calibration.

**Step 2: Run tests**

Run: `pytest tests/test_engine.py -q`
Expected: FAIL because realm helpers and model fields do not exist yet.

**Step 3: Implement model and inference**

Create `CultivationRealm`, add config fields, and compute realm from the selected score.

**Step 4: Run tests**

Run: `pytest tests/test_engine.py -q`
Expected: PASS.

### Task 2: Renderer Migration

**Files:**
- Modify: `src/danling/renderers/statusline.py`
- Modify: `src/danling/renderers/textual_app.py`
- Test: `tests/test_renderers.py`
- Test: `tests/test_textual_tui.py`

**Step 1: Write failing tests**

Update renderer tests to expect realm text and no `Lv.N` dependency.

**Step 2: Run tests**

Run: `pytest tests/test_renderers.py tests/test_textual_tui.py -q`
Expected: FAIL until renderers use `state.realm`.

**Step 3: Implement renderer changes**

Replace `Lv.{level}` display with `realm.name`; append a realm line to the TUI pet panel.

**Step 4: Run tests**

Run: `pytest tests/test_renderers.py tests/test_textual_tui.py -q`
Expected: PASS.

### Task 3: Config, CLI Storage, And Docs

**Files:**
- Modify: `src/danling/cli.py`
- Modify: `README.md`
- Modify: `docs/configuration.md`
- Modify: `examples/danling.yaml`
- Test: `tests/test_models.py`
- Test: `tests/test_storage_config.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

Assert config defaults expose realm calibration fields and generated config includes them.

**Step 2: Run tests**

Run: `pytest tests/test_models.py tests/test_storage_config.py tests/test_cli.py -q`
Expected: FAIL until config/docs/storage are updated.

**Step 3: Implement cleanup**

Stop persisting derived `level`, document realm calibration, and refresh examples.

**Step 4: Run full verification**

Run: `pytest -q`
Run: `ruff check src tests`
Expected: PASS.
