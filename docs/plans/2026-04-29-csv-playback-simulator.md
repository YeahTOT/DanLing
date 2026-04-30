# CSV Playback Simulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a simulator that replays `logs/ultralytics/results.csv` into `logs/run/results.csv` one row at a time, so a second terminal can visualize `logs/run` with DanLing TUI or watch mode.

**Architecture:** Add a small standard-library simulator module responsible only for CSV playback. Wire it into the Typer CLI as `danling simulate`; keep visualization in the existing `tui`/`watch` commands, and upgrade the optional Textual app to refresh state from existing readers instead of showing static text.

**Tech Stack:** Python standard library CSV/path/time, Typer, existing ReaderRegistry/StateEngine/Renderers, optional Textual.

---

### Task 1: Simulator Core

**Files:**
- Create: `src/danling/simulator.py`
- Test: `tests/test_simulator.py`

**Step 1: Write failing tests**

Test that `playback_results_csv(..., interval=0, max_rows=2)` creates the target directory, writes a header and two rows, and yields two progress snapshots.

**Step 2: Implement minimal simulator**

Use `csv.reader`/`csv.writer`, create parent directories, optionally overwrite target file, flush after each row, sleep between rows, and support `max_rows` for tests/dry-runs.

### Task 2: CLI Command

**Files:**
- Modify: `src/danling/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

Test `danling simulate SRC DEST --interval 0 --max-rows 2` returns 0 and writes `DEST/results.csv`.

**Step 2: Implement command**

Add defaults:
- source: `logs/ultralytics/results.csv`
- output: `logs/run`
- interval: `2.0`

### Task 3: TUI Refresh

**Files:**
- Modify: `src/danling/renderers/textual_app.py`
- Test: `tests/test_textual_tui.py`

**Step 1: Write tests around helper behavior**

Test a pure helper can build display text from a `DanLingState`, so Textual rendering has a stable, testable core even when Textual is not installed.

**Step 2: Implement live Textual shell**

When Textual is installed, create an app that refreshes state by calling existing readers/state engine and updates a text widget. Keep the missing dependency message unchanged.

### Task 4: Docs And Verification

**Files:**
- Modify: `README.md`

**Step 1: Document two-terminal workflow**

Terminal A:
`danling simulate logs/ultralytics/results.csv logs/run --interval 2`

Terminal B:
`danling tui logs/run --source csv --no-hardware`
or `danling watch logs/run --source csv --no-hardware`

**Step 2: Verify**

Run focused simulator tests, full pytest, ruff, and a short CLI dry-run.

