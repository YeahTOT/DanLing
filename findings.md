# DanLing Findings

## 2026-04-29 Initial Context

- Repository root contains `CLAUDE.md`, `README.md`, `pyproject.toml`, `src/`, `tests/`, `logs/`, and `开发计划.md`.
- `开发计划.md` defines phases from v0.0.1 through v1.0.0, with the user's current request starting after phase 2.
- Planned phase 3-11 areas are terminal rendering, hardware monitoring, config and persistence, diagnostics, TensorBoard reading, reader registry, Textual TUI, and release hardening.
- Current source tree includes `src/danling/cli.py`, `models.py`, `config.py`, `readers/base.py`, and `readers/ultralytics_csv.py`.
- Current tests include `tests/test_cli.py`, `tests/test_reader.py`, and an Ultralytics fixture.
- Later-stage modules such as `hardware/`, `engine/trends.py`, `engine/state.py`, `renderers/console.py`, `renderers/statusline.py`, and storage implementation are not present in the initial file list.
- `pyproject.toml` already declares optional extras for `tensorboard`, `yaml`, `tui`, `web`, `mlflow`, and `dev`.
- Phase 5-8 requirements emphasize graceful degradation when optional tools or CLIs are unavailable.
- Current `models.py` already has the main dataclasses/enums and recursive `to_dict()`.
- Current `config.py` is still a placeholder with only `pet_name`, so phase 1 config fields must be completed before later phases.
- Current CLI `status` and `watch` are still placeholders, and `inspect` is wired directly to `UltralyticsCSVReader`.
- Current `tests/test_cli.py` still expects placeholder `status`/`watch` behavior.
- Baseline test suite passes before new implementation: 15 tests passed.
- Git status shows the repository contents are currently untracked, so changes should be kept clearly scoped and not rely on existing commit history.
- Git worktree isolation is unavailable because `master` has no commits yet.
- `.gitignore` ignores `docs/plans`, so planning files under that directory are local execution artifacts rather than release docs.

## 2026-04-29 Simulation Request

- User wants a simulation program that reads `logs/ultralytics/results.csv` and writes one row every 2 seconds to `logs/run/results.csv`.
- Existing CLI already has `status`, `watch`, and optional `tui` commands that read a training log directory containing `results.csv`.
- Existing `tui` command is a lightweight optional Textual shell; `watch` is the currently functional terminal visualization path based on Rich Live.
- Source CSV has enough historical rows for playback; target path should be a run directory because the existing CSV reader detects `PATH/results.csv`.
