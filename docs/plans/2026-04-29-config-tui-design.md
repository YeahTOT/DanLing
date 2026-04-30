# DanLing Config TUI Design

## Goal

Add `danling config tui` so users can edit `danling.yaml` from the terminal.

## Command

`danling config tui` opens a Textual form for the current directory `danling.yaml`.
`danling config tui --config path/to/danling.yaml` edits a specific config file.
If the target file does not exist, the editor starts from default values and creates the file on save.

## Scope

The first version edits stable core fields only:

- `pet_name`
- `primary_score`
- `baseline_score`
- `sota_score`
- five `realm_thresholds` entries
- `watch_interval`
- `loss_window`
- `no_improve_patience`
- `stale_seconds`

Monitoring TUI and configuration TUI stay separate so the long-running dashboard does not own settings persistence.

## Data Flow

The TUI loads `DanLingConfig`, fills Textual `Input` widgets, converts form values back into a config dataclass, and writes a stable YAML-like file. Pure conversion and serialization helpers remain testable without Textual installed.
