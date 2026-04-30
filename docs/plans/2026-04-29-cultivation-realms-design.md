# DanLing Cultivation Realms Design

## Goal

Replace the old numeric pet level with a first-class cultivation realm derived from the current primary score.

## Model

DanLing state should expose a `realm` object instead of treating `level` as the core concept. The realm carries the Chinese phase name, rank, score bounds, and progress inside the current phase. The supported phases are:

- 炼器期
- 筑基期
- 结丹期
- 元婴期
- 化神期

`化神期` starts at the configured SOTA score. Scores above SOTA remain in `化神期`.

## Configuration

Configuration adds:

- `baseline_score`: score floor for `炼器期`
- `sota_score`: score floor for `化神期`
- `realm_thresholds`: optional manual score floors, keyed by realm name

When `realm_thresholds` is absent and both baseline and SOTA are configured, DanLing divides the baseline-to-SOTA range evenly into four intervals.

## Rendering

All user-facing renderers should display the realm text directly, for example `DanLing 结丹期 happy`. The Textual pet panel should append a realm line to the current temporary text art so later graphical realm art can replace that line cleanly.

## Compatibility

This is a long-term cleanup. New state JSON should prefer `realm` and stop using `level` as a first-class state field. Persisted historic `level` values are ignored.
