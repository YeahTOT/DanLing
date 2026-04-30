"""测试配置 TUI 的纯表单逻辑。"""

from __future__ import annotations

import pytest

from danling.config import DanLingConfig, load_config
from danling.renderers.config_tui import (
    config_from_form_values,
    config_to_form_values,
    create_config_app,
    realm_field_id,
    render_realm_preview,
    save_config_form,
    suggest_config_from_results,
    validate_config_form,
)


def test_config_to_form_values_flattens_realm_thresholds() -> None:
    values = config_to_form_values(
        DanLingConfig(
            pet_name="小丹灵",
            primary_score="precision",
            baseline_score=0.5,
            sota_score=0.9,
            realm_thresholds={"炼器期": 0.5, "化神期": 0.9},
        )
    )

    assert values["pet_name"] == "小丹灵"
    assert values["primary_score"] == "precision"
    assert values["baseline_score"] == "0.5"
    assert values["sota_score"] == "0.9"
    assert values[realm_field_id("炼器期")] == "0.5"
    assert values[realm_field_id("筑基期")] == ""
    assert values[realm_field_id("化神期")] == "0.9"


def test_config_from_form_values_parses_config_fields() -> None:
    config = config_from_form_values(
        {
            "pet_name": "小丹灵",
            "primary_score": "precision",
            "primary_loss": "",
            "baseline_score": "0.5",
            "sota_score": "0.9",
            "watch_interval": "1.5",
            "loss_window": "7",
            "no_improve_patience": "12",
            "stale_seconds": "600",
            "high_memory_ratio": "0.92",
            "hot_util_percent": "88",
            realm_field_id("炼器期"): "0.50",
            realm_field_id("筑基期"): "0.62",
            realm_field_id("结丹期"): "0.74",
            realm_field_id("元婴期"): "0.84",
            realm_field_id("化神期"): "0.90",
        }
    )

    assert config.pet_name == "小丹灵"
    assert config.primary_score == "precision"
    assert config.primary_loss is None
    assert config.baseline_score == 0.5
    assert config.sota_score == 0.9
    assert config.watch_interval == 1.5
    assert config.loss_window == 7
    assert config.no_improve_patience == 12
    assert config.stale_seconds == 600
    assert config.high_memory_ratio == 0.92
    assert config.hot_util_percent == 88
    assert config.realm_thresholds == {
        "炼器期": 0.50,
        "筑基期": 0.62,
        "结丹期": 0.74,
        "元婴期": 0.84,
        "化神期": 0.90,
    }


def test_config_from_form_values_ignores_partial_realm_thresholds() -> None:
    config = config_from_form_values(
        {
            "pet_name": "DanLing",
            realm_field_id("炼器期"): "0.50",
            realm_field_id("筑基期"): "",
        }
    )

    assert config.realm_thresholds is None


def test_config_from_form_values_reports_invalid_numbers() -> None:
    with pytest.raises(ValueError, match="baseline_score"):
        config_from_form_values({"baseline_score": "not-a-number"})


def test_render_realm_preview_uses_equal_baseline_sota_ranges() -> None:
    preview = render_realm_preview(
        {
            "baseline_score": "0.60",
            "sota_score": "0.80",
        }
    )

    assert "境界预览" in preview
    assert "炼器期: 0.6000 - 0.6500" in preview
    assert "筑基期: 0.6500 - 0.7000" in preview
    assert "结丹期: 0.7000 - 0.7500" in preview
    assert "元婴期: 0.7500 - 0.8000" in preview
    assert "化神期: >= 0.8000" in preview


def test_render_realm_preview_uses_manual_thresholds_when_all_are_present() -> None:
    preview = render_realm_preview(
        {
            realm_field_id("炼器期"): "0.50",
            realm_field_id("筑基期"): "0.62",
            realm_field_id("结丹期"): "0.74",
            realm_field_id("元婴期"): "0.84",
            realm_field_id("化神期"): "0.90",
        }
    )

    assert "炼器期: 0.5000 - 0.6200" in preview
    assert "元婴期: 0.8400 - 0.9000" in preview
    assert "化神期: >= 0.9000" in preview


def test_render_realm_preview_shows_disabled_message_when_realm_is_unconfigured() -> None:
    preview = render_realm_preview({})

    assert "境界未启用" in preview


def test_suggest_config_from_ultralytics_results_uses_20_percent_row_and_best_score(
    tmp_path,
) -> None:
    path = tmp_path / "results.csv"
    rows = [
        "epoch,metrics/mAP50(B),metrics/precision(B)",
        *[f"{index},{index / 10:.2f},{0.50 + index / 100:.2f}" for index in range(1, 11)],
    ]
    path.write_text("\n".join(rows), encoding="utf-8")

    values = suggest_config_from_results(path, "mAP50")

    assert values["primary_score"] == "mAP50"
    assert values["baseline_score"] == "0.2"
    assert values["sota_score"] == "1.0"


def test_suggest_config_from_ultralytics_results_supports_raw_metric_name(tmp_path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "\n".join(
            [
                "epoch,metrics/mAP50(B),metrics/mAP50-95(B)",
                "1,0.10,0.05",
                "2,0.20,0.15",
                "3,0.30,0.25",
                "4,0.40,0.35",
                "5,0.50,0.45",
            ]
        ),
        encoding="utf-8",
    )

    values = suggest_config_from_results(path, "metrics/mAP50-95(B)")

    assert values["primary_score"] == "metrics/mAP50-95(B)"
    assert values["baseline_score"] == "0.05"
    assert values["sota_score"] == "0.45"


def test_suggest_config_from_ultralytics_results_rejects_missing_metric(tmp_path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "\n".join(["epoch,metrics/mAP50(B)", "1,0.10", "2,0.20"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未找到指标"):
        suggest_config_from_results(path, "precision")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"baseline_score": "0.80", "sota_score": "0.60"}, "baseline_score 必须小于 sota_score"),
        ({"baseline_score": "0.60"}, "baseline_score 和 sota_score 需要同时填写"),
        ({realm_field_id("炼器期"): "0.50"}, "境界阈值需要全部留空，或五个全部填写"),
        (
            {
                realm_field_id("炼器期"): "0.50",
                realm_field_id("筑基期"): "0.62",
                realm_field_id("结丹期"): "0.62",
                realm_field_id("元婴期"): "0.84",
                realm_field_id("化神期"): "0.90",
            },
            "境界阈值必须严格递增",
        ),
        ({"baseline_score": "not-a-number"}, "baseline_score 必须是数字"),
        ({"loss_window": "1.5"}, "loss_window 必须是整数"),
    ],
)
def test_validate_config_form_reports_invalid_realm_settings(
    values: dict[str, str], message: str
) -> None:
    assert message in validate_config_form(values)


def test_save_config_form_writes_loadable_yaml(tmp_path) -> None:
    path = tmp_path / "danling.yaml"

    save_config_form(
        path,
        {
            "pet_name": "小丹灵",
            "baseline_score": "0.5",
            "sota_score": "0.9",
        },
    )

    config = load_config(str(path))
    assert config.pet_name == "小丹灵"
    assert config.baseline_score == 0.5
    assert config.sota_score == 0.9


def test_save_config_form_rejects_invalid_realm_config_without_writing(tmp_path) -> None:
    path = tmp_path / "danling.yaml"

    with pytest.raises(ValueError, match="baseline_score 必须小于 sota_score"):
        save_config_form(path, {"baseline_score": "0.9", "sota_score": "0.5"})

    assert not path.exists()


def test_save_config_form_rejects_invalid_realm_config_without_overwriting(tmp_path) -> None:
    path = tmp_path / "danling.yaml"
    original = "pet_name: 原丹灵\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="境界阈值需要全部留空"):
        save_config_form(path, {realm_field_id("炼器期"): "0.5"})

    assert path.read_text(encoding="utf-8") == original


def test_create_config_app_instantiates_when_textual_is_available(tmp_path) -> None:
    try:
        import textual  # noqa: F401
    except ImportError:
        return

    app = create_config_app(tmp_path / "danling.yaml")

    assert app is not None
