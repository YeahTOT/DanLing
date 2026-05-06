"""DanLing CLI 入口。"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Annotated

import typer
from rich.live import Live
from rich.table import Table

from danling.config import DanLingConfig, default_config_text, load_config
from danling.engine.diagnostics import diagnose
from danling.engine.state import build_state
from danling.hardware import read_all_hardware
from danling.models import DanLingState, PetMood
from danling.readers.registry import read_history
from danling.remote import (
    RemoteCommandError,
    RemoteProfile,
    RemoteTrainingMonitor,
    load_remote_profile,
    read_remote_nvidia_hardware,
)
from danling.renderers.console import render_state_panel
from danling.renderers.statusline import render_statusline
from danling.simulator import playback_training_log
from danling.storage.json_store import JsonStateStore

app = typer.Typer(
    name="danling",
    help="炼丹宠物 — 深度学习训练状态解释层",
    no_args_is_help=True,
)
config_app = typer.Typer(help="配置文件工具")
SOURCE_HELP = "数据源：auto/csv/ultralytics-log/tensorboard"


class RemoteProfileGroup(typer.core.TyperGroup):
    """Custom group that routes `danling remote <profile> <command>`."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands:
            profile_name = args.pop(0)
            ctx.ensure_object(dict)
            ctx.obj["profile_name"] = profile_name
        return super().resolve_command(ctx, args)


remote_app = typer.Typer(
    cls=RemoteProfileGroup,
    help="通过 SSH 可视化远程训练过程",
)
remote_config_app = typer.Typer(help="远程配置文件管理")
remote_app.add_typer(remote_config_app, name="config")
app.add_typer(config_app, name="config")
app.add_typer(remote_app, name="remote")

console = None


def _get_console():
    global console
    if console is None:
        from rich.console import Console

        console = Console()
    return console


def _version_callback(value: bool) -> None:
    if value:
        from danling import __version__

        typer.echo(f"danling v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="显示版本号"),
    ] = None,
) -> None:
    """DanLing — 炼丹宠物 CLI。"""


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="训练日志目录、CSV 文件或 TensorBoard event 路径")],
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """检查训练日志并输出最新指标快照。"""
    history = _read_history_or_exit(path, source, debug=debug)
    if not history:
        _fail("无法读取指标数据", debug=debug)
    metric = history[-1]

    if json_output:
        _echo_json(metric.to_dict())
        return

    table = _metric_table(metric.to_dict(), title=f"DanLing Inspect — {path}")
    _get_console().print(table)


@app.command()
def state(
    path: Annotated[Path, typer.Argument(help="训练日志目录或文件路径")],
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    no_hardware: Annotated[bool, typer.Option("--no-hardware", help="跳过硬件读取")] = False,
    no_save: Annotated[bool, typer.Option("--no-save", help="不更新本地状态")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """输出 DanLing 聚合状态。"""
    current = _build_state_for_path(path, source, config, no_hardware, debug)
    if not no_save:
        _update_store(path, current)
    if json_output:
        _echo_json(current.to_dict())
    else:
        _get_console().print(render_state_panel(current))


@app.command()
def status(
    path: Annotated[Path, typer.Argument(help="训练日志目录或文件路径")],
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    no_hardware: Annotated[bool, typer.Option("--no-hardware", help="跳过硬件读取")] = False,
    no_save: Annotated[bool, typer.Option("--no-save", help="不更新本地状态")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """渲染一次 Rich 状态面板。"""
    current = _build_state_for_path(path, source, config, no_hardware, debug)
    if not no_save:
        _update_store(path, current)
    _get_console().print(render_state_panel(current))


@app.command()
def statusline(
    path: Annotated[Path, typer.Argument(help="训练日志目录或文件路径")],
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    no_hardware: Annotated[bool, typer.Option("--no-hardware", help="跳过硬件读取")] = False,
    no_save: Annotated[bool, typer.Option("--no-save", help="不更新本地状态")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """输出适合 shell prompt 的单行状态。"""
    current = _build_state_for_path(path, source, config, no_hardware, debug)
    if not no_save:
        _update_store(path, current)
    typer.echo(render_statusline(current))


@app.command()
def watch(
    path: Annotated[Path, typer.Argument(help="训练日志目录或文件路径")],
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    interval: Annotated[float | None, typer.Option("--interval", help="刷新间隔秒数")] = None,
    no_hardware: Annotated[bool, typer.Option("--no-hardware", help="跳过硬件读取")] = False,
    no_save: Annotated[bool, typer.Option("--no-save", help="不更新本地状态")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """实时监控训练日志并刷新状态面板。"""
    cfg = _load_config_or_exit(config, debug)
    refresh_interval = interval if interval is not None else cfg.watch_interval
    try:
        with Live(refresh_per_second=4, screen=False) as live:
            while True:
                current = _build_state_for_path(path, source, config, no_hardware, debug)
                if not no_save:
                    _update_store(path, current)
                live.update(render_state_panel(current))
                time.sleep(refresh_interval)
    except KeyboardInterrupt:
        typer.echo("DanLing watch stopped.")


@app.command()
def simulate(
    source_path: Annotated[
        Path,
        typer.Argument(help="源 results.csv、训练目录或 TensorBoard event 路径"),
    ] = Path("logs/ultralytics/results.csv"),
    output: Annotated[
        Path,
        typer.Argument(help="输出 run 目录或 results.csv 路径"),
    ] = Path("logs/run"),
    source: Annotated[
        str,
        typer.Option("--source", help=SOURCE_HELP),
    ] = "csv",
    interval: Annotated[float, typer.Option("--interval", help="每行写入间隔秒数")] = 2.0,
    max_rows: Annotated[int | None, typer.Option("--max-rows", help="最多回放行数")] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--append", help="覆盖或追加目标 results.csv"),
    ] = True,
) -> None:
    """按行回放训练日志，用于仿真训练中的日志增长。"""
    try:
        last_progress = None
        for progress in playback_training_log(
            source_path,
            output,
            source=source,
            interval=interval,
            max_rows=max_rows,
            overwrite=overwrite,
        ):
            last_progress = progress
            typer.echo(
                f"wrote {progress.rows_written}/{progress.total_rows} -> "
                f"{progress.target_path}"
            )
        if last_progress is None:
            typer.echo("no rows written")
    except Exception as exc:
        _fail(str(exc), debug=False, exc=exc)


@app.command()
def hardware(
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
) -> None:
    """显示当前 GPU/NPU 硬件状态。"""
    snapshots = read_all_hardware()
    if json_output:
        _echo_json([snapshot.to_dict() for snapshot in snapshots])
        return

    table = Table(title="DanLing Hardware")
    table.add_column("设备")
    table.add_column("利用率")
    table.add_column("显存")
    for snapshot in snapshots:
        table.add_row(
            f"{snapshot.device_type}:{snapshot.device_id or '-'} {snapshot.name or ''}".strip(),
            f"{snapshot.util_percent:.0f}%" if snapshot.util_percent is not None else "-",
            (
                f"{snapshot.memory_used_mb / 1024:.1f}/{snapshot.memory_total_mb / 1024:.1f}G"
                if snapshot.memory_used_mb is not None and snapshot.memory_total_mb is not None
                else "-"
            ),
        )
    if not snapshots:
        table.add_row("unknown", "-", "-")
    _get_console().print(table)


@app.command()
def doctor(
    path: Annotated[Path | None, typer.Argument(help="可选训练日志路径")] = None,
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    no_hardware: Annotated[bool, typer.Option("--no-hardware", help="跳过硬件读取")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """环境检查；提供 PATH 时输出训练诊断。"""
    if path is None:
        checks = _environment_checks()
        if json_output:
            _echo_json(checks)
        else:
            for item in checks:
                typer.echo(f"{item['name']}: {item['value']}")
        return

    cfg = _load_config_or_exit(config, debug)
    history = _read_history_or_exit(path, source, debug)
    hardware_snapshots = [] if no_hardware else read_all_hardware()
    diagnostics = diagnose(history, hardware_snapshots, cfg)
    if json_output:
        _echo_json([item.to_dict() for item in diagnostics])
    else:
        for item in diagnostics:
            typer.echo(f"[{item.severity.value}] {item.code}: {item.title} - {item.message}")


@app.command()
def init() -> None:
    """在当前目录生成 `danling.yaml` 示例配置。"""
    path = Path.cwd() / "danling.yaml"
    if path.exists():
        typer.echo("danling.yaml already exists")
        return
    path.write_text(default_config_text(), encoding="utf-8")
    typer.echo(f"created {path}")


@app.command()
def reset(
    yes: Annotated[bool, typer.Option("--yes", help="确认删除本地状态")] = False,
) -> None:
    """删除 `~/.danling/state.json`。"""
    if not yes and not typer.confirm("Delete ~/.danling/state.json?"):
        raise typer.Exit(code=1)
    path = JsonStateStore().path
    if path.exists():
        path.unlink()
    typer.echo("state reset")


@config_app.command("show")
def config_show(
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
) -> None:
    """显示当前配置。"""
    cfg = load_config(str(config) if config is not None else None)
    if json_output:
        _echo_json(cfg.to_dict())
    else:
        table = _metric_table(cfg.to_dict(), title="DanLing Config")
        _get_console().print(table)


@config_app.command("tui")
def config_tui(
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
) -> None:
    """编辑 danling.yaml 配置；Ctrl+S 保存，q 退出。"""
    if importlib.util.find_spec("textual") is None:
        typer.echo("Textual config UI requires: pip install danling[tui]")
        raise typer.Exit(code=1)
    from danling.renderers.config_tui import create_config_app

    create_config_app(config).run()


@app.command()
def tui(
    path: Annotated[Path, typer.Argument(help="训练日志目录或文件路径")],
    source: Annotated[str, typer.Option("--source", help=SOURCE_HELP)] = "auto",
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    interval: Annotated[float, typer.Option("--interval", help="刷新间隔秒数")] = 2.0,
    no_hardware: Annotated[bool, typer.Option("--no-hardware", help="跳过硬件读取")] = False,
) -> None:
    """启动可选 Textual 全屏监控 TUI。"""
    if importlib.util.find_spec("textual") is None:
        typer.echo("Textual UI requires: pip install danling[tui]")
        raise typer.Exit(code=1)
    from danling.renderers.textual_app import create_app

    create_app(path, source=source, interval=interval, config=config, no_hardware=no_hardware).run()


# --- Remote helpers ---

def _resolve_config_for_remote(profile: RemoteProfile, cli_config: Path | None) -> Path | None:
    """优先 CLI --config，其次 profile.config_path。"""
    if cli_config is not None:
        return cli_config
    if profile.config_path:
        return Path(profile.config_path)
    return None


def _get_profile_from_context(ctx: typer.Context, debug: bool) -> RemoteProfile:
    profile_name = (ctx.obj or {}).get("profile_name")
    if not profile_name:
        typer.echo("错误：请指定远程配置名称，如 danling remote A30 status", err=True)
        raise typer.Exit(code=1)
    return _load_remote_profile_or_exit(profile_name, debug)


# --- remote setup ---

@remote_app.command("setup")
def remote_setup(
    profile: Annotated[str, typer.Argument(help="远程配置名称")] = "default",
) -> None:
    """打开远程 SSH 密钥配置 TUI。"""
    if importlib.util.find_spec("textual") is None:
        typer.echo("Textual remote setup requires: pip install danling[tui]")
        raise typer.Exit(code=1)
    from danling.renderers.remote_setup_tui import create_remote_setup_app

    create_remote_setup_app(profile).run()


# --- remote state ---

@remote_app.command("state")
def remote_state(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 读取远程训练日志并输出聚合状态。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            current = monitor.read_state()
        if json_output:
            _echo_json(current.to_dict())
        else:
            _get_console().print(render_state_panel(current))
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)


# --- remote status ---

@remote_app.command("status")
def remote_status(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 读取远程训练日志并渲染一次状态面板。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            current = monitor.read_state()
        _get_console().print(render_state_panel(current))
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)


# --- remote statusline ---

@remote_app.command("statusline")
def remote_statusline(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 读取远程训练日志并输出单行状态。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            current = monitor.read_state()
        typer.echo(render_statusline(current))
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)


# --- remote watch ---

@remote_app.command("watch")
def remote_watch(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    interval: Annotated[float | None, typer.Option("--interval", help="刷新间隔秒数")] = None,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 循环刷新远程训练状态面板。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        cfg = _load_config_or_exit(config_path, debug)
    except typer.Exit:
        raise
    except Exception:
        cfg = DanLingConfig()

    refresh_interval = interval if interval is not None else cfg.watch_interval
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            with Live(refresh_per_second=4, screen=False) as live:
                while True:
                    live.update(render_state_panel(monitor.read_state()))
                    time.sleep(refresh_interval)
    except KeyboardInterrupt:
        typer.echo("DanLing remote watch stopped.")
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)


# --- remote tui ---

@remote_app.command("tui")
def remote_tui(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    interval: Annotated[float, typer.Option("--interval", help="刷新间隔秒数")] = 2.0,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 启动远程训练 Textual 全屏监控 TUI。"""
    if importlib.util.find_spec("textual") is None:
        typer.echo("Textual UI requires: pip install danling[tui]")
        raise typer.Exit(code=1)
    from danling.renderers.textual_app import create_app

    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            create_app(
                Path(profile.remote_path),
                source="csv",
                interval=interval,
                config=config_path,
                state_provider=monitor.read_state,
                display_label=monitor.display_label,
            ).run()
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)


# --- remote hardware ---

@remote_app.command("hardware")
def remote_hardware(
    ctx: typer.Context,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 读取远程 NVIDIA GPU 状态。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        snapshots = read_remote_nvidia_hardware(
            profile.to_options(
                ssh_options=tuple(ssh_option or ()),
                sync_timeout=sync_timeout,
            )
        )
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)

    if json_output:
        _echo_json([snapshot.to_dict() for snapshot in snapshots])
        return

    table = Table(title=f"DanLing Remote Hardware — {profile.name}")
    table.add_column("设备")
    table.add_column("利用率")
    table.add_column("显存")
    table.add_column("温度")
    table.add_column("功耗")
    if not snapshots:
        table.add_row("unknown", "-", "-", "-", "-")
    for snapshot in snapshots:
        memory = "-"
        if snapshot.memory_used_mb is not None and snapshot.memory_total_mb is not None:
            memory = f"{snapshot.memory_used_mb / 1024:.1f}/{snapshot.memory_total_mb / 1024:.1f}G"
        table.add_row(
            f"{snapshot.device_type}:{snapshot.device_id or '-'} {snapshot.name or ''}".strip(),
            f"{snapshot.util_percent:.0f}%" if snapshot.util_percent is not None else "-",
            memory,
            f"{snapshot.temperature_c:.0f}C" if snapshot.temperature_c is not None else "-",
            f"{snapshot.power_w:.0f}W" if snapshot.power_w is not None else "-",
        )
    _get_console().print(table)


# --- remote doctor ---

@remote_app.command("doctor")
def remote_doctor(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 对远程训练进行诊断分析。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            current = monitor.read_state()
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)

    diagnostics = current.events if current.events else []
    if json_output:
        _echo_json([e.to_dict() for e in diagnostics])
        return

    if not diagnostics:
        _get_console().print("[green]未发现训练异常[/green]")
        return

    table = Table(title="远程训练诊断报告")
    table.add_column("严重程度")
    table.add_column("事件")
    for event in diagnostics:
        sev_style = {
            "info": "blue",
            "warning": "yellow",
            "error": "red",
            "critical": "bold red",
        }.get(event.severity.value, "")
        table.add_row(f"[{sev_style}]{event.severity.value}[/{sev_style}]", event.message)
    _get_console().print(table)


# --- remote inspect ---

@remote_app.command("inspect")
def remote_inspect(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="配置文件路径")] = None,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    ssh_option: Annotated[
        list[str] | None,
        typer.Option("--ssh-option", help="传给 ssh 的 -o 选项，可重复"),
    ] = None,
    remote_hardware: Annotated[
        bool,
        typer.Option("--remote-hardware/--no-remote-hardware", help="读取远程 NVIDIA GPU 状态"),
    ] = True,
    sync_timeout: Annotated[float, typer.Option("--sync-timeout", help="SSH 超时秒数")] = 10.0,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """通过 SSH 检查远程训练日志并输出最新指标快照。"""
    try:
        profile = _get_profile_from_context(ctx, debug)
        config_path = _resolve_config_for_remote(profile, config)
        with RemoteTrainingMonitor(
            profile,
            config_path,
            ssh_options=tuple(ssh_option or ()),
            remote_hardware=remote_hardware,
            sync_timeout=sync_timeout,
        ) as monitor:
            current = monitor.read_state()
    except (RemoteCommandError, RuntimeError, ValueError) as exc:
        _fail(str(exc), debug=debug, exc=exc)

    metric = current.metric
    if metric is None:
        _get_console().print("[yellow]未读取到训练指标[/yellow]")
        return

    if json_output:
        _echo_json(metric.to_dict())
        return

    data = metric.to_dict()
    _get_console().print(_metric_table(data, "远程训练指标"))


# --- remote config show/tui ---

@remote_config_app.command("show")
def remote_config_show(
    ctx: typer.Context,
    json_output: Annotated[bool, typer.Option("--json/--no-json", help="以 JSON 格式输出")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """显示远程 profile 关联的配置文件。"""
    from dataclasses import asdict

    profile = _get_profile_from_context(ctx, debug)
    config_path = _resolve_config_for_remote(profile, None)
    cfg = _load_config_or_exit(config_path, debug)

    if json_output:
        _echo_json(asdict(cfg))
        return

    data = {k: str(v) for k, v in asdict(cfg).items()}
    _get_console().print(_metric_table(data, f"DanLing 配置 ({profile.name})"))


@remote_config_app.command("tui")
def remote_config_tui(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="配置文件路径（覆盖 profile 默认）"),
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="显示调试 traceback")] = False,
) -> None:
    """编辑远程 profile 关联的配置文件。"""
    if importlib.util.find_spec("textual") is None:
        typer.echo("Textual UI requires: pip install danling[tui]")
        raise typer.Exit(code=1)
    from danling.renderers.config_tui import create_config_app

    profile = _get_profile_from_context(ctx, debug)
    config_path = _resolve_config_for_remote(profile, config)
    create_config_app(config_path).run()


def _load_remote_profile_or_exit(profile: str, debug: bool):
    try:
        return load_remote_profile(profile)
    except KeyError as exc:
        _fail(
            f"远程配置不存在: {profile}. Run: danling remote setup {profile}",
            debug=debug,
            exc=exc,
        )


def _build_state_for_path(
    path: Path,
    source: str,
    config_path: Path | None,
    no_hardware: bool,
    debug: bool,
) -> DanLingState:
    cfg = _load_config_or_exit(config_path, debug)
    history = _read_history_or_exit(path, source, debug)
    hardware_snapshots = [] if no_hardware else read_all_hardware()
    persisted = JsonStateStore().load()
    return build_state(history, hardware_snapshots, cfg, persisted=persisted)


def _read_history_or_exit(path: Path, source: str, debug: bool) -> list:
    try:
        return read_history(str(path), source=source)
    except Exception as exc:
        _fail(str(exc), debug=debug, exc=exc)


def _load_config_or_exit(config_path: Path | None, debug: bool) -> DanLingConfig:
    try:
        return load_config(str(config_path) if config_path is not None else None)
    except Exception as exc:
        _fail(str(exc), debug=debug, exc=exc)


def _update_store(path: Path, state: DanLingState) -> None:
    store = JsonStateStore()
    data = store.load()
    previous_updates = int(data.get("total_updates", 0) or 0)
    total_updates = previous_updates + 1
    data.pop("level", None)

    metric = state.metric
    if metric is not None and metric.score is not None:
        best_score = data.get("best_score")
        if not isinstance(best_score, (int, float)) or metric.score > float(best_score):
            data["best_score"] = metric.score
            data["best_score_name"] = metric.score_name
    if state.pet_mood == PetMood.failed:
        data["failed_runs"] = int(data.get("failed_runs", 0) or 0) + 1

    data.update(
        {
            "pet_name": state.pet_name,
            "total_updates": total_updates,
            "best_loss": _best_loss(data.get("best_loss"), metric.train_loss if metric else None),
        }
    )
    run_id = str(path.resolve())
    runs = data.setdefault("runs", {})
    runs[run_id] = {
        "run_path": run_id,
        "last_epoch": metric.epoch if metric else None,
        "best_score": data.get("best_score"),
        "best_loss": data.get("best_loss"),
        "last_seen_timestamp": metric.timestamp if metric else None,
        "events_count": len(state.events),
    }
    try:
        store.save(data)
    except OSError:
        return


def _best_loss(old: object, new: float | None) -> float | None:
    if new is None:
        return old if isinstance(old, (int, float)) else None
    if not isinstance(old, (int, float)):
        return new
    return min(float(old), new)


def _environment_checks() -> list[dict[str, str]]:
    from danling import __version__

    state_dir = Path.home() / ".danling"
    writable = True
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        writable = False
    return [
        {"name": "Python", "value": sys.version.split()[0]},
        {"name": "danling", "value": __version__},
        {"name": "nvidia-smi", "value": "yes" if shutil.which("nvidia-smi") else "no"},
        {"name": "npu-smi", "value": "yes" if shutil.which("npu-smi") else "no"},
        {"name": "~/.danling writable", "value": "yes" if writable else "no"},
    ]


def _metric_table(data: dict, title: str) -> Table:
    table = Table(title=title)
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue
        table.add_row(str(key), "-" if value is None else str(value))
    return table


def _echo_json(data) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _fail(message: str, debug: bool, exc: Exception | None = None):
    if debug and exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    else:
        typer.echo(f"错误：{message}", err=True)
    raise typer.Exit(code=1)
