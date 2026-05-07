"""Remote SSH-key setup TUI and testable helper logic."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from danling.remote import (
    RemoteOptions,
    RemoteProfile,
    RemoteProfileStore,
    build_ssh_command,
    read_remote_nvidia_hardware,
    save_remote_profile,
)

DEFAULT_REMOTE_KEY_NAME = "danling_remote_ed25519"


@dataclass(frozen=True)
class RemoteSetupValues:
    """String values collected from the remote setup TUI."""

    name: str = "default"
    host: str = ""
    remote_path: str = ""
    source: str = "csv"
    identity: str = ""
    port: str = ""
    sync_timeout: str = "10"
    config_path: str = ""


def default_identity_path() -> Path:
    return Path.home() / ".ssh" / DEFAULT_REMOTE_KEY_NAME


def validate_remote_setup_values(values: RemoteSetupValues) -> list[str]:
    errors: list[str] = []
    if not values.name.strip():
        errors.append("配置名称不能为空")
    if not values.host.strip():
        errors.append("SSH 主机不能为空")
    if not values.remote_path.strip():
        errors.append("远程日志目录不能为空")
    if _normalize_source(values.source) not in {"csv", "ultralytics-log"}:
        errors.append("数据源必须是 csv 或 ultralytics-log")
    if not values.identity.strip():
        errors.append("SSH 私钥路径不能为空")
    if values.port.strip() and _parse_int(values.port) is None:
        errors.append("SSH 端口必须是整数")
    if values.sync_timeout.strip() and _parse_float(values.sync_timeout) is None:
        errors.append("同步超时必须是数字")
    return errors


def generate_ssh_keypair(identity: str | Path) -> Path:
    path = Path(identity).expanduser()
    if path.exists() and path.with_suffix(path.suffix + ".pub").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ssh-keygen failed: {detail}")
    return path


def check_ssh_key_connection(
    host: str,
    identity: str | Path,
    *,
    port: int | None = None,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    key_path = Path(identity).expanduser()
    if not key_path.is_file():
        return False, f"SSH 私钥文件不存在: {key_path}"
    from danling.remote import RemoteOptions

    try:
        result = subprocess.run(
            build_ssh_command(
                RemoteOptions(
                    host=host,
                    port=port,
                    identity=key_path,
                    sync_timeout=timeout,
                ),
                "true",
            ),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return False, f"连接失败: {exc}"
    if result.returncode == 0:
        return True, "连接成功"
    detail = (result.stderr or result.stdout).strip()
    return False, f"连接失败: {detail}"


def check_remote_gpu(
    host: str,
    identity: str | Path,
    *,
    port: int | None = None,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    key_path = Path(identity).expanduser()
    if not key_path.is_file():
        return False, f"SSH 私钥文件不存在: {key_path}"
    hardware = read_remote_nvidia_hardware(
        RemoteOptions(
            host=host,
            port=port,
            identity=key_path,
            sync_timeout=timeout,
        )
    )
    if not hardware:
        return False, "未读取到远程 NVIDIA GPU 信息"
    return True, "；".join(_gpu_summary(item) for item in hardware)


def save_remote_setup_values(
    values: RemoteSetupValues,
    *,
    store: RemoteProfileStore | None = None,
) -> RemoteProfile:
    errors = validate_remote_setup_values(values)
    if errors:
        raise ValueError("; ".join(errors))

    identity = Path(values.identity).expanduser()
    if not identity.is_file():
        raise ValueError(f"SSH 私钥文件不存在: {identity}")

    profile = RemoteProfile(
        name=values.name.strip(),
        host=values.host.strip(),
        remote_path=values.remote_path.strip(),
        identity=identity,
        port=_parse_int(values.port),
        source=_normalize_source(values.source),
        sync_timeout=_parse_float(values.sync_timeout) or 10.0,
        config_path=values.config_path.strip() or None,
    )
    return save_remote_profile(profile, store=store)


def create_remote_setup_app(profile_name: str = "default"):
    """Create Textual app for configuring SSH-key-backed remote profiles."""
    try:
        import textual  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Textual remote setup requires: pip install danling[tui]") from exc

    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, Input, Label, Static

    field_ids = {
        "name": "remote-profile-name",
        "host": "remote-host",
        "port": "remote-port",
        "remote_path": "remote-path",
        "source": "remote-source",
        "identity": "remote-identity",
        "sync_timeout": "remote-sync-timeout",
        "config_path": "remote-config-path",
    }
    default_values = RemoteSetupValues(
        name=profile_name,
        identity=str(default_identity_path()),
    )

    class RemoteSetupApp(App):
        BINDINGS = [
            ("ctrl+k", "generate_key", "生成密钥"),
            ("ctrl+t", "test", "测试连接"),
            ("ctrl+h", "test_gpu", "测试GPU"),
            ("ctrl+s", "save", "保存"),
            ("q", "quit", "退出"),
        ]
        CSS = """
        Screen {
            background: #061016;
            color: #d6fff8;
        }

        #remote-setup {
            height: 1fr;
        }

        #remote-left {
            width: 50%;
            padding: 1 2;
        }

        #remote-right {
            width: 50%;
            padding: 1 2;
        }

        #remote-help {
            border: round #16d9c5;
            padding: 1 2;
            background: #0b1720;
            height: 1fr;
        }

        #remote-message {
            padding: 0 2 1 2;
            color: #16d9c5;
        }

        Input {
            margin-bottom: 1;
        }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(
                "远程 SSH 密钥配置 | Ctrl+K 生成密钥 | Ctrl+T 测试 | Ctrl+H 测试GPU | Ctrl+S 保存",
                id="remote-message",
            )
            with Horizontal(id="remote-setup"):
                with VerticalScroll(id="remote-left"):
                    yield Label("配置名称")
                    yield Input(value=default_values.name, id=field_ids["name"])
                    yield Label("SSH 主机（如 user@server）")
                    yield Input(value=default_values.host, id=field_ids["host"])
                    yield Label("SSH 端口（可留空）")
                    yield Input(value=default_values.port, id=field_ids["port"])
                    yield Label("远程日志目录或文件")
                    yield Input(value=default_values.remote_path, id=field_ids["remote_path"])
                    yield Label("数据源（csv / ultralytics-log）")
                    yield Input(value=default_values.source, id=field_ids["source"])
                    yield Label("SSH 私钥路径")
                    yield Input(value=default_values.identity, id=field_ids["identity"])
                    yield Label("同步超时秒数")
                    yield Input(value=default_values.sync_timeout, id=field_ids["sync_timeout"])
                    yield Label("配置文件路径（可留空，如 ./danling.yaml）")
                    yield Input(value=default_values.config_path, id=field_ids["config_path"])
                with Vertical(id="remote-right"):
                    yield Static(_help_text(default_values), id="remote-help")
            yield Footer()

        def action_generate_key(self) -> None:
            message = self.query_one("#remote-message", Static)
            help_panel = self.query_one("#remote-help", Static)
            values = self._values()
            try:
                identity = generate_ssh_keypair(values.identity or default_identity_path())
            except RuntimeError as exc:
                message.update(str(exc))
                return
            self.query_one(f"#{field_ids['identity']}", Input).value = str(identity)
            message.update(f"已生成密钥: {identity}")
            help_panel.update(_help_text(self._values()))

        def action_test(self) -> None:
            message = self.query_one("#remote-message", Static)
            values = self._values()
            errors = validate_remote_setup_values(values)
            if errors:
                message.update("输入错误: " + "; ".join(errors))
                return
            ok, detail = check_ssh_key_connection(
                values.host.strip(),
                values.identity.strip(),
                port=_parse_int(values.port),
                timeout=_parse_float(values.sync_timeout) or 10.0,
            )
            message.update(detail if ok else f"测试失败: {detail}")

        def action_test_gpu(self) -> None:
            message = self.query_one("#remote-message", Static)
            values = self._values()
            errors = validate_remote_setup_values(values)
            if errors:
                message.update("输入错误: " + "; ".join(errors))
                return
            ok, detail = check_remote_gpu(
                values.host.strip(),
                values.identity.strip(),
                port=_parse_int(values.port),
                timeout=_parse_float(values.sync_timeout) or 10.0,
            )
            message.update(f"远程GPU: {detail}" if ok else f"GPU测试失败: {detail}")

        def action_save(self) -> None:
            message = self.query_one("#remote-message", Static)
            try:
                profile = save_remote_setup_values(self._values())
            except ValueError as exc:
                message.update(f"输入错误: {exc}")
                return
            message.update(
                f"已保存远程配置: {profile.name} | 运行 danling remote tui {profile.name}"
            )

        def on_input_changed(self, _event) -> None:
            self.query_one("#remote-help", Static).update(_help_text(self._values()))

        def _values(self) -> RemoteSetupValues:
            return RemoteSetupValues(
                name=self.query_one(f"#{field_ids['name']}", Input).value,
                host=self.query_one(f"#{field_ids['host']}", Input).value,
                port=self.query_one(f"#{field_ids['port']}", Input).value,
                remote_path=self.query_one(f"#{field_ids['remote_path']}", Input).value,
                source=self.query_one(f"#{field_ids['source']}", Input).value,
                identity=self.query_one(f"#{field_ids['identity']}", Input).value,
                sync_timeout=self.query_one(f"#{field_ids['sync_timeout']}", Input).value,
                config_path=self.query_one(f"#{field_ids['config_path']}", Input).value,
            )

    return RemoteSetupApp()


def _help_text(values: RemoteSetupValues) -> str:
    identity = values.identity.strip() or str(default_identity_path())
    public_key = f"{identity}.pub"
    host = values.host.strip() or "user@server"
    return "\n".join(
        [
            "SSH 密钥配置说明",
            "",
            "1. Ctrl+K 生成 DanLing 专用密钥。",
            "2. 在终端执行以下命令把公钥放到服务器：",
            f"   ssh-copy-id -i {public_key} {host}",
            "3. 回到这里按 Ctrl+T 测试无密码登录。",
            "4. 测试成功后按 Ctrl+S 保存。",
            "5. 可按 Ctrl+H 检查远程 NVIDIA GPU 信息。",
            "",
            "监控阶段会强制使用:",
            "   ssh -i <key> -o BatchMode=yes ...",
            "",
            "远程数据源支持 csv 和 ultralytics-log。",
            "长 epoch 的训练建议使用 ultralytics-log 读取实时控制台进度。",
        ]
    )


def _gpu_summary(device) -> str:
    name = getattr(device, "name", None) or ""
    device_id = getattr(device, "device_id", None) or "-"
    util = getattr(device, "util_percent", None)
    used = getattr(device, "memory_used_mb", None)
    total = getattr(device, "memory_total_mb", None)
    util_text = f"{util:.0f}%" if isinstance(util, (int, float)) else "-"
    memory_text = "-"
    if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
        memory_text = f"{used / 1024:.1f}/{total / 1024:.1f}G"
    return f"GPU:{device_id} {name} util={util_text} mem={memory_text}".strip()


def _parse_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _parse_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _normalize_source(source: str) -> str:
    value = source.strip()
    if value in {"log", "txt", "ultralytics_log"}:
        return "ultralytics-log"
    if value == "ultralytics":
        return "csv"
    return value or "csv"
