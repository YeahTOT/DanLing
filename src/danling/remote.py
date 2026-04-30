"""SSH-key-backed remote training log polling and state building."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from danling.config import load_config
from danling.engine.state import build_state
from danling.hardware.nvidia import NVIDIA_SMI_QUERY_ARGS, parse_nvidia_smi_csv
from danling.models import DanLingState, HardwareSnapshot
from danling.readers.registry import read_history

REMOTE_PROFILE_PATH = Path.home() / ".danling" / "remote_profiles.json"
REMOTE_CSV_PATH_MARKER = "__DANLING_CSV_PATH__"
REMOTE_CSV_MTIME_MARKER = "__DANLING_CSV_MTIME__"


class RemoteCommandError(RuntimeError):
    """Raised when an SSH command cannot complete."""


class RemoteKeyRequired(RemoteCommandError):
    """Raised when remote monitoring is attempted without an SSH identity key."""


class RemoteLogNotFound(RemoteCommandError):
    """Raised when no supported remote log can be read."""


@dataclass(frozen=True)
class RemoteOptions:
    """Connection options for system OpenSSH commands."""

    host: str
    port: int | None = None
    identity: Path | None = None
    ssh_options: tuple[str, ...] = ()
    sync_timeout: float = 10.0


@dataclass(frozen=True)
class RemoteProfile:
    """Saved remote training target."""

    name: str
    host: str
    remote_path: str
    identity: Path
    port: int | None = None
    source: str = "csv"
    remote_hardware: bool = True
    sync_timeout: float = 10.0
    ssh_options: tuple[str, ...] = ()

    def to_options(
        self,
        *,
        ssh_options: tuple[str, ...] = (),
        sync_timeout: float | None = None,
    ) -> RemoteOptions:
        merged_options = (*self.ssh_options, *ssh_options)
        return RemoteOptions(
            host=self.host,
            port=self.port,
            identity=self.identity,
            ssh_options=merged_options,
            sync_timeout=self.sync_timeout if sync_timeout is None else sync_timeout,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["identity"] = str(self.identity)
        data["ssh_options"] = list(self.ssh_options)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemoteProfile:
        return cls(
            name=str(data["name"]),
            host=str(data["host"]),
            remote_path=str(data["remote_path"]),
            identity=Path(str(data["identity"])).expanduser(),
            port=_optional_int(data.get("port")),
            source=str(data.get("source") or "csv"),
            remote_hardware=bool(data.get("remote_hardware", True)),
            sync_timeout=float(data.get("sync_timeout", 10.0)),
            ssh_options=tuple(str(item) for item in data.get("ssh_options", [])),
        )


class RemoteProfileStore:
    """JSON store for remote profiles."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else REMOTE_PROFILE_PATH

    def load_all(self) -> dict[str, RemoteProfile]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        profiles = raw.get("profiles", raw)
        if not isinstance(profiles, dict):
            return {}
        result: dict[str, RemoteProfile] = {}
        for name, data in profiles.items():
            if isinstance(data, dict):
                try:
                    profile = RemoteProfile.from_dict({"name": name, **data})
                except (KeyError, TypeError, ValueError):
                    continue
                result[profile.name] = profile
        return result

    def save(self, profile: RemoteProfile) -> None:
        profiles = self.load_all()
        profiles[profile.name] = profile
        data = {"profiles": {name: item.to_dict() for name, item in profiles.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, self.path)

    def get(self, name: str) -> RemoteProfile:
        profiles = self.load_all()
        if name not in profiles:
            raise KeyError(name)
        return profiles[name]


class RemoteTrainingMonitor:
    """Poll a remote CSV log over SSH and build a DanLing state snapshot."""

    def __init__(
        self,
        remote_profile: RemoteProfile,
        config_path: Path | None,
        *,
        ssh_options: tuple[str, ...] = (),
        remote_hardware: bool | None = None,
        sync_timeout: float | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if remote_profile.source != "csv":
            raise ValueError("远程监控当前仅支持 Ultralytics results.csv")
        self.profile = remote_profile
        self.options = remote_profile.to_options(
            ssh_options=ssh_options,
            sync_timeout=sync_timeout,
        )
        self.config_path = config_path
        self.remote_hardware = (
            remote_profile.remote_hardware if remote_hardware is None else remote_hardware
        )
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="danling-remote-"))
        self._owns_cache = cache_dir is None

    @classmethod
    def from_profile(
        cls,
        remote_profile: RemoteProfile,
        config_path: Path | None,
        **kwargs,
    ) -> RemoteTrainingMonitor:
        return cls(remote_profile, config_path, **kwargs)

    @property
    def display_label(self) -> str:
        return f"{self.profile.name}: {self.profile.host}:{self.profile.remote_path}"

    def __enter__(self) -> RemoteTrainingMonitor:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._owns_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def read_state(self) -> DanLingState:
        local_path = read_remote_csv_to_cache(
            self.options,
            self.profile.remote_path,
            self.cache_dir,
        )
        config = load_config(str(self.config_path) if self.config_path is not None else None)
        history = read_history(str(local_path), source="csv")
        hardware = read_remote_nvidia_hardware(self.options) if self.remote_hardware else []
        return build_state(history, hardware, config)


def save_remote_profile(
    profile: RemoteProfile,
    *,
    store: RemoteProfileStore | None = None,
) -> RemoteProfile:
    target_store = store or RemoteProfileStore()
    target_store.save(profile)
    return profile


def load_remote_profile(
    name: str,
    *,
    store: RemoteProfileStore | None = None,
) -> RemoteProfile:
    target_store = store or RemoteProfileStore()
    return target_store.get(name)


def build_ssh_command(options: RemoteOptions, remote_command: str) -> list[str]:
    """Build a system `ssh` command that cannot fall back to password prompts."""
    if options.identity is None:
        raise RemoteKeyRequired("SSH key is required. Run: danling remote setup <profile>")
    return [
        "ssh",
        *_ssh_common_args(options),
        options.host,
        remote_command,
    ]


def read_remote_csv_to_cache(
    options: RemoteOptions,
    remote_path: str,
    cache_dir: Path,
) -> Path:
    """Read remote `results.csv` through SSH stdout and cache it as a local file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = _run_ssh(
        options,
        _remote_csv_read_command(remote_path),
        allowed_returncodes=(0, 2),
    )
    if result.returncode == 2:
        raise RemoteLogNotFound(
            f"No remote results.csv found at {options.host}:{remote_path}"
        )

    mtime, csv_text = _parse_remote_csv_output(result.stdout)
    if not csv_text.strip():
        raise RemoteLogNotFound(f"Remote results.csv is empty at {options.host}:{remote_path}")

    local_csv = cache_dir / "results.csv"
    local_csv.write_text(csv_text, encoding="utf-8")
    if mtime is not None:
        os.utime(local_csv, (mtime, mtime))
    return cache_dir


def read_remote_nvidia_hardware(options: RemoteOptions) -> list[HardwareSnapshot]:
    """Read remote NVIDIA GPU state through key-only SSH."""
    try:
        result = subprocess.run(
            build_ssh_command(options, _remote_nvidia_smi_command()),
            capture_output=True,
            text=True,
            timeout=options.sync_timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_nvidia_smi_csv(result.stdout)


def _remote_nvidia_smi_command() -> str:
    query_args = " ".join(shlex.quote(part) for part in NVIDIA_SMI_QUERY_ARGS[1:])
    return (
        'nvidia_smi="$(command -v nvidia-smi 2>/dev/null || true)"; '
        'if [ -z "$nvidia_smi" ] && [ -x /usr/bin/nvidia-smi ]; then '
        'nvidia_smi=/usr/bin/nvidia-smi; fi; '
        'if [ -z "$nvidia_smi" ] && [ -x /usr/local/cuda/bin/nvidia-smi ]; then '
        'nvidia_smi=/usr/local/cuda/bin/nvidia-smi; fi; '
        'if [ -z "$nvidia_smi" ]; then exit 127; fi; '
        f'"$nvidia_smi" {query_args}'
    )


def _run_ssh(
    options: RemoteOptions,
    remote_command: str,
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            build_ssh_command(options, remote_command),
            capture_output=True,
            text=True,
            timeout=options.sync_timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RemoteCommandError("Required command not found: ssh") from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteCommandError(
            f"SSH command timed out after {options.sync_timeout:g}s"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RemoteCommandError(f"SSH command failed: {exc}") from exc

    if result.returncode not in allowed_returncodes:
        detail = (result.stderr or result.stdout).strip()
        if result.returncode == 255:
            raise RemoteCommandError(
                f"SSH key authentication failed for {options.host}: {detail}. "
                "Run: danling remote setup"
            )
        raise RemoteCommandError(f"SSH command failed on {options.host}: {detail}")
    return result


def _remote_csv_read_command(remote_path: str) -> str:
    quoted = shlex.quote(remote_path)
    return (
        f"p={quoted}; "
        'if [ -f "$p" ]; then f="$p"; '
        'elif [ -f "$p/results.csv" ]; then f="$p/results.csv"; '
        "else exit 2; fi; "
        f'printf "{REMOTE_CSV_PATH_MARKER}%s\\n" "$f"; '
        f'printf "{REMOTE_CSV_MTIME_MARKER}"; '
        '(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || date +%s); '
        'cat "$f"'
    )


def _parse_remote_csv_output(output: str) -> tuple[float | None, str]:
    lines = output.splitlines(keepends=True)
    if len(lines) < 3:
        raise RemoteLogNotFound("Remote results.csv output was incomplete")

    path_line = lines[0].strip()
    mtime_line = lines[1].strip()
    if not path_line.startswith(REMOTE_CSV_PATH_MARKER):
        raise RemoteLogNotFound("Remote results.csv output did not include path marker")
    if not mtime_line.startswith(REMOTE_CSV_MTIME_MARKER):
        raise RemoteLogNotFound("Remote results.csv output did not include mtime marker")

    mtime_text = mtime_line.removeprefix(REMOTE_CSV_MTIME_MARKER)
    try:
        mtime = float(mtime_text)
    except ValueError:
        mtime = None
    return mtime, "".join(lines[2:])


def _ssh_common_args(options: RemoteOptions) -> list[str]:
    args: list[str] = []
    if options.port is not None:
        args.extend(["-p", str(options.port)])
    if options.identity is not None:
        args.extend(["-i", str(options.identity)])
    args.extend(["-o", "BatchMode=yes"])
    for option in options.ssh_options:
        if option.strip().lower().startswith("batchmode="):
            continue
        args.extend(["-o", option])
    return args


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)
