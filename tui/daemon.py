"""Daemon management — spawn, track, and query background benchmark jobs."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DAEMON_DIR = Path.home() / ".cache" / "inference-bench" / "daemons"


def _ensure_dir() -> None:
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DaemonJob:
    job_id: str
    benchmark: str
    label: str
    started_at: str
    status: str = "running"       # running | completed | failed
    finished_at: str | None = None
    exit_code: int | None = None
    result_files: list[str] = field(default_factory=list)

    @property
    def pid_file(self) -> Path:
        return DAEMON_DIR / f"{self.job_id}.pid"

    @property
    def log_file(self) -> Path:
        return DAEMON_DIR / f"{self.job_id}.log"

    @property
    def meta_file(self) -> Path:
        return DAEMON_DIR / f"{self.job_id}.meta.json"

    def is_alive(self) -> bool:
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, PermissionError):
            return False

    def save_meta(self) -> None:
        _ensure_dir()
        data = {
            "job_id": self.job_id,
            "benchmark": self.benchmark,
            "label": self.label,
            "started_at": self.started_at,
            "status": self.status,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "result_files": self.result_files,
        }
        self.meta_file.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_meta(cls, path: Path) -> DaemonJob:
        data = json.loads(path.read_text())
        return cls(**data)

    def kill(self) -> bool:
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.status = "failed"
            self.finished_at = datetime.now().isoformat()
            self.exit_code = -9
            self.save_meta()
            return True
        except (ProcessLookupError, ValueError, PermissionError):
            return False

    def tail_log(self, lines: int = 40) -> str:
        if not self.log_file.exists():
            return "(no log yet)"
        text = self.log_file.read_text()
        return "\n".join(text.splitlines()[-lines:])


# ---------------------------------------------------------------------------
# Spawn & discovery
# ---------------------------------------------------------------------------

def spawn_daemon(
    benchmark: str,
    label: str = "",
    label_large: str = "",
    label_small: str = "",
    project_dir: str | Path = ".",
) -> DaemonJob:
    """Launch a benchmark as a nohup background process."""
    _ensure_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = label or f"{label_large}+{label_small}" or "all"
    job_id = f"{benchmark}_{slug}_{ts}"

    # Build the make command
    if benchmark == "co-deploy":
        cmd_parts = [f"make {benchmark}"]
        if label_large:
            cmd_parts.append(f"LABEL_LARGE={label_large}")
        if label_small:
            cmd_parts.append(f"LABEL_SMALL={label_small}")
    else:
        cmd_parts = [f"make {benchmark}"]
        if label:
            cmd_parts.append(f"LABEL={label}")
    make_cmd = " ".join(cmd_parts)

    job = DaemonJob(
        job_id=job_id,
        benchmark=benchmark,
        label=slug,
        started_at=datetime.now().isoformat(),
        status="running",
    )

    log_path = job.log_file
    pid_path = job.pid_file

    # Spawn via nohup
    shell_cmd = (
        f"cd {Path(project_dir).resolve()} && "
        f"nohup {make_cmd} > {log_path} 2>&1 & echo $! > {pid_path}"
    )
    subprocess.run(["bash", "-c", shell_cmd])
    job.save_meta()
    return job


def list_daemon_jobs() -> list[DaemonJob]:
    """Discover all daemon jobs, refreshing status from PID files."""
    _ensure_dir()
    jobs = []
    for meta_path in sorted(DAEMON_DIR.glob("*.meta.json"), reverse=True):
        try:
            job = DaemonJob.from_meta(meta_path)
        except Exception:
            continue
        # Reconcile status with PID
        if job.status == "running" and not job.is_alive():
            job.status = "completed"  # best guess; could parse log for errors
            job.finished_at = job.finished_at or datetime.now().isoformat()
            job.save_meta()
        jobs.append(job)
    return jobs
