"""Daemon Status Tab — monitor, view logs, and kill background benchmark jobs."""
from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Button, DataTable, Label, RichLog, Static

from tui.daemon import list_daemon_jobs, DaemonJob


class DaemonTab(Widget):
    """Table of background benchmark jobs with actions."""

    BINDINGS = [
        Binding("v", "view_log", "View Log", show=True),
        Binding("k", "kill_job", "Kill Job", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._jobs: list[DaemonJob] = []

    def compose(self) -> ComposeResult:
        yield Label("🔄 Background Jobs", classes="info-panel")
        yield DataTable(id="daemon-table")
        yield Static("[v] View log  [k] Kill job  [r] Refresh", id="daemon-help")
        yield RichLog(id="daemon-log", wrap=True, highlight=True, markup=True)

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#daemon-table", DataTable)
        table.add_columns("Job ID", "Benchmark", "Model(s)", "Status", "Started", "Elapsed")
        table.cursor_type = "row"
        self.refresh_data()

    def refresh_data(self) -> None:
        self._jobs = list_daemon_jobs()
        table: DataTable = self.query_one("#daemon-table", DataTable)
        table.clear()

        for job in self._jobs:
            # Status with colour
            if job.status == "running":
                status = "🟢 running"
            elif job.status == "completed":
                status = "✅ completed"
            else:
                status = "❌ failed"

            # Elapsed
            try:
                started = datetime.fromisoformat(job.started_at)
                if job.status == "running":
                    elapsed = datetime.now() - started
                elif job.finished_at:
                    elapsed = datetime.fromisoformat(job.finished_at) - started
                else:
                    elapsed = datetime.now() - started
                elapsed_str = str(elapsed).split(".")[0]  # drop microseconds
            except Exception:
                elapsed_str = "—"

            table.add_row(
                job.job_id,
                job.benchmark,
                job.label,
                status,
                job.started_at[:19] if job.started_at else "—",
                elapsed_str,
            )

    def _selected_job(self) -> DaemonJob | None:
        table: DataTable = self.query_one("#daemon-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self._jobs):
            return None
        return self._jobs[table.cursor_row]

    def action_view_log(self) -> None:
        job = self._selected_job()
        log = self.query_one("#daemon-log", RichLog)
        log.clear()
        if job is None:
            log.write("No job selected.")
            return
        log.write(f"[bold]Log for {job.job_id}:[/bold]\n")
        log.write(job.tail_log(lines=50))

    def action_kill_job(self) -> None:
        job = self._selected_job()
        if job is None:
            self.app.notify("No job selected.", severity="warning")
            return
        if job.status != "running":
            self.app.notify("Job is not running.", severity="warning")
            return
        killed = job.kill()
        if killed:
            self.app.notify(f"Killed {job.job_id}", title="Job Killed")
        else:
            self.app.notify(f"Could not kill {job.job_id}", severity="error")
        self.refresh_data()
