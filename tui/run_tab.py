"""Run Benchmark Tab — form-based benchmark launcher."""
from __future__ import annotations

import asyncio
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Button, Label, RichLog, Select, Static, RadioButton, RadioSet, SelectionList

from tui.data import load_models, models_by_role
from tui.daemon import spawn_daemon

PROJECT_DIR = Path(__file__).resolve().parent.parent


class RunTab(Widget):
    """Form to launch benchmarks."""

    def compose(self) -> ComposeResult:
        models = load_models()
        # "All Models" as the first option, value="all"
        selection_items = [("All Models", "all")] + [(m["label"], m["label"]) for m in models]
        
        large_labels = [("All", "")] + [(m["label"], m["label"]) for m in models_by_role("large")]
        small_labels = [("All", "")] + [(m["label"], m["label"]) for m in models_by_role("small")]

        yield Label("Launch Benchmark", classes="info-panel")

        yield Label("Benchmark type:")
        yield Select(
            [
                ("Concurrency Bench", "concurrency-bench"),
                ("Co-Deploy", "co-deploy"),
                ("Sanity Check", "sanity"),
            ],
            id="bench-type",
            value="concurrency-bench",
        )

        yield Label("Model filter (single-model benchmarks):")
        # Use SelectionList for multiple choices
        yield SelectionList(*selection_items, id="model-filter")

        yield Label("Large model (co-deploy only):")
        yield Select(large_labels, id="large-model", value="")

        yield Label("Small model (co-deploy only):")
        yield Select(small_labels, id="small-model", value="")

        yield Label("Run mode:")
        with RadioSet(id="run-mode"):
            yield RadioButton("Foreground (stream logs)", value=True, id="fg")
            yield RadioButton("Daemon (background)", id="bg")

        yield Button("▶ Run", variant="primary", id="run-btn")
        yield RichLog(id="run-log", wrap=True, highlight=True, markup=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self._launch()

    def _launch(self) -> None:
        bench = self.query_one("#bench-type", Select).value
        
        # Handle multiple model selection
        selected_models = self.query_one("#model-filter", SelectionList).selected
        if "all" in selected_models or not selected_models:
            label = ""
        else:
            label = ",".join(selected_models)

        label_large = self.query_one("#large-model", Select).value
        label_small = self.query_one("#small-model", Select).value
        log = self.query_one("#run-log", RichLog)

        # Determine run mode
        is_daemon = self.query_one("#bg", RadioButton).value

        if is_daemon:
            self._launch_daemon(bench, label, label_large, label_small, log)
        else:
            self._launch_foreground(bench, label, label_large, label_small, log)

    def _launch_daemon(self, bench, label, label_large, label_small, log: RichLog) -> None:
        log.clear()
        job = spawn_daemon(
            benchmark=bench,
            label=label,
            label_large=label_large,
            label_small=label_small,
            project_dir=PROJECT_DIR,
        )
        log.write(f"[green]Daemon launched:[/green] {job.job_id}")
        log.write(f"Log: {job.log_file}")
        log.write("Switch to the Daemon Status tab to monitor progress.")
        self.app.notify(f"Daemon started: {job.job_id}", title="Benchmark")

    def _launch_foreground(self, bench, label, label_large, label_small, log: RichLog) -> None:
        # Build the make command
        if bench == "co-deploy":
            parts = ["make", bench]
            if label_large:
                parts.append(f"LABEL_LARGE={label_large}")
            if label_small:
                parts.append(f"LABEL_SMALL={label_small}")
        else:
            parts = ["make", bench]
            if label:
                # Pass the comma-separated list of labels
                parts.append(f"LABEL={label}")

        cmd = " ".join(parts)
        log.clear()
        log.write(f"[bold cyan]$ {cmd}[/bold cyan]\n")

        # Run asynchronously to keep TUI responsive
        self.run_worker(self._stream_command(cmd, log), exclusive=True)

    async def _stream_command(self, cmd: str, log: RichLog) -> None:
        """Run a command and stream output to the log widget."""
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            log.write(line.decode("utf-8", errors="replace").rstrip())

        exit_code = await process.wait()
        if exit_code == 0:
            log.write("\n[green bold]✓ Benchmark completed successfully.[/green bold]")
            self.app.notify("Benchmark completed!", title="Done")
        else:
            log.write(f"\n[red bold]✗ Benchmark failed (exit code {exit_code}).[/red bold]")
            self.app.notify(f"Benchmark failed (exit {exit_code})", title="Error", severity="error")

        # Trigger results refresh
        from tui.results_tab import ResultsTab
        for tab in self.app.query(ResultsTab):
            tab.refresh_data()
