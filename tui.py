"""Blackwell Inference Testbench — TUI Dashboard.

Usage:
    python tui.py          # or: make tui
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from tui.results_tab import ResultsTab
from tui.run_tab import RunTab
from tui.daemon_tab import DaemonTab


class BenchApp(App):
    """Main Textual application."""

    TITLE = "Blackwell Inference Testbench"
    SUB_TITLE = "RTX PRO 6000 · 96 GB"
    CSS_PATH = "tui/styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("e", "screenshot", "Export SVG"),
        Binding("f1", "help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="results"):
            with TabPane("Results", id="results"):
                yield ResultsTab()
            with TabPane("Run Benchmark", id="run"):
                yield RunTab()
            with TabPane("Daemon Status", id="daemon"):
                yield DaemonTab()
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh all tabs."""
        for tab in self.query(ResultsTab):
            tab.refresh_data()
        for tab in self.query(DaemonTab):
            tab.refresh_data()

    def action_screenshot(self) -> None:
        self.save_screenshot()
        self.notify("Screenshot saved.")

    def action_help(self) -> None:
        HELP = (
            "◀▶  Change output tokens    ▲▼  Change prompt tokens\n"
            "[m]  Cycle metric            [s]  Toggle scorecard\n"
            "[r]  Refresh data            [e]  Export SVG\n"
            "[q]  Quit                    [F1] This help"
        )
        self.notify(HELP, title="Keybindings", timeout=10)


if __name__ == "__main__":
    app = BenchApp()
    app.run()
