"""Results Tab — the Tuner.

Arrow-key navigable grid explorer with sorted bar chart, minimap, and scorecard.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static, Tree, DataTable, SelectionList

from rich.text import Text
from rich.table import Table as RichTable

import pandas as pd

from tui.data import (
    ResultSet,
    discover_results,
    load_decision_csv,
    load_split_load_summary,
    load_sanity_summary,
    get_concurrency_grid,
    get_models_in_decision,
    slice_decision,
    compute_win_board,
    scorecard,
    get_split_load_grid,
    slice_split_load,
    get_pairs_in_split_load,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

BAR_FULL = "█"
BAR_EMPTY = "░"
BAR_WIDTH = 30

METRICS = ["P95_ttft_ms", "P50_ttft_ms", "P95_itl_ms", "mean_throughput_tok_s"]
METRIC_LABELS = {
    "P95_ttft_ms": "P95 TTFT (ms)",
    "P50_ttft_ms": "P50 TTFT (ms)",
    "P95_itl_ms": "P95 ITL (ms)",
    "mean_throughput_tok_s": "Throughput (tok/s)",
}

# Co-deploy metrics
CO_METRICS = ["large_P95_ttft_ms", "large_P50_ttft_ms", "large_P95_itl_ms", "combined_throughput_tok_s"]
CO_METRIC_LABELS = {
    "large_P95_ttft_ms": "Large P95 TTFT (ms)",
    "large_P50_ttft_ms": "Large P50 TTFT (ms)",
    "large_P95_itl_ms": "Large P95 ITL (ms)",
    "combined_throughput_tok_s": "Combined Throughput (tok/s)",
}


def _short_model(name: str) -> str:
    """Shorten 'org/Model-Name' → 'Model-Name', truncate at 24 chars."""
    if "/" in name:
        name = name.split("/")[-1]
    if len(name) > 24:
        name = name[:22] + "…"
    return name


def _itl_badge(itl_val: float) -> Text:
    """Return a coloured ITL badge."""
    if pd.isna(itl_val):
        return Text("? itl", style="dim")
    val = round(itl_val, 0)
    if val < 50:
        return Text(f"✓ itl {val:.0f}ms", style="green")
    elif val <= 100:
        return Text(f"⚠ itl {val:.0f}ms", style="yellow")
    else:
        return Text(f"✗ itl {val:.0f}ms", style="red")


def _bar_color(value: float, metric: str) -> str:
    """Choose bar colour based on metric thresholds."""
    if "throughput" in metric:
        # Higher is better — reverse logic; just use cyan
        return "cyan"
    if "itl" in metric:
        if value < 50:
            return "green"
        elif value <= 100:
            return "yellow"
        return "red"
    # TTFT
    if value < 200:
        return "green"
    elif value <= 1000:
        return "yellow"
    return "red"


def _render_bar(value: float, max_val: float, metric: str) -> Text:
    """Render a proportional bar with colour."""
    if max_val <= 0:
        frac = 0
    else:
        frac = min(value / max_val, 1.0)
    filled = round(frac * BAR_WIDTH)
    empty = BAR_WIDTH - filled
    color = _bar_color(value, metric)
    bar = Text(BAR_FULL * filled, style=color) + Text(BAR_EMPTY * empty, style="dim")
    return bar


# ── Minimap renderer ─────────────────────────────────────────────────────────

# Assign a stable colour to each model across the minimap
_MODEL_COLORS = [
    "bright_green", "bright_cyan", "bright_magenta", "bright_yellow",
    "bright_red", "bright_blue", "green", "cyan", "magenta", "yellow",
]


def _minimap_rich(
    win_board: pd.DataFrame,
    prompts: list[int],
    outputs: list[int],
    cur_prompt: int,
    cur_output: int,
) -> RichTable:
    """Build a Rich Table showing the winner at each grid cell."""
    # Build a colour map
    winners = win_board["winner"].unique().tolist()
    cmap = {}
    for i, w in enumerate(winners):
        if w == "—":
            cmap[w] = "dim"
        else:
            cmap[w] = _MODEL_COLORS[i % len(_MODEL_COLORS)]

    table = RichTable(title="Winner Board", box=None, pad_edge=False, show_header=True)
    table.add_column("prompt\\out", style="bold", width=10)
    for o in outputs:
        table.add_column(str(o), justify="center", width=max(10, 18))

    for p in prompts:
        cells = []
        for o in outputs:
            row = win_board[(win_board["prompt"] == p) & (win_board["output"] == o)]
            if row.empty:
                cell_text = "—"
                style = "dim"
            else:
                cell_text = row.iloc[0]["winner"]
                style = cmap.get(cell_text, "white")
            # Highlight the currently selected cell
            if p == cur_prompt and o == cur_output:
                cell_text = f"[{cell_text}]"
                style += " bold reverse"
            cells.append(Text(cell_text, style=style))
        table.add_row(str(p), *cells)
    return table


# ══════════════════════════════════════════════════════════════════════════════
# ResultsTab Widget
# ══════════════════════════════════════════════════════════════════════════════

class ResultsPane(Vertical):
    """Focusable container for the main results area."""
    can_focus = True
    
    def on_click(self) -> None:
        self.focus()


class ResultsTab(Widget):
    """The main Results view with sidebar + tuner."""

    # Bindings bubble up from focused children (like ResultsPane)
    BINDINGS = [
        Binding("left", "prev_output", "◀ Output", show=True),
        Binding("right", "next_output", "▶ Output", show=True),
        Binding("up", "prev_prompt", "▲ Prompt", show=True),
        Binding("down", "next_prompt", "▼ Prompt", show=True),
        Binding("m", "cycle_metric", "Metric", show=True),
        Binding("s", "toggle_scorecard", "Scorecard", show=True),
    ]

    # Reactive state
    prompt_idx: reactive[int] = reactive(0)
    output_idx: reactive[int] = reactive(0)
    metric_idx: reactive[int] = reactive(0)
    show_scorecard: reactive[bool] = reactive(True)

    def __init__(self) -> None:
        super().__init__()
        self._result_sets: list[ResultSet] = []
        self._current_rs: ResultSet | None = None
        self._df: pd.DataFrame | None = None
        self._prompts: list[int] = []
        self._outputs: list[int] = []
        self._is_codeploy: bool = False

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="results-sidebar"):
                yield Label("📂 Benchmark Runs", classes="info-panel")
                yield Tree("Results", id="result-tree")
                yield Label("Model Filter", classes="info-panel")
                yield SelectionList(id="results-model-filter")
            with ResultsPane(id="results-main"):
                yield Static("", id="scorecard")
                with Horizontal(classes="tuner-controls"):
                    yield Label("prompt:")
                    yield Static("—", id="prompt-val", classes="value")
                    yield Label("output:")
                    yield Static("—", id="output-val", classes="value")
                    yield Label("metric:")
                    yield Static("—", id="metric-val", classes="value")
                    yield Static("", id="grid-pos")
                yield Static("Select a benchmark run from the sidebar.", id="bar-chart-area")
                yield Static("", id="minimap")


    def on_mount(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        """Rescan results/ and rebuild the sidebar tree."""
        self._result_sets = discover_results()
        tree: Tree = self.query_one("#result-tree", Tree)
        tree.clear()
        tree.root.expand()

        # Group by bench type
        groups: dict[str, list[ResultSet]] = {}
        for rs in self._result_sets:
            groups.setdefault(rs.bench_type, []).append(rs)

        bench_labels = {
            "concurrency_bench": "⚡ Concurrency Bench",
            "split_load": "🔀 Co-Deploy",
            "context_stress": "📏 Context Stress",
            "sanity_check": "✅ Sanity Check",
        }
        
        # Enforce specific order
        bench_order = ["concurrency_bench", "split_load", "context_stress", "sanity_check"]

        for bench_type in bench_order:
            if bench_type in groups:
                runs = groups[bench_type]
                branch = tree.root.add(bench_labels.get(bench_type, bench_type))
                for rs in runs:
                    branch.add_leaf(rs.label, data=rs)
        
        # Handle any unknown bench types
        for bench_type, runs in groups.items():
            if bench_type not in bench_order:
                branch = tree.root.add(bench_type)
                for rs in runs:
                    branch.add_leaf(rs.label, data=rs)

    # ── Sidebar selection ────────────────────────────────────────────────

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        rs: ResultSet | None = event.node.data
        if rs is None:
            return
        self._current_rs = rs
        self._is_codeploy = rs.bench_type == "split_load"

        # Clear/hide the filter for sanity checks or unknown types
        selector = self.query_one("#results-model-filter", SelectionList)
        selector.clear_options()

        if rs.bench_type == "sanity_check":
            self._show_sanity(rs)
            return

        # Load the appropriate CSV
        csv_path = rs.decision_csv or rs.summary_csv
        if csv_path is None:
            self.query_one("#bar-chart-area", Static).update("No CSV data found for this run.")
            return

        if self._is_codeploy:
            self._df = load_split_load_summary(csv_path)
            self._prompts, self._outputs = get_split_load_grid(self._df)
            
            # Populate filter
            pairs = get_pairs_in_split_load(self._df)
            options = []
            for lg, sm in pairs:
                label = f"{_short_model(lg)} + {_short_model(sm)}"
                value = f"{lg}|{sm}"
                options.append((label, value))
            selector.add_options(options)
            selector.select_all()

        else:
            self._df = load_decision_csv(csv_path)
            self._prompts, self._outputs = get_concurrency_grid(self._df)
            
            # Populate filter
            models = get_models_in_decision(self._df)
            options = [( _short_model(m), m ) for m in models]
            selector.add_options(options)
            selector.select_all()

        self.prompt_idx = 0
        self.output_idx = 0
        self.metric_idx = 0
        self._render_tuner()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        if event.selection_list.id == "results-model-filter":
            self._render_tuner()

    # ── Sanity view (simple table) ───────────────────────────────────────

    def _show_sanity(self, rs: ResultSet) -> None:
        path = rs.summary_csv
        if path is None:
            self.query_one("#bar-chart-area", Static).update("No summary CSV.")
            return
        df = load_sanity_summary(path)
        table = RichTable(title="Sanity Check", box=None)
        for col in df.columns:
            table.add_column(col, justify="right" if df[col].dtype != object else "left")
        for _, row in df.head(20).iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    cells.append(f"{val:.2f}")
                else:
                    cells.append(str(val))
            table.add_row(*cells)
        self.query_one("#bar-chart-area", Static).update(table)
        self.query_one("#minimap", Static).update("")
        self.query_one("#scorecard", Static).update("")
        for wid in ("prompt-val", "output-val", "metric-val", "grid-pos"):
            self.query_one(f"#{wid}", Static).update("—")

    # ── Tuner rendering ──────────────────────────────────────────────────

    def _render_tuner(self) -> None:
        if self._df is None or not self._prompts or not self._outputs:
            return

        # 1. Apply Model Filter
        selector = self.query_one("#results-model-filter", SelectionList)
        selected_values = set(selector.selected)
        
        if not selected_values:
            filtered_df = pd.DataFrame(columns=self._df.columns) # Empty
        elif self._is_codeploy:
            # Vectorized filter for pairs
            mask = (self._df["large_model"] + "|" + self._df["small_model"]).isin(selected_values)
            filtered_df = self._df[mask].copy()
        else:
            filtered_df = self._df[self._df["model"].isin(selected_values)].copy()

        cur_prompt = self._prompts[self.prompt_idx % len(self._prompts)]
        cur_output = self._outputs[self.output_idx % len(self._outputs)]

        metrics = CO_METRICS if self._is_codeploy else METRICS
        metric_labels = CO_METRIC_LABELS if self._is_codeploy else METRIC_LABELS
        metric = metrics[self.metric_idx % len(metrics)]

        # Update controls
        self.query_one("#prompt-val", Static).update(f"◀ {cur_prompt} ▶")
        self.query_one("#output-val", Static).update(f"◀ {cur_output} ▶")
        self.query_one("#metric-val", Static).update(metric_labels.get(metric, metric))

        total_cells = len(self._prompts) * len(self._outputs)
        cell_num = self.prompt_idx * len(self._outputs) + self.output_idx + 1
        self.query_one("#grid-pos", Static).update(f"[{cell_num}/{total_cells}]")

        # Slice data for current cell from FILTERED dataframe
        if self._is_codeploy:
            sliced = slice_split_load(filtered_df, cur_prompt, cur_output, metric)
        else:
            sliced = slice_decision(filtered_df, cur_prompt, cur_output, metric)

        # ── Bar chart ────────────────────────────────────────────────────
        if sliced.empty:
            self.query_one("#bar-chart-area", Static).update(
                "No data for this (prompt, output) combination (or all models filtered out)."
            )
        else:
            higher_better = "throughput" in metric
            if higher_better:
                sliced = sliced.sort_values(metric, ascending=False).reset_index(drop=True)

            max_val = sliced[metric].max()
            if max_val <= 0:
                max_val = 1

            # Build rich output line-by-line
            lines = Text()
            header_text = metric_labels.get(metric, metric)
            direction = "higher is better" if higher_better else "lower is better"
            lines.append(f"{header_text} — {direction}\n\n", style="bold")

            for _, row in sliced.iterrows():
                value = row[metric]
                if self._is_codeploy:
                    name = _short_model(row["large_model"]) + " + " + _short_model(row["small_model"])
                    itl_col = "small_P95_itl_ms"
                else:
                    name = _short_model(row["model"])
                    itl_col = "P95_itl_ms"

                # Pad name to 28 chars
                padded = f"{name:<28}"
                lines.append(padded)
                lines.append_text(_render_bar(value, max_val, metric))
                lines.append(f"  {value:>8.0f}  ")

                if itl_col in row.index:
                    lines.append_text(_itl_badge(row[itl_col]))

                lines.append("\n")

            self.query_one("#bar-chart-area", Static).update(lines)

        # ── Minimap ──────────────────────────────────────────────────────
        if not self._is_codeploy and self._df is not None:
            # Use filtered_df for minimap/scorecard to reflect the filter
            if filtered_df.empty:
                self.query_one("#minimap", Static).update("No data selected.")
                self.query_one("#scorecard", Static).update("")
            else:
                win_board = compute_win_board(filtered_df, metric)
                minimap = _minimap_rich(
                    win_board, self._prompts, self._outputs, cur_prompt, cur_output
                )
                self.query_one("#minimap", Static).update(minimap)

                # ── Scorecard ────────────────────────────────────────────────
                if self.show_scorecard:
                    sc = scorecard(filtered_df, metric)
                    parts = []
                    for model, wins, total in sc:
                        parts.append(f"{model} wins {wins}/{total}")
                    sc_text = "Scorecard: " + " │ ".join(parts) if parts else ""
                    self.query_one("#scorecard", Static).update(
                        Text(sc_text, style="bold")
                    )
                else:
                    self.query_one("#scorecard", Static).update("")
        else:
            self.query_one("#minimap", Static).update("")
            self.query_one("#scorecard", Static).update("")

    # ── Key actions ──────────────────────────────────────────────────────

    def action_prev_output(self) -> None:
        if self._outputs:
            self.output_idx = (self.output_idx - 1) % len(self._outputs)
            self._render_tuner()

    def action_next_output(self) -> None:
        if self._outputs:
            self.output_idx = (self.output_idx + 1) % len(self._outputs)
            self._render_tuner()

    def action_prev_prompt(self) -> None:
        if self._prompts:
            self.prompt_idx = (self.prompt_idx - 1) % len(self._prompts)
            self._render_tuner()

    def action_next_prompt(self) -> None:
        if self._prompts:
            self.prompt_idx = (self.prompt_idx + 1) % len(self._prompts)
            self._render_tuner()

    def action_cycle_metric(self) -> None:
        metrics = CO_METRICS if self._is_codeploy else METRICS
        self.metric_idx = (self.metric_idx + 1) % len(metrics)
        self._render_tuner()

    def action_toggle_scorecard(self) -> None:
        self.show_scorecard = not self.show_scorecard
        self._render_tuner()
