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
    SweepGroup,
    discover_results,
    group_into_sweeps,
    load_decision_csv,
    load_merged_decision_csvs,
    load_split_load_summary,
    load_sanity_summary,
    load_stt_summary,
    load_stt_streaming_summary,
    load_mixed_co_deploy_summary,
    get_stt_concurrency_levels,
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

# TTFT variants the user can toggle between with 'm'
TTFT_METRICS = ["P95_ttft_ms", "P50_ttft_ms"]
TTFT_LABELS = {
    "P95_ttft_ms": "P95 TTFT (ms)",
    "P50_ttft_ms": "P50 TTFT (ms)",
}
THRU_METRIC = "mean_throughput_tok_s"
THRU_LABEL = "Throughput (tok/s)"

# Co-deploy equivalents
CO_TTFT_METRICS = ["large_P95_ttft_ms", "large_P50_ttft_ms"]
CO_TTFT_LABELS = {
    "large_P95_ttft_ms": "Large P95 TTFT (ms)",
    "large_P50_ttft_ms": "Large P50 TTFT (ms)",
}
CO_THRU_METRIC = "combined_throughput_tok_s"
CO_THRU_LABEL = "Combined Throughput (tok/s)"

# Bench types that get STT-specific rendering
STT_BENCH_TYPES = {
    "stt_sanity", "stt_concurrency_bench",
    "stt_streaming_sanity", "stt_streaming_bench",
}
STT_STREAMING_BENCH_TYPES = {
    "stt_streaming_sanity", "stt_streaming_bench",
}


def _short_model(name: str) -> str:
    """Shorten 'org/Model-Name' → 'Model-Name', truncate at 30 chars."""
    if "/" in name:
        name = name.split("/")[-1]
    if len(name) > 30:
        name = name[:28] + "…"
    return name


def _bar_color(value: float, metric: str) -> str:
    """Choose bar colour based on metric thresholds."""
    if "throughput" in metric:
        return "cyan"
    # WER — lower is better; green < 5%, yellow < 15%, red >= 15%
    if "wer" in metric:
        if value < 0.05:
            return "green"
        elif value <= 0.15:
            return "yellow"
        return "red"
    # RTF — lower is better; green < 0.5, yellow < 1.0, red >= 1.0
    if "rtf" in metric:
        if value < 0.5:
            return "green"
        elif value < 1.0:
            return "yellow"
        return "red"
    # TTFW / inter-delta / latency — treat like TTFT
    if value < 200:
        return "green"
    elif value <= 1000:
        return "yellow"
    return "red"


NAME_PAD = 30  # column width for model names in bar charts


def _render_chart(
    sliced: pd.DataFrame,
    metric: str,
    label: str,
    higher_better: bool,
    unit: str,
    fmt: str,
    name_fn=None,
) -> Text:
    """Render a single bar chart as Rich Text."""
    lines = Text()
    direction = "higher is better" if higher_better else "lower is better"
    lines.append(f"{label}  —  {direction}\n", style="bold")
    lines.append("\n")

    sorted_df = sliced.sort_values(
        metric, ascending=not higher_better
    ).reset_index(drop=True)

    max_val = sorted_df[metric].max()
    if max_val <= 0:
        max_val = 1

    for _, row in sorted_df.iterrows():
        val = row[metric]
        name = name_fn(row) if callable(name_fn) else _short_model(str(row.get("model", "")))
        padded = f"  {name:<{NAME_PAD}}"
        lines.append(padded)
        lines.append_text(_render_bar(val, max_val, metric))
        lines.append(f"  {val:{fmt}} {unit}\n")

    return lines


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
        Binding("m", "cycle_metric", "P95/P50", show=True),
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
                    yield Label("TTFT:")
                    yield Static("P95", id="metric-val", classes="value")
                    yield Static("", id="grid-pos")
                with Horizontal(id="charts-row"):
                    yield Static("Select a benchmark run from the sidebar.", id="chart-ttft")
                    yield Static("", id="chart-thru")
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
            "stt_sanity": "🎤 STT Sanity",
            "stt_concurrency_bench": "🎤 STT Bench",
            "stt_streaming_sanity": "🎤 STT Streaming Sanity",
            "stt_streaming_bench": "🎤 STT Streaming Bench",
            "mixed_co_deploy": "🎤+⚡ Mixed Co-Deploy",
            "sanity_check": "✅ Sanity Check",
        }
        
        # Enforce specific order
        bench_order = [
            "concurrency_bench", "split_load",
            "stt_sanity", "stt_concurrency_bench",
            "stt_streaming_sanity", "stt_streaming_bench",
            "mixed_co_deploy",
            "sanity_check",
        ]

        for bench_type in bench_order:
            if bench_type in groups:
                runs = groups[bench_type]
                branch = tree.root.add(bench_labels.get(bench_type, bench_type))

                if bench_type in ("concurrency_bench",):
                    sweeps = group_into_sweeps(runs)
                    for sweep in sweeps:
                        if len(sweep.result_sets) == 1:
                            branch.add_leaf(sweep.result_sets[0].label, data=sweep.result_sets[0])
                        else:
                            sweep_node = branch.add(sweep.label, data=sweep)
                            sweep_node.expand()
                            for rs in sweep.result_sets:
                                sweep_node.add_leaf(rs.label, data=rs)
                else:
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
        node_data = event.node.data
        if node_data is None:
            return

        # ── SweepGroup: merge all decision CSVs in the sweep ─────────
        if isinstance(node_data, SweepGroup):
            sweep: SweepGroup = node_data
            self._current_rs = None
            self._is_codeploy = False

            selector = self.query_one("#results-model-filter", SelectionList)
            selector.clear_options()

            csv_paths = sweep.decision_csvs
            if not csv_paths:
                self.query_one("#bar-chart-area", Static).update(
                    "No decision CSV data found for this sweep."
                )
                return

            self._df = load_merged_decision_csvs(csv_paths)
            self._prompts, self._outputs = get_concurrency_grid(self._df)

            models = get_models_in_decision(self._df)
            options = [(_short_model(m), m) for m in models]
            selector.add_options(options)
            selector.select_all()

            self.prompt_idx = 0
            self.output_idx = 0
            self.metric_idx = 0
            self._render_tuner()
            return

        # ── Single ResultSet ─────────────────────────────────────────
        rs: ResultSet = node_data
        self._current_rs = rs
        self._is_codeploy = rs.bench_type == "split_load"

        # Clear/hide the filter for sanity checks or unknown types
        selector = self.query_one("#results-model-filter", SelectionList)
        selector.clear_options()

        if rs.bench_type == "sanity_check":
            self._show_sanity(rs)
            return

        # ── STT bench types: dedicated renderer ──────────────────────
        if rs.bench_type in STT_BENCH_TYPES:
            self._show_stt(rs)
            return

        # ── Mixed co-deploy: dedicated renderer ──────────────────────
        if rs.bench_type == "mixed_co_deploy":
            self._show_mixed_co_deploy(rs)
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
        self.query_one("#chart-ttft", Static).update(table)
        self.query_one("#chart-thru", Static).update("")
        self.query_one("#minimap", Static).update("")
        self.query_one("#scorecard", Static).update("")
        for wid in ("prompt-val", "output-val", "metric-val", "grid-pos"):
            self.query_one(f"#{wid}", Static).update("—")

    # ── STT view (bar charts + summary table) ────────────────────────────

    def _show_stt(self, rs: ResultSet) -> None:
        """Render STT (offline or streaming) benchmark results.

        Shows:
          Left chart:  WER bar chart (lower is better) — one bar per concurrency level
          Right chart: RTF bar chart (lower is better)
                       OR for streaming: TTFW bar chart (lower is better)
          Below:       Full summary table with all metrics
        """
        path = rs.summary_csv
        if path is None:
            self.query_one("#chart-ttft", Static).update("No summary CSV found for this run.")
            self.query_one("#chart-thru", Static).update("")
            self.query_one("#minimap", Static).update("")
            self.query_one("#scorecard", Static).update("")
            for wid in ("prompt-val", "output-val", "metric-val", "grid-pos"):
                self.query_one(f"#{wid}", Static).update("—")
            return

        is_streaming = rs.bench_type in STT_STREAMING_BENCH_TYPES
        df = load_stt_streaming_summary(path) if is_streaming else load_stt_summary(path)

        if df.empty:
            self.query_one("#chart-ttft", Static).update("Summary CSV is empty.")
            self.query_one("#chart-thru", Static).update("")
            self.query_one("#minimap", Static).update("")
            self.query_one("#scorecard", Static).update("")
            for wid in ("prompt-val", "output-val", "metric-val", "grid-pos"):
                self.query_one(f"#{wid}", Static).update("—")
            return

        # Label for concurrency level in chart bars
        def _conc_label(row) -> str:
            level = row.get("concurrency_level", "?")
            n = row.get("n_requests", "?")
            return f"conc={level} (n={n})"

        # ── Left chart: WER ──────────────────────────────────────────────
        wer_col = "mean_wer"
        if wer_col in df.columns:
            wer_chart = _render_chart(
                df, wer_col, "Mean WER  —  lower is better",
                higher_better=False, unit="", fmt=">8.4f",
                name_fn=_conc_label,
            )
        else:
            wer_chart = Text("No WER data available.", style="dim")

        # ── Right chart: RTF (offline) or TTFW (streaming) ──────────────
        if is_streaming and "P95_ttfw_ms" in df.columns:
            right_chart = _render_chart(
                df, "P95_ttfw_ms", "P95 TTFW (ms)  —  lower is better",
                higher_better=False, unit="ms", fmt=">8.0f",
                name_fn=_conc_label,
            )
        elif "mean_rtf" in df.columns:
            right_chart = _render_chart(
                df, "mean_rtf", "Mean RTF  —  lower is better (< 1.0 = real-time)",
                higher_better=False, unit="", fmt=">8.4f",
                name_fn=_conc_label,
            )
        else:
            right_chart = Text("No RTF/TTFW data available.", style="dim")

        self.query_one("#chart-ttft", Static).update(wer_chart)
        self.query_one("#chart-thru", Static).update(right_chart)

        # ── Summary table below charts ───────────────────────────────────
        bench_label = "Streaming STT" if is_streaming else "STT"
        table = RichTable(
            title=f"{bench_label} Summary — {rs.model_tag or rs.label}",
            box=None,
        )
        for col in df.columns:
            table.add_column(col, justify="right" if df[col].dtype != object else "left")
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    if "wer" in col or "rtf" in col:
                        cells.append(f"{val:.4f}")
                    else:
                        cells.append(f"{val:.2f}")
                else:
                    cells.append(str(val))
            table.add_row(*cells)

        self.query_one("#minimap", Static).update(table)

        # ── Controls + scorecard: show bench type label ──────────────────
        model_name = df["model"].iloc[0] if "model" in df.columns else "?"
        levels = get_stt_concurrency_levels(df)
        self.query_one("#prompt-val", Static).update(f"{_short_model(str(model_name))}")
        self.query_one("#output-val", Static).update(f"levels: {levels}")
        self.query_one("#metric-val", Static).update(bench_label)
        self.query_one("#grid-pos", Static).update(f"[{len(df)} rows]")
        self.query_one("#scorecard", Static).update("")

    # ── Mixed co-deploy view ─────────────────────────────────────────────

    def _show_mixed_co_deploy(self, rs: ResultSet) -> None:
        """Render mixed co-deploy benchmark results.

        Shows:
          Left chart:  Text model P95 TTFT (lower is better)
          Right chart: STT model Mean WER (lower is better)
          Below:       Full summary table with all metrics
        """
        path = rs.summary_csv
        if path is None:
            self.query_one("#chart-ttft", Static).update("No summary CSV found for this run.")
            self.query_one("#chart-thru", Static).update("")
            self.query_one("#minimap", Static).update("")
            self.query_one("#scorecard", Static).update("")
            for wid in ("prompt-val", "output-val", "metric-val", "grid-pos"):
                self.query_one(f"#{wid}", Static).update("—")
            return

        df = load_mixed_co_deploy_summary(path)
        if df.empty:
            self.query_one("#chart-ttft", Static).update("Summary CSV is empty.")
            self.query_one("#chart-thru", Static).update("")
            self.query_one("#minimap", Static).update("")
            self.query_one("#scorecard", Static).update("")
            for wid in ("prompt-val", "output-val", "metric-val", "grid-pos"):
                self.query_one(f"#{wid}", Static).update("—")
            return

        text_df = df[df["endpoint"] == "text"]
        stt_df = df[df["endpoint"] == "stt"]

        def _text_label(row) -> str:
            p = row.get("prompt_tokens_target", "?")
            o = row.get("output_tokens_target", "?")
            return f"p={p} o={o}"

        def _stt_label(row) -> str:
            p = row.get("prompt_tokens_target", "?")
            o = row.get("output_tokens_target", "?")
            return f"p={p} o={o}"

        # Left chart: Text P95 TTFT
        if not text_df.empty and "P95_ttft_ms" in text_df.columns:
            left_chart = _render_chart(
                text_df, "P95_ttft_ms", "Text LLM — P95 TTFT (ms)",
                higher_better=False, unit="ms", fmt=">8.0f",
                name_fn=_text_label,
            )
        else:
            left_chart = Text("No text endpoint data available.", style="dim")

        # Right chart: STT Mean WER
        if not stt_df.empty and "mean_wer" in stt_df.columns:
            right_chart = _render_chart(
                stt_df, "mean_wer", "STT — Mean WER",
                higher_better=False, unit="", fmt=">8.4f",
                name_fn=_stt_label,
            )
        elif not stt_df.empty and "mean_rtf" in stt_df.columns:
            right_chart = _render_chart(
                stt_df, "mean_rtf", "STT — Mean RTF",
                higher_better=False, unit="", fmt=">8.4f",
                name_fn=_stt_label,
            )
        else:
            right_chart = Text("No STT endpoint data available.", style="dim")

        self.query_one("#chart-ttft", Static).update(left_chart)
        self.query_one("#chart-thru", Static).update(right_chart)

        # Full summary table
        table = RichTable(
            title=f"Mixed Co-Deploy Summary — {rs.model_tag or rs.label}",
            box=None,
        )
        for col in df.columns:
            table.add_column(col, justify="right" if df[col].dtype != object else "left")
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    if "wer" in col or "rtf" in col:
                        cells.append(f"{val:.4f}")
                    else:
                        cells.append(f"{val:.2f}")
                else:
                    cells.append(str(val))
            table.add_row(*cells)

        self.query_one("#minimap", Static).update(table)

        # Controls
        text_model = df["text_model"].iloc[0] if "text_model" in df.columns else "?"
        stt_model = df["stt_model"].iloc[0] if "stt_model" in df.columns else "?"
        self.query_one("#prompt-val", Static).update(_short_model(str(text_model)))
        self.query_one("#output-val", Static).update(_short_model(str(stt_model)))
        self.query_one("#metric-val", Static).update("Mixed Co-Deploy")
        self.query_one("#grid-pos", Static).update(f"[{len(df)} rows]")
        self.query_one("#scorecard", Static).update("")

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

        # Resolve the two metrics for the dual-chart view
        if self._is_codeploy:
            ttft_variants = CO_TTFT_METRICS
            ttft_labels = CO_TTFT_LABELS
            thru_metric = CO_THRU_METRIC
            thru_label = CO_THRU_LABEL
        else:
            ttft_variants = TTFT_METRICS
            ttft_labels = TTFT_LABELS
            thru_metric = THRU_METRIC
            thru_label = THRU_LABEL

        ttft_metric = ttft_variants[self.metric_idx % len(ttft_variants)]
        ttft_label = ttft_labels[ttft_metric]

        # Update controls
        self.query_one("#prompt-val", Static).update(f"◀ {cur_prompt} ▶")
        self.query_one("#output-val", Static).update(f"◀ {cur_output} ▶")
        self.query_one("#metric-val", Static).update(ttft_label.split("(")[0].strip())

        total_cells = len(self._prompts) * len(self._outputs)
        cell_num = self.prompt_idx * len(self._outputs) + self.output_idx + 1
        self.query_one("#grid-pos", Static).update(f"[{cell_num}/{total_cells}]")

        # Slice data for current cell from FILTERED dataframe (sort by TTFT)
        if self._is_codeploy:
            sliced = slice_split_load(filtered_df, cur_prompt, cur_output, ttft_metric)
        else:
            sliced = slice_decision(filtered_df, cur_prompt, cur_output, ttft_metric)

        # ── Dual bar charts (side by side) ──────────────────────────────────
        if sliced.empty:
            self.query_one("#chart-ttft", Static).update(
                "No data for this (prompt, output) combination (or all models filtered out)."
            )
            self.query_one("#chart-thru", Static).update("")
        else:
            if self._is_codeploy:
                name_fn = lambda row: _short_model(row["large_model"]) + " + " + _short_model(row["small_model"])
            else:
                name_fn = lambda row: _short_model(row["model"])

            ttft_chart = _render_chart(
                sliced, ttft_metric, ttft_label,
                higher_better=False, unit="ms", fmt=">8.0f",
                name_fn=name_fn,
            )
            thru_chart = _render_chart(
                sliced, thru_metric, thru_label,
                higher_better=True, unit="tok/s", fmt=">8.1f",
                name_fn=name_fn,
            )
            self.query_one("#chart-ttft", Static).update(ttft_chart)
            self.query_one("#chart-thru", Static).update(thru_chart)

        # ── Minimap (uses TTFT metric for "who wins") ────────────────────
        if not self._is_codeploy and self._df is not None:
            if filtered_df.empty:
                self.query_one("#minimap", Static).update("No data selected.")
                self.query_one("#scorecard", Static).update("")
            else:
                win_board = compute_win_board(filtered_df, ttft_metric)
                minimap = _minimap_rich(
                    win_board, self._prompts, self._outputs, cur_prompt, cur_output
                )
                self.query_one("#minimap", Static).update(minimap)

                # ── Scorecard ────────────────────────────────────────────────
                if self.show_scorecard:
                    sc = scorecard(filtered_df, ttft_metric)
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
        ttft_variants = CO_TTFT_METRICS if self._is_codeploy else TTFT_METRICS
        self.metric_idx = (self.metric_idx + 1) % len(ttft_variants)
        self._render_tuner()

    def action_toggle_scorecard(self) -> None:
        self.show_scorecard = not self.show_scorecard
        self._render_tuner()
