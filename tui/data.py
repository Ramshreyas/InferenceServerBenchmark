"""Data loading and aggregation for TUI views.

Reads CSV/JSON result files from results/ and models.yaml,
and provides structured DataFrames for rendering.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_YAML = Path(__file__).resolve().parent.parent / "models.yaml"


# ---------------------------------------------------------------------------
# models.yaml loader
# ---------------------------------------------------------------------------

def load_models() -> list[dict]:
    """Return the model list from models.yaml."""
    if not MODELS_YAML.exists():
        return []
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def models_by_role(role: str) -> list[dict]:
    return [m for m in load_models() if m.get("role") == role]


# ---------------------------------------------------------------------------
# Result-set discovery
# ---------------------------------------------------------------------------

@dataclass
class ResultSet:
    """A group of files produced by a single benchmark run."""
    bench_type: str        # concurrency_bench | split_load | sanity_check | context_stress
    timestamp: str         # e.g. 20260225_162047
    decision_csv: str | None = None
    summary_csv: str | None = None
    detailed_json: str | None = None
    telemetry_json: str | None = None

    @property
    def label(self) -> str:
        ts = self.timestamp
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"


def discover_results() -> list[ResultSet]:
    """Scan results/ and group files into ResultSets."""
    if not RESULTS_DIR.exists():
        return []

    pattern = re.compile(
        r"^(?P<bench>.+?)_(?P<ts>\d{8}_\d{6})_(?P<kind>decision|summary|detailed|telemetry)\."
    )
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for p in sorted(RESULTS_DIR.iterdir()):
        m = pattern.match(p.name)
        if not m:
            continue
        key = (m.group("bench"), m.group("ts"))
        groups.setdefault(key, {})[m.group("kind")] = str(p)

    result_sets = []
    for (bench, ts), files in sorted(groups.items(), key=lambda x: x[0][1], reverse=True):
        rs = ResultSet(
            bench_type=bench,
            timestamp=ts,
            decision_csv=files.get("decision"),
            summary_csv=files.get("summary"),
            detailed_json=files.get("detailed"),
            telemetry_json=files.get("telemetry"),
        )
        result_sets.append(rs)
    return result_sets


# ---------------------------------------------------------------------------
# Concurrency bench data
# ---------------------------------------------------------------------------

def load_decision_csv(path: str) -> pd.DataFrame:
    """Load a _decision.csv into a DataFrame."""
    return pd.read_csv(path)


def load_summary_csv(path: str) -> pd.DataFrame:
    """Load a _summary.csv (detailed per-request) into a DataFrame."""
    return pd.read_csv(path)


def get_concurrency_grid(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Return sorted (prompt_tiers, output_tiers) from a decision CSV."""
    prompts = sorted(df["prompt_tokens_target"].unique().tolist())
    outputs = sorted(df["output_tokens_target"].unique().tolist())
    return prompts, outputs


def get_models_in_decision(df: pd.DataFrame) -> list[str]:
    """Return unique model names from a decision CSV."""
    return sorted(df["model"].unique().tolist())


def slice_decision(
    df: pd.DataFrame,
    prompt_tokens: int,
    output_tokens: int,
    metric: str = "P95_ttft_ms",
) -> pd.DataFrame:
    """Filter decision CSV to one (prompt, output) cell, sorted by metric ascending."""
    mask = (
        (df["prompt_tokens_target"] == prompt_tokens)
        & (df["output_tokens_target"] == output_tokens)
    )
    sliced = df[mask].copy()
    if sliced.empty:
        return sliced
    sliced = sliced.sort_values(metric, ascending=True).reset_index(drop=True)
    return sliced


def compute_win_board(df: pd.DataFrame, metric: str = "P95_ttft_ms") -> pd.DataFrame:
    """Build a pivot table: rows=prompt, cols=output, values=winning model label.

    Also adds a short label (last path component or label slug).
    """
    prompts, outputs = get_concurrency_grid(df)
    records = []
    for p in prompts:
        for o in outputs:
            s = slice_decision(df, p, o, metric)
            if s.empty:
                winner = "—"
            else:
                winner = s.iloc[0]["model"]
                # Shorten: "mistralai/Mistral-7B-Instruct-v0.3" → "Mistral-7B-Instruct-v0.3"
                if "/" in winner:
                    winner = winner.split("/")[-1]
                # Truncate long names
                if len(winner) > 18:
                    winner = winner[:16] + "…"
            records.append({"prompt": p, "output": o, "winner": winner})
    return pd.DataFrame(records)


def scorecard(df: pd.DataFrame, metric: str = "P95_ttft_ms") -> list[tuple[str, int, int]]:
    """Return [(model_short, wins, total_cells)] sorted by wins descending."""
    wb = compute_win_board(df, metric)
    total = len(wb)
    counts = wb["winner"].value_counts().to_dict()
    result = [(model, count, total) for model, count in counts.items() if model != "—"]
    result.sort(key=lambda x: -x[1])
    return result


# ---------------------------------------------------------------------------
# Split-load (co-deploy) data
# ---------------------------------------------------------------------------

def load_split_load_summary(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_split_load_grid(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    prompts = sorted(df["prompt_tokens_target"].unique().tolist())
    outputs = sorted(df["output_tokens_target"].unique().tolist())
    return prompts, outputs


def get_pairs_in_split_load(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return unique (large_model, small_model) pairs."""
    pairs = df[["large_model", "small_model"]].drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))


def slice_split_load(
    df: pd.DataFrame,
    prompt_tokens: int,
    output_tokens: int,
    metric: str = "large_P95_ttft_ms",
) -> pd.DataFrame:
    mask = (
        (df["prompt_tokens_target"] == prompt_tokens)
        & (df["output_tokens_target"] == output_tokens)
    )
    sliced = df[mask].copy()
    if sliced.empty:
        return sliced
    return sliced.sort_values(metric, ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity check data
# ---------------------------------------------------------------------------

def load_sanity_summary(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
