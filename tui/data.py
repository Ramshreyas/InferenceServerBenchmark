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
from datetime import datetime as _dt, timedelta as _td
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
    bench_type: str        # concurrency_bench | split_load | sanity_check
    timestamp: str         # e.g. 20260225_162047  (sweep-level when launched via sweep.py)
    model_tag: str = ""    # e.g. qwen35-122b-a10b-awq4  (empty for legacy runs)
    decision_csv: str | None = None
    summary_csv: str | None = None
    detailed_json: str | None = None
    telemetry_json: str | None = None

    @property
    def label(self) -> str:
        if self.model_tag:
            return self.model_tag
        ts = self.timestamp
        return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"


def discover_results() -> list[ResultSet]:
    """Scan results/ and group files into ResultSets.

    Handles two filename conventions:
      New:  {bench}_{YYYYMMDD_HHMMSS}_{model-tag}_{kind}.{ext}
      Old:  {bench}_{YYYYMMDD_HHMMSS}_{kind}.{ext}
    """
    if not RESULTS_DIR.exists():
        return []

    # Try new format first (with model tag), then fall back to old format.
    # The tag can contain alphanumerics, hyphens, dots, and plus signs.
    new_pat = re.compile(
        r"^(?P<bench>.+?)_(?P<ts>\d{8}_\d{6})_(?P<tag>[A-Za-z0-9][A-Za-z0-9._+\-]*)_(?P<kind>decision|summary|detailed|telemetry)\."
    )
    old_pat = re.compile(
        r"^(?P<bench>.+?)_(?P<ts>\d{8}_\d{6})_(?P<kind>decision|summary|detailed|telemetry)\."
    )

    # key = (bench, timestamp, model_tag)
    groups: dict[tuple[str, str, str], dict[str, str]] = {}
    for p in sorted(RESULTS_DIR.iterdir()):
        m = new_pat.match(p.name)
        if m:
            key = (m.group("bench"), m.group("ts"), m.group("tag"))
            groups.setdefault(key, {})[m.group("kind")] = str(p)
            continue
        m = old_pat.match(p.name)
        if m:
            key = (m.group("bench"), m.group("ts"), "")
            groups.setdefault(key, {})[m.group("kind")] = str(p)

    result_sets_raw: dict[tuple[str, str, str], ResultSet] = {}
    for (bench, ts, tag), files in sorted(groups.items(), key=lambda x: x[0][1], reverse=True):
        rs = ResultSet(
            bench_type=bench,
            timestamp=ts,
            model_tag=tag,
            decision_csv=files.get("decision"),
            summary_csv=files.get("summary"),
            detailed_json=files.get("detailed"),
            telemetry_json=files.get("telemetry"),
        )
        result_sets_raw[(bench, ts, tag)] = rs

    # Merge near-neighbour orphans (legacy data only — no model_tag):
    # if a ResultSet has only a decision CSV and another ResultSet of the
    # same bench type exists within a few seconds, fold the decision CSV in.
    merged_keys: set[tuple[str, str, str]] = set()
    for (bench, ts, tag), rs in result_sets_raw.items():
        if tag:
            continue  # new-format files don't have this problem
        if rs.decision_csv and not rs.summary_csv and not rs.detailed_json:
            ts_dt = _parse_ts(ts)
            for (b2, t2, tag2), rs2 in result_sets_raw.items():
                if b2 != bench or t2 == ts or tag2:
                    continue
                if abs((_parse_ts(t2) - ts_dt).total_seconds()) <= 5:
                    if rs2.summary_csv or rs2.detailed_json:
                        if not rs2.decision_csv:
                            rs2.decision_csv = rs.decision_csv
                        merged_keys.add((bench, ts, tag))
                        break

    result_sets = [
        rs for key, rs in result_sets_raw.items() if key not in merged_keys
    ]
    return result_sets


# ---------------------------------------------------------------------------
# Sweep grouping — merge sequential runs of the same bench type
# ---------------------------------------------------------------------------

def _parse_ts(ts: str) -> _dt:
    """Parse '20260225_162047' → datetime."""
    return _dt.strptime(ts, "%Y%m%d_%H%M%S")


@dataclass
class SweepGroup:
    """A group of ResultSets from one multi-model sweep."""
    bench_type: str
    result_sets: list[ResultSet]  # newest-first

    @property
    def label(self) -> str:
        n = len(self.result_sets)
        if n == 1:
            return self.result_sets[0].label
        oldest = self.result_sets[-1]
        newest = self.result_sets[0]
        start = _parse_ts(oldest.timestamp)
        end = _parse_ts(newest.timestamp)
        if start.date() == end.date():
            date_range = start.strftime("%b %d")
        else:
            date_range = f"{start.strftime('%b %d')} \u2013 {end.strftime('%b %d')}"
        return f"\U0001f4ca {date_range} ({n} runs)"

    @property
    def decision_csvs(self) -> list[str]:
        return [rs.decision_csv for rs in self.result_sets if rs.decision_csv]


def group_into_sweeps(
    result_sets: list[ResultSet],
    gap_hours: float = 18.0,
) -> list[SweepGroup]:
    """Group ResultSets into sweeps.

    New-format runs (with model_tag) that share the same timestamp are
    deterministically grouped — they came from one ``sweep.py`` invocation.

    Legacy runs (no model_tag) fall back to the gap-based heuristic:
    runs whose timestamps are within *gap_hours* of each other are
    considered part of the same sweep.

    Returns newest-sweep-first; each sweep is internally newest-first.
    """
    if not result_sets:
        return []

    # Partition into tagged (new) and untagged (legacy) sets
    tagged: dict[str, list[ResultSet]] = {}   # ts → list of ResultSets
    untagged: list[ResultSet] = []
    for rs in result_sets:
        if rs.model_tag:
            tagged.setdefault(rs.timestamp, []).append(rs)
        else:
            untagged.append(rs)

    buckets: list[list[ResultSet]] = []

    # Tagged: each shared timestamp is one sweep
    for ts in sorted(tagged, reverse=True):
        # Sort within sweep: alphabetical by model_tag for stable ordering
        buckets.append(sorted(tagged[ts], key=lambda r: r.model_tag))

    # Untagged: gap-based grouping (legacy)
    if untagged:
        by_time = sorted(untagged, key=lambda rs: rs.timestamp)
        gap = _td(hours=gap_hours)
        legacy_buckets: list[list[ResultSet]] = [[by_time[0]]]
        for rs in by_time[1:]:
            prev = _parse_ts(legacy_buckets[-1][-1].timestamp)
            cur = _parse_ts(rs.timestamp)
            if cur - prev <= gap:
                legacy_buckets[-1].append(rs)
            else:
                legacy_buckets.append([rs])
        buckets.extend(legacy_buckets)

    # Sort all buckets newest-first by their most recent timestamp
    buckets.sort(key=lambda b: max(rs.timestamp for rs in b), reverse=True)

    return [
        SweepGroup(
            bench_type=bucket[0].bench_type,
            result_sets=list(reversed(bucket)),
        )
        for bucket in buckets
    ]


# ---------------------------------------------------------------------------
# Concurrency bench data
# ---------------------------------------------------------------------------

def load_decision_csv(path: str) -> pd.DataFrame:
    """Load a _decision.csv into a DataFrame."""
    return pd.read_csv(path)


def load_merged_decision_csvs(paths: list[str]) -> pd.DataFrame:
    """Concatenate multiple decision CSVs into one DataFrame."""
    dfs = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


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
                if len(winner) > 24:
                    winner = winner[:22] + "…"
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


# ---------------------------------------------------------------------------
# STT (offline) data
# ---------------------------------------------------------------------------

def load_stt_summary(path: str) -> pd.DataFrame:
    """Load an STT summary CSV.

    Columns: model, concurrency_level, n_requests, n_success,
             mean_wer, median_wer, P95_wer,
             mean_rtf, median_rtf, P95_rtf,
             P50_latency_ms, P95_latency_ms, P99_latency_ms,
             total_audio_duration_s
    """
    return pd.read_csv(path)


def get_stt_concurrency_levels(df: pd.DataFrame) -> list:
    """Return sorted concurrency levels from an STT summary."""
    return sorted(df["concurrency_level"].unique().tolist(), key=str)


# ---------------------------------------------------------------------------
# STT streaming data
# ---------------------------------------------------------------------------

def load_stt_streaming_summary(path: str) -> pd.DataFrame:
    """Load a streaming STT summary CSV.

    Columns: model, concurrency_level, n_requests, n_success,
             mean_wer, median_wer, P95_wer,
             mean_rtf, median_rtf,
             mean_ttfw_ms, P50_ttfw_ms, P95_ttfw_ms,
             mean_inter_delta_ms,
             P50_latency_ms, P95_latency_ms, P99_latency_ms
    """
    return pd.read_csv(path)
