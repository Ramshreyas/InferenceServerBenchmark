#!/usr/bin/env python3
"""
STT (Speech-to-Text) benchmark runner.

Supports two modes:
  1. **Sanity** — Sequential transcription of a few files, basic WER check.
  2. **Concurrency bench** — Sweep across concurrency levels, measure RTF + WER.

Audio is sent to the vLLM server via the OpenAI-compatible audio transcription
endpoint.  Voxtral on vLLM exposes /v1/audio/transcriptions (offline) and
/v1/realtime (streaming).  This runner uses the offline endpoint for
reproducible WER benchmarking.

Entry point:
    python core/stt_runner.py --config /configs/stt_sanity.yaml
    python core/stt_runner.py --config /configs/stt_concurrency_bench.yaml
"""

import argparse
import io
import json
import os
import re
import sys
import time
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from utils import setup_logger, format_latency, save_json, save_csv
from telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# WER computation (standard word-error-rate via edit distance)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Compute Word Error Rate between reference and hypothesis.

    Returns dict with wer, substitutions, insertions, deletions, ref_words, hyp_words.
    """
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()

    # Levenshtein distance at word level
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrace for S/I/D counts
    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    wer = dp[n][m] / max(n, 1)

    return {
        "wer": round(wer, 4),
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_word_count": n,
        "hyp_word_count": m,
        "edit_distance": dp[n][m],
    }


# ---------------------------------------------------------------------------
# LibriSpeech dataset loader
# ---------------------------------------------------------------------------

def load_librispeech_samples(
    dataset_path: str,
    max_files: int = 0,
) -> List[Dict[str, str]]:
    """Load LibriSpeech audio files and reference transcripts.

    Returns list of {"audio_path": str, "reference": str, "speaker": str, "chapter": str}.
    """
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}.\n"
            f"Run: bash assets/download_librispeech.sh"
        )

    samples: list[Dict[str, str]] = []

    # LibriSpeech structure: <speaker>/<chapter>/<speaker>-<chapter>-<uttid>.flac
    # Transcripts: <speaker>/<chapter>/<speaker>-<chapter>.trans.txt
    for trans_file in sorted(root.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        parts = chapter_dir.parts
        speaker = parts[-2] if len(parts) >= 2 else "unknown"
        chapter = parts[-1] if len(parts) >= 1 else "unknown"

        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: <uttid> <transcript text>
                utt_id, _, transcript = line.partition(" ")
                audio_path = chapter_dir / f"{utt_id}.flac"
                if audio_path.exists():
                    samples.append({
                        "audio_path": str(audio_path),
                        "reference": transcript,
                        "speaker": speaker,
                        "chapter": chapter,
                        "utt_id": utt_id,
                    })

    if max_files and max_files > 0:
        samples = samples[:max_files]

    return samples


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except ImportError:
        # Fallback: try wave module (only works for .wav)
        try:
            with wave.open(audio_path, "r") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# STT Benchmark Runner
# ---------------------------------------------------------------------------

class STTBenchmarkRunner:
    """Benchmarks STT models via vLLM's audio transcription endpoint."""

    def __init__(self, config_path: str, sweep_ts: str | None = None, model_tag: str | None = None):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config["benchmark"]["name"])
        self.output_dir = Path("/results")
        self.output_dir.mkdir(exist_ok=True)

        self.sweep_ts = sweep_ts
        self.model_tag = model_tag

        # vLLM endpoint
        vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8001/v1")
        self.base_url = vllm_endpoint.rstrip("/")

        # We use requests library for multipart audio upload
        import requests as _requests
        self._session = _requests.Session()

        self.telemetry = (
            TelemetryCollector()
            if self.config.get("telemetry", {}).get("collect_gpu_stats")
            else None
        )

        self.exit_code = 0

        # Detect model name
        self._model_name = self._detect_model_name()

    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _detect_model_name(self) -> str:
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            name = data["data"][0]["id"]
            self.logger.info(f"Auto-detected STT model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect STT model: {e}")

    def _wait_for_server(self, timeout: int = 300):
        start = time.time()
        base = self.base_url.rstrip("/").removesuffix("/v1")
        while time.time() - start < timeout:
            try:
                resp = self._session.get(f"{base}/health", timeout=5)
                if resp.status_code == 200:
                    self.logger.info("✓ STT server is ready")
                    return
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError("STT server did not become ready in time")

    # ── Single transcription request ─────────────────────────────────────────

    def _transcribe_file(
        self,
        audio_path: str,
        reference: str,
        request_id: int,
        extra_tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Send one audio file to the server and measure performance."""
        duration = get_audio_duration(audio_path)
        start_time = time.perf_counter()

        try:
            # OpenAI-compatible /v1/audio/transcriptions endpoint
            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/flac")}
                data = {
                    "model": self._model_name,
                    "temperature": self.config["requests"].get("temperature", 0.0),
                    "response_format": "json",
                }
                resp = self._session.post(
                    f"{self.base_url}/audio/transcriptions",
                    files=files,
                    data=data,
                    timeout=300,
                )
                resp.raise_for_status()

            end_time = time.perf_counter()
            result_data = resp.json()
            hypothesis = result_data.get("text", "")

            total_latency = (end_time - start_time) * 1000  # ms
            rtf = (end_time - start_time) / duration if duration > 0 else float("inf")

            wer_result = compute_wer(reference, hypothesis)

            result: Dict[str, Any] = {
                "request_id": request_id,
                "model": self._model_name,
                "success": True,
                "audio_path": Path(audio_path).name,
                "audio_duration_s": round(duration, 2),
                "total_latency_ms": round(total_latency, 2),
                "rtf": round(rtf, 4),  # Real-Time Factor: <1.0 means faster than real-time
                "wer": wer_result["wer"],
                "wer_details": wer_result,
                "reference": reference,
                "hypothesis": hypothesis,
                "timestamp": datetime.now().isoformat(),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

        except Exception as e:
            end_time = time.perf_counter()
            self.logger.error(f"Request {request_id} failed: {e}")
            result = {
                "request_id": request_id,
                "model": self._model_name,
                "success": False,
                "audio_path": Path(audio_path).name,
                "audio_duration_s": round(duration, 2),
                "total_latency_ms": round((end_time - start_time) * 1000, 2),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

    # ── Mode detection and execution ─────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting STT benchmark: {self.config['benchmark']['name']}")
        self._wait_for_server()

        # Load audio samples
        ds = self.config["dataset"]
        samples = load_librispeech_samples(ds["path"], ds.get("max_files", 0))
        if not samples:
            raise RuntimeError(f"No audio samples found at {ds['path']}")
        self.logger.info(f"Loaded {len(samples)} audio samples from {ds['name']}")

        if self.telemetry:
            self.telemetry.start_collection(
                interval=self.config["telemetry"]["sample_interval_sec"]
            )

        # Detect mode
        if "concurrency_levels" in self.config:
            all_results = self._run_concurrency_sweep(samples)
        else:
            all_results = self._run_sanity(samples)

        if self.telemetry:
            self.telemetry.stop_collection()

        self._save_results(all_results)
        self._save_summary_csv(all_results)
        self._print_summary(all_results)
        self.logger.info(f"\n✓ STT benchmark complete. Results saved to {self.output_dir}")

    # ── Sanity mode ──────────────────────────────────────────────────────────

    def _run_sanity(self, samples: List[Dict]) -> List[Dict]:
        """Sequential transcription of a small number of files."""
        num_concurrent = self.config["requests"].get("num_concurrent", 1)
        results: list[Dict] = []

        with tqdm(total=len(samples), desc="STT Sanity") as pbar:
            for i, sample in enumerate(samples):
                result = self._transcribe_file(
                    audio_path=sample["audio_path"],
                    reference=sample["reference"],
                    request_id=i,
                    extra_tags={"speaker": sample.get("speaker"), "utt_id": sample.get("utt_id")},
                )
                results.append(result)
                pbar.update(1)

        return results

    # ── Concurrency sweep mode ───────────────────────────────────────────────

    def _run_concurrency_sweep(self, samples: List[Dict]) -> List[Dict]:
        """Sweep across concurrency levels, measuring RTF + WER at each level."""
        concurrency_levels = self.config["concurrency_levels"]
        num_requests = self.config["requests"].get("num_requests", 50)
        all_results: list[Dict] = []

        for concurrency in concurrency_levels:
            self.logger.info(
                f"\n{'='*60}\n"
                f"  STT Concurrency sweep: concurrency={concurrency}  requests={num_requests}\n"
                f"{'='*60}"
            )

            tags = {"concurrency_level": concurrency}
            results: list[Dict] = []

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for i in range(num_requests):
                    sample = samples[i % len(samples)]
                    futures.append(
                        executor.submit(
                            self._transcribe_file,
                            audio_path=sample["audio_path"],
                            reference=sample["reference"],
                            request_id=i,
                            extra_tags={
                                **tags,
                                "speaker": sample.get("speaker"),
                                "utt_id": sample.get("utt_id"),
                            },
                        )
                    )

                with tqdm(total=num_requests, desc=f"concurrency={concurrency}") as pbar:
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        pbar.update(1)

            all_results.extend(results)

        return all_results

    # ── Persistence ──────────────────────────────────────────────────────────

    def _make_stem(self) -> str:
        ts = self.sweep_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.config["benchmark"]["output_prefix"]
        if self.model_tag:
            return f"{prefix}_{ts}_{self.model_tag}"
        return f"{prefix}_{ts}"

    def _save_results(self, results: List[Dict]):
        stem = self._make_stem()

        # Strip full reference/hypothesis from detailed JSON to keep it manageable
        clean = []
        for r in results:
            cr = dict(r)
            # Keep reference and hypothesis in detailed but truncate for readability
            clean.append(cr)

        json_path = self.output_dir / f"{stem}_detailed.json"
        save_json(clean, json_path)
        self.logger.info(f"Detailed results → {json_path}")

        if self.telemetry:
            tel_path = self.output_dir / f"{stem}_telemetry.json"
            save_json(self.telemetry.get_data(), tel_path)
            self.logger.info(f"Telemetry → {tel_path}")

    def _save_summary_csv(self, results: List[Dict]):
        """One row per concurrency level with aggregate metrics."""
        stem = self._make_stem()
        csv_path = self.output_dir / f"{stem}_summary.csv"

        successful = [r for r in results if r.get("success")]
        if not successful:
            self.logger.warning("No successful requests — cannot write summary CSV.")
            return

        # Group by concurrency_level (or "all" for sanity mode)
        groups: dict[Any, list] = defaultdict(list)
        for r in successful:
            key = r.get("concurrency_level", "all")
            groups[key].append(r)

        rows: list[Dict] = []
        for level, group in sorted(groups.items(), key=lambda x: str(x[0])):
            wers = [r["wer"] for r in group if "wer" in r]
            rtfs = [r["rtf"] for r in group if "rtf" in r]
            latencies = [r["total_latency_ms"] for r in group if "total_latency_ms" in r]
            durations = [r["audio_duration_s"] for r in group if "audio_duration_s" in r]

            row: Dict[str, Any] = {
                "model": self._model_name,
                "concurrency_level": level,
                "n_requests": len(group),
                "n_success": len([r for r in group if r.get("success")]),
            }

            if wers:
                row["mean_wer"] = round(float(np.mean(wers)), 4)
                row["median_wer"] = round(float(np.median(wers)), 4)
                row["P95_wer"] = round(float(np.percentile(wers, 95)), 4)

            if rtfs:
                row["mean_rtf"] = round(float(np.mean(rtfs)), 4)
                row["median_rtf"] = round(float(np.median(rtfs)), 4)
                row["P95_rtf"] = round(float(np.percentile(rtfs, 95)), 4)

            if latencies:
                row["P50_latency_ms"] = round(float(np.percentile(latencies, 50)), 2)
                row["P95_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)
                row["P99_latency_ms"] = round(float(np.percentile(latencies, 99)), 2)

            if durations:
                row["total_audio_duration_s"] = round(float(np.sum(durations)), 2)

            rows.append(row)

        save_csv(rows, csv_path)
        self.logger.info(f"Summary CSV → {csv_path}")

    def _print_summary(self, results: List[Dict]):
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        if not results:
            self.exit_code = 2
            return

        if len(failed) == len(results):
            self.exit_code = 2
        elif failed:
            self.exit_code = 1

        self.logger.info("\n" + "=" * 60)
        self.logger.info("STT BENCHMARK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self._model_name}")
        self.logger.info(f"Total Requests: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")

        if successful:
            wers = [r["wer"] for r in successful if "wer" in r]
            rtfs = [r["rtf"] for r in successful if "rtf" in r]
            latencies = [r["total_latency_ms"] for r in successful]
            durations = [r["audio_duration_s"] for r in successful]

            if wers:
                self.logger.info(f"\nWord Error Rate (WER):")
                self.logger.info(f"  Mean:   {np.mean(wers):.2%}")
                self.logger.info(f"  Median: {np.median(wers):.2%}")
                self.logger.info(f"  P95:    {np.percentile(wers, 95):.2%}")
                self.logger.info(f"  Min:    {np.min(wers):.2%}")
                self.logger.info(f"  Max:    {np.max(wers):.2%}")

            if rtfs:
                self.logger.info(f"\nReal-Time Factor (RTF < 1.0 = faster than real-time):")
                self.logger.info(f"  Mean:   {np.mean(rtfs):.4f}")
                self.logger.info(f"  Median: {np.median(rtfs):.4f}")
                self.logger.info(f"  P95:    {np.percentile(rtfs, 95):.4f}")

            if latencies:
                self.logger.info(f"\nLatency:")
                self.logger.info(f"  P50: {format_latency(np.percentile(latencies, 50))}")
                self.logger.info(f"  P95: {format_latency(np.percentile(latencies, 95))}")
                self.logger.info(f"  P99: {format_latency(np.percentile(latencies, 99))}")

            if durations:
                total_dur = np.sum(durations)
                self.logger.info(f"\nAudio processed: {total_dur:.1f}s total")

        if failed:
            self.logger.info(f"\nFailed requests:")
            from collections import Counter
            error_counts = Counter(r.get("error", "unknown") for r in failed)
            for error, count in error_counts.most_common(5):
                truncated = (error[:120] + "…") if len(error) > 120 else error
                self.logger.info(f"  [{count}x] {truncated}")

        self.logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Standalone STT benchmark function (called from mixed co-deploy runner)
# ---------------------------------------------------------------------------

def run_stt_benchmark_batch(
    base_url: str,
    model_name: str,
    samples: List[Dict],
    concurrency: int,
    num_requests: int,
    temperature: float = 0.0,
) -> List[Dict]:
    """Run a batch of STT requests. Used by mixed_co_deploy_runner.

    Returns list of per-request result dicts.
    """
    import requests as _requests
    session = _requests.Session()

    def _transcribe_one(request_id: int) -> Dict:
        sample = samples[request_id % len(samples)]
        audio_path = sample["audio_path"]
        reference = sample["reference"]
        duration = get_audio_duration(audio_path)
        start = time.perf_counter()

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/flac")}
                data = {
                    "model": model_name,
                    "temperature": temperature,
                    "response_format": "json",
                }
                resp = session.post(
                    f"{base_url}/audio/transcriptions",
                    files=files,
                    data=data,
                    timeout=300,
                )
                resp.raise_for_status()

            end = time.perf_counter()
            hypothesis = resp.json().get("text", "")
            wer_result = compute_wer(reference, hypothesis)

            return {
                "request_id": request_id,
                "endpoint": "stt",
                "model": model_name,
                "success": True,
                "audio_path": Path(audio_path).name,
                "audio_duration_s": round(duration, 2),
                "total_latency_ms": round((end - start) * 1000, 2),
                "rtf": round((end - start) / max(duration, 0.001), 4),
                "wer": wer_result["wer"],
                "wer_details": wer_result,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            end = time.perf_counter()
            return {
                "request_id": request_id,
                "endpoint": "stt",
                "model": model_name,
                "success": False,
                "audio_path": Path(audio_path).name,
                "error": str(e),
                "total_latency_ms": round((end - start) * 1000, 2),
                "timestamp": datetime.now().isoformat(),
            }

    results: list[Dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_transcribe_one, i) for i in range(num_requests)]
        for fut in as_completed(futures):
            results.append(fut.result())

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STT benchmark runner")
    parser.add_argument("--config", required=True, help="Path to STT benchmark YAML config")
    parser.add_argument("--sweep-ts", default=None, help="Sweep-level timestamp")
    parser.add_argument("--model-tag", default=None, help="Short model label for filenames")
    args = parser.parse_args()

    runner = STTBenchmarkRunner(args.config, sweep_ts=args.sweep_ts, model_tag=args.model_tag)
    runner.run()
    sys.exit(runner.exit_code)


if __name__ == "__main__":
    main()
