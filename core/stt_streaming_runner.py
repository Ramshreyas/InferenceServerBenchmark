#!/usr/bin/env python3
"""
Streaming STT benchmark runner via /v1/realtime WebSocket API.

Simulates real-time audio streaming by sending PCM16 audio chunks over
WebSocket at (configurable) real-time rate.  Measures streaming-specific
metrics:
  - TTFW   (Time-to-First-Word):  first chunk sent → first transcription.delta
  - Inter-delta latency:          gaps between successive delta events
  - Final latency:                stream start → transcription.done
  - WER:                          final transcript vs. reference
  - RTF:                          processing time / audio duration

Uses the same LibriSpeech test-clean dataset as the offline STT bench for
direct WER comparison.

Modes:
  1. **Sanity** — Sequential streaming of a few files.
  2. **Concurrency bench** — Sweep across concurrent WebSocket sessions.

Entry point:
    python core/stt_streaming_runner.py --config /configs/stt_streaming_sanity.yaml
    python core/stt_streaming_runner.py --config /configs/stt_streaming_bench.yaml
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm

from utils import setup_logger, format_latency, save_json, save_csv
from telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Reuse WER + dataset loaders from stt_runner
# ---------------------------------------------------------------------------

from stt_runner import compute_wer, load_librispeech_samples, get_audio_duration


# ---------------------------------------------------------------------------
# Audio conversion: FLAC → PCM16 base64 chunks
# ---------------------------------------------------------------------------

def audio_to_pcm16_bytes(audio_path: str, target_sr: int = 16000) -> bytes:
    """Load an audio file and convert to raw PCM16 bytes @ target_sr, mono."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


# ---------------------------------------------------------------------------
# Single WebSocket streaming session
# ---------------------------------------------------------------------------

async def _stream_one_file(
    ws_uri: str,
    model_name: str,
    audio_path: str,
    reference: str,
    request_id: int,
    chunk_size: int = 4096,
    realtime_factor: float = 1.0,
    extra_tags: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Stream one audio file over WebSocket and collect metrics.

    Args:
        ws_uri:         ws://host:port/v1/realtime
        model_name:     Model ID for session.update
        audio_path:     Path to FLAC/WAV file
        reference:      Ground-truth transcript
        request_id:     Unique ID for this request
        chunk_size:     Raw PCM16 bytes per chunk (default 4096 = 128ms @ 16kHz)
        realtime_factor: Playback speed multiplier. 1.0 = real-time, 0 = blast.
        extra_tags:     Additional metadata to merge into result dict.

    Returns:
        Result dict with latency, WER, streaming metrics.
    """
    import websockets

    duration = get_audio_duration(audio_path)
    pcm_bytes = audio_to_pcm16_bytes(audio_path)

    # Timing anchors
    delta_texts: List[str] = []
    delta_timestamps: List[float] = []
    first_delta_time: Optional[float] = None
    stream_start: Optional[float] = None
    stream_done_time: Optional[float] = None
    final_text: str = ""
    error_msg: Optional[str] = None

    try:
        async with websockets.connect(ws_uri, max_size=None, close_timeout=30) as ws:
            # 1. Wait for session.created
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            resp = json.loads(raw)
            if resp.get("type") != "session.created":
                raise RuntimeError(f"Expected session.created, got: {resp.get('type')}")

            # 2. Send session.update with model
            await ws.send(json.dumps({
                "type": "session.update",
                "model": model_name,
            }))

            # 3. Signal ready to start
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # 4. Stream audio chunks
            stream_start = time.perf_counter()
            total_chunks = (len(pcm_bytes) + chunk_size - 1) // chunk_size

            # Compute inter-chunk delay for real-time playback simulation
            # chunk_size bytes = chunk_size/2 samples @ 16kHz = chunk_size/(2*16000) seconds
            chunk_duration_s = chunk_size / (2 * 16000)
            delay_per_chunk = chunk_duration_s * realtime_factor if realtime_factor > 0 else 0

            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i : i + chunk_size]
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }))
                if delay_per_chunk > 0:
                    await asyncio.sleep(delay_per_chunk)

            # 5. Signal audio complete
            await ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
                "final": True,
            }))

            # 6. Collect transcription events
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                except asyncio.TimeoutError:
                    error_msg = "Timed out waiting for transcription.done"
                    break

                resp = json.loads(raw)
                event_type = resp.get("type", "")

                if event_type == "transcription.delta":
                    now = time.perf_counter()
                    delta_text = resp.get("delta", "")
                    delta_texts.append(delta_text)
                    delta_timestamps.append(now)
                    if first_delta_time is None:
                        first_delta_time = now

                elif event_type == "transcription.done":
                    stream_done_time = time.perf_counter()
                    final_text = resp.get("text", "".join(delta_texts))
                    break

                elif event_type == "error":
                    error_msg = resp.get("error", {}).get("message", str(resp))
                    break

    except Exception as e:
        error_msg = str(e)
        stream_done_time = time.perf_counter()

    # ── Compute metrics ────────────────────────────────────────────────
    if stream_start is None:
        stream_start = time.perf_counter()
    if stream_done_time is None:
        stream_done_time = time.perf_counter()

    total_latency_ms = (stream_done_time - stream_start) * 1000
    rtf = (stream_done_time - stream_start) / max(duration, 0.001)

    # Time-to-first-word
    ttfw_ms: Optional[float] = None
    if first_delta_time is not None and stream_start is not None:
        ttfw_ms = (first_delta_time - stream_start) * 1000

    # Inter-delta latencies
    inter_delta_ms: List[float] = []
    for j in range(1, len(delta_timestamps)):
        inter_delta_ms.append((delta_timestamps[j] - delta_timestamps[j - 1]) * 1000)

    # WER
    wer_result: Optional[Dict] = None
    if not error_msg and final_text:
        wer_result = compute_wer(reference, final_text)

    success = error_msg is None and wer_result is not None

    result: Dict[str, Any] = {
        "request_id": request_id,
        "model": model_name,
        "endpoint": "streaming_stt",
        "success": success,
        "audio_path": Path(audio_path).name,
        "audio_duration_s": round(duration, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "rtf": round(rtf, 4),
        "ttfw_ms": round(ttfw_ms, 2) if ttfw_ms is not None else None,
        "num_deltas": len(delta_texts),
        "reference": reference,
        "hypothesis": final_text,
        "timestamp": datetime.now().isoformat(),
    }

    if inter_delta_ms:
        result["inter_delta_mean_ms"] = round(float(np.mean(inter_delta_ms)), 2)
        result["inter_delta_p50_ms"] = round(float(np.percentile(inter_delta_ms, 50)), 2)
        result["inter_delta_p95_ms"] = round(float(np.percentile(inter_delta_ms, 95)), 2)

    if wer_result:
        result["wer"] = wer_result["wer"]
        result["wer_details"] = wer_result

    if error_msg:
        result["error"] = error_msg

    if extra_tags:
        result.update(extra_tags)

    return result


# ---------------------------------------------------------------------------
# Streaming STT Benchmark Runner
# ---------------------------------------------------------------------------

class StreamingSTTBenchmarkRunner:
    """Benchmarks STT models via vLLM's /v1/realtime WebSocket endpoint."""

    def __init__(self, config_path: str, sweep_ts: str | None = None, model_tag: str | None = None):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config["benchmark"]["name"])
        self.output_dir = Path("/results")
        self.output_dir.mkdir(exist_ok=True)

        self.sweep_ts = sweep_ts
        self.model_tag = model_tag

        # Build WebSocket URI from vLLM endpoint
        vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8001/v1")
        # Convert http://host:port/v1 → ws://host:port/v1/realtime
        base = vllm_endpoint.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]  # strip /v1
        self.ws_uri = base.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"
        self.http_base = base

        self.telemetry = (
            TelemetryCollector()
            if self.config.get("telemetry", {}).get("collect_gpu_stats")
            else None
        )

        self.exit_code = 0

        # Auto-detect model name via HTTP
        self._model_name = self._detect_model_name()

    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _detect_model_name(self) -> str:
        import requests as _requests
        try:
            resp = _requests.get(f"{self.http_base}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            name = data["data"][0]["id"]
            self.logger.info(f"Auto-detected STT model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect STT model: {e}")

    def _wait_for_server(self, timeout: int = 300):
        import requests as _requests
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = _requests.get(f"{self.http_base}/health", timeout=5)
                if resp.status_code == 200:
                    self.logger.info("✓ STT server is ready")
                    return
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError("STT server did not become ready in time")

    # ── Async helpers ────────────────────────────────────────────────────

    def _stream_file_sync(
        self,
        audio_path: str,
        reference: str,
        request_id: int,
        extra_tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper around the async streaming function."""
        chunk_size = self.config.get("streaming", {}).get("chunk_size", 4096)
        realtime_factor = self.config.get("streaming", {}).get("realtime_factor", 1.0)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _stream_one_file(
                    ws_uri=self.ws_uri,
                    model_name=self._model_name,
                    audio_path=audio_path,
                    reference=reference,
                    request_id=request_id,
                    chunk_size=chunk_size,
                    realtime_factor=realtime_factor,
                    extra_tags=extra_tags,
                )
            )
        finally:
            loop.close()

    # ── Mode detection and execution ────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting streaming STT benchmark: {self.config['benchmark']['name']}")
        self.logger.info(f"WebSocket URI: {self.ws_uri}")
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
        self.logger.info(f"\n✓ Streaming STT benchmark complete. Results saved to {self.output_dir}")

    # ── Sanity mode ────────────────────────────────────────────────────

    def _run_sanity(self, samples: List[Dict]) -> List[Dict]:
        """Sequential streaming of a small number of files."""
        results: List[Dict] = []
        with tqdm(total=len(samples), desc="Streaming STT Sanity") as pbar:
            for i, sample in enumerate(samples):
                result = self._stream_file_sync(
                    audio_path=sample["audio_path"],
                    reference=sample["reference"],
                    request_id=i,
                    extra_tags={
                        "speaker": sample.get("speaker"),
                        "utt_id": sample.get("utt_id"),
                    },
                )
                results.append(result)
                pbar.update(1)
        return results

    # ── Concurrency sweep ──────────────────────────────────────────────

    def _run_concurrency_sweep(self, samples: List[Dict]) -> List[Dict]:
        """Sweep across concurrency levels — each is N simultaneous WebSocket sessions."""
        concurrency_levels = self.config["concurrency_levels"]
        num_requests = self.config["requests"].get("num_requests", 30)
        all_results: List[Dict] = []

        for concurrency in concurrency_levels:
            self.logger.info(
                f"\n{'='*60}\n"
                f"  Streaming STT Concurrency: level={concurrency}  requests={num_requests}\n"
                f"{'='*60}"
            )

            results = self._run_concurrent_batch(samples, concurrency, num_requests)
            all_results.extend(results)

        return all_results

    def _run_concurrent_batch(
        self,
        samples: List[Dict],
        concurrency: int,
        num_requests: int,
    ) -> List[Dict]:
        """Run N concurrent WebSocket streaming sessions using asyncio."""
        chunk_size = self.config.get("streaming", {}).get("chunk_size", 4096)
        realtime_factor = self.config.get("streaming", {}).get("realtime_factor", 1.0)

        async def _run_all():
            semaphore = asyncio.Semaphore(concurrency)
            results: List[Dict] = []
            pbar = tqdm(total=num_requests, desc=f"concurrency={concurrency}")

            async def _guarded(request_id: int):
                sample = samples[request_id % len(samples)]
                async with semaphore:
                    result = await _stream_one_file(
                        ws_uri=self.ws_uri,
                        model_name=self._model_name,
                        audio_path=sample["audio_path"],
                        reference=sample["reference"],
                        request_id=request_id,
                        chunk_size=chunk_size,
                        realtime_factor=realtime_factor,
                        extra_tags={
                            "concurrency_level": concurrency,
                            "speaker": sample.get("speaker"),
                            "utt_id": sample.get("utt_id"),
                        },
                    )
                    results.append(result)
                    pbar.update(1)
                    return result

            tasks = [_guarded(i) for i in range(num_requests)]
            await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()
            return results

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run_all())
        finally:
            loop.close()

    # ── Persistence ────────────────────────────────────────────────────

    def _make_stem(self) -> str:
        ts = self.sweep_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.config["benchmark"]["output_prefix"]
        if self.model_tag:
            return f"{prefix}_{ts}_{self.model_tag}"
        return f"{prefix}_{ts}"

    def _save_results(self, results: List[Dict]):
        stem = self._make_stem()
        json_path = self.output_dir / f"{stem}_detailed.json"
        save_json(results, json_path)
        self.logger.info(f"Detailed results → {json_path}")

        if self.telemetry:
            tel_path = self.output_dir / f"{stem}_telemetry.json"
            save_json(self.telemetry.get_data(), tel_path)
            self.logger.info(f"Telemetry → {tel_path}")

    def _save_summary_csv(self, results: List[Dict]):
        """One row per concurrency level with aggregate streaming metrics."""
        stem = self._make_stem()
        csv_path = self.output_dir / f"{stem}_summary.csv"

        successful = [r for r in results if r.get("success")]
        if not successful:
            self.logger.warning("No successful requests — cannot write summary CSV.")
            return

        groups: Dict[Any, List] = defaultdict(list)
        for r in successful:
            key = r.get("concurrency_level", "all")
            groups[key].append(r)

        rows: List[Dict] = []
        for level, group in sorted(groups.items(), key=lambda x: str(x[0])):
            wers = [r["wer"] for r in group if "wer" in r]
            rtfs = [r["rtf"] for r in group if "rtf" in r]
            latencies = [r["total_latency_ms"] for r in group]
            ttfws = [r["ttfw_ms"] for r in group if r.get("ttfw_ms") is not None]
            inter_delta_means = [r["inter_delta_mean_ms"] for r in group if r.get("inter_delta_mean_ms") is not None]

            row: Dict[str, Any] = {
                "model": self._model_name,
                "concurrency_level": level,
                "n_requests": len(group),
                "n_success": len(group),
            }

            if wers:
                row["mean_wer"] = round(float(np.mean(wers)), 4)
                row["median_wer"] = round(float(np.median(wers)), 4)
                row["P95_wer"] = round(float(np.percentile(wers, 95)), 4)

            if rtfs:
                row["mean_rtf"] = round(float(np.mean(rtfs)), 4)
                row["median_rtf"] = round(float(np.median(rtfs)), 4)

            if ttfws:
                row["mean_ttfw_ms"] = round(float(np.mean(ttfws)), 2)
                row["P50_ttfw_ms"] = round(float(np.percentile(ttfws, 50)), 2)
                row["P95_ttfw_ms"] = round(float(np.percentile(ttfws, 95)), 2)

            if inter_delta_means:
                row["mean_inter_delta_ms"] = round(float(np.mean(inter_delta_means)), 2)

            if latencies:
                row["P50_latency_ms"] = round(float(np.percentile(latencies, 50)), 2)
                row["P95_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)
                row["P99_latency_ms"] = round(float(np.percentile(latencies, 99)), 2)

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
        self.logger.info("STREAMING STT BENCHMARK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self._model_name}")
        self.logger.info(f"WebSocket: {self.ws_uri}")
        self.logger.info(f"Total Requests: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")

        if successful:
            wers = [r["wer"] for r in successful if "wer" in r]
            rtfs = [r["rtf"] for r in successful if "rtf" in r]
            latencies = [r["total_latency_ms"] for r in successful]
            ttfws = [r["ttfw_ms"] for r in successful if r.get("ttfw_ms") is not None]
            inter_delta_means = [r["inter_delta_mean_ms"] for r in successful if r.get("inter_delta_mean_ms") is not None]
            durations = [r["audio_duration_s"] for r in successful]

            if wers:
                self.logger.info(f"\nWord Error Rate (WER):")
                self.logger.info(f"  Mean:   {np.mean(wers):.2%}")
                self.logger.info(f"  Median: {np.median(wers):.2%}")
                self.logger.info(f"  P95:    {np.percentile(wers, 95):.2%}")
                self.logger.info(f"  Min:    {np.min(wers):.2%}")
                self.logger.info(f"  Max:    {np.max(wers):.2%}")

            if ttfws:
                self.logger.info(f"\nTime-to-First-Word (TTFW):")
                self.logger.info(f"  Mean:   {format_latency(np.mean(ttfws))}")
                self.logger.info(f"  P50:    {format_latency(np.percentile(ttfws, 50))}")
                self.logger.info(f"  P95:    {format_latency(np.percentile(ttfws, 95))}")

            if inter_delta_means:
                self.logger.info(f"\nInter-Delta Latency (mean per session):")
                self.logger.info(f"  Mean:   {format_latency(np.mean(inter_delta_means))}")
                self.logger.info(f"  P50:    {format_latency(np.percentile(inter_delta_means, 50))}")
                self.logger.info(f"  P95:    {format_latency(np.percentile(inter_delta_means, 95))}")

            if rtfs:
                self.logger.info(f"\nReal-Time Factor (RTF < 1.0 = faster than real-time):")
                self.logger.info(f"  Mean:   {np.mean(rtfs):.4f}")
                self.logger.info(f"  Median: {np.median(rtfs):.4f}")
                self.logger.info(f"  P95:    {np.percentile(rtfs, 95):.4f}")

            if latencies:
                self.logger.info(f"\nTotal Session Latency:")
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
# Standalone batch function (for mixed co-deploy usage)
# ---------------------------------------------------------------------------

def run_streaming_stt_batch(
    ws_uri: str,
    model_name: str,
    samples: List[Dict],
    concurrency: int,
    num_requests: int,
    chunk_size: int = 4096,
    realtime_factor: float = 1.0,
) -> List[Dict]:
    """Run a batch of streaming STT requests. Used by mixed_co_deploy_runner.

    Returns list of per-request result dicts.
    """
    async def _run_all():
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict] = []

        async def _guarded(request_id: int):
            sample = samples[request_id % len(samples)]
            async with semaphore:
                result = await _stream_one_file(
                    ws_uri=ws_uri,
                    model_name=model_name,
                    audio_path=sample["audio_path"],
                    reference=sample["reference"],
                    request_id=request_id,
                    chunk_size=chunk_size,
                    realtime_factor=realtime_factor,
                )
                results.append(result)
                return result

        tasks = [_guarded(i) for i in range(num_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_all())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Streaming STT benchmark runner")
    parser.add_argument("--config", required=True, help="Path to streaming STT benchmark YAML config")
    parser.add_argument("--sweep-ts", default=None, help="Sweep-level timestamp")
    parser.add_argument("--model-tag", default=None, help="Short model label for filenames")
    args = parser.parse_args()

    runner = StreamingSTTBenchmarkRunner(args.config, sweep_ts=args.sweep_ts, model_tag=args.model_tag)
    runner.run()
    sys.exit(runner.exit_code)


if __name__ == "__main__":
    main()
