#!/usr/bin/env python3
"""
Mixed co-deploy benchmark runner: Text LLM + STT model simultaneously.

Simulates real-world usage (e.g., meeting transcription + LLM queries)
by running both benchmark workloads AT THE SAME TIME, measuring mutual
performance degradation.

Entry point:
    python core/mixed_co_deploy_runner.py --config /configs/mixed_co_deploy.yaml
"""

import argparse
import json
import os
import sys
import time
import urllib.request as _req
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from openai import OpenAI
from tqdm import tqdm

from utils import setup_logger, format_latency, format_throughput, save_json, save_csv
from telemetry import TelemetryCollector
from stt_runner import load_librispeech_samples, run_stt_benchmark_batch, compute_wer, get_audio_duration


class MixedCoDeployRunner:
    """Benchmarks text + STT models running simultaneously on shared GPU."""

    def __init__(self, config_path: str, sweep_ts: str | None = None, model_tag: str | None = None):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config["benchmark"]["name"])
        self.output_dir = Path("/results")
        self.output_dir.mkdir(exist_ok=True)

        self.sweep_ts = sweep_ts
        self.model_tag = model_tag

        # ── Text endpoint (port 8000 by default) ───────────────────────────
        text_url = os.getenv("VLLM_ENDPOINT_LARGE", "http://vllm-8000:8000/v1")
        self.client_text = OpenAI(base_url=text_url, api_key="dummy")
        self.text_model = os.getenv("LARGE_MODEL_NAME") or self._detect_text(self.client_text)

        # ── STT endpoint (port 8001 by default) ────────────────────────────
        self.stt_url = os.getenv("VLLM_ENDPOINT_SMALL", "http://vllm-8001:8001/v1")
        self.stt_model = os.getenv("SMALL_MODEL_NAME") or self._detect_stt()

        self.telemetry = (
            TelemetryCollector()
            if self.config.get("telemetry", {}).get("collect_gpu_stats")
            else None
        )

    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _detect_text(self, client: OpenAI) -> str:
        try:
            name = client.models.list().data[0].id
            self.logger.info(f"Auto-detected text model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect text model: {e}")

    def _detect_stt(self) -> str:
        import requests as _requests
        try:
            resp = _requests.get(f"{self.stt_url}/models", timeout=10)
            resp.raise_for_status()
            name = resp.json()["data"][0]["id"]
            self.logger.info(f"Auto-detected STT model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect STT model: {e}")

    def _wait_for_text(self, timeout: int = 300):
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.client_text.models.list()
                self.logger.info("✓ Text server is ready")
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError("Text server did not become ready in time")

    def _wait_for_stt(self, timeout: int = 300):
        import requests as _requests
        start = time.time()
        base = self.stt_url.rstrip("/").removesuffix("/v1")
        while time.time() - start < timeout:
            try:
                resp = _requests.get(f"{base}/health", timeout=5)
                if resp.status_code == 200:
                    self.logger.info("✓ STT server is ready")
                    return
            except Exception:
                time.sleep(5)
        raise TimeoutError("STT server did not become ready in time")

    def _get_text_max_model_len(self) -> int:
        base = os.getenv("VLLM_ENDPOINT_LARGE", "http://vllm-8000:8000/v1")
        base = base.rstrip("/").removesuffix("/v1")
        try:
            with _req.urlopen(f"{base}/v1/models", timeout=10) as resp:
                data = json.loads(resp.read())
                return int(data["data"][0].get("max_model_len", 2**31))
        except Exception:
            return 2**31

    def _generate_prompt(self, num_tokens: int) -> str:
        base_text = "Explain the following concept in detail with examples and use cases: "
        filler = (
            "artificial intelligence and machine learning in modern applications "
            * (num_tokens // 10 + 1)
        )
        return base_text + filler[: num_tokens * 4]

    # ── Single text request ──────────────────────────────────────────────────

    def _run_text_request(
        self,
        prompt: str,
        max_tokens: int,
        request_id: int,
        extra_tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        try:
            system_prompt = self.config["text"].get("system_prompt", "")
            response = self.client_text.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                stream=True,
            )

            first_token_time = None
            token_times: list[float] = []
            tokens: list[str] = []

            for chunk in response:
                delta = chunk.choices[0].delta
                text = delta.content or (delta.model_extra or {}).get("reasoning")
                if text:
                    t = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = t
                    token_times.append(t)
                    tokens.append(text)

            end_time = time.perf_counter()
            ttft = (first_token_time - start_time) * 1000 if first_token_time else None
            total_latency = (end_time - start_time) * 1000
            num_generated = len(tokens)
            itl = (
                float(np.mean(np.diff(token_times)) * 1000)
                if len(token_times) > 1
                else None
            )
            throughput = (
                num_generated / (end_time - start_time)
                if (end_time - start_time) > 0
                else 0
            )

            result: Dict[str, Any] = {
                "request_id": request_id,
                "endpoint": "text",
                "model": self.text_model,
                "success": True,
                "ttft_ms": ttft,
                "itl_ms": itl,
                "total_latency_ms": total_latency,
                "tokens_generated": num_generated,
                "throughput_tokens_per_sec": throughput,
                "timestamp": datetime.now().isoformat(),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

        except Exception as e:
            end_time = time.perf_counter()
            result = {
                "request_id": request_id,
                "endpoint": "text",
                "model": self.text_model,
                "success": False,
                "error": str(e),
                "total_latency_ms": (end_time - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

    # ── Main benchmark ───────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting mixed co-deploy benchmark: {self.config['benchmark']['name']}")
        self.logger.info(f"Text model:  {self.text_model}")
        self.logger.info(f"STT model:   {self.stt_model}")

        self._wait_for_text()
        self._wait_for_stt()

        text_max_len = self._get_text_max_model_len()
        self.logger.info(f"Text max_model_len: {text_max_len:,}")

        # Load STT samples
        stt_cfg = self.config["stt"]
        samples = load_librispeech_samples(
            stt_cfg["dataset"]["path"],
            stt_cfg["dataset"].get("max_files", 0),
        )
        if not samples:
            raise RuntimeError(f"No STT samples found at {stt_cfg['dataset']['path']}")
        self.logger.info(f"Loaded {len(samples)} STT audio samples")

        if self.telemetry:
            self.telemetry.start_collection(
                interval=self.config["telemetry"]["sample_interval_sec"]
            )

        text_cfg = self.config["text"]
        prompt_lengths = text_cfg["prompt_token_lengths"]
        output_lengths = text_cfg["output_token_lengths"]
        text_concurrency = text_cfg["concurrency"]
        text_num_requests = text_cfg["num_requests"]

        stt_concurrency = stt_cfg["concurrency"]
        stt_num_requests = stt_cfg["num_requests"]
        stt_temperature = stt_cfg.get("temperature", 0.0)

        all_results: list[Dict] = []

        for prompt_tokens in prompt_lengths:
            for output_tokens in output_lengths:
                total_needed = prompt_tokens + output_tokens
                if total_needed > text_max_len:
                    self.logger.warning(
                        f"SKIP prompt={prompt_tokens} output={output_tokens} — "
                        f"total {total_needed:,} > max_model_len {text_max_len:,}"
                    )
                    continue

                self.logger.info(
                    f"\n{'='*60}\n"
                    f"  Mixed sweep: prompt={prompt_tokens}  output={output_tokens}\n"
                    f"  Text: {text_concurrency} concurrent, {text_num_requests} requests\n"
                    f"  STT:  {stt_concurrency} concurrent, {stt_num_requests} requests\n"
                    f"  → Running SIMULTANEOUSLY\n"
                    f"{'='*60}"
                )

                text_tags = {
                    "prompt_tokens_target": prompt_tokens,
                    "output_tokens_target": output_tokens,
                }

                # Launch BOTH workloads simultaneously using a shared executor
                total_workers = text_concurrency + stt_concurrency
                results_this_point: list[Dict] = []

                with ThreadPoolExecutor(max_workers=total_workers) as executor:
                    futures: list[Future] = []

                    # Submit text requests
                    for i in range(text_num_requests):
                        futures.append(
                            executor.submit(
                                self._run_text_request,
                                prompt=self._generate_prompt(prompt_tokens),
                                max_tokens=output_tokens,
                                request_id=i,
                                extra_tags=text_tags,
                            )
                        )

                    # Submit STT requests (using the standalone function)
                    # We submit them individually to interleave with text
                    import requests as _requests
                    session = _requests.Session()

                    def _stt_one(req_id: int) -> Dict:
                        sample = samples[req_id % len(samples)]
                        audio_path = sample["audio_path"]
                        reference = sample["reference"]
                        duration = get_audio_duration(audio_path)
                        start = time.perf_counter()
                        try:
                            with open(audio_path, "rb") as f:
                                files = {"file": (Path(audio_path).name, f, "audio/flac")}
                                data = {
                                    "model": self.stt_model,
                                    "temperature": stt_temperature,
                                    "response_format": "json",
                                }
                                resp = session.post(
                                    f"{self.stt_url}/audio/transcriptions",
                                    files=files,
                                    data=data,
                                    timeout=300,
                                )
                                resp.raise_for_status()
                            end = time.perf_counter()
                            hypothesis = resp.json().get("text", "")
                            wer_result = compute_wer(reference, hypothesis)
                            return {
                                "request_id": req_id,
                                "endpoint": "stt",
                                "model": self.stt_model,
                                "success": True,
                                "audio_path": Path(audio_path).name,
                                "audio_duration_s": round(duration, 2),
                                "total_latency_ms": round((end - start) * 1000, 2),
                                "rtf": round((end - start) / max(duration, 0.001), 4),
                                "wer": wer_result["wer"],
                                "prompt_tokens_target": prompt_tokens,
                                "output_tokens_target": output_tokens,
                                "timestamp": datetime.now().isoformat(),
                            }
                        except Exception as e:
                            end = time.perf_counter()
                            return {
                                "request_id": req_id,
                                "endpoint": "stt",
                                "model": self.stt_model,
                                "success": False,
                                "error": str(e),
                                "total_latency_ms": round((end - start) * 1000, 2),
                                "prompt_tokens_target": prompt_tokens,
                                "output_tokens_target": output_tokens,
                                "timestamp": datetime.now().isoformat(),
                            }

                    for i in range(stt_num_requests):
                        futures.append(executor.submit(_stt_one, i))

                    total = text_num_requests + stt_num_requests
                    with tqdm(total=total, desc=f"p={prompt_tokens} o={output_tokens} (mixed)") as pbar:
                        for fut in as_completed(futures):
                            results_this_point.append(fut.result())
                            pbar.update(1)

                all_results.extend(results_this_point)

        if self.telemetry:
            self.telemetry.stop_collection()

        self._save_results(all_results)
        self._save_summary_csv(all_results)
        self._print_summary(all_results)
        self.logger.info(f"\n✓ Mixed co-deploy benchmark complete. Results saved to {self.output_dir}")

    # ── Persistence ──────────────────────────────────────────────────────────

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
        """One row per (prompt, output, endpoint) with aggregate metrics."""
        stem = self._make_stem()
        csv_path = self.output_dir / f"{stem}_summary.csv"

        successful = [r for r in results if r.get("success")]
        if not successful:
            self.logger.warning("No successful requests — cannot write summary CSV.")
            return

        # Group by (prompt_tokens_target, output_tokens_target, endpoint)
        groups: dict[tuple, list] = defaultdict(list)
        for r in successful:
            key = (
                r.get("prompt_tokens_target"),
                r.get("output_tokens_target"),
                r.get("endpoint"),
            )
            groups[key].append(r)

        rows: list[Dict] = []
        for (prompt_t, output_t, endpoint), group in sorted(groups.items()):
            row: Dict[str, Any] = {
                "text_model": self.text_model,
                "stt_model": self.stt_model,
                "prompt_tokens_target": prompt_t,
                "output_tokens_target": output_t,
                "endpoint": endpoint,
                "n_requests": len(group),
            }

            if endpoint == "text":
                ttfts = [r["ttft_ms"] for r in group if r.get("ttft_ms") is not None]
                itls = [r["itl_ms"] for r in group if r.get("itl_ms") is not None]
                throughputs = [r["throughput_tokens_per_sec"] for r in group]
                if ttfts:
                    row["P50_ttft_ms"] = round(float(np.percentile(ttfts, 50)), 2)
                    row["P95_ttft_ms"] = round(float(np.percentile(ttfts, 95)), 2)
                    row["P99_ttft_ms"] = round(float(np.percentile(ttfts, 99)), 2)
                if itls:
                    row["P50_itl_ms"] = round(float(np.percentile(itls, 50)), 2)
                    row["P95_itl_ms"] = round(float(np.percentile(itls, 95)), 2)
                if throughputs:
                    row["mean_throughput_tok_s"] = round(float(np.mean(throughputs)), 2)

            elif endpoint == "stt":
                wers = [r["wer"] for r in group if "wer" in r]
                rtfs = [r["rtf"] for r in group if "rtf" in r]
                latencies = [r["total_latency_ms"] for r in group]
                if wers:
                    row["mean_wer"] = round(float(np.mean(wers)), 4)
                    row["median_wer"] = round(float(np.median(wers)), 4)
                if rtfs:
                    row["mean_rtf"] = round(float(np.mean(rtfs)), 4)
                    row["P95_rtf"] = round(float(np.percentile(rtfs, 95)), 4)
                if latencies:
                    row["P50_latency_ms"] = round(float(np.percentile(latencies, 50)), 2)
                    row["P95_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)

            rows.append(row)

        save_csv(rows, csv_path)
        self.logger.info(f"Summary CSV → {csv_path}")

    def _print_summary(self, results: List[Dict]):
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        self.logger.info("\n" + "=" * 60)
        self.logger.info("MIXED CO-DEPLOY BENCHMARK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Text model: {self.text_model}")
        self.logger.info(f"STT model:  {self.stt_model}")
        self.logger.info(f"Total Requests: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")

        # Text endpoint summary
        text_ok = [r for r in successful if r.get("endpoint") == "text"]
        if text_ok:
            ttfts = [r["ttft_ms"] for r in text_ok if r.get("ttft_ms")]
            itls = [r["itl_ms"] for r in text_ok if r.get("itl_ms")]
            throughputs = [r["throughput_tokens_per_sec"] for r in text_ok]
            self.logger.info(f"\n  [TEXT endpoint — {len(text_ok)} requests]")
            if ttfts:
                self.logger.info(f"    TTFT  P50={format_latency(np.percentile(ttfts, 50))}  "
                                 f"P95={format_latency(np.percentile(ttfts, 95))}  "
                                 f"P99={format_latency(np.percentile(ttfts, 99))}")
            if itls:
                self.logger.info(f"    ITL   P50={format_latency(np.percentile(itls, 50))}  "
                                 f"P95={format_latency(np.percentile(itls, 95))}")
            if throughputs:
                self.logger.info(f"    Throughput  mean={format_throughput(np.mean(throughputs))}")

        # STT endpoint summary
        stt_ok = [r for r in successful if r.get("endpoint") == "stt"]
        if stt_ok:
            wers = [r["wer"] for r in stt_ok if "wer" in r]
            rtfs = [r["rtf"] for r in stt_ok if "rtf" in r]
            self.logger.info(f"\n  [STT endpoint — {len(stt_ok)} requests]")
            if wers:
                self.logger.info(f"    WER   mean={np.mean(wers):.2%}  median={np.median(wers):.2%}  "
                                 f"P95={np.percentile(wers, 95):.2%}")
            if rtfs:
                self.logger.info(f"    RTF   mean={np.mean(rtfs):.4f}  P95={np.percentile(rtfs, 95):.4f}")

        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Mixed co-deploy benchmark (text + STT)")
    parser.add_argument("--config", required=True, help="Path to mixed_co_deploy YAML config")
    parser.add_argument("--sweep-ts", default=None, help="Sweep-level timestamp")
    parser.add_argument("--model-tag", default=None, help="Short model-pair label for filenames")
    args = parser.parse_args()

    runner = MixedCoDeployRunner(args.config, sweep_ts=args.sweep_ts, model_tag=args.model_tag)
    runner.run()


if __name__ == "__main__":
    main()
