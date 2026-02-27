#!/usr/bin/env python3
"""
Benchmark runner for vLLM inference server.

Supports three benchmark modes, auto-detected from the YAML config:

  1. **Sanity check** (legacy) — config has `context_lengths`.
     Sequential requests, low concurrency. Unchanged from original.

  2. **Concurrency bench** (Goal 1) — config has `output_token_lengths`
     AND `prompt_token_lengths`.  2-D sweep grid of (prompt, output) with
     a fixed-depth ThreadPoolExecutor (queue always full).

  3. **Context stress** — config has `prompt_token_lengths` and
     `requests.output_tokens` (scalar, not a list).  1-D sweep of prompt
     lengths at fixed output tokens and concurrency.
"""

import argparse
import json
import os
import sys
import time
import urllib.request as _req
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from openai import OpenAI
from tqdm import tqdm

from utils import setup_logger, format_latency, format_throughput, save_json, save_csv
from telemetry import TelemetryCollector


class BenchmarkRunner:
    """Orchestrates benchmark execution and result collection."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config["benchmark"]["name"])
        self.output_dir = Path("/results")
        self.output_dir.mkdir(exist_ok=True)

        # OpenAI-compatible client
        vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
        self.client = OpenAI(base_url=vllm_endpoint, api_key="dummy")

        self.telemetry = (
            TelemetryCollector()
            if self.config.get("telemetry", {}).get("collect_gpu_stats")
            else None
        )

        self.exit_code = 0  # 0 = pass, 1 = partial failure, 2 = total failure

        # Resolve model name
        configured_name = self.config["model"]["name"]
        self._model_name = (
            self._detect_model_name() if configured_name == "auto" else configured_name
        )

    # ── Config & server helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _detect_model_name(self) -> str:
        try:
            models = self.client.models.list()
            name = models.data[0].id
            self.logger.info(f"Auto-detected model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect model name from vLLM: {e}")

    def _wait_for_server(self, timeout: int = 300):
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.client.models.list()
                self.logger.info("✓ vLLM server is ready")
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError("vLLM server did not become ready in time")

    def _get_server_max_model_len(self) -> int:
        endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
        base = endpoint.rstrip("/").removesuffix("/v1")
        try:
            with _req.urlopen(f"{base}/v1/models", timeout=10) as resp:
                data = json.loads(resp.read())
                max_len = int(data["data"][0].get("max_model_len", 2**31))
                self.logger.info(f"Server max_model_len: {max_len:,} tokens")
                return max_len
        except Exception as e:
            self.logger.warning(
                f"Could not query server max_model_len: {e} — using config values as-is"
            )
            return 2**31

    # ── Prompt generation ────────────────────────────────────────────────────

    def _generate_prompt(self, num_tokens: int) -> str:
        """Generate a synthetic prompt of approximately *num_tokens* length."""
        base_text = "Explain the following concept in detail with examples and use cases: "
        filler = (
            "artificial intelligence and machine learning in modern applications "
            * (num_tokens // 10 + 1)
        )
        return base_text + filler[: num_tokens * 4]

    # ── Single request execution ─────────────────────────────────────────────

    def _run_single_request(
        self,
        prompt: str,
        max_tokens: int,
        request_id: int,
        extra_tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute one streaming inference request and return metrics."""
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.config["requests"].get("system_prompt", ""),
                    },
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
                "model": self._model_name,
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
            self.logger.error(f"Request {request_id} failed: {e}")
            result = {
                "request_id": request_id,
                "model": self._model_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

    # ── Concurrent sweep (Goal 1 / context-stress) ───────────────────────────

    def _run_concurrent_sweep(
        self,
        prompt_tokens: int,
        output_tokens: int,
        num_requests: int,
        concurrency: int,
    ) -> List[Dict]:
        """Run *num_requests* at *concurrency* depth for one (prompt, output) point."""
        tags = {
            "prompt_tokens_target": prompt_tokens,
            "output_tokens_target": output_tokens,
        }
        results: list[Dict] = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self._run_single_request,
                    prompt=self._generate_prompt(prompt_tokens),
                    max_tokens=output_tokens,
                    request_id=i,
                    extra_tags=tags,
                )
                for i in range(num_requests)
            ]
            with tqdm(
                total=num_requests,
                desc=f"prompt={prompt_tokens} out={output_tokens}",
            ) as pbar:
                for fut in as_completed(futures):
                    results.append(fut.result())
                    pbar.update(1)

        return results

    # ── Legacy sequential sweep (sanity check) ───────────────────────────────

    def _run_legacy_sweep(self, context_length: int) -> List[Dict]:
        """Original per-context-length sequential loop (sanity check)."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running benchmark for context length: {context_length}")
        self.logger.info(f"{'='*60}")

        cfg = self.config["requests"]
        num_requests = cfg["num_requests"]
        results: list[Dict] = []

        if self.telemetry:
            self.telemetry.start_collection(
                interval=self.config["telemetry"]["sample_interval_sec"]
            )

        with tqdm(total=num_requests, desc=f"Context {context_length}") as pbar:
            for i in range(num_requests):
                prompt_tokens = np.random.randint(
                    cfg["prompt_tokens_min"], cfg["prompt_tokens_max"]
                )
                prompt = self._generate_prompt(prompt_tokens)
                result = self._run_single_request(
                    prompt=prompt,
                    max_tokens=cfg["completion_tokens"],
                    request_id=i,
                )
                result["context_length"] = context_length
                result["prompt_tokens"] = prompt_tokens
                results.append(result)
                pbar.update(1)

                if cfg.get("arrival_pattern") == "poisson" and cfg.get(
                    "rate_per_second"
                ):
                    delay = np.random.exponential(1.0 / cfg["rate_per_second"])
                    time.sleep(delay)

        if self.telemetry:
            self.telemetry.stop_collection()

        return results

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting benchmark: {self.config['benchmark']['name']}")
        self.logger.info(f"Description: {self.config['benchmark']['description']}")
        self._wait_for_server()

        server_max_len = self._get_server_max_model_len()

        # ── Detect mode from config keys ─────────────────────────────────────

        has_context_lengths = "context_lengths" in self.config
        has_output_token_lengths = "output_token_lengths" in self.config
        has_prompt_token_lengths = "prompt_token_lengths" in self.config
        has_fixed_output = "output_tokens" in self.config.get("requests", {})

        # Mode 1: Legacy sanity check
        if has_context_lengths:
            self._run_legacy_mode(server_max_len)
            return

        # Mode 2: Concurrency bench (2-D grid)
        if has_output_token_lengths and has_prompt_token_lengths:
            self._run_concurrent_mode(server_max_len)
            return

        # Mode 3: Context stress (1-D prompt sweep, fixed output)
        if has_prompt_token_lengths and has_fixed_output:
            self._run_context_stress_mode(server_max_len)
            return

        raise ValueError(
            "Could not determine benchmark mode from config. "
            "Expected one of: context_lengths, output_token_lengths + prompt_token_lengths, "
            "or prompt_token_lengths + requests.output_tokens."
        )

    # ── Mode 1: Legacy (sanity check) ────────────────────────────────────────

    def _run_legacy_mode(self, server_max_len: int):
        all_ctx = self.config.get("context_lengths", [8192])
        ctx_ok = [c for c in all_ctx if c <= server_max_len]
        skipped = [c for c in all_ctx if c > server_max_len]
        if skipped:
            self.logger.warning(
                f"Skipping context lengths {skipped} — exceed server max_model_len ({server_max_len:,})."
            )
        if not ctx_ok:
            raise ValueError(
                f"All configured context lengths exceed server max_model_len ({server_max_len:,})."
            )

        all_results: list[Dict] = []
        for c in skipped:
            all_results.append(
                {
                    "model": self._model_name,
                    "context_length": c,
                    "skipped": True,
                    "skip_reason": f"exceeds server max_model_len ({server_max_len:,})",
                }
            )

        for ctx_len in ctx_ok:
            all_results.extend(self._run_legacy_sweep(ctx_len))

        self._save_results(all_results)
        self._print_summary(all_results)
        self.logger.info(f"\n✓ Benchmark complete. Results saved to {self.output_dir}")

    # ── Mode 2: Concurrency bench (2-D prompt × output sweep) ────────────────

    def _run_concurrent_mode(self, server_max_len: int):
        prompt_lengths = self.config["prompt_token_lengths"]
        output_lengths = self.config["output_token_lengths"]
        cfg = self.config["requests"]
        num_requests = cfg["num_requests"]
        concurrency = cfg["concurrency"]

        if self.telemetry:
            self.telemetry.start_collection(
                interval=self.config["telemetry"]["sample_interval_sec"]
            )

        all_results: list[Dict] = []

        for prompt_tokens in prompt_lengths:
            for output_tokens in output_lengths:
                total_needed = prompt_tokens + output_tokens
                if total_needed > server_max_len:
                    self.logger.warning(
                        f"SKIP prompt={prompt_tokens} output={output_tokens} — "
                        f"total {total_needed:,} > max_model_len {server_max_len:,}"
                    )
                    all_results.append(
                        {
                            "model": self._model_name,
                            "prompt_tokens_target": prompt_tokens,
                            "output_tokens_target": output_tokens,
                            "skipped": True,
                            "skip_reason": f"prompt+output ({total_needed:,}) > max_model_len ({server_max_len:,})",
                        }
                    )
                    continue

                self.logger.info(
                    f"\n{'='*60}\n"
                    f"  Sweep point: prompt={prompt_tokens}  output={output_tokens}  "
                    f"concurrency={concurrency}  requests={num_requests}\n"
                    f"{'='*60}"
                )
                results = self._run_concurrent_sweep(
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    num_requests=num_requests,
                    concurrency=concurrency,
                )
                all_results.extend(results)

        if self.telemetry:
            self.telemetry.stop_collection()

        self._save_results(all_results)
        self._save_decision_csv(all_results)
        self._print_summary(all_results)
        self.logger.info(f"\n✓ Benchmark complete. Results saved to {self.output_dir}")

    # ── Mode 3: Context stress (1-D prompt sweep) ────────────────────────────

    def _run_context_stress_mode(self, server_max_len: int):
        prompt_lengths = self.config["prompt_token_lengths"]
        cfg = self.config["requests"]
        output_tokens = cfg["output_tokens"]
        num_requests = cfg["num_requests"]
        concurrency = cfg["concurrency"]

        if self.telemetry:
            self.telemetry.start_collection(
                interval=self.config["telemetry"]["sample_interval_sec"]
            )

        all_results: list[Dict] = []

        for prompt_tokens in prompt_lengths:
            total_needed = prompt_tokens + output_tokens
            if total_needed > server_max_len:
                self.logger.warning(
                    f"SKIP prompt={prompt_tokens} — "
                    f"total {total_needed:,} > max_model_len {server_max_len:,}"
                )
                all_results.append(
                    {
                        "model": self._model_name,
                        "prompt_tokens_target": prompt_tokens,
                        "output_tokens_target": output_tokens,
                        "skipped": True,
                        "skip_reason": f"prompt+output ({total_needed:,}) > max_model_len ({server_max_len:,})",
                    }
                )
                continue

            self.logger.info(
                f"\n{'='*60}\n"
                f"  Context stress: prompt={prompt_tokens}  output={output_tokens}  "
                f"concurrency={concurrency}  requests={num_requests}\n"
                f"{'='*60}"
            )
            results = self._run_concurrent_sweep(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                num_requests=num_requests,
                concurrency=concurrency,
            )
            all_results.extend(results)

        if self.telemetry:
            self.telemetry.stop_collection()

        self._save_results(all_results)
        self._print_summary(all_results)
        self.logger.info(f"\n✓ Benchmark complete. Results saved to {self.output_dir}")

    # ── Result persistence ───────────────────────────────────────────────────

    def _save_results(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.config["benchmark"]["output_prefix"]

        for r in results:
            r.setdefault("model", self._model_name)

        json_path = self.output_dir / f"{prefix}_{timestamp}_detailed.json"
        save_json(results, json_path)

        csv_path = self.output_dir / f"{prefix}_{timestamp}_summary.csv"
        save_csv(results, csv_path)

        if self.telemetry:
            telemetry_data = self.telemetry.get_data()
            telemetry_path = self.output_dir / f"{prefix}_{timestamp}_telemetry.json"
            save_json(telemetry_data, telemetry_path)

        self.logger.info(f"Results saved to {json_path}")
        self.logger.info(f"Summary saved to {csv_path}")

    def _save_decision_csv(self, results: List[Dict]):
        """Write Goal 1 decision CSV: one row per (model, prompt, output) with percentiles."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.config["benchmark"]["output_prefix"]
        csv_path = self.output_dir / f"{prefix}_{timestamp}_decision.csv"

        successful = [r for r in results if r.get("success")]
        if not successful:
            self.logger.warning("No successful requests — cannot write decision CSV.")
            return

        # Group by (prompt_tokens_target, output_tokens_target)
        from collections import defaultdict

        groups: dict[tuple, list] = defaultdict(list)
        for r in successful:
            key = (r.get("prompt_tokens_target"), r.get("output_tokens_target"))
            groups[key].append(r)

        rows: list[Dict] = []
        for (prompt_t, output_t), group in sorted(groups.items()):
            ttfts = [r["ttft_ms"] for r in group if r.get("ttft_ms") is not None]
            itls = [r["itl_ms"] for r in group if r.get("itl_ms") is not None]
            throughputs = [r["throughput_tokens_per_sec"] for r in group]

            row: Dict[str, Any] = {
                "model": self._model_name,
                "prompt_tokens_target": prompt_t,
                "output_tokens_target": output_t,
                "n_requests": len(group),
                "n_success": len([r for r in group if r.get("success")]),
            }
            if ttfts:
                row["P50_ttft_ms"] = round(float(np.percentile(ttfts, 50)), 2)
                row["P95_ttft_ms"] = round(float(np.percentile(ttfts, 95)), 2)
                row["P99_ttft_ms"] = round(float(np.percentile(ttfts, 99)), 2)
            if itls:
                row["P50_itl_ms"] = round(float(np.percentile(itls, 50)), 2)
                row["P95_itl_ms"] = round(float(np.percentile(itls, 95)), 2)
            if throughputs:
                row["mean_throughput_tok_s"] = round(float(np.mean(throughputs)), 2)
            row["total_tokens_generated"] = sum(
                r.get("tokens_generated", 0) for r in group
            )
            rows.append(row)

        save_csv(rows, csv_path)
        self.logger.info(f"Decision CSV saved to {csv_path}")

    # ── Summary printing ─────────────────────────────────────────────────────

    def _print_summary(self, results: List[Dict]):
        from collections import Counter

        skipped = [r for r in results if r.get("skipped")]
        non_skipped = [r for r in results if not r.get("skipped")]
        successful = [r for r in non_skipped if r.get("success")]
        failed = [r for r in non_skipped if not r.get("success")]

        # ── Set exit code ────────────────────────────────────────────────
        if len(non_skipped) == 0 or len(failed) == len(non_skipped):
            self.exit_code = 2  # total failure
        elif len(failed) > 0:
            self.exit_code = 1  # partial failure
        else:
            self.exit_code = 0  # clean pass

        # ── Summary header ───────────────────────────────────────────────
        self.logger.info("\n" + "=" * 60)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Requests: {len(non_skipped)}  (+ {len(skipped)} skipped)")
        self.logger.info(f"Successful:     {len(successful)}")
        self.logger.info(f"Failed:         {len(failed)}")

        # ── Latency & throughput (only when there are successes) ─────────
        if successful:
            ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms")]
            itls = [r["itl_ms"] for r in successful if r.get("itl_ms")]
            throughputs = [r["throughput_tokens_per_sec"] for r in successful]

            if ttfts:
                self.logger.info(f"\nTime to First Token (TTFT):")
                self.logger.info(f"  Mean: {format_latency(np.mean(ttfts))}")
                self.logger.info(f"  P50:  {format_latency(np.percentile(ttfts, 50))}")
                self.logger.info(f"  P95:  {format_latency(np.percentile(ttfts, 95))}")
                self.logger.info(f"  P99:  {format_latency(np.percentile(ttfts, 99))}")

            if itls:
                self.logger.info(f"\nInter-Token Latency (ITL):")
                self.logger.info(f"  Mean: {format_latency(np.mean(itls))}")
                self.logger.info(f"  P50:  {format_latency(np.percentile(itls, 50))}")
                self.logger.info(f"  P95:  {format_latency(np.percentile(itls, 95))}")

            if throughputs:
                self.logger.info(f"\nThroughput:")
                self.logger.info(f"  Mean: {format_throughput(np.mean(throughputs))}")
                self.logger.info(
                    f"  Total tokens: {sum(r.get('tokens_generated', 0) for r in successful)}"
                )

        # ── Failure report ───────────────────────────────────────────────
        if failed:
            self.logger.info("\n" + "-" * 60)
            self.logger.info("FAILURE REPORT")
            self.logger.info("-" * 60)
            error_counts = Counter(r.get("error", "unknown error") for r in failed)
            for rank, (error, count) in enumerate(error_counts.most_common(10), 1):
                truncated = (error[:150] + "…") if len(error) > 150 else error
                self.logger.info(f"  {rank}. [{count}x] {truncated}")
            if len(error_counts) > 10:
                self.logger.info(f"  … and {len(error_counts) - 10} more distinct error(s)")
            self.logger.info("-" * 60)

        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.config)
    runner.run()
    sys.exit(runner.exit_code)


if __name__ == "__main__":
    main()
