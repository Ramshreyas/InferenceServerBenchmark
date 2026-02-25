#!/usr/bin/env python3
"""
Co-deploy split-load benchmark runner (Goal 2).

Benchmarks two simultaneously-running vLLM instances (large + small) on the
same GPU.  Traffic is split 70/30 (configurable) and routed to the
appropriate endpoint.

Entry point:
    python core/co_deploy_runner.py --config /configs/split_load.yaml
"""

import argparse
import json
import os
import time
import urllib.request as _req
from collections import defaultdict
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


class CoDeployRunner:
    """Benchmarks two co-deployed models under a traffic split."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(self.config["benchmark"]["name"])
        self.output_dir = Path("/results")
        self.output_dir.mkdir(exist_ok=True)

        # ── Endpoints ────────────────────────────────────────────────────────
        large_url = os.getenv("VLLM_ENDPOINT_LARGE", "http://vllm-large:8000/v1")
        small_url = os.getenv("VLLM_ENDPOINT_SMALL", "http://vllm-small:8001/v1")

        self.client_large = OpenAI(base_url=large_url, api_key="dummy")
        self.client_small = OpenAI(base_url=small_url, api_key="dummy")

        # Model names (env or auto-detect)
        self.large_model = os.getenv("LARGE_MODEL_NAME") or self._detect(self.client_large, "large")
        self.small_model = os.getenv("SMALL_MODEL_NAME") or self._detect(self.client_small, "small")

        self.telemetry = (
            TelemetryCollector()
            if self.config.get("telemetry", {}).get("collect_gpu_stats")
            else None
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _detect(self, client: OpenAI, label: str) -> str:
        try:
            name = client.models.list().data[0].id
            self.logger.info(f"Auto-detected {label} model: {name}")
            return name
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect {label} model: {e}")

    def _wait_for_server(self, client: OpenAI, label: str, timeout: int = 300):
        start = time.time()
        while time.time() - start < timeout:
            try:
                client.models.list()
                self.logger.info(f"✓ {label} server is ready")
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError(f"{label} server did not become ready in {timeout}s")

    def _get_max_model_len(self, base_url: str) -> int:
        base = base_url.rstrip("/").removesuffix("/v1")
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

    # ── Single request ───────────────────────────────────────────────────────

    def _run_single_request(
        self,
        client: OpenAI,
        model_name: str,
        endpoint_label: str,
        prompt: str,
        max_tokens: int,
        request_id: int,
        extra_tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=model_name,
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
                "endpoint": endpoint_label,
                "model": model_name,
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
            self.logger.error(f"Request {request_id} ({endpoint_label}) failed: {e}")
            result: Dict[str, Any] = {
                "request_id": request_id,
                "endpoint": endpoint_label,
                "model": model_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

    # ── Main ─────────────────────────────────────────────────────────────────

    def run(self):
        self.logger.info(f"Starting co-deploy benchmark: {self.config['benchmark']['name']}")
        self.logger.info(f"Large model: {self.large_model}")
        self.logger.info(f"Small model: {self.small_model}")

        self._wait_for_server(self.client_large, "large")
        self._wait_for_server(self.client_small, "small")

        large_base = os.getenv("VLLM_ENDPOINT_LARGE", "http://vllm-large:8000/v1")
        small_base = os.getenv("VLLM_ENDPOINT_SMALL", "http://vllm-small:8001/v1")
        large_max_len = self._get_max_model_len(large_base)
        small_max_len = self._get_max_model_len(small_base)
        min_max_len = min(large_max_len, small_max_len)
        self.logger.info(
            f"max_model_len  large={large_max_len:,}  small={small_max_len:,}  min={min_max_len:,}"
        )

        # Split config
        split = self.config["split"]
        total_concurrency = split["total_concurrency"]
        n_large = round(total_concurrency * split["large_fraction"])
        n_small = total_concurrency - n_large
        self.logger.info(f"Traffic split: {n_large} → large, {n_small} → small")

        prompt_lengths = self.config["prompt_token_lengths"]
        output_lengths = self.config["output_token_lengths"]
        num_requests = self.config["requests"]["num_requests"]

        if self.telemetry:
            self.telemetry.start_collection(
                interval=self.config["telemetry"]["sample_interval_sec"]
            )

        all_results: list[Dict] = []

        for prompt_tokens in prompt_lengths:
            for output_tokens in output_lengths:
                total_needed = prompt_tokens + output_tokens
                if total_needed > min_max_len:
                    self.logger.warning(
                        f"SKIP prompt={prompt_tokens} output={output_tokens} — "
                        f"total {total_needed:,} > min max_model_len {min_max_len:,}"
                    )
                    continue

                self.logger.info(
                    f"\n{'='*60}\n"
                    f"  Co-deploy sweep: prompt={prompt_tokens}  output={output_tokens}\n"
                    f"  large={n_large} slots  small={n_small} slots  total_requests={num_requests}\n"
                    f"{'='*60}"
                )

                tags = {
                    "prompt_tokens_target": prompt_tokens,
                    "output_tokens_target": output_tokens,
                }

                results: list[Dict] = []
                with ThreadPoolExecutor(max_workers=total_concurrency) as executor:
                    futures = []
                    for i in range(num_requests):
                        if i % total_concurrency < n_large:
                            client = self.client_large
                            model = self.large_model
                            label = "large"
                        else:
                            client = self.client_small
                            model = self.small_model
                            label = "small"

                        futures.append(
                            executor.submit(
                                self._run_single_request,
                                client=client,
                                model_name=model,
                                endpoint_label=label,
                                prompt=self._generate_prompt(prompt_tokens),
                                max_tokens=output_tokens,
                                request_id=i,
                                extra_tags=tags,
                            )
                        )

                    with tqdm(
                        total=num_requests,
                        desc=f"p={prompt_tokens} o={output_tokens}",
                    ) as pbar:
                        for fut in as_completed(futures):
                            results.append(fut.result())
                            pbar.update(1)

                all_results.extend(results)

        if self.telemetry:
            self.telemetry.stop_collection()

        self._save_results(all_results)
        self._save_summary_csv(all_results)
        self._print_summary(all_results)
        self.logger.info(f"\n✓ Co-deploy benchmark complete. Results saved to {self.output_dir}")

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_results(self, results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.config["benchmark"]["output_prefix"]

        json_path = self.output_dir / f"{prefix}_{timestamp}_detailed.json"
        save_json(results, json_path)
        self.logger.info(f"Detailed results → {json_path}")

        if self.telemetry:
            tel_path = self.output_dir / f"{prefix}_{timestamp}_telemetry.json"
            save_json(self.telemetry.get_data(), tel_path)
            self.logger.info(f"Telemetry → {tel_path}")

    def _save_summary_csv(self, results: List[Dict]):
        """One row per (large_model, small_model, prompt, output) with per-endpoint percentiles."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.config["benchmark"]["output_prefix"]
        csv_path = self.output_dir / f"{prefix}_{timestamp}_summary.csv"

        successful = [r for r in results if r.get("success")]
        if not successful:
            self.logger.warning("No successful requests — cannot write summary CSV.")
            return

        # Group by (prompt_tokens_target, output_tokens_target)
        groups: dict[tuple, list] = defaultdict(list)
        for r in successful:
            key = (r.get("prompt_tokens_target"), r.get("output_tokens_target"))
            groups[key].append(r)

        rows: list[Dict] = []
        for (prompt_t, output_t), group in sorted(groups.items()):
            large_reqs = [r for r in group if r.get("endpoint") == "large"]
            small_reqs = [r for r in group if r.get("endpoint") == "small"]

            row: Dict[str, Any] = {
                "large_model": self.large_model,
                "small_model": self.small_model,
                "prompt_tokens_target": prompt_t,
                "output_tokens_target": output_t,
            }

            for label, reqs in [("large", large_reqs), ("small", small_reqs)]:
                ttfts = [r["ttft_ms"] for r in reqs if r.get("ttft_ms") is not None]
                itls = [r["itl_ms"] for r in reqs if r.get("itl_ms") is not None]
                if ttfts:
                    row[f"{label}_P50_ttft_ms"] = round(float(np.percentile(ttfts, 50)), 2)
                    row[f"{label}_P95_ttft_ms"] = round(float(np.percentile(ttfts, 95)), 2)
                    row[f"{label}_P99_ttft_ms"] = round(float(np.percentile(ttfts, 99)), 2)
                if itls:
                    row[f"{label}_P50_itl_ms"] = round(float(np.percentile(itls, 50)), 2)
                    row[f"{label}_P95_itl_ms"] = round(float(np.percentile(itls, 95)), 2)

            # Combined throughput
            all_throughputs = [r["throughput_tokens_per_sec"] for r in group]
            if all_throughputs:
                row["combined_throughput_tok_s"] = round(float(np.sum(all_throughputs)), 2)

            rows.append(row)

        save_csv(rows, csv_path)
        self.logger.info(f"Summary CSV → {csv_path}")

    def _print_summary(self, results: List[Dict]):
        successful = [r for r in results if r.get("success")]
        if not successful:
            self.logger.error("No successful requests!")
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("CO-DEPLOY BENCHMARK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Large model: {self.large_model}")
        self.logger.info(f"Small model: {self.small_model}")
        self.logger.info(f"Total Requests: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(results) - len(successful)}")

        for label in ("large", "small"):
            reqs = [r for r in successful if r.get("endpoint") == label]
            if not reqs:
                continue
            ttfts = [r["ttft_ms"] for r in reqs if r.get("ttft_ms")]
            itls = [r["itl_ms"] for r in reqs if r.get("itl_ms")]
            self.logger.info(f"\n  [{label.upper()} endpoint]")
            if ttfts:
                self.logger.info(f"    TTFT  P50={format_latency(np.percentile(ttfts, 50))}  "
                                 f"P95={format_latency(np.percentile(ttfts, 95))}  "
                                 f"P99={format_latency(np.percentile(ttfts, 99))}")
            if itls:
                self.logger.info(f"    ITL   P50={format_latency(np.percentile(itls, 50))}  "
                                 f"P95={format_latency(np.percentile(itls, 95))}")

        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Co-deploy split-load benchmark")
    parser.add_argument("--config", required=True, help="Path to split_load YAML config")
    args = parser.parse_args()

    runner = CoDeployRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
