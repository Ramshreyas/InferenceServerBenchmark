#!/usr/bin/env python3
"""
Sweep runner  —  reads models.yaml, serves each model, runs benchmarks.

Run from repo root:
    python core/sweep.py --bench context-sweep
    python core/sweep.py --bench kv-analysis
    python core/sweep.py --bench sanity --label mistral-7b
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Benchmark config paths (inside bench-runner container)
# ---------------------------------------------------------------------------
BENCH_CONFIGS = {
    "sanity":        "/configs/sanity_check.yaml",
    "context-sweep": "/configs/context_sweep.yaml",
    "kv-analysis":   "/configs/kv_cache.yaml",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_models(models_file: str = "models.yaml") -> list:
    with open(models_file) as f:
        data = yaml.safe_load(f)
    return data["models"]


def write_env(model: dict, enable_prefix_caching: bool = True, hf_token: str | None = None):
    quant = model.get("quantization", "none")
    quant_flag = f"--quantization {quant}" if quant != "none" else ""
    prefix_flag = "--enable-prefix-caching" if enable_prefix_caching else ""

    lines = [
        f"MODEL_NAME={model['name']}",
        f"TENSOR_PARALLEL={model.get('tensor_parallel', 1)}",
        f"GPU_MEMORY_UTIL={model.get('gpu_memory_util', 0.92)}",
        f"MAX_MODEL_LEN={model.get('max_model_len', 32768)}",
        f"QUANTIZATION_FLAG={quant_flag}",
        f"ENABLE_PREFIX_CACHING_FLAG={prefix_flag}",
    ]
    if hf_token:
        lines.append(f"HF_TOKEN={hf_token}")

    with open(".env", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=== .env written ===")
    for line in lines:
        print(f"  {line}")
    print()


def run(cmd: list, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def wait_for_vllm(host: str = "localhost", port: int = 8000, timeout: int = 600):
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    print(f">>> Waiting for vLLM at {url} (up to {timeout}s)…")
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            print(">>> vLLM is ready!\n")
            return
        except Exception:
            time.sleep(10)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s")


def serve_model(model: dict, enable_prefix_caching: bool = True):
    hf_token = os.environ.get("HF_TOKEN")
    write_env(model, enable_prefix_caching=enable_prefix_caching, hf_token=hf_token)
    run(["docker", "compose", "down"])
    run(["docker", "compose", "up", "-d", "vllm"])
    wait_for_vllm()


def run_bench(bench_key: str):
    config_path = BENCH_CONFIGS[bench_key]
    run([
        "docker", "compose", "run", "--rm", "bench-runner",
        "python", "/app/core/bench_runner.py", "--config", config_path,
    ])


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep benchmarks over all models listed in models.yaml"
    )
    parser.add_argument(
        "--bench", choices=list(BENCH_CONFIGS.keys()),
        help="Which benchmark config to run (omit with --serve-only)",
    )
    parser.add_argument(
        "--serve-only", action="store_true",
        help="Only serve the model — skip the benchmark (useful for manual bench runs)",
    )
    parser.add_argument(
        "--models-file", default="models.yaml",
        help="Path to models YAML file (default: models.yaml)",
    )
    parser.add_argument(
        "--label",
        help="Run only the model with this label",
    )
    args = parser.parse_args()

    if not args.serve_only and not args.bench:
        parser.error("--bench is required unless --serve-only is set")

    models = load_models(args.models_file)

    # Filter by label if requested
    if args.label:
        models = [m for m in models if m.get("label") == args.label]
        if not models:
            print(f"ERROR: No model with label '{args.label}' found in {args.models_file}", file=sys.stderr)
            sys.exit(1)

    if args.serve_only:
        if len(models) != 1:
            parser.error("--serve-only requires exactly one model; use --label to select one")
        label = models[0].get("label", models[0]["name"].split("/")[-1])
        section(f"SERVE: {models[0]['name']}  (label={label})")
        serve_model(models[0], enable_prefix_caching=True)
        print("Server is ready. Run `make bench-<name>` to benchmark against it.")
        return

    section(f"SWEEP: {args.bench}  |  {len(models)} model(s)")

    # ── KV analysis: run each model TWICE (prefix caching on vs off) ──────────
    if args.bench == "kv-analysis":
        for i, model in enumerate(models, 1):
            label = model.get("label", model["name"].split("/")[-1])

            print(f"\n[{i}/{len(models)}] Model: {model['name']}  (label={label})")

            for caching in (True, False):
                tag = "prefix-ON" if caching else "prefix-OFF"
                print(f"\n  ── {tag} ──")
                serve_model(model, enable_prefix_caching=caching)
                run_bench(args.bench)

    # ── All other benchmarks: serve once, bench once ──────────────────────────
    else:
        for i, model in enumerate(models, 1):
            label = model.get("label", model["name"].split("/")[-1])
            print(f"\n[{i}/{len(models)}] Serving: {model['name']}  (label={label})")
            serve_model(model, enable_prefix_caching=True)
            print(f"[{i}/{len(models)}] Benchmarking: {args.bench}")
            run_bench(args.bench)

    section(f"SWEEP COMPLETE: {args.bench}")


if __name__ == "__main__":
    main()
