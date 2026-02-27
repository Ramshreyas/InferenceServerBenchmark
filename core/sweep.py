#!/usr/bin/env python3
"""
Sweep runner  —  reads models.yaml, serves each model, runs benchmarks.

Run from repo root:
    python core/sweep.py --bench concurrency-bench
    python core/sweep.py --bench sanity --label mistral-7b
    python core/sweep.py --bench co-deploy --label-large gpt-oss-120b --label-small qwen3-8b
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from itertools import product
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Benchmark config paths (inside bench-runner container)
# ---------------------------------------------------------------------------
BENCH_CONFIGS = {
    "sanity":            "/configs/sanity_check.yaml",
    "concurrency-bench": "/configs/concurrency_bench.yaml",
    # co-deploy uses a separate code path (co_deploy_sweep)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_models(models_file: str = "models.yaml") -> list:
    with open(models_file) as f:
        data = yaml.safe_load(f)
    return data["models"]


def load_config(bench_key: str) -> dict:
    """Load a benchmark config from the local configs/ directory."""
    config_path = BENCH_CONFIGS[bench_key].replace("/configs/", "configs/")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _model_flag(key: str, value, prefix: str = "") -> str:
    """Build a CLI flag string.  Returns '' if the value is falsy / 'none'."""
    if not value or str(value) == "none":
        return ""
    return f"--{key} {value}"


def write_env(model: dict, enable_prefix_caching: bool = True, hf_token: str | None = None):
    quant = model.get("quantization", "none")
    quant_flag = f"--quantization {quant}" if quant != "none" else ""
    prefix_flag = "--enable-prefix-caching" if enable_prefix_caching else ""

    max_len = model.get("max_model_len", 0)
    if max_len and str(max_len) != "auto":
        max_len_flag = f"--max-model-len {max_len}"
    else:
        max_len_flag = ""

    extra_flags = model.get("vllm_extra_flags", "")

    lines = [
        f"MODEL_NAME={model['name']}",
        f"TENSOR_PARALLEL={model.get('tensor_parallel', 1)}",
        f"GPU_MEMORY_UTIL={model.get('gpu_memory_util', 0.92)}",
        f"MAX_MODEL_LEN_FLAG={max_len_flag}",
        f"QUANTIZATION_FLAG={quant_flag}",
        f"ENABLE_PREFIX_CACHING_FLAG={prefix_flag}",
        f"EXTRA_VLLM_FLAGS={extra_flags}",
        # Placeholders so docker compose doesn't warn about unset vars
        "SMALL_MODEL_NAME=",
        "SMALL_TENSOR_PARALLEL=1",
        "SMALL_GPU_MEMORY_UTIL=0.50",
        "SMALL_MAX_MODEL_LEN_FLAG=",
        "SMALL_QUANTIZATION_FLAG=",
        "SMALL_EXTRA_VLLM_FLAGS=",
    ]
    if hf_token:
        lines.append(f"HF_TOKEN={hf_token}")

    with open(".env", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=== .env written ===")
    for line in lines:
        print(f"  {line}")
    print()


def write_env_dual(large: dict, small: dict, lg_util: float = 0.65, sm_util: float = 0.30, hf_token: str | None = None):
    """Write .env for a co-deploy run (two models on one GPU).
    
    lg_util / sm_util are the auto-computed gpu_memory_util values from
    compute_co_deploy_memory(), NOT the solo gpu_memory_util from models.yaml.
    """
    lg_quant = large.get("quantization", "none")
    sm_quant = small.get("quantization", "none")

    lg_max = large.get("max_model_len", 0)
    sm_max = small.get("max_model_len", 0)

    lg_extra = large.get("vllm_extra_flags", "")
    sm_extra = small.get("vllm_extra_flags", "")

    lines = [
        # ── Large model (vllm-large, port 8000) ──
        f"MODEL_NAME={large['name']}",
        f"TENSOR_PARALLEL={large.get('tensor_parallel', 1)}",
        f"GPU_MEMORY_UTIL={lg_util}",
        f"MAX_MODEL_LEN_FLAG={f'--max-model-len {lg_max}' if lg_max and str(lg_max) != 'auto' else ''}",
        f"QUANTIZATION_FLAG={f'--quantization {lg_quant}' if lg_quant != 'none' else ''}",
        f"ENABLE_PREFIX_CACHING_FLAG=--enable-prefix-caching",
        f"EXTRA_VLLM_FLAGS={lg_extra}",
        # ── Small model (vllm-small, port 8001) ──
        f"SMALL_MODEL_NAME={small['name']}",
        f"SMALL_TENSOR_PARALLEL={small.get('tensor_parallel', 1)}",
        f"SMALL_GPU_MEMORY_UTIL={sm_util}",
        f"SMALL_MAX_MODEL_LEN_FLAG={f'--max-model-len {sm_max}' if sm_max and str(sm_max) != 'auto' else ''}",
        f"SMALL_QUANTIZATION_FLAG={f'--quantization {sm_quant}' if sm_quant != 'none' else ''}",
        f"SMALL_EXTRA_VLLM_FLAGS={sm_extra}",
    ]
    if hf_token:
        lines.append(f"HF_TOKEN={hf_token}")

    with open(".env", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("=== .env written (co-deploy) ===")
    for line in lines:
        print(f"  {line}")
    print()


def probe_max_model_len(host: str = "localhost", port: int = 8000) -> int | None:
    """Query vLLM for the max_model_len it actually started with."""
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return int(data["data"][0].get("max_model_len", 0)) or None
    except Exception as e:
        print(f"WARNING: Could not probe max_model_len: {e}")
        return None


def run(cmd: list, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print(f">>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kwargs)


def compose_down_all(check: bool = True):
    """Tear down ALL services (including co-deploy profile) and prune stale networks."""
    run(["docker", "compose", "--profile", "co-deploy", "down", "--remove-orphans"], check=check)
    run(["docker", "network", "prune", "-f"], check=False)


def compose_up(args: list[str], retries: int = 3, delay: float = 5.0):
    """Run 'docker compose ... up' with retries to handle Docker network race conditions.

    After 'docker compose down' removes the network, Docker may still reference
    the stale network ID internally.  Between retries we tear down ALL services
    (including profiled ones), prune dangling networks, and sleep.
    """
    cmd = ["docker", "compose"] + args
    for attempt in range(1, retries + 1):
        result = run(cmd, check=False)
        if result.returncode == 0:
            return
        print(f">>> compose up failed (attempt {attempt}/{retries}), retrying in {delay}s …", flush=True)
        compose_down_all(check=False)
        time.sleep(delay)
    # Final attempt — let it raise on failure
    run(cmd, check=True)


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


def _get_running_model(port: int = 8000) -> str | None:
    """Return the model name currently served by vLLM, or None if unreachable."""
    url = f"http://localhost:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return data["data"][0]["id"]
    except Exception:
        return None


def serve_model(model: dict, enable_prefix_caching: bool = True):
    hf_token = os.environ.get("HF_TOKEN")
    wanted = model["name"]

    # Skip teardown if the same model is already healthy on port 8000
    running = _get_running_model(port=8000)
    if running == wanted:
        print(f">>> {wanted} is already running — reusing container", flush=True)
        write_env(model, enable_prefix_caching=enable_prefix_caching, hf_token=hf_token)
        return

    write_env(model, enable_prefix_caching=enable_prefix_caching, hf_token=hf_token)
    compose_down_all()
    compose_up(["up", "-d", "vllm-large"])
    wait_for_vllm(timeout=1800)  # large models can take 15-30 min to load on first run


def run_bench(bench_key: str) -> int:
    """Run benchmark container; return exit code (0=pass, 1=partial fail, 2+=total fail)."""
    config_path = BENCH_CONFIGS[bench_key]
    result = run([
        "docker", "compose", "run", "--rm", "bench-runner",
        "python", "/app/core/bench_runner.py", "--config", config_path,
    ], check=False)
    return result.returncode


def section(title: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# Context-window fitness filter
# ---------------------------------------------------------------------------

def context_window_check(model: dict, bench_key: str) -> bool:
    """Return True if the model can satisfy at least the smallest prompt+output tier."""
    max_len = model.get("max_model_len", 0)
    if not max_len or str(max_len) == "auto":
        return True  # unknown ceiling → let runtime decide

    cfg = load_config(bench_key)
    prompt_lens = cfg.get("prompt_token_lengths", [0])
    output_lens = cfg.get("output_token_lengths", [0])

    min_needed = min(prompt_lens) + min(output_lens)
    if min_needed > int(max_len):
        return False
    return True


# ---------------------------------------------------------------------------
# Co-deploy memory allocation
# ---------------------------------------------------------------------------

GPU_VRAM_GB = 96.0
CO_DEPLOY_TOTAL_BUDGET = 0.92   # leave 8% for CUDA context / driver overhead
CO_DEPLOY_HEADROOM = 1.15       # 15% headroom over loaded_gb for KV cache / activations

def compute_co_deploy_memory(large: dict, small: dict) -> tuple[float | None, float | None]:
    """Return (large_util, small_util) proportional to loaded_gb, or (None, None) if pair cannot fit."""
    lg_gb = large.get("loaded_gb", 0)
    sm_gb = small.get("loaded_gb", 0)
    if not lg_gb or not sm_gb:
        print(
            f"WARNING: missing loaded_gb for {large.get('label')} or {small.get('label')} — skipping pair"
        )
        return None, None

    # Estimated VRAM required (weights + KV headroom)
    lg_needed = lg_gb * CO_DEPLOY_HEADROOM
    sm_needed = sm_gb * CO_DEPLOY_HEADROOM
    total_needed = lg_needed + sm_needed

    if total_needed > GPU_VRAM_GB * CO_DEPLOY_TOTAL_BUDGET:
        return None, None  # won't physically fit

    # Proportional split within the total budget
    ratio = lg_needed / total_needed
    lg_util = round(CO_DEPLOY_TOTAL_BUDGET * ratio, 2)
    sm_util = round(CO_DEPLOY_TOTAL_BUDGET * (1 - ratio), 2)
    return lg_util, sm_util


# ---------------------------------------------------------------------------
# Co-deploy sweep  (Goal 2)
# ---------------------------------------------------------------------------

def co_deploy_sweep(models: list, label_large: str | None = None, label_small: str | None = None):
    large_pool = [m for m in models if m.get("role") == "large"]
    small_pool = [m for m in models if m.get("role") == "small"]

    if label_large:
        large_pool = [m for m in large_pool if m.get("label") == label_large]
    if label_small:
        small_pool = [m for m in small_pool if m.get("label") == label_small]

    if not large_pool:
        print("ERROR: No 'large' role models found (or label filter matched nothing).", file=sys.stderr)
        sys.exit(1)
    if not small_pool:
        print("ERROR: No 'small' role models found (or label filter matched nothing).", file=sys.stderr)
        sys.exit(1)

    # Build viable pairs — auto-compute memory splits from loaded_gb
    pairs: list[tuple[dict, dict, float, float]] = []
    for lg, sm in product(large_pool, small_pool):
        lg_util, sm_util = compute_co_deploy_memory(lg, sm)
        if lg_util is None:
            print(
                f"SKIP: {lg.get('label')} + {sm.get('label')} — "
                f"combined VRAM estimate ({lg.get('loaded_gb', '?')}+{sm.get('loaded_gb', '?')} GB) "
                f"exceeds budget ({GPU_VRAM_GB * CO_DEPLOY_TOTAL_BUDGET:.0f} GB)"
            )
        else:
            pairs.append((lg, sm, lg_util, sm_util))

    if not pairs:
        print("ERROR: No viable (large, small) pair fits in VRAM.", file=sys.stderr)
        sys.exit(1)

    section(f"CO-DEPLOY SWEEP  |  {len(pairs)} pair(s)")
    hf_token = os.environ.get("HF_TOKEN")
    failed = []

    for lg, sm, lg_util, sm_util in pairs:
        lg_label = lg.get("label", lg["name"].split("/")[-1])
        sm_label = sm.get("label", sm["name"].split("/")[-1])
        section(f"CO-DEPLOY: {lg_label} ({lg_util:.0%}) + {sm_label} ({sm_util:.0%})")

        write_env_dual(lg, sm, lg_util=lg_util, sm_util=sm_util, hf_token=hf_token)

        # Skip teardown if both models are already running on correct ports
        running_lg = _get_running_model(port=8000)
        running_sm = _get_running_model(port=8001)
        if running_lg == lg["name"] and running_sm == sm["name"]:
            print(f">>> {lg_label} + {sm_label} already running — reusing containers", flush=True)
        else:
            compose_down_all()

            # Start models SEQUENTIALLY to avoid GPU memory allocation race.
            # If both vLLM instances call torch.cuda.mem_get_info() at the same
            # time, they both see all 96 GB free and both try to over-allocate.
            print(">>> Starting vllm-large first (sequential to avoid memory race)…", flush=True)
            compose_up(["--profile", "co-deploy", "up", "-d", "vllm-large"])
            try:
                wait_for_vllm(port=8000, timeout=1800)
            except TimeoutError as e:
                print(f"  *** TIMEOUT (vllm-large): {e}", file=sys.stderr)
                print("  --- vllm-large logs (last 80 lines) ---", flush=True)
                run(["docker", "logs", "--tail", "80", "vllm-large"], check=False)
                failed.append(f"{lg_label}+{sm_label}")
                compose_down_all(check=False)
                continue

            print(">>> vllm-large healthy — now starting vllm-small…", flush=True)
            compose_up(["--profile", "co-deploy", "up", "-d", "vllm-small"])
            try:
                wait_for_vllm(port=8001, timeout=1800)
            except TimeoutError as e:
                print(f"  *** TIMEOUT (vllm-small): {e}", file=sys.stderr)
                print("  --- vllm-small logs (last 80 lines) ---", flush=True)
                run(["docker", "logs", "--tail", "80", "vllm-small"], check=False)
                failed.append(f"{lg_label}+{sm_label}")
                compose_down_all(check=False)
                continue

        try:
            run([
                "docker", "compose", "--profile", "co-deploy",
                "run", "--rm", "co-runner",
                "python", "/app/core/co_deploy_runner.py",
                "--config", "/configs/split_load.yaml",
            ])
        except Exception as e:
            print(f"  *** BENCH FAILED: {e}", file=sys.stderr)
            failed.append(f"{lg_label}+{sm_label}")
        finally:
            compose_down_all(check=False)

    section("CO-DEPLOY SWEEP COMPLETE")
    if failed:
        print(f"WARNING: {len(failed)} pair(s) failed: {failed}")


# ---------------------------------------------------------------------------
# Report card
# ---------------------------------------------------------------------------

def _print_report_card(bench_key: str, report: list[tuple[str, str, str]]):
    """Print a clear pass/fail report card at the end of a sweep."""
    section(f"REPORT CARD: {bench_key}")

    if not report:
        print("  (no models were tested)")
        return

    symbols = {
        "PASS":       "\u2713 PASS",
        "PARTIAL":    "\u25d0 PARTIAL",
        "FAIL":       "\u2717 FAIL",
        "SERVE_FAIL": "\u2717 SERVE_FAIL",
        "SKIP":       "\u2298 SKIP",
    }

    max_label = max(len(label) for label, _, _ in report)
    for label, status, detail in report:
        sym = symbols.get(status, status)
        print(f"  {label:<{max_label}}  {sym:<16} {detail}")

    # ── Tallies ───────────────────────────────────────────────────────────
    n_pass    = sum(1 for _, s, _ in report if s == "PASS")
    n_partial = sum(1 for _, s, _ in report if s == "PARTIAL")
    n_fail    = sum(1 for _, s, _ in report if s in ("FAIL", "SERVE_FAIL"))
    n_skip    = sum(1 for _, s, _ in report if s == "SKIP")
    total     = len(report)

    parts = [f"{n_pass}/{total} passed"]
    if n_partial:
        parts.append(f"{n_partial} partial")
    if n_fail:
        parts.append(f"{n_fail} failed")
    if n_skip:
        parts.append(f"{n_skip} skipped")
    print(f"\n  {',  '.join(parts)}")

    # ── Action items ─────────────────────────────────────────────────────
    action_items = [(l, d) for l, s, d in report if s in ("FAIL", "SERVE_FAIL", "PARTIAL")]
    if action_items:
        print(f"\n  Action items:")
        for label, detail in action_items:
            print(f"    \u2192 {label}: {detail}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_bench_choices = list(BENCH_CONFIGS.keys()) + ["co-deploy"]

    parser = argparse.ArgumentParser(
        description="Sweep benchmarks over all models listed in models.yaml"
    )
    parser.add_argument(
        "--bench", choices=all_bench_choices,
        help="Which benchmark to run (omit with --serve-only)",
    )
    parser.add_argument(
        "--serve-only", action="store_true",
        help="Only serve the model — skip the benchmark (useful for manual bench runs)",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Start each model without --max-model-len and print the auto-detected value. "
             "Use this once to find correct max_model_len values for models.yaml.",
    )
    parser.add_argument(
        "--models-file", default="models.yaml",
        help="Path to models YAML file (default: models.yaml)",
    )
    parser.add_argument(
        "--label",
        help="Run only the model with this label (single-model benchmarks)",
    )
    parser.add_argument(
        "--label-large",
        help="(co-deploy only) Filter large-role models to this label",
    )
    parser.add_argument(
        "--label-small",
        help="(co-deploy only) Filter small-role models to this label",
    )
    args = parser.parse_args()

    if not args.serve_only and not args.bench and not args.probe:
        parser.error("--bench or --probe is required unless --serve-only is set")

    models = load_models(args.models_file)

    # ── Co-deploy: separate code path ─────────────────────────────────────────
    if args.bench == "co-deploy":
        co_deploy_sweep(models, label_large=args.label_large, label_small=args.label_small)
        return

    # Filter by label if requested (for single-model benchmarks)
    if args.label:
        target_labels = set(args.label.split(","))
        models = [m for m in models if m.get("label") in target_labels]
        if not models:
            print(f"ERROR: No models found matching labels '{args.label}' in {args.models_file}", file=sys.stderr)
            sys.exit(1)

    if args.serve_only:
        if len(models) != 1:
            parser.error("--serve-only requires exactly one model; use --label to select one")
        label = models[0].get("label", models[0]["name"].split("/")[-1])
        section(f"SERVE: {models[0]['name']}  (label={label})")
        serve_model(models[0], enable_prefix_caching=True)
        print("Server is ready. Run `make bench-<name>` to benchmark against it.")
        return

    # ── Probe mode: auto-detect max_model_len for each model ─────────────────
    if args.probe:
        section(f"PROBE  |  {len(models)} model(s)")
        print("Starting each model WITHOUT --max-model-len so vLLM auto-caps from gpu_memory_util.\n")
        suggestions = []
        for i, model in enumerate(models, 1):
            label = model.get("label", model["name"].split("/")[-1])
            print(f"\n[{i}/{len(models)}] Probing: {model['name']}  (label={label})")
            probe_model = {**model, "max_model_len": 0}
            try:
                serve_model(probe_model, enable_prefix_caching=False)
                detected = probe_max_model_len()
                if detected:
                    suggestions.append((label, model["name"], detected))
                    print(f"  → Detected max_model_len: {detected:,}")
                else:
                    print(f"  → Could not detect max_model_len")
            except Exception as e:
                print(f"  *** FAILED {label}: {e}", file=sys.stderr)
            finally:
                compose_down_all(check=False)

        section("PROBE RESULTS — paste into models.yaml")
        for label, name, detected in suggestions:
            print(f"  {label} ({name})")
            print(f"    max_model_len: {detected}")
            print()
        return

    # ── Single-model sweep: serve → bench for each model ──────────────────────
    bench_key = args.bench
    section(f"SWEEP: {bench_key}  |  {len(models)} model(s)")

    report: list[tuple[str, str, str]] = []  # (label, status, detail)

    for i, model in enumerate(models, 1):
        label = model.get("label", model["name"].split("/")[-1])

        # Context-window fitness check
        if bench_key in ("concurrency-bench",):
            if not context_window_check(model, bench_key):
                max_len = model.get("max_model_len", "?")
                detail = f"max_model_len={max_len} too small for {bench_key}"
                print(f"  SKIP {label}: {detail}")
                report.append((label, "SKIP", detail))
                continue

        print(f"\n[{i}/{len(models)}] Serving: {model['name']}  (label={label})")
        try:
            serve_model(model, enable_prefix_caching=True)
        except Exception as e:
            msg = str(e)
            print(f"  *** FAILED to start {label}: {msg} — skipping", file=sys.stderr)
            report.append((label, "SERVE_FAIL", msg))
            continue

        print(f"[{i}/{len(models)}] Benchmarking: {bench_key}")
        exit_code = run_bench(bench_key)
        if exit_code == 0:
            report.append((label, "PASS", "all requests succeeded"))
        elif exit_code == 1:
            report.append((label, "PARTIAL", "some requests failed — check detailed JSON"))
        else:
            report.append((label, "FAIL", f"bench exited with code {exit_code}"))

    # Final cleanup
    compose_down_all(check=False)

    # ── Report card ──────────────────────────────────────────────────────
    _print_report_card(bench_key, report)
    sys.stdout.flush()
    return


if __name__ == "__main__":
    main()
