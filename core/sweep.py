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
from datetime import datetime
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

STT_BENCH_CONFIGS = {
    "stt-sanity":            "/configs/stt_sanity.yaml",
    "stt-concurrency-bench": "/configs/stt_concurrency_bench.yaml",
}

STT_STREAMING_BENCH_CONFIGS = {
    "stt-streaming-sanity": "/configs/stt_streaming_sanity.yaml",
    "stt-streaming-bench":  "/configs/stt_streaming_bench.yaml",
}

MIXED_BENCH_CONFIGS = {
    "mixed-co-deploy": "/configs/mixed_co_deploy.yaml",
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
    vllm_image = model.get("vllm_image", "vllm/vllm-openai:cu130-nightly")
    # The gemma4 image bakes in a patched transformers; reinstalling vllm[audio]
    # in the entrypoint would clobber it and break model loading.
    skip_pip = "1" if "gemma4" in vllm_image else "0"

    lines = [
        f"VLLM_IMAGE={vllm_image}",
        f"MODEL_NAME={model['name']}",
        f"TENSOR_PARALLEL={model.get('tensor_parallel', 1)}",
        f"GPU_MEMORY_UTIL={model.get('gpu_memory_util', 0.92)}",
        f"MAX_MODEL_LEN_FLAG={max_len_flag}",
        f"QUANTIZATION_FLAG={quant_flag}",
        f"ENABLE_PREFIX_CACHING_FLAG={prefix_flag}",
        f"EXTRA_VLLM_FLAGS={extra_flags}",
        f"SKIP_ENTRYPOINT_PIP={skip_pip}",
        # Placeholders so docker compose doesn't warn about unset vars
        "SMALL_VLLM_IMAGE=vllm/vllm-openai:cu130-nightly",
        "SMALL_MODEL_NAME=",
        "SMALL_TENSOR_PARALLEL=1",
        "SMALL_GPU_MEMORY_UTIL=0.50",
        "SMALL_MAX_MODEL_LEN_FLAG=",
        "SMALL_QUANTIZATION_FLAG=",
        "SMALL_EXTRA_VLLM_FLAGS=",
        "SMALL_SKIP_ENTRYPOINT_PIP=0",
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

    # Auto-cap max_model_len for co-deploy: models with huge default context
    # windows (e.g. 262K) will OOM on KV/Mamba-state cache when gpu_memory_util
    # is reduced for shared-GPU deployment.  Inject a safe cap so vLLM starts.
    # STT models get a much smaller cap — they produce short decoder outputs
    # and large values waste VRAM on encoder cache + KV allocation.
    lg_default_max = CO_DEPLOY_DEFAULT_STT_MAX_MODEL_LEN if large.get("modality") == "stt" else CO_DEPLOY_DEFAULT_MAX_MODEL_LEN
    sm_default_max = CO_DEPLOY_DEFAULT_STT_MAX_MODEL_LEN if small.get("modality") == "stt" else CO_DEPLOY_DEFAULT_MAX_MODEL_LEN
    if not lg_max and "--max-model-len" not in lg_extra:
        lg_max = lg_default_max
    if not sm_max and "--max-model-len" not in sm_extra:
        sm_max = sm_default_max

    # Auto-cap max_num_seqs for co-deploy: fewer concurrent sequences = less
    # peak KV cache and activation memory, critical on shared-GPU deployments.
    if "--max-num-seqs" not in lg_extra:
        lg_extra = f"{lg_extra} --max-num-seqs {CO_DEPLOY_DEFAULT_MAX_NUM_SEQS}".strip()
    if "--max-num-seqs" not in sm_extra:
        sm_extra = f"{sm_extra} --max-num-seqs {CO_DEPLOY_DEFAULT_MAX_NUM_SEQS}".strip()
    lg_image = large.get("vllm_image", "vllm/vllm-openai:cu130-nightly")
    sm_image = small.get("vllm_image", "vllm/vllm-openai:cu130-nightly")
    lg_skip_pip = "1" if "gemma4" in lg_image else "0"
    sm_skip_pip = "1" if "gemma4" in sm_image else "0"

    lines = [
        # ── Port-8000 model (vllm-8000) ──
        f"VLLM_IMAGE={lg_image}",
        f"MODEL_NAME={large['name']}",
        f"TENSOR_PARALLEL={large.get('tensor_parallel', 1)}",
        f"GPU_MEMORY_UTIL={lg_util}",
        f"MAX_MODEL_LEN_FLAG={f'--max-model-len {lg_max}' if lg_max and str(lg_max) != 'auto' else ''}",
        f"QUANTIZATION_FLAG={f'--quantization {lg_quant}' if lg_quant != 'none' else ''}",
        f"ENABLE_PREFIX_CACHING_FLAG=--enable-prefix-caching",
        f"EXTRA_VLLM_FLAGS={lg_extra}",
        f"SKIP_ENTRYPOINT_PIP={lg_skip_pip}",
        # ── Port-8001 model (vllm-8001) ──
        f"SMALL_VLLM_IMAGE={sm_image}",
        f"SMALL_MODEL_NAME={small['name']}",
        f"SMALL_TENSOR_PARALLEL={small.get('tensor_parallel', 1)}",
        f"SMALL_GPU_MEMORY_UTIL={sm_util}",
        f"SMALL_MAX_MODEL_LEN_FLAG={f'--max-model-len {sm_max}' if sm_max and str(sm_max) != 'auto' else ''}",
        f"SMALL_QUANTIZATION_FLAG={f'--quantization {sm_quant}' if sm_quant != 'none' else ''}",
        f"SMALL_EXTRA_VLLM_FLAGS={sm_extra}",
        f"SMALL_SKIP_ENTRYPOINT_PIP={sm_skip_pip}",
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


def _container_is_running(container_name: str) -> bool | None:
    """Check if a Docker container is still running.

    Returns True if running, False if exited/dead, None if inspect failed.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        status = result.stdout.strip().lower()
        return status in ("running", "created", "restarting")
    except Exception:
        return None


def _get_container_restart_count(container_name: str) -> int | None:
    """Return the container's RestartCount, or None if inspect failed."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.RestartCount}}", container_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except Exception:
        return None


def _dump_container_logs(
    container_name: str,
    context: str = "",
    results_dir: str = "results",
    tail_lines: int = 200,
) -> str | None:
    """Capture full container logs to a timestamped file in results/.

    Also prints the last *tail_lines* to stdout for immediate visibility.
    Returns the path of the saved log file, or None on failure.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = context.replace(" ", "_").replace("/", "-") if context else "crash"
    log_filename = f"{container_name}_{slug}_{ts}.log"
    log_path = Path(results_dir) / log_filename
    Path(results_dir).mkdir(exist_ok=True)

    # Capture full logs to file
    try:
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True, text=True, timeout=60,
        )
        full_log = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        # Try direct write first; fall back to sudo tee if permission denied
        try:
            log_path.write_text(full_log)
        except PermissionError:
            subprocess.run(
                ["sudo", "tee", str(log_path)],
                input=full_log, text=True, capture_output=True, timeout=10,
            )
        print(f"  ** Full container logs saved → {log_path}", flush=True)
    except Exception as e:
        print(f"  WARNING: Could not capture full logs for {container_name}: {e}", file=sys.stderr)
        # Still print logs to stdout even if file write failed
        print(f"  --- {container_name} logs (last {tail_lines} lines) ---", flush=True)
        run(["docker", "logs", "--tail", str(tail_lines), container_name], check=False)
        return None

    # Print tail for immediate visibility
    print(f"  --- {container_name} logs (last {tail_lines} lines) ---", flush=True)
    run(["docker", "logs", "--tail", str(tail_lines), container_name], check=False)

    return str(log_path)


# Maximum number of container restarts before declaring a crash-loop and
# failing immediately — prevents silently hanging for the full timeout when
# Docker's ``restart: unless-stopped`` keeps reviving a broken container.
CRASH_LOOP_RESTART_THRESHOLD = 2


def wait_for_vllm(
    host: str = "localhost",
    port: int = 8000,
    timeout: int = 600,
    container_name: str | None = None,
):
    """Wait for a vLLM health endpoint, with early exit if the container dies.

    If *container_name* is given the function checks ``docker inspect`` every
    iteration and fails immediately when the container is no longer running
    **or** has entered a crash-loop (RestartCount ≥ threshold), printing
    container logs so the operator can see why startup failed.
    """
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    label = f" ({container_name})" if container_name else ""
    print(f">>> Waiting for vLLM at {url}{label} (up to {timeout}s)…")

    initial_restarts: int | None = None  # set on first successful inspect

    while time.time() < deadline:
        # ---- fail-fast: container crashed or crash-looping? ------------------
        if container_name:
            alive = _container_is_running(container_name)
            if alive is False:
                print(
                    f"\n  *** Container '{container_name}' is no longer running!",
                    file=sys.stderr, flush=True,
                )
                log_path = _dump_container_logs(
                    container_name,
                    context="startup-crash",
                    tail_lines=200,
                )
                raise RuntimeError(
                    f"Container '{container_name}' exited before becoming healthy. "
                    f"Full logs saved to {log_path or '(capture failed)'}. "
                    f"Check for OOM, model-load error, CUDA graph crash, or missing deps."
                )

            # Detect crash-loop: Docker restarts the container via
            # 'restart: unless-stopped' so it stays "running" even though
            # vLLM keeps crashing on startup.
            restarts = _get_container_restart_count(container_name)
            if restarts is not None:
                if initial_restarts is None:
                    initial_restarts = restarts
                delta = restarts - initial_restarts
                if delta >= CRASH_LOOP_RESTART_THRESHOLD:
                    print(
                        f"\n  *** Container '{container_name}' has restarted "
                        f"{delta} time(s) — crash loop detected!",
                        file=sys.stderr, flush=True,
                    )
                    log_path = _dump_container_logs(
                        container_name,
                        context="crash-loop",
                        tail_lines=200,
                    )
                    raise RuntimeError(
                        f"Container '{container_name}' is crash-looping "
                        f"(restarted {delta}× since launch). "
                        f"Full logs saved to {log_path or '(capture failed)'}. "
                        f"Check for import errors, CUDA/torch ABI mismatch, or missing deps."
                    )

        # ---- health probe ----------------------------------------------------
        try:
            urllib.request.urlopen(url, timeout=3)
            print(">>> vLLM is ready!\n")
            return
        except Exception:
            time.sleep(10)
    # Timeout — dump logs before raising so the operator can diagnose
    if container_name:
        _dump_container_logs(container_name, context="timeout", tail_lines=120)
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
    compose_up(["up", "-d", "vllm-8000"])
    wait_for_vllm(timeout=1800, container_name="vllm-8000")  # large models can take 15-30 min to load on first run


def run_bench(bench_key: str, sweep_ts: str | None = None, model_tag: str | None = None) -> int:
    """Run benchmark container; return exit code (0=pass, 1=partial fail, 2+=total fail)."""
    config_path = BENCH_CONFIGS[bench_key]
    cmd = [
        "docker", "compose", "run", "--rm", "bench-runner",
        "python", "/app/core/bench_runner.py", "--config", config_path,
    ]
    if sweep_ts:
        cmd += ["--sweep-ts", sweep_ts]
    if model_tag:
        cmd += ["--model-tag", model_tag]
    result = run(cmd, check=False)
    return result.returncode


def run_stt_bench(bench_key: str, sweep_ts: str | None = None, model_tag: str | None = None) -> int:
    """Run STT benchmark container; return exit code."""
    config_path = STT_BENCH_CONFIGS[bench_key]
    cmd = [
        "docker", "compose", "run", "--rm", "stt-runner",
        "python", "/app/core/stt_runner.py", "--config", config_path,
    ]
    if sweep_ts:
        cmd += ["--sweep-ts", sweep_ts]
    if model_tag:
        cmd += ["--model-tag", model_tag]
    result = run(cmd, check=False)
    return result.returncode


def run_stt_streaming_bench(bench_key: str, sweep_ts: str | None = None, model_tag: str | None = None) -> int:
    """Run streaming STT benchmark container; return exit code."""
    config_path = STT_STREAMING_BENCH_CONFIGS[bench_key]
    cmd = [
        "docker", "compose", "run", "--rm", "stt-streaming-runner",
        "python", "/app/core/stt_streaming_runner.py", "--config", config_path,
    ]
    if sweep_ts:
        cmd += ["--sweep-ts", sweep_ts]
    if model_tag:
        cmd += ["--model-tag", model_tag]
    result = run(cmd, check=False)
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
CO_DEPLOY_TOTAL_BUDGET = 0.90   # leave 10% for CUDA context / driver / Triton scratch
CO_DEPLOY_HEADROOM = 1.20       # 20% headroom over loaded_gb for KV cache / activations
CO_DEPLOY_STT_HEADROOM = 1.50   # 50% headroom for STT: audio encoder cache + spectrogram activations
CO_DEPLOY_DEFAULT_MAX_MODEL_LEN = 65536 # safe cap for text/vlm models without explicit --max-model-len in co-deploy
CO_DEPLOY_DEFAULT_STT_MAX_MODEL_LEN = 8192  # STT encoder tokens can be large; 8K covers ~30s audio + decoder output
CO_DEPLOY_DEFAULT_MAX_NUM_SEQS = 4      # limit concurrent sequences to bound peak KV/activation memory

def compute_co_deploy_memory(large: dict, small: dict) -> tuple[float | None, float | None]:
    """Return (large_util, small_util) as gpu_memory_utilization fractions.

    Strategy: give the smaller model just enough VRAM (loaded_gb × headroom),
    then give everything remaining to the larger model.  This avoids wasting
    VRAM on a small model that can't use it (e.g. a 9 GB STT model getting
    21 GB under proportional split).
    """
    lg_gb = large.get("loaded_gb", 0)
    sm_gb = small.get("loaded_gb", 0)
    if not lg_gb or not sm_gb:
        print(
            f"WARNING: missing loaded_gb for {large.get('label')} or {small.get('label')} — skipping pair"
        )
        return None, None

    # Estimated VRAM required (weights + KV headroom)
    # STT models need extra headroom for audio encoder cache profiling
    lg_headroom = CO_DEPLOY_STT_HEADROOM if large.get("modality") == "stt" else CO_DEPLOY_HEADROOM
    sm_headroom = CO_DEPLOY_STT_HEADROOM if small.get("modality") == "stt" else CO_DEPLOY_HEADROOM
    lg_needed = lg_gb * lg_headroom
    sm_needed = sm_gb * sm_headroom
    total_needed = lg_needed + sm_needed
    total_budget_gb = GPU_VRAM_GB * CO_DEPLOY_TOTAL_BUDGET

    if total_needed > total_budget_gb:
        return None, None  # won't physically fit

    # Give the small model exactly what it needs, rest goes to the large model
    sm_util = round(sm_needed / GPU_VRAM_GB, 2)
    lg_util = round(CO_DEPLOY_TOTAL_BUDGET - sm_util, 2)
    return lg_util, sm_util


# ---------------------------------------------------------------------------
# Co-serve  (serve two models without benchmarking)
# ---------------------------------------------------------------------------

def co_serve(
    models: list,
    label_a: str,
    label_b: str,
):
    """Serve two models simultaneously: label_a on port 8000, label_b on port 8001.

    Computes VRAM split automatically from loaded_gb in models.yaml.
    Does NOT run any benchmark — just boots and waits.
    """
    pool = {m.get("label"): m for m in models}
    model_a = pool.get(label_a)
    model_b = pool.get(label_b)
    if not model_a:
        print(f"ERROR: No model with label '{label_a}' in models.yaml", file=sys.stderr)
        sys.exit(1)
    if not model_b:
        print(f"ERROR: No model with label '{label_b}' in models.yaml", file=sys.stderr)
        sys.exit(1)

    # Determine which model is larger — compute_co_deploy_memory gives the
    # remainder budget to the first ("large") arg, so we must pass the heavier
    # model first regardless of port assignment.
    a_gb = model_a.get("loaded_gb", 0)
    b_gb = model_b.get("loaded_gb", 0)
    if a_gb >= b_gb:
        lg_util, sm_util = compute_co_deploy_memory(model_a, model_b)
        a_util, b_util = lg_util, sm_util
    else:
        lg_util, sm_util = compute_co_deploy_memory(model_b, model_a)
        a_util, b_util = sm_util, lg_util

    if a_util is None:
        print(
            f"ERROR: {label_a} ({a_gb} GB) + "
            f"{label_b} ({b_gb} GB) won't fit in "
            f"{GPU_VRAM_GB * CO_DEPLOY_TOTAL_BUDGET:.0f} GB budget.",
            file=sys.stderr,
        )
        sys.exit(1)

    section(
        f"CO-SERVE: {label_a} ({a_util:.0%}) on :8000  +  {label_b} ({b_util:.0%}) on :8001"
    )

    hf_token = os.environ.get("HF_TOKEN")
    write_env_dual(model_a, model_b, lg_util=a_util, sm_util=b_util, hf_token=hf_token)
    compose_down_all()

    print(">>> Starting port-8000 model first…", flush=True)
    compose_up(["--profile", "co-deploy", "up", "-d", "vllm-8000"])
    wait_for_vllm(port=8000, timeout=1800, container_name="vllm-8000")

    print(">>> Port-8000 healthy — starting port-8001 model…", flush=True)
    compose_up(["--profile", "co-deploy", "up", "-d", "vllm-8001"])
    wait_for_vllm(port=8001, timeout=1800, container_name="vllm-8001")

    section("BOTH MODELS READY")
    print(f"  :8000  →  {model_a['name']}  (label={label_a})")
    print(f"  :8001  →  {model_b['name']}  (label={label_b})")
    print(f"\nContainers will stay up. Stop with: make stop")


# ---------------------------------------------------------------------------
# Co-deploy sweep  (Goal 2)
# ---------------------------------------------------------------------------

def mixed_co_deploy_sweep(models: list, label_large: str | None = None, label_stt: str | None = None, stt_primary: bool = False):
    """Run mixed co-deploy: text (large) + STT (small) simultaneously.

    When stt_primary=True, the STT model is placed on port 8000 (vllm-8000)
    and the text model on port 8001 (vllm-8001).
    """
    text_pool = [m for m in models if m.get("role") == "large" and m.get("modality", "text") == "text"]
    stt_pool = [m for m in models if m.get("modality") == "stt"]

    if label_large:
        text_pool = [m for m in text_pool if m.get("label") == label_large]
    if label_stt:
        stt_pool = [m for m in stt_pool if m.get("label") == label_stt]

    if not text_pool:
        print("ERROR: No text 'large' role models found.", file=sys.stderr)
        sys.exit(1)
    if not stt_pool:
        print("ERROR: No STT models found (or label filter matched nothing).", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[dict, dict, float, float]] = []
    for lg, stt in product(text_pool, stt_pool):
        lg_util, stt_util = compute_co_deploy_memory(lg, stt)
        if lg_util is None:
            print(
                f"SKIP: {lg.get('label')} + {stt.get('label')} — "
                f"combined VRAM estimate ({lg.get('loaded_gb', '?')}+{stt.get('loaded_gb', '?')} GB) "
                f"exceeds budget ({GPU_VRAM_GB * CO_DEPLOY_TOTAL_BUDGET:.0f} GB)"
            )
        else:
            pairs.append((lg, stt, lg_util, stt_util))

    if not pairs:
        print("ERROR: No viable (text, stt) pair fits in VRAM.", file=sys.stderr)
        sys.exit(1)

    section(f"MIXED CO-DEPLOY SWEEP  |  {len(pairs)} pair(s)")
    hf_token = os.environ.get("HF_TOKEN")
    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed = []

    for lg, stt, lg_util, stt_util in pairs:
        lg_label = lg.get("label", lg["name"].split("/")[-1])
        stt_label = stt.get("label", stt["name"].split("/")[-1])
        port_note = f"STT:{8000 if stt_primary else 8001} Text:{8001 if stt_primary else 8000}"
        section(f"MIXED CO-DEPLOY: {lg_label} ({lg_util:.0%}) + {stt_label} ({stt_util:.0%})  [{port_note}]")

        if stt_primary:
            # STT on port 8000 (vllm-8000), text on port 8001 (vllm-8001)
            write_env_dual(stt, lg, lg_util=stt_util, sm_util=lg_util, hf_token=hf_token)
            expected_8000, expected_8001 = stt["name"], lg["name"]
            text_endpoint = "http://vllm-8001:8001/v1"
            stt_endpoint = "http://vllm-8000:8000/v1"
        else:
            # Text on port 8000 (vllm-8000), STT on port 8001 (vllm-8001)
            write_env_dual(lg, stt, lg_util=lg_util, sm_util=stt_util, hf_token=hf_token)
            expected_8000, expected_8001 = lg["name"], stt["name"]
            text_endpoint = "http://vllm-8000:8000/v1"
            stt_endpoint = "http://vllm-8001:8001/v1"

        running_8000 = _get_running_model(port=8000)
        running_8001 = _get_running_model(port=8001)
        if running_8000 == expected_8000 and running_8001 == expected_8001:
            print(f">>> {lg_label} + {stt_label} already running — reusing containers", flush=True)
        else:
            compose_down_all()

            print(">>> Starting vllm-8000 first (sequential to avoid memory race)…", flush=True)
            compose_up(["--profile", "co-deploy", "up", "-d", "vllm-8000"])
            try:
                wait_for_vllm(port=8000, timeout=1800, container_name="vllm-8000")
            except (TimeoutError, RuntimeError) as e:
                print(f"  *** FAILED (vllm-8000): {e}", file=sys.stderr)
                _dump_container_logs("vllm-8000", context=f"mixed_{lg_label}")
                failed.append(f"{lg_label}+{stt_label}")
                compose_down_all(check=False)
                continue

            print(">>> vllm-8000 healthy — now starting vllm-8001 (STT)…", flush=True)
            compose_up(["--profile", "co-deploy", "up", "-d", "vllm-8001"])
            try:
                wait_for_vllm(port=8001, timeout=1800, container_name="vllm-8001")
            except (TimeoutError, RuntimeError) as e:
                print(f"  *** FAILED (vllm-8001/STT): {e}", file=sys.stderr)
                _dump_container_logs("vllm-8001", context=f"mixed_{stt_label}")
                _dump_container_logs("vllm-8000", context=f"mixed_{lg_label}_when-stt-failed")
                failed.append(f"{lg_label}+{stt_label}")
                compose_down_all(check=False)
                continue

        try:
            pair_tag = f"{lg_label}+{stt_label}"
            run([
                "docker", "compose", "--profile", "co-deploy",
                "run", "--rm",
                "-e", f"VLLM_ENDPOINT_LARGE={text_endpoint}",
                "-e", f"VLLM_ENDPOINT_SMALL={stt_endpoint}",
                "-e", f"LARGE_MODEL_NAME={lg['name']}",
                "-e", f"SMALL_MODEL_NAME={stt['name']}",
                "mixed-runner",
                "python", "/app/core/mixed_co_deploy_runner.py",
                "--config", "/configs/mixed_co_deploy.yaml",
                "--sweep-ts", sweep_ts,
                "--model-tag", pair_tag,
            ])
        except Exception as e:
            print(f"  *** BENCH FAILED: {e}", file=sys.stderr)
            _dump_container_logs("vllm-8000", context=f"mixed-benchfail_{lg_label}")
            _dump_container_logs("vllm-8001", context=f"mixed-benchfail_{stt_label}")
            failed.append(f"{lg_label}+{stt_label}")
        finally:
            compose_down_all(check=False)

    section("MIXED CO-DEPLOY SWEEP COMPLETE")
    if failed:
        print(f"WARNING: {len(failed)} pair(s) failed: {failed}")


def co_deploy_sweep(models: list, label_large: str | None = None, label_small: str | None = None):
    large_pool = [m for m in models if m.get("role") == "large" and m.get("modality", "text") == "text"]
    small_pool = [m for m in models if m.get("role") == "small" and m.get("modality", "text") == "text"]

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
    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            print(">>> Starting vllm-8000 first (sequential to avoid memory race)…", flush=True)
            compose_up(["--profile", "co-deploy", "up", "-d", "vllm-8000"])
            try:
                wait_for_vllm(port=8000, timeout=1800, container_name="vllm-8000")
            except (TimeoutError, RuntimeError) as e:
                print(f"  *** FAILED (vllm-8000): {e}", file=sys.stderr)
                _dump_container_logs("vllm-8000", context=f"co-deploy_{lg_label}")
                failed.append(f"{lg_label}+{sm_label}")
                compose_down_all(check=False)
                continue

            print(">>> vllm-8000 healthy — now starting vllm-8001…", flush=True)
            compose_up(["--profile", "co-deploy", "up", "-d", "vllm-8001"])
            try:
                wait_for_vllm(port=8001, timeout=1800, container_name="vllm-8001")
            except (TimeoutError, RuntimeError) as e:
                print(f"  *** FAILED (vllm-8001): {e}", file=sys.stderr)
                _dump_container_logs("vllm-8001", context=f"co-deploy_{sm_label}")
                # Also capture the large container logs — it may have OOM'd after small started
                _dump_container_logs("vllm-8000", context=f"co-deploy_{lg_label}_when-small-failed")
                failed.append(f"{lg_label}+{sm_label}")
                compose_down_all(check=False)
                continue

        try:
            pair_tag = f"{lg_label}+{sm_label}"
            run([
                "docker", "compose", "--profile", "co-deploy",
                "run", "--rm", "co-runner",
                "python", "/app/core/co_deploy_runner.py",
                "--config", "/configs/split_load.yaml",
                "--sweep-ts", sweep_ts,
                "--model-tag", pair_tag,
            ])
        except Exception as e:
            print(f"  *** BENCH FAILED: {e}", file=sys.stderr)
            _dump_container_logs("vllm-8000", context=f"co-deploy-benchfail_{lg_label}")
            _dump_container_logs("vllm-8001", context=f"co-deploy-benchfail_{sm_label}")
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
    all_bench_choices = (
        list(BENCH_CONFIGS.keys())
        + list(STT_BENCH_CONFIGS.keys())
        + list(STT_STREAMING_BENCH_CONFIGS.keys())
        + ["co-deploy", "mixed-co-deploy"]
    )

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
        help="(co-deploy / mixed-co-deploy) Filter large-role models to this label",
    )
    parser.add_argument(
        "--label-small",
        help="(co-deploy only) Filter small-role text models to this label",
    )
    parser.add_argument(
        "--label-stt",
        help="(mixed-co-deploy only) Filter STT models to this label",
    )
    parser.add_argument(
        "--stt-primary", action="store_true",
        help="(mixed-co-deploy only) Put STT model on port 8000 and text model on port 8001",
    )
    parser.add_argument(
        "--co-serve", nargs=2, metavar=("LABEL_8000", "LABEL_8001"),
        help="Serve two models: first label on port 8000, second on port 8001 (no benchmark)",
    )
    args = parser.parse_args()

    if not args.serve_only and not args.bench and not args.probe and not args.co_serve:
        parser.error("--bench, --probe, --serve-only, or --co-serve is required")

    models = load_models(args.models_file)

    # ── Co-serve: boot two models, no benchmark ───────────────────────────────
    if args.co_serve:
        co_serve(models, label_a=args.co_serve[0], label_b=args.co_serve[1])
        return

    # ── Co-deploy: separate code paths ─────────────────────────────────────────
    if args.bench == "co-deploy":
        co_deploy_sweep(models, label_large=args.label_large, label_small=args.label_small)
        return

    if args.bench == "mixed-co-deploy":
        mixed_co_deploy_sweep(models, label_large=args.label_large, label_stt=args.label_stt, stt_primary=args.stt_primary)
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
        modality = models[0].get("modality", "text")
        section(f"SERVE: {models[0]['name']}  (label={label}, modality={modality})")
        serve_model(models[0], enable_prefix_caching=(modality == "text"))
        if modality == "stt":
            print("STT server is ready. Run `make stt-sanity` or `make stt-bench` to benchmark.")
        else:
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

    # ── Determine if this is an STT, streaming STT, or text bench ────────────
    bench_key = args.bench
    is_stt_bench = bench_key in STT_BENCH_CONFIGS
    is_stt_streaming_bench = bench_key in STT_STREAMING_BENCH_CONFIGS

    # Filter models by modality: text benches only run text models, STT only STT
    if is_stt_bench or is_stt_streaming_bench:
        models = [m for m in models if m.get("modality") == "stt"]
        if not models:
            print("ERROR: No STT models found in models.yaml (need modality: stt)", file=sys.stderr)
            sys.exit(1)
    else:
        models = [m for m in models if m.get("modality", "text") == "text"]
        if not models:
            print("ERROR: No text models found in models.yaml", file=sys.stderr)
            sys.exit(1)

    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    section(f"SWEEP: {bench_key}  |  {len(models)} model(s)  |  sweep_ts={sweep_ts}")

    report: list[tuple[str, str, str]] = []  # (label, status, detail)

    for i, model in enumerate(models, 1):
        label = model.get("label", model["name"].split("/")[-1])
        modality = model.get("modality", "text")

        # Context-window fitness check (text models only)
        if bench_key in ("concurrency-bench",):
            if not context_window_check(model, bench_key):
                max_len = model.get("max_model_len", "?")
                detail = f"max_model_len={max_len} too small for {bench_key}"
                print(f"  SKIP {label}: {detail}")
                report.append((label, "SKIP", detail))
                continue

        print(f"\n[{i}/{len(models)}] Serving: {model['name']}  (label={label}, modality={modality})")
        try:
            serve_model(model, enable_prefix_caching=(modality == "text"))
        except Exception as e:
            msg = str(e)
            print(f"  *** FAILED to start {label}: {msg} — skipping", file=sys.stderr)
            report.append((label, "SERVE_FAIL", msg))
            continue

        print(f"[{i}/{len(models)}] Benchmarking: {bench_key}")
        if is_stt_streaming_bench:
            exit_code = run_stt_streaming_bench(bench_key, sweep_ts=sweep_ts, model_tag=label)
        elif is_stt_bench:
            exit_code = run_stt_bench(bench_key, sweep_ts=sweep_ts, model_tag=label)
        else:
            exit_code = run_bench(bench_key, sweep_ts=sweep_ts, model_tag=label)
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
