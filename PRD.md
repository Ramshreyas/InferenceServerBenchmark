# PRD: Blackwell Private Inference Testbench

## 1. Project Overview

Provide the data needed to make two specific deployment decisions for a **10 concurrent user** workload on an **NVIDIA RTX PRO 6000 (96 GB Blackwell)**:

| Decision | Question |
|---|---|
| **Goal 1** | Which single model delivers the best P95 TTFT and ITL under sustained 10-user concurrency? |
| **Goal 2** | Which combination of large + small model delivers the best P95 TTFT and ITL under sustained 10-user concurrency? |

Everything in this repository exists to answer those two questions. Nothing else.

---

## 2. Hardware Context

- **GPU:** NVIDIA RTX PRO 6000 (Blackwell)
- **VRAM:** 96 GB GDDR7
- **Driver/CUDA:** 580.105.08 / CUDA 13.0
- **Host OS:** Linux

---

## 3. Architecture (unchanged)

**Decoupled Controller-Executor:**

- **Control Plane (local):** YAML configs, Python sweep scripts, Jupyter analysis.
- **Data Plane (server):** Dockerized vLLM engine(s) and telemetry.
- **Bridge:** Git sync; SSH/rsync for pulling results.

---

## 4. Benchmarks

### 4.1 Sanity Check (`configs/sanity_check.yaml`)

**Purpose:** Validate that the stack is wired up correctly before running a long sweep. Unchanged from current implementation.

- 10 sequential requests, 2 in-flight, 8k context, short completions.
- Run first against any new model to confirm it loads and responds.
- **Do not modify this config.**

---

### 4.2 Goal 1 — Single-Tenant Concurrency Benchmark (`configs/concurrency_bench.yaml`)

**Purpose:** Find which model has the best user-facing latency under constant 10-user pressure.

**Design decisions:**

- **Fixed queue depth of 10.** A new request is submitted the moment a slot frees (ThreadPoolExecutor, not Poisson arrival). Poisson at 2 rps gives an average queue depth of <<10 for fast models; this benchmark must keep the queue full.
- **Output token sweep, not context sweep.** The productivity-limiting variable for reasoning models (DeepSeek-R1, QwQ, Qwen3-Thinking) is generation length, not prompt length. A 512-token completion cap would make those models look faster than they are in production where they emit 2k–4k token chains-of-thought. We sweep `[256, 512, 1024, 2048, 4096]` to capture their real behaviour.
- **Prompt length sweep.** Run at three representative prompt sizes — **short (512 tokens)**, **medium (4 096 tokens)**, and **long (16 384 tokens)** — alongside the output token sweep. This surfaces latency degradation across the context window. A model that looks fast at 512-token prompts but collapses at 16k is unsuitable for a mixed workload. If `prompt_tokens + output_tokens > model.max_model_len`, that tier is skipped for that model (not a disqualification; the model still runs at shorter tiers).
- **200 requests per point.** Required for stable P95/P99 estimates.
- **P50/P95/P99 TTFT and ITL.** Mean latency is misleading. P95 TTFT is the user-visible tail that determines whether the system feels acceptable.

**Decision output:** For each model, a table:

```
model | prompt_tokens | output_tokens | P50_ttft_ms | P95_ttft_ms | P99_ttft_ms | P50_itl_ms | P95_itl_ms | throughput_tok_s
```

The winning model is the one with the best P95 TTFT at the `(prompt_tokens, output_tokens)` combination representative of your workload. A model that degrades sharply at longer prompts is naturally penalised when you select the row that matches your actual workload context.

---

### 4.3 Context Stress Benchmark (`configs/context_stress.yaml`)

**Purpose:** Profile how P95 TTFT and ITL degrade as prompt length grows from short to near each model's `max_model_len`. This is not a ranking benchmark — it is a fitness check run before Goal 1. A model that is fast at short context but degrades catastrophically at medium context is unsuitable for a mixed workload and should be dropped from consideration.

**Design decisions:**

- **Fixed output tokens (256).** Isolates prefill pressure; generation length is held constant.
- **Fixed concurrency of 10.** Same as Goal 1 so degradation is measured under realistic queue pressure.
- **Prompt length sweep: `[512, 2048, 8192, 32768, 65536]`.** Any value where `prompt_tokens + 256 > model.max_model_len` is skipped automatically.
- **100 requests per point.** Fewer than Goal 1; this is a profile, not a ranking. P95 stability is sufficient.

**Decision output:** For each model, a degradation curve:

```
model | prompt_tokens | P50_ttft_ms | P95_ttft_ms | P99_ttft_ms | P50_itl_ms | P95_itl_ms
```

A model is context-sensitive if its P95 TTFT at any prompt tier present in your workload exceeds 3× its short-context (512-token) P95 TTFT. Such models are flagged in the Goal 1 decision table but not automatically excluded — the flag informs the final call.

---

### 4.4 Goal 2 — Co-Deploy Split-Load Benchmark (`configs/split_load.yaml`)

**Purpose:** Find which (large, small) model pair delivers the best user-facing latency across 10 concurrent users when both models run simultaneously on the same GPU.

**Design decisions:**

- **Single GPU, two vLLM instances.** The Blackwell has 96 GB. A large model at `gpu_memory_util=0.65` (~62 GB) + a small model at `gpu_memory_util=0.30` (~29 GB) = 95% utilization. Pairs are only viable if `large.gpu_memory_util + small.gpu_memory_util ≤ 0.95`.
- **Fixed 70/30 split.** 7 concurrent slots routed to the large model endpoint, 3 to the small. This reflects a realistic load where most queries go to the capable model.
- **Same prompt × output token sweep as Goal 1** to keep results directly comparable and allow ranking across Goals 1 and 2 at the same workload context point.
- **Per-endpoint AND combined metrics.** Both the large and small endpoint P95 TTFT/ITL are reported independently so the ranking reflects real user experience on each tier.

**Decision output:** For each (large, small) pair:

```
pair | endpoint | prompt_tokens | output_tokens | P50_ttft_ms | P95_ttft_ms | P50_itl_ms | P95_itl_ms | throughput_tok_s
```

The winning pair is the one with the best P95 TTFT on the large endpoint at the output token length representative of your workload, subject to the small endpoint also meeting the `P95_itl_ms < 100 ms` smoothness threshold.

---

## 5. File Inventory

### Kept unchanged
| File | Reason |
|---|---|
| `configs/sanity_check.yaml` | Correct and useful as-is |
| `core/telemetry.py` | Solid nvidia-smi wrapper |
| `core/utils.py` | Solid logging/serialization utilities |
| `core/prefetch.py` | Works correctly |
| `Dockerfile` | No changes needed |

### Removed
| File | Reason |
|---|---|
| `configs/context_sweep.yaml` | Context length is not the decision variable for 10-user concurrency; replaced by output token sweep |
| `configs/kv_cache.yaml` | Answers a different question (cache hit rate); not needed for model selection |
| `configs/multitenancy_test.yaml` | References models not in the benchmark set; LoRA scenario not applicable; VRAM partitioning physically impossible on a single GPU |
| `configs/llama3_70b_sweep.yaml` | Model-specific, superseded by general configs |

### New / Modified
| File | Status | Notes |
|---|---|---|
| `configs/concurrency_bench.yaml` | **New** | Goal 1 benchmark config |
| `configs/context_stress.yaml` | **New** | Context fitness check config |
| `configs/split_load.yaml` | **New** | Goal 2 benchmark config |
| `models.yaml` | **Modified** | Add `role: large \| small` field to each entry |
| `docker-compose.yml` | **Modified** | Rename `vllm` → `vllm-large`; add `vllm-small` on port 8001; add `co-runner` service |
| `core/bench_runner.py` | **Modified** | Add concurrent execution mode; output token sweep; backward-compat with sanity check |
| `core/co_deploy_runner.py` | **New** | Goal 2 split-load runner against two endpoints |
| `core/sweep.py` | **Modified** | Add `concurrency-bench` key; add `co-deploy` sweep mode; remove obsolete keys |
| `Makefile` | **Modified** | New targets; remove obsolete ones |
| `PRD.md` | **Modified** | This document |

---

## 6. Implementation Specification

### 6.1 `models.yaml` — add `role` field

Every model entry gets a `role` field:

```yaml
role: large   # large | small
```

- `large`: considered for the primary endpoint in co-deploy (port 8000).
- `small`: considered for the secondary endpoint in co-deploy (port 8001).

The co-deploy sweep generates all `(large, small)` pairs where:
```
large.gpu_memory_util + small.gpu_memory_util <= 0.95
```
Any pair exceeding 95% total utilization is skipped with a warning. This check is performed in `sweep.py` before starting any containers.

---

### 6.2 `core/bench_runner.py` — concurrency refactor

**Backward compatibility:** If the config has a `context_lengths` list (sanity check), use the existing sequential sweep behaviour unchanged.

**New concurrent mode:** If the config has an `output_token_lengths` list, use `concurrent.futures.ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def _run_concurrent_sweep(self, prompt_tokens: int, output_tokens: int) -> list[dict]:
    config = self.config['requests']
    num_requests = config['num_requests']
    concurrency  = config['concurrency']   # replaces concurrent_requests

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                self._run_single_request,
                prompt=self._generate_prompt(prompt_tokens),
                max_tokens=output_tokens,
                request_id=i,
            )
            for i in range(num_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results
```

`output_token_lengths` and `prompt_token_lengths` define a 2-D sweep grid. For each `(prompt_tokens, output_tokens)` pair, call `_run_concurrent_sweep(prompt_tokens, output_tokens)`. Skip any pair where `prompt_tokens + output_tokens > model.max_model_len` and log a warning. Tag every result with both `prompt_tokens_target` and `output_tokens_target`.

**Per-request result dict** (extend existing schema):

```python
{
    'model': str,
    'prompt_tokens_target': int,          # requested prompt length
    'output_tokens_target': int,          # requested max_tokens
    'tokens_generated': int,              # actual tokens received
    'ttft_ms': float,
    'itl_ms': float,
    'total_latency_ms': float,
    'throughput_tokens_per_sec': float,
    'success': bool,
    'timestamp': str,
}
```

**Summary output:** Extend `_print_summary` and `_save_results` to group by `(prompt_tokens_target, output_tokens_target)` and write an additional `{prefix}_{ts}_decision.csv` with one row per `(model, prompt_tokens_target, output_tokens_target)`:

```
model, prompt_tokens_target, output_tokens_target, n_requests, n_success,
P50_ttft_ms, P95_ttft_ms, P99_ttft_ms,
P50_itl_ms, P95_itl_ms,
mean_throughput_tok_s, total_tokens_generated
```

---

### 6.3 `configs/context_stress.yaml` — new file

```yaml
bench: context-stress
requests:
  num_requests: 100
  concurrency: 10
  output_tokens: 256
prompt_token_lengths: [512, 2048, 8192, 32768, 65536]
```

`sweep.py` skips any `prompt_tokens` value where `prompt_tokens + output_tokens > model.max_model_len`. Results are saved as `context_stress_{ts}_detailed.json` and `context_stress_{ts}_summary.csv`.

---

### 6.4 `core/co_deploy_runner.py` — new file

Entry point: `python core/co_deploy_runner.py --config /configs/split_load.yaml`

**Environment variables consumed:**
- `VLLM_ENDPOINT_LARGE` (default: `http://vllm-large:8000/v1`)
- `VLLM_ENDPOINT_SMALL` (default: `http://vllm-small:8001/v1`)
- `LARGE_MODEL_NAME` — used for result labelling
- `SMALL_MODEL_NAME` — used for result labelling

**Logic:**

1. Create two OpenAI clients (large endpoint + small endpoint). Auto-detect model names if env vars absent.
2. Read `split.total_concurrency` and `split.large_fraction` from config.
3. `n_large = round(total_concurrency * large_fraction)`, `n_small = total_concurrency - n_large`.
4. For each `(prompt_tokens, output_tokens)` pair in the 2-D sweep:
   - Skip if `prompt_tokens + output_tokens > min(large.max_model_len, small.max_model_len)`.
   - Build `num_requests` request tasks. Route request `i` to large if `i % total_concurrency < n_large`, else small.
   - Submit all to a single `ThreadPoolExecutor(max_workers=total_concurrency)`.
   - Tag each result with `endpoint: large | small`, `prompt_tokens_target`, and `output_tokens_target`.
5. Compute per-endpoint and combined P50/P95/P99 TTFT/ITL/throughput.
6. Save `split_load_{ts}_detailed.json`, `split_load_{ts}_summary.csv`, `split_load_{ts}_telemetry.json`.

**Summary CSV schema:**

```
large_model, small_model, prompt_tokens_target, output_tokens_target,
large_P50_ttft_ms, large_P95_ttft_ms, large_P99_ttft_ms,
large_P50_itl_ms, large_P95_itl_ms,
small_P50_ttft_ms, small_P95_ttft_ms, small_P99_ttft_ms,
small_P50_itl_ms, small_P95_itl_ms,
combined_throughput_tok_s
```

---

### 6.5 `core/sweep.py` — changes

**`BENCH_CONFIGS` dict — replace entirely:**

```python
BENCH_CONFIGS = {
    "sanity":            "/configs/sanity_check.yaml",
    "concurrency-bench": "/configs/concurrency_bench.yaml",
    "context-stress":    "/configs/context_stress.yaml",
    # co-deploy uses a separate code path (co_deploy_sweep)
}
```

Remove `"context-sweep"` and `"kv-analysis"` keys.

**Context window filter:** Before starting any container in both the single-model sweep and `co_deploy_sweep`, check `model.max_model_len` against `max(prompt_token_lengths) + max(output_token_lengths)`. If the model cannot satisfy even the smallest tier (`min(prompt_token_lengths) + min(output_token_lengths) > model.max_model_len`), skip the model entirely with a warning. Otherwise proceed — individual tiers that exceed `max_model_len` are skipped at runtime by `bench_runner.py`.

**Add `--bench co-deploy` as a new argparse choice** (not a key in `BENCH_CONFIGS`) that triggers `co_deploy_sweep()`.

**New `write_env_dual(large, small)`** function — writes `.env` with both model variable sets:

```
MODEL_NAME=<large.name>
GPU_MEMORY_UTIL=<large.gpu_memory_util>
MAX_MODEL_LEN_FLAG=--max-model-len <large.max_model_len>  # if set
QUANTIZATION_FLAG=--quantization <large.quantization>     # if not none
SMALL_MODEL_NAME=<small.name>
SMALL_GPU_MEMORY_UTIL=<small.gpu_memory_util>
SMALL_MAX_MODEL_LEN_FLAG=--max-model-len <small.max_model_len>  # if set
SMALL_QUANTIZATION_FLAG=--quantization <small.quantization>     # if not none
SMALL_TENSOR_PARALLEL=<small.tensor_parallel>
```

**New `co_deploy_sweep(models, label_large, label_small)` function:**

```python
def co_deploy_sweep(models, label_large=None, label_small=None):
    large_pool = [m for m in models if m.get('role') == 'large']
    small_pool = [m for m in models if m.get('role') == 'small']

    if label_large:
        large_pool = [m for m in large_pool if m['label'] == label_large]
    if label_small:
        small_pool = [m for m in small_pool if m['label'] == label_small]

    pairs = [
        (lg, sm)
        for lg in large_pool
        for sm in small_pool
        if lg['gpu_memory_util'] + sm['gpu_memory_util'] <= 0.95
    ]

    skipped = [
        (lg['label'], sm['label'])
        for lg in large_pool
        for sm in small_pool
        if lg['gpu_memory_util'] + sm['gpu_memory_util'] > 0.95
    ]
    for lg_lbl, sm_lbl in skipped:
        print(f"SKIP: {lg_lbl} + {sm_lbl} — combined gpu_memory_util > 0.95")

    failed = []
    for lg, sm in pairs:
        section(f"CO-DEPLOY: {lg['label']} + {sm['label']}")
        write_env_dual(lg, sm)
        run(["docker", "compose", "down"])
        run(["docker", "compose", "--profile", "co-deploy", "up", "-d", "vllm-large", "vllm-small"])
        try:
            wait_for_vllm(port=8000)
            wait_for_vllm(port=8001)
        except TimeoutError as e:
            print(f"  *** TIMEOUT: {e}", file=sys.stderr)
            failed.append(f"{lg['label']}+{sm['label']}")
            run(["docker", "compose", "down"], check=False)
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
            failed.append(f"{lg['label']}+{sm['label']}")
        finally:
            run(["docker", "compose", "down"], check=False)

    if failed:
        print(f"\nWARNING: {len(failed)} pair(s) failed: {failed}")
```

**`--label` for `co-deploy`** is replaced by `--label-large` and `--label-small` args (filter the respective pool).

---

### 6.6 `docker-compose.yml` — service changes

Rename the existing `vllm` service to `vllm-large`. No other changes to it.

**Add `vllm-small` service:**

```yaml
vllm-small:
  image: vllm/vllm-openai:cu130-nightly
  container_name: vllm-small
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - CUDA_VISIBLE_DEVICES=0
    - LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
    - HF_TOKEN=${HF_TOKEN:-}
  ports:
    - "8001:8001"
  volumes:
    - ~/.cache/huggingface:/root/.cache/huggingface
    - /usr/local/cuda-13.1/compat:/usr/local/cuda/compat:ro
    - ./results:/results
  command: >
    --host 0.0.0.0
    --port 8001
    --model ${SMALL_MODEL_NAME}
    --tensor-parallel-size ${SMALL_TENSOR_PARALLEL:-1}
    --gpu-memory-utilization ${SMALL_GPU_MEMORY_UTIL:-0.25}
    ${SMALL_MAX_MODEL_LEN_FLAG:-}
    ${SMALL_QUANTIZATION_FLAG:-}
    --trust-remote-code
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 120s
  profiles:
    - co-deploy
  networks:
    - inference-net
```

Note: both `vllm-large` and `vllm-small` share `CUDA_VISIBLE_DEVICES=0`. vLLM uses `gpu_memory_util` to cap its own VRAM allocation — this is the mechanism that prevents OOM when two instances share the same physical GPU. The sum of their `gpu_memory_util` values must be ≤ 0.95. There is no hard OS-level VRAM partition; it is vLLM's soft ceiling only. If a model's actual memory footprint exceeds its allocation, it will OOM at startup, not silently.

**Add `co-runner` service:**

```yaml
co-runner:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: co-runner
  environment:
    - VLLM_ENDPOINT_LARGE=http://vllm-large:8000/v1
    - VLLM_ENDPOINT_SMALL=http://vllm-small:8001/v1
    - LARGE_MODEL_NAME=${MODEL_NAME:-}
    - SMALL_MODEL_NAME=${SMALL_MODEL_NAME:-}
  volumes:
    - ./configs:/configs:ro
    - ./core:/app/core:ro
    - ./results:/results
  depends_on:
    vllm-large:
      condition: service_healthy
    vllm-small:
      condition: service_healthy
  profiles:
    - co-deploy
  networks:
    - inference-net
```

Update existing `bench-runner` service: rename env var `VLLM_ENDPOINT` remains at `http://vllm-large:8000/v1` (now that vllm is renamed to vllm-large).

---

## 7. Execution Workflow

```
# 0. Setup
make prefetch                       # Download all models in models.yaml

# 1. Validate stack
make sanity LABEL=<any-model>       # Quick 10-request smoke test

# 2. Context fitness check (run before Goal 1)
make context-stress                 # All models in models.yaml
make context-stress LABEL=qwq-32b   # One model only

# 3. Goal 1 — best single-tenant model
make concurrency-bench              # All models in models.yaml
make concurrency-bench LABEL=qwq-32b  # One model only

# 4. Goal 2 — best co-deploy pair
make co-deploy                      # All viable (large, small) pairs
make co-deploy LABEL_LARGE=gpt-oss-120b LABEL_SMALL=qwen3-8b  # One pair

# Utilities
make probe LABEL=<model>            # Auto-detect max_model_len
make serve LABEL=<model>            # Boot one model, no benchmark
make bench-sanity                   # Run sanity against whatever is up
make bench-concurrency              # Run concurrency bench against whatever is up
make logs | status | stop | results | gpu-monitor
```

---

## 8. Output Files

All files land in `results/` with timestamps.

| File | Contents |
|---|---|
| `sanity_check_{ts}_detailed.json` | Raw per-request results (unchanged) |
| `sanity_check_{ts}_summary.csv` | Basic stats (unchanged) |
| `context_stress_{ts}_detailed.json` | One row per request; tagged with `prompt_tokens_target` |
| `context_stress_{ts}_summary.csv` | Per-model degradation curve across prompt tiers |
| `concurrency_bench_{ts}_detailed.json` | One row per request; all raw metrics |
| `concurrency_bench_{ts}_summary.csv` | Full stats grouped by `(model, prompt_tokens_target, output_tokens_target)` |
| `concurrency_bench_{ts}_decision.csv` | **Primary Goal 1 artefact.** P95 TTFT/ITL ranking table across prompt × output tiers. |
| `split_load_{ts}_detailed.json` | One row per request, tagged `endpoint: large\|small` |
| `split_load_{ts}_summary.csv` | Per-endpoint P50/P95/P99 TTFT/ITL and combined throughput; one row per `(large_model, small_model, prompt_tokens_target, output_tokens_target)` |
| `split_load_{ts}_telemetry.json` | GPU telemetry for the entire co-deploy run |

---

## 9. Decision Framework

### Goal 1 — Best Single-Tenant Model

**Step 0 — Context fitness check.** From `context_stress_*_summary.csv`, flag any model whose P95 TTFT at your expected prompt length exceeds 3× its 512-token baseline. These models are context-sensitive — note the flag but do not automatically exclude them; factor it into the final call.

**Step 1 — Rank on Goal 1 data.** From `concurrency_bench_*_decision.csv`, select rows where `prompt_tokens` matches your expected workload prompt length and `output_tokens` matches your expected completion length:

1. Rank models by `P95_ttft_ms` ascending.
2. If the top model's `P95_ttft_ms > 3000 ms`, no model meets the bar at that combination — either the workload assumptions are wrong, or no model in the set is fast enough.
3. Also check `P95_itl_ms < 100 ms` — token streaming feels smooth at ≤ 100 ms ITL; above that users perceive stuttering.

### Goal 2 — Best Co-Deploy Pair

From `split_load_*_summary.csv`, for the `(prompt_tokens, output_tokens)` combination representative of your workload:

1. Rank pairs by `large_P95_ttft_ms` ascending — this is the primary user-visible metric for the dominant (70%) traffic tier.
2. If the top pair's `large_P95_ttft_ms > 3000 ms`, no viable pair meets the bar at that output length.
3. Also check `small_P95_itl_ms < 100 ms` — token streaming on the small endpoint must feel smooth. Discard any pair where the small model exceeds this threshold.
4. Among pairs that pass the ITL check, the winner is the one with the lowest `large_P95_ttft_ms`.

---

## 10. What This Benchmark Explicitly Does NOT Answer

- **RAG / prefix caching benefit** — omitted. Add `configs/prefix_cache.yaml` if this becomes a selection criterion.
- **Fine-tuned / LoRA adapters** — out of scope.
- **Multi-GPU tensor parallel** — out of scope for this single-GPU testbench.
- **Token budget optimisation for reasoning models** — DeepSeek-R1 and Qwen3-Thinking support `budget_token` / thinking mode controls; these are model-specific and out of scope here.
