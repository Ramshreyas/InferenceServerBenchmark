# Blackwell Private Inference Testbench

A containerized benchmarking suite for making two specific deployment decisions on an **NVIDIA RTX PRO 6000 (96 GB Blackwell)**:

| Decision | Question |
|---|---|
| **Goal 1** | Which single model delivers the best P95 TTFT and ITL under sustained 10-user concurrency? |
| **Goal 2** | Which (large + small) model pair delivers the best P95 TTFT and ITL under sustained 10-user concurrency? |

Everything in this repository exists to answer those two questions. Nothing else.

---

## Architecture

```
Layer 1 — You edit this:    models.yaml        ← model list, roles, VRAM budget
Layer 2 — Benchmark params: configs/*.yaml     ← prompt/output sweeps, concurrency, telemetry
Layer 3 — Infrastructure:   Makefile + docker-compose.yml  ← never touch
```

**The only file you need to edit is `models.yaml`.**
Everything else is driven by `make`.

### Services (docker-compose.yml)

| Service | Port | Purpose |
|---|---|---|
| `vllm-large` | 8000 | Primary inference server (always used) |
| `vllm-small` | 8001 | Secondary server (co-deploy profile only) |
| `bench-runner` | — | Benchmark runner for single-model benchmarks |
| `co-runner` | — | Benchmark runner for co-deploy (Goal 2) |

---

## Prerequisites

### Target Server
- **GPU**: NVIDIA RTX PRO 6000 (Blackwell / GB202) — 96 GB VRAM
- **Driver**: 580.105.08+ / CUDA 13.0+
- **OS**: Ubuntu 22.04+
- **Docker**: Engine 24.0+ with NVIDIA Container Toolkit
- **Docker Compose**: v2.0+
- **make**: `sudo apt install make -y`
- **HuggingFace CLI** (`hf`): required for `make prefetch`

  ```bash
  sudo apt install python3-pip pipx -y
  pipx install huggingface_hub[cli]
  pipx ensurepath
  source ~/.bashrc
  ```

### Local Development Machine
- Git (for code sync)
- Python 3.10+ (for analysis notebooks)
- SSH / Teleport access to target server

---

## Quick Start

### 1. Clone to Server

```bash
git clone <repository-url>
cd InferenceServerBenchmark
```

### 2. Set HuggingFace Token (if needed)

```bash
export HF_TOKEN=hf_...   # only required for gated models
```

### 3. Pre-download Models

```bash
make prefetch     # downloads all models in models.yaml to HF cache
```

### 4. Validate Stack

```bash
make sanity LABEL=mistral-7b
```

Starts vLLM with Mistral 7B, runs 10 test requests, prints TTFT / ITL / throughput.

### 5. Run Benchmarks

```bash
# Step 1 — Context fitness check (run first)
make context-stress

# Step 2 — Goal 1: Find the best single model
make concurrency-bench

# Step 3 — Goal 2: Find the best co-deploy pair
make co-deploy
```

---

## Configuring Models

Edit `models.yaml`:

```yaml
models:
  - name: Qwen/QwQ-32B
    label: qwq-32b
    role: large                  # large | small
    quantization: fp8
    gpu_memory_util: 0.85        # for solo benchmarks
    tensor_parallel: 1
    topology: dense
    loaded_gb: 33                # approximate VRAM when loaded

  - name: Qwen/Qwen3-8B
    label: qwen3-8b
    role: small
    quantization: none
    gpu_memory_util: 0.85
    tensor_parallel: 1
    topology: dense
    loaded_gb: 16
```

| Field | Options | Notes |
|---|---|---|
| `name` | HuggingFace model ID | Set `HF_TOKEN` for gated models |
| `label` | any slug | Used in CLI (`LABEL=`) and output filenames |
| `role` | `large`, `small` | Determines endpoint in co-deploy |
| `quantization` | `none`, `fp8`, `awq`, `gptq` | FP8 recommended for 70B+ on Blackwell |
| `gpu_memory_util` | 0.0 – 1.0 | For solo benchmarks; co-deploy splits are auto-computed |
| `tensor_parallel` | integer | 1 for single-GPU |
| `topology` | `dense`, `sparse_moe` | MoE models load all expert weights into VRAM |
| `loaded_gb` | integer | Approximate loaded VRAM; used to auto-compute co-deploy memory splits |

### Co-deploy Memory Allocation

`gpu_memory_util` is only used for solo benchmarks (Goals 1). For co-deploy (Goal 2), `sweep.py` auto-computes proportional memory splits from `loaded_gb`:

- **Budget:** 92% of 96 GB = 88 GB (8% reserved for CUDA context)
- **Headroom:** 15% over `loaded_gb` for KV cache and activations
- Pairs where `(large.loaded_gb + small.loaded_gb) × 1.15 > 88 GB` are skipped

---

## Benchmarks

### 0. Sanity Check

> "Is the stack wired up correctly?"

```bash
make sanity LABEL=mistral-7b
```

10 sequential requests, short completions. Run first against any new model.

### 1. Context Stress (fitness check)

> "Does this model collapse at longer prompts?"

```bash
make context-stress                   # all models
make context-stress LABEL=qwq-32b    # one model
```

Profiles TTFT/ITL degradation from 512 to 65k tokens at fixed output length (256) and concurrency 10. A model whose P95 TTFT at your workload prompt length exceeds 3× its 512-token baseline is flagged as context-sensitive.

### 2. Goal 1 — Single-Tenant Concurrency Bench

> "Which model has the best P95 TTFT under sustained 10-user load?"

```bash
make concurrency-bench                   # all models
make concurrency-bench LABEL=qwq-32b    # one model
```

2-D sweep across `prompt_token_lengths × output_token_lengths` with fixed queue depth of 10. 200 requests per point. Produces a `_decision.csv` ranking table.

### 3. Goal 2 — Co-Deploy Split-Load

> "Which (large, small) pair is best when sharing the GPU?"

```bash
make co-deploy                                                       # all viable pairs
make co-deploy LABEL_LARGE=qwq-32b LABEL_SMALL=qwen3-8b            # one pair
```

Two vLLM instances on one GPU. 70% traffic to large, 30% to small. Same 2-D sweep as Goal 1. Per-endpoint P95 TTFT/ITL reported independently.

---

## Reference — All Make Targets

| Target | Description |
|---|---|
| `make sanity [LABEL=]` | Quick 10-request validation |
| `make context-stress [LABEL=]` | Context fitness check (run before Goal 1) |
| `make concurrency-bench [LABEL=]` | Goal 1 — rank single-tenant models |
| `make co-deploy [LABEL_LARGE= LABEL_SMALL=]` | Goal 2 — rank co-deploy pairs |
| `make probe [LABEL=]` | Auto-detect max_model_len for models |
| `make serve LABEL=<label>` | Start vLLM for one model (no bench) |
| `make prefetch` | Pre-download all models to HF cache |
| `make bench-sanity` | Run sanity against whatever is up |
| `make bench-concurrency` | Run concurrency bench against whatever is up |
| `make logs` | Tail vLLM logs |
| `make status` | Containers + GPU stats |
| `make stop` | Stop all containers |
| `make results` | List result files |
| `make gpu-monitor` | One-shot GPU snapshot |

---

## Project Structure

```
.
├── models.yaml                  ← EDIT THIS — model list with roles and VRAM estimates
├── PRD.md                       ← Full design specification
├── Makefile                     ← All make targets
├── docker-compose.yml           ← vllm-large, vllm-small, bench-runner, co-runner
├── Dockerfile                   ← bench-runner / co-runner image
├── core/
│   ├── sweep.py                 ← Iterates models.yaml, drives docker compose
│   ├── bench_runner.py          ← Single-model benchmark (sanity, concurrency, context stress)
│   ├── co_deploy_runner.py      ← Split-load benchmark against two endpoints (Goal 2)
│   ├── prefetch.py              ← Pre-downloads all models to HF cache
│   ├── telemetry.py             ← GPU monitoring via nvidia-smi
│   └── utils.py                 ← Logging, serialization helpers
├── configs/
│   ├── sanity_check.yaml        ← 10 sequential requests, quick validation
│   ├── concurrency_bench.yaml   ← Goal 1: 2-D prompt×output sweep, 10 concurrent, 200 req
│   ├── context_stress.yaml      ← Context fitness: prompt sweep 512–65k, fixed output 256
│   └── split_load.yaml          ← Goal 2: same 2-D sweep, 70/30 traffic split
├── results/                     ← Output directory
│   ├── *_detailed.json          ← Per-request metrics
│   ├── *_summary.csv            ← Aggregated P50/P95/P99
│   ├── *_decision.csv           ← Goal 1 ranking table
│   └── *_telemetry.json         ← GPU telemetry
└── notebooks/                   ← Local analysis
```

---

## Output Files

| File | Contents |
|---|---|
| `sanity_check_{ts}_detailed.json` | Raw per-request results |
| `sanity_check_{ts}_summary.csv` | Basic stats |
| `context_stress_{ts}_detailed.json` | Per-request, tagged with `prompt_tokens_target` |
| `context_stress_{ts}_summary.csv` | Per-model degradation curve across prompt tiers |
| `concurrency_bench_{ts}_detailed.json` | Per-request, all raw metrics |
| `concurrency_bench_{ts}_summary.csv` | Stats grouped by `(model, prompt, output)` |
| `concurrency_bench_{ts}_decision.csv` | **Goal 1 ranking table** — P95 TTFT/ITL per tier |
| `split_load_{ts}_detailed.json` | Per-request, tagged `endpoint: large\|small` |
| `split_load_{ts}_summary.csv` | **Goal 2 ranking table** — per-endpoint P50/P95/P99 |
| `split_load_{ts}_telemetry.json` | GPU telemetry for co-deploy run |

### Key Metrics

- **TTFT** (Time to First Token) — latency until first token streams back. P95 is the primary ranking metric.
- **ITL** (Inter-Token Latency) — average time between consecutive tokens. Must be < 100 ms for smooth streaming.
- **Throughput** — tokens generated per second.

---

## Decision Framework

### Goal 1 — Best Single Model

1. From `context_stress_*_summary.csv`, flag models whose P95 TTFT at your workload prompt length > 3× their 512-token baseline.
2. From `concurrency_bench_*_decision.csv`, select the `(prompt, output)` row matching your workload.
3. Rank by `P95_ttft_ms` ascending. Winner must also have `P95_itl_ms < 100 ms`.

### Goal 2 — Best Co-Deploy Pair

1. From `split_load_*_summary.csv`, select the `(prompt, output)` row matching your workload.
2. Rank by `large_P95_ttft_ms` ascending.
3. Discard pairs where `small_P95_itl_ms > 100 ms`.

See [PRD.md](PRD.md) §9 for the full decision framework.

---

## Analysing Results

```bash
# Copy results to local machine
rsync -avz server:~/InferenceServerBenchmark/results/ ./results/

# Open analysis notebook
cd notebooks && jupyter notebook
```

---

## Server Setup Notes

### CUDA Compatibility (Blackwell)

The RTX PRO 6000 uses CUDA 13.0+ drivers. The vLLM image (`cu130-nightly`) is
bridged by the CUDA Forward Compatibility layer:

```yaml
# docker-compose.yml (already configured)
volumes:
  - /usr/local/cuda-13.1/compat:/usr/local/cuda/compat:ro
environment:
  - LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
```

The host package `cuda-compat-13-1` must be installed.

### vLLM Image

Use `vllm/vllm-openai:cu130-nightly` — the default `latest` tag is CUDA 12.x
and is incompatible with Blackwell drivers.

---

## Troubleshooting

### vLLM reports CUDA Error 803

Driver / CUDA version mismatch. Verify:
1. `cuda-compat-13-1` is installed on the host
2. The compat volume mount exists in `docker-compose.yml`
3. `LD_LIBRARY_PATH` includes `/usr/local/cuda/compat`

### vLLM startup timeout on first run

Large models (70B+) can take 15–30 minutes to download on first use:

```bash
make prefetch   # pre-download before benchmarking
```

If the HF cache was previously written by Docker (root-owned):

```bash
sudo chown -R $USER:$USER ~/.cache/huggingface/
```

### OOM / Out of Memory

Reduce `max_model_len` or `gpu_memory_util` in `models.yaml`. For co-deploy, reduce `loaded_gb` estimates or remove pairs that are too large.

### Model not found (404)

All configs use `name: auto` — the bench runner auto-detects the loaded model via `/v1/models`.

### Benchmark runner can't connect

```bash
curl http://localhost:8000/health                                    # from host
docker compose exec bench-runner curl http://vllm-large:8000/health  # from container
```

---

## Development Workflow

`core/` is bind-mounted into containers — Python changes take effect without a rebuild:

```bash
nano core/bench_runner.py
make bench-sanity          # no rebuild needed
```

---

## Resources

- [PRD.md](PRD.md) — Full design specification with rationale
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
