# Blackwell Private Inference Testbench

A professional, containerized benchmarking suite for evaluating Large Language Model (LLM) inference performance on NVIDIA RTX PRO 6000 (96 GB Blackwell).

- **Goal**: Determine feasible model and serving configurations on the RTX PRO 6000 (96 GB) by measuring throughput and latency under realistic load.
- **Tech Approach**: Dockerized `vLLM` inference server plus a benchmark runner that drives OpenAI-compatible requests; telemetry captures GPU power, thermals, and VRAM during runs.
- **Evals**: Sweep model/quantization, context windows, and concurrency to report TTFT, ITL, and tokens/sec; summarize results in CSV/JSON.
- **Hygiene**: Host stays clean; all execution is containerized, configs are versioned YAML, and outputs live under `./results` for easy rsync.
- **Outcome**: Clear guidance on which model sizes, context windows, and tokens/sec targets are achievable for 10 concurrent users on this server.

---

## Architecture — Three Layers

```
Layer 1 — You edit this:    models.yaml        ← model list, quant, VRAM budget
Layer 2 — Benchmark params: configs/*.yaml     ← context lengths, concurrency, telemetry
Layer 3 — Infrastructure:   Makefile + docker-compose.yml  ← never touch
```

**The only file you need to edit is `models.yaml`.**
Everything else is driven automatically by `make`.

---

## Prerequisites

### Target Server
- **GPU**: NVIDIA RTX PRO 6000 (Blackwell / GB202) — 96 GB VRAM
- **Driver**: 590.48.01+ (CUDA 13.1)
- **OS**: Ubuntu 22.04+
- **Docker**: Engine 24.0+ with NVIDIA Container Toolkit
- **Docker Compose**: v2.0+

### Local Development Machine
- Git (for code sync)
- Python 3.10+ (for analysis notebooks)
- SSH / Teleport access to target server

---

## Quick Start

### 1. Clone to Server

```bash
git clone <repository-url>
cd InferenceServer
```

### 2. Set HuggingFace Token (if needed)

```bash
export HF_TOKEN=hf_...   # only required for gated models
```

### 3. Verify Setup

```bash
make sanity LABEL=mistral-7b
```

This starts vLLM with Mistral 7B, waits for it to be ready, runs 10 test requests, and prints TTFT / ITL / throughput.

### 4. Run Sweeps

```bash
make sweep          # context window sweep across all models in models.yaml
make kv-analysis    # KV cache analysis — prefix caching ON vs OFF
```

---

## Configuring Models

Edit `models.yaml` — this is the only file you need to touch:

```yaml
models:
  - name: mistralai/Mistral-7B-Instruct-v0.3
    label: mistral-7b          # used in CLI and output filenames
    quantization: none
    max_model_len: 32768
    gpu_memory_util: 0.90
    tensor_parallel: 1

  - name: Qwen/Qwen2.5-72B-Instruct
    label: qwen2.5-72b-fp8
    quantization: fp8
    max_model_len: 16384
    gpu_memory_util: 0.95
    tensor_parallel: 1
```

| Field | Options | Notes |
|---|---|---|
| `name` | HuggingFace model ID | Set `HF_TOKEN` for gated models |
| `label` | any slug | Used in CLI (`LABEL=`) and output filenames |
| `quantization` | `none`, `fp8`, `awq`, `gptq` | FP8 recommended for 70B+ on Blackwell |
| `max_model_len` | integer | Reduce for large models to stay within 96 GB |
| `gpu_memory_util` | 0.0 – 1.0 | 0.90–0.95 typical |
| `tensor_parallel` | integer | 1 for single-GPU |

---

## User Stories

### Story 1 — Context Window Sweep

> "How does performance degrade as context length increases?"

```bash
make sweep                       # test all models
make sweep LABEL=mistral-7b      # test one model
```

Runs [`configs/context_sweep.yaml`](configs/context_sweep.yaml) for each model:
- Context lengths: 8 k → 32 k → 65 k → 98 k → 131 k tokens
- 50 requests, 10 concurrent, Poisson arrival
- Outputs: TTFT, ITL, throughput per context length

---

### Story 2 — KV Cache Analysis

> "What does prefix caching actually buy us?"

```bash
make kv-analysis                 # all models
make kv-analysis LABEL=mistral-7b
```

Runs [`configs/kv_cache.yaml`](configs/kv_cache.yaml) **twice** per model — once with prefix caching enabled, once disabled — so pass-1 vs pass-2 deltas show the cache benefit directly.

- Context lengths: 4 k → 8 k → 16 k → 32 k
- 60 requests, 6 concurrent
- High-frequency telemetry (0.5 s) to capture transient VRAM spikes

---

### Story 3 — Multi-Tenancy (coming soon)

> "Can we serve multiple workloads efficiently on one GPU?"

Two sub-stories with fundamentally different infrastructure:

| | 3A — Shared Engine | 3B — Partitioned VRAM |
|---|---|---|
| Approach | Single vLLM + LoRA adapters | Two vLLM containers, split VRAM |
| Make target | `make multitenancy-shared` | `make multitenancy-partitioned` |
| Status | Planned | Planned |

---

## Manual / One-Off Commands

```bash
# Boot a single model, then run bench targets manually
make serve LABEL=mistral-7b

# Run individual benchmarks against whatever vLLM is currently running
make bench-sanity
make bench-context-sweep
make bench-kv-analysis
```

---

## Reference — All Make Targets

| Target | Description |
|---|---|
| `make sweep [LABEL=]` | Context sweep across all (or one) model |
| `make kv-analysis [LABEL=]` | KV cache analysis, prefix ON vs OFF |
| `make sanity [LABEL=]` | Quick 10-request validation |
| `make serve LABEL=<label>` | Start vLLM for one model (no bench) |
| `make bench-sanity` | Bench against current server |
| `make bench-context-sweep` | Bench against current server |
| `make bench-kv-analysis` | Bench against current server |
| `make logs` | Tail vLLM logs |
| `make status` | Containers + GPU stats |
| `make stop` | Stop all containers |
| `make results` | List result files |
| `make gpu-monitor` | One-shot GPU snapshot |

---

## Project Structure

```
.
├── models.yaml                  ← EDIT THIS — your model list
├── Makefile                     ← run commands
├── docker-compose.yml           ← infrastructure (don't touch)
├── Dockerfile                   ← bench-runner image
├── core/
│   ├── sweep.py                 ← iterates models.yaml, drives docker compose
│   ├── bench_runner.py          ← sends requests, collects metrics
│   ├── telemetry.py             ← GPU monitoring
│   └── utils.py                 ← helpers
├── configs/
│   ├── sanity_check.yaml        ← 10 requests, quick validation
│   ├── context_sweep.yaml       ← 5 context lengths, 50 req, 10 concurrent
│   ├── kv_cache.yaml            ← prefix cache analysis
│   └── multitenancy_test.yaml   ← (future) LoRA + partitioned VRAM
├── results/                     ← output directory (gitignored)
│   ├── *_detailed.json          ← per-request metrics
│   ├── *_summary.csv            ← aggregated statistics
│   └── *_telemetry.json         ← GPU telemetry
└── notebooks/                   ← local analysis
```

---

## Output Files

Each benchmark run produces three files under `./results/`:

| File | Contents |
|---|---|
| `<prefix>_<timestamp>_detailed.json` | Per-request: TTFT, ITL, latency, tokens |
| `<prefix>_<timestamp>_summary.csv` | Aggregated: P50/P95/P99 across requests |
| `<prefix>_<timestamp>_telemetry.json` | GPU power, temp, VRAM at sample interval |

### Key Metrics

- **TTFT** (Time to First Token) — latency until first token streams back
- **ITL** (Inter-Token Latency) — average time between consecutive tokens
- **Throughput** — tokens generated per second
- **VRAM Usage** — peak memory consumption during run
- **Power Draw** — GPU watt-hours (useful for cost modelling)

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

### CUDA Compatibility (Blackwell + Driver 590)

The RTX PRO 6000 uses CUDA 13.1 drivers. The vLLM image (`cu130-nightly`) ships
with CUDA 13.0 libs — bridged by the CUDA Forward Compatibility layer:

```yaml
# docker-compose.yml (already configured)
volumes:
  - /usr/local/cuda-13.1/compat:/usr/local/cuda/compat:ro
environment:
  - LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64
```

The host package `cuda-compat-13-1` must also be installed.

### vLLM Image

Use `vllm/vllm-openai:cu130-nightly` — the default `latest` tag is CUDA 12.x
and is incompatible with this driver.

---

## Troubleshooting

### vLLM reports CUDA Error 803

Driver / CUDA version mismatch. Verify:
1. `cuda-compat-13-1` is installed on the host
2. The compat volume mount exists in `docker-compose.yml`
3. `LD_LIBRARY_PATH` includes `/usr/local/cuda/compat`

### OOM / Out of Memory

Reduce `max_model_len` or `gpu_memory_util` in `models.yaml` for the offending model.

### Model not found (404)

All configs use `name: auto` — the bench runner queries vLLM's `/v1/models` endpoint
to discover the loaded model automatically.

### Benchmark runner can't connect

```bash
# Verify vLLM health from host
curl http://localhost:8000/health

# Verify networking inside bench-runner container
docker compose exec bench-runner curl http://vllm:8000/health
```

---

## Development Workflow

The `core/` directory is bind-mounted into the bench-runner container, so Python changes take effect immediately without a rebuild:

```bash
# Edit locally
nano core/bench_runner.py

# Re-run (no rebuild needed)
make bench-sanity
```

To add a new benchmark scenario:
1. Create `configs/my_scenario.yaml`
2. Add it to `BENCH_CONFIGS` in `core/sweep.py`
3. Add a `make my-scenario` target in `Makefile`

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
