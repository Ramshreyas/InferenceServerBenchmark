# Blackwell Private Inference Testbench

A containerized benchmarking suite for making deployment decisions on an **NVIDIA RTX PRO 6000 (96 GB Blackwell)**:

| Decision | Question |
|---|---|
| **Goal 1** | Which single model delivers the best P95 TTFT and ITL under sustained 10-user concurrency? |
| **Goal 2** | Which (large + small) model pair delivers the best P95 TTFT and ITL under sustained 10-user concurrency? |
| **Goal 3** | What STT (speech-to-text) throughput and WER can we achieve? |
| **Goal 4** | Can we co-deploy a text LLM + STT model and handle both workloads simultaneously? |

The suite supports three modalities: **text** (chat/completion), **STT** (speech-to-text with WER scoring), and **VLM** (vision-language — future).

---

## Architecture

```
Layer 1 — You edit this:    models.yaml        ← model list, roles, VRAM budget, modality
Layer 2 — Benchmark params: configs/*.yaml     ← prompt/output sweeps, concurrency, telemetry
Layer 3 — Infrastructure:   Makefile + docker-compose.yml  ← never touch
Layer 4 — Assets:           assets/            ← STT datasets (LibriSpeech)
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
| `stt-runner` | — | STT benchmark runner (Goal 3) |
| `mixed-runner` | — | Mixed text+STT co-deploy runner (Goal 4) |

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
make sanity LABEL=ministral3-8b
```

Starts vLLM with Ministral-3 8B, runs 10 test requests, prints TTFT / ITL / throughput.

### 5. Run Benchmarks

```bash
# Goal 1: Find the best single model
make concurrency-bench

# Goal 2: Find the best co-deploy pair
make co-deploy

# Goal 3: STT benchmarking (download dataset first)
make download-stt-data
make stt-sanity LABEL=voxtral-mini-4b
make stt-bench LABEL=voxtral-mini-4b

# Goal 4: Mixed text + STT co-deploy
make mixed-co-deploy LABEL_LARGE=gpt-oss-120b LABEL_STT=voxtral-mini-4b
```

---

## Configuring Models

Edit `models.yaml`:

```yaml
models:
  # Text model
  - name: openai/gpt-oss-120b
    label: gpt-oss-120b
    role: large                  # large | small
    modality: text               # text | stt | vlm
    quantization: none           # Pre-quantized mxfp4
    gpu_memory_util: 0.95        # for solo benchmarks
    tensor_parallel: 1
    topology: dense
    loaded_gb: 60                # approximate VRAM when loaded

  # STT model
  - name: mistralai/Voxtral-Mini-4B-Realtime-2602
    label: voxtral-mini-4b
    role: small
    modality: stt
    quantization: none
    gpu_memory_util: 0.85
    tensor_parallel: 1
    topology: dense
    loaded_gb: 9
    vllm_extra_flags: "--compilation_config '{\"cudagraph_mode\": \"PIECEWISE\"}'"

  # VLM-capable model — dual entries (text + vlm)
  - name: Qwen/Qwen3.5-27B
    label: qwen35-27b-text       # text mode: skip vision encoder
    role: large
    modality: text
    quantization: fp8
    loaded_gb: 28
    vllm_extra_flags: "--language-model-only --reasoning-parser qwen3"

  - name: Qwen/Qwen3.5-27B
    label: qwen35-27b-vlm        # vlm mode: vision encoder loaded
    role: large
    modality: vlm
    quantization: fp8
    loaded_gb: 32
    vllm_extra_flags: "--reasoning-parser qwen3"
```

| Field | Options | Notes |
|---|---|---|
| `name` | HuggingFace model ID | Set `HF_TOKEN` for gated models |
| `label` | any slug | Used in CLI (`LABEL=`) and output filenames |
| `role` | `large`, `small` | Determines endpoint in co-deploy |
| `modality` | `text`, `stt`, `vlm` | Routes to the correct benchmark runner |
| `quantization` | `none`, `fp8`, `awq`, `gptq` | FP8 recommended for 70B+ on Blackwell |
| `gpu_memory_util` | 0.0 – 1.0 | For solo benchmarks; co-deploy splits are auto-computed |
| `tensor_parallel` | integer | 1 for single-GPU |
| `topology` | `dense`, `sparse_moe` | MoE models load all expert weights into VRAM |
| `loaded_gb` | integer | Approximate loaded VRAM; used to auto-compute co-deploy memory splits |
| `vllm_extra_flags` | string (optional) | Additional vLLM CLI flags passed verbatim (e.g. `--language-model-only`) |

### Modality & VLM Dual Entries

Models that support both text and vision (e.g., Qwen3.5, Ministral-3) appear **twice** in `models.yaml`:

- **text entry**: uses `--language-model-only` to skip the vision encoder (lower VRAM, text-only benchmarks)
- **vlm entry**: loads the full model with vision encoder (higher `loaded_gb`, future VLM benchmarks)

`sweep.py` filters models by modality — text benchmarks only see `modality: text`, STT benchmarks only see `modality: stt`, etc.

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
make sanity LABEL=ministral3-8b
```

10 sequential requests, short completions. Run first against any new model.

### 1. Goal 1 — Single-Tenant Concurrency Bench

> "Which model has the best P95 TTFT under sustained 10-user load?"

```bash
make concurrency-bench                        # all models
make concurrency-bench LABEL=gpt-oss-120b    # one model
```

2-D sweep across `prompt_token_lengths × output_token_lengths` with fixed queue depth of 10. 200 requests per point. Produces a `_decision.csv` ranking table.

### 2. Goal 2 — Co-Deploy Split-Load

> "Which (large, small) pair is best when sharing the GPU?"

```bash
make co-deploy                                                              # all viable pairs
make co-deploy LABEL_LARGE=gpt-oss-120b LABEL_SMALL=ministral3-8b          # one pair
```

Two vLLM instances on one GPU. 70% traffic to large, 30% to small. Same 2-D sweep as Goal 1. Per-endpoint P95 TTFT/ITL reported independently.

### 3. Goal 3 — STT (Speech-to-Text) Benchmark

> "What WER and throughput can we get from the STT model?"

```bash
# Download the LibriSpeech test-clean dataset first
make download-stt-data

# Quick smoke test (10 audio files)
make stt-sanity LABEL=voxtral-mini-4b

# Full concurrency benchmark (sweep over concurrent streams)
make stt-bench LABEL=voxtral-mini-4b
```

Transcribes audio files from LibriSpeech test-clean via `/v1/audio/transcriptions`, computes **WER** (Word Error Rate) against reference transcripts, and measures **RTF** (Real-Time Factor). Concurrency sweep tests throughput under parallel streams.

### 4. Goal 4 — Mixed Co-Deploy (Text + STT)

> "Can we run text and STT simultaneously on one GPU?"

```bash
make mixed-co-deploy LABEL_LARGE=gpt-oss-120b LABEL_STT=voxtral-mini-4b
```

Co-deploys a text LLM + STT model on the same GPU and benchmarks both simultaneously. Text requests exercise the chat/completion endpoint while STT requests transcribe audio files — mimicking real-world usage (e.g., meeting transcription + LLM queries at the same time). Reports independent metrics for each endpoint.

---

## Reference — All Make Targets

| Target | Description |
|---|---|
| `make sanity [LABEL=]` | Quick 10-request validation |
| `make concurrency-bench [LABEL=]` | Goal 1 — rank single-tenant models |
| `make co-deploy [LABEL_LARGE= LABEL_SMALL=]` | Goal 2 — rank co-deploy pairs |
| `make download-stt-data` | Download LibriSpeech test-clean dataset |
| `make stt-sanity [LABEL=]` | Goal 3 — quick 10-file STT smoke test |
| `make stt-bench [LABEL=]` | Goal 3 — STT concurrency benchmark |
| `make mixed-co-deploy [LABEL_LARGE= LABEL_STT=]` | Goal 4 — text + STT simultaneous benchmark |
| `make probe [LABEL=]` | Auto-detect max_model_len for models |
| `make serve LABEL=<label>` | Start vLLM for one model (no bench) |
| `make prefetch` | Pre-download all models to HF cache |
| `make tui` | Interactive results explorer (terminal UI) |
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
├── models.yaml                  ← EDIT THIS — model list with roles, VRAM, modality
├── PRD.md                       ← Full design specification
├── GATEWAY.md                   ← Gateway integration notes
├── Makefile                     ← All make targets
├── docker-compose.yml           ← vllm-large/small, bench/co/stt/mixed runners
├── Dockerfile                   ← Runner images (includes soundfile for STT)
├── core/
│   ├── sweep.py                 ← Iterates models.yaml, drives docker compose
│   ├── bench_runner.py          ← Single-model benchmark (sanity, concurrency)
│   ├── co_deploy_runner.py      ← Split-load benchmark against two endpoints (Goal 2)
│   ├── stt_runner.py            ← STT benchmark — WER, RTF, concurrency sweep (Goal 3)
│   ├── mixed_co_deploy_runner.py ← Simultaneous text+STT benchmark (Goal 4)
│   ├── prefetch.py              ← Pre-downloads all models to HF cache
│   ├── telemetry.py             ← GPU monitoring via nvidia-smi
│   └── utils.py                 ← Logging, serialization helpers
├── configs/
│   ├── sanity_check.yaml        ← 10 sequential requests, quick validation
│   ├── concurrency_bench.yaml   ← Goal 1: 2-D prompt×output sweep, 10 concurrent, 200 req
│   ├── split_load.yaml          ← Goal 2: same 2-D sweep, 70/30 traffic split
│   ├── stt_sanity.yaml          ← Goal 3: 10-file STT smoke test
│   ├── stt_concurrency_bench.yaml ← Goal 3: STT concurrency sweep [1,2,4,8]
│   └── mixed_co_deploy.yaml     ← Goal 4: text + STT simultaneous benchmark
├── assets/
│   ├── download_librispeech.sh  ← Downloads LibriSpeech test-clean (~346 MB)
│   ├── librispeech-test-clean/  ← Dataset (gitignored, created by download script)
│   └── README.md                ← Asset documentation
├── tui/
│   ├── data.py                  ← Result discovery, sweep grouping, CSV merging
│   ├── results_tab.py           ← Charts, minimap, scorecard, model filter
│   ├── run_tab.py               ← (future) launch benchmarks from TUI
│   ├── daemon.py                ← Background process management
│   ├── daemon_tab.py            ← (future) manage vLLM daemon
│   └── styles.tcss              ← Textual CSS for layout
├── tui.py                       ← TUI entry point
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
| `concurrency_bench_{ts}_detailed.json` | Per-request, all raw metrics |
| `concurrency_bench_{ts}_summary.csv` | Stats grouped by `(model, prompt, output)` |
| `concurrency_bench_{ts}_decision.csv` | **Goal 1 ranking table** — P95 TTFT/ITL per tier |
| `split_load_{ts}_detailed.json` | Per-request, tagged `endpoint: large\|small` |
| `split_load_{ts}_summary.csv` | **Goal 2 ranking table** — per-endpoint P50/P95/P99 |
| `split_load_{ts}_telemetry.json` | GPU telemetry for co-deploy run |
| `stt_sanity_{ts}_detailed.json` | Per-file STT results (transcriptions, WER, RTF) |
| `stt_sanity_{ts}_summary.csv` | STT sanity stats |
| `stt_concurrency_{ts}_detailed.json` | STT results under concurrent load |
| `stt_concurrency_{ts}_summary.csv` | **Goal 3** — WER, RTF, throughput by concurrency level |
| `mixed_co_deploy_{ts}_detailed.json` | Text + STT per-request results |
| `mixed_co_deploy_{ts}_summary.csv` | **Goal 4** — independent metrics for both endpoints |

### Key Metrics

**Text / VLM:**
- **TTFT** (Time to First Token) — latency until first token streams back. P95 is the primary ranking metric.
- **ITL** (Inter-Token Latency) — average time between consecutive tokens. Must be < 100 ms for smooth streaming.
- **Throughput** — tokens generated per second.

**STT:**
- **WER** (Word Error Rate) — edit distance between transcription and reference, normalized by reference length. Lower is better.
- **RTF** (Real-Time Factor) — processing time / audio duration. RTF < 1.0 means faster than real-time.
- **Throughput** — audio seconds processed per wall-clock second under concurrent load.

---

## Decision Framework

### Goal 1 — Best Single Model

1. From `concurrency_bench_*_decision.csv`, select the `(prompt, output)` row matching your workload.
2. Rank by `P95_ttft_ms` ascending. Winner must also have `P95_itl_ms < 100 ms`.

### Goal 2 — Best Co-Deploy Pair

1. From `split_load_*_summary.csv`, select the `(prompt, output)` row matching your workload.
2. Rank by `large_P95_ttft_ms` ascending.
3. Discard pairs where `small_P95_itl_ms > 100 ms`.

### Goal 3 — STT Quality & Throughput

1. From `stt_*_summary.csv`, check `mean_wer` — acceptable range depends on domain (< 5% for clean speech).
2. Check `mean_rtf` — must be < 1.0 for real-time transcription.
3. Review throughput at target concurrency level.

### Goal 4 — Mixed Co-Deploy Feasibility

1. From mixed co-deploy results, verify text metrics (TTFT, ITL) remain acceptable under STT co-load.
2. Verify STT WER does not degrade compared to solo STT benchmarks.
3. If both metrics hold, the pair is viable for production co-deployment.

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

## TUI — Interactive Results Explorer

A terminal UI for browsing and comparing benchmark results across models.

```bash
make tui
```

### Layout

| Area | Description |
|---|---|
| **Left sidebar — Benchmark Runs** | Tree of all result files grouped by bench type. Concurrency bench runs from the same sweep are auto-grouped so you can view all models together. |
| **Left sidebar — Model Filter** | Checkboxes to show/hide individual models in the charts. |
| **Main pane — Charts** | Side-by-side bar charts: **TTFT** (lower is better) on the left, **Throughput tok/s** (higher is better) on the right. |
| **Main pane — Minimap** | Grid showing which model wins each (prompt, output) cell. |
| **Main pane — Scorecard** | Win counts per model across all grid cells. |

### Navigation

| Key | Action |
|---|---|
| `←` / `→` | Change output token tier |
| `↑` / `↓` | Change prompt token tier |
| `m` | Toggle TTFT between P95 and P50 |
| `s` | Toggle scorecard visibility |

### Sweep Grouping

When `make concurrency-bench` runs all models, each model produces its own timestamped result files. The TUI automatically groups sequential runs (within 8 hours) into a single sweep entry. Clicking the sweep node merges all decision CSVs so you can compare every model side-by-side in the charts, minimap, and scorecard.

Individual runs within a sweep can still be expanded and viewed separately.

### Supported Bench Types

- **⚡ Concurrency Bench** — dual charts + minimap + scorecard (sweep-grouped)
- **🔀 Co-Deploy** — dual charts for (large + small) model pairs
- **🎤 STT Bench** — WER, RTF, throughput under concurrent streams
- **🎤+⚡ Mixed Co-Deploy** — text + STT simultaneous benchmark
- **✅ Sanity Check** — simple table view

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
