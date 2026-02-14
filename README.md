# Blackwell Private Inference Testbench

A professional, containerized benchmarking suite for evaluating Large Language Model (LLM) inference performance on NVIDIA RTX PRO 6000 (96GB Blackwell). This project provides data-driven insights for model selection, VRAM allocation strategies, and multi-tenant hosting optimization.

- **Goal**: Determine feasible model and serving configurations on the RTX PRO 6000 (96GB) by measuring throughput and latency under realistic load.
- **Tech Approach**: Dockerized `vLLM` inference server plus a benchmark runner that drives OpenAI-compatible requests; telemetry captures GPU power, thermals, and VRAM during runs.
- **Evals**: Sweep model/quantization, context windows, and concurrency (incl. 10 concurrent users) to report TTFT, ITL, and tokens/sec; summarize results in CSV/JSON.
- **Hygiene**: Host stays clean; all execution is containerized, configs are versioned YAML, and outputs are stored under `./results` for easy rsync.
- **Outcome**: Clear guidance on which model sizes, context windows, and tokens/sec targets are achievable for 10 concurrent users on this server.

## 🎯 Features

- **Automated Performance Benchmarking**: Measure TTFT, ITL, and throughput across different context lengths
- **KV Cache Analysis**: Sweep context windows (8k-128k) and measure VRAM usage with prefix caching
- **Multi-Tenancy Simulation**: Test shared engine vs. partitioned VRAM allocation strategies
- **Real-time Telemetry**: Track GPU power, temperature, and memory usage during benchmarks
- **Zero-Cruft Architecture**: All execution happens in containers; host remains clean
- **Decoupled Design**: Control plane on local machine, data plane on server

## 📋 Prerequisites

### Target Server
- **GPU**: NVIDIA RTX PRO 6000 (Blackwell Architecture) with 96GB VRAM
- **Driver**: NVIDIA Driver 580.105.08+ / CUDA 13.0+
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Docker**: Docker Engine 24.0+ with NVIDIA Container Toolkit
- **Docker Compose**: v2.0+

### Local Development Machine
- Git for syncing code
- Python 3.10+ (for local analysis notebooks)
- SSH access to target server

## 🚀 Quick Start

### 1. Clone Repository to Server

```bash
# On your server
git clone <repository-url>
cd InferenceServer
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Key settings:
```bash
MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
TENSOR_PARALLEL=1
GPU_MEMORY_UTIL=0.95
MAX_MODEL_LEN=32768
```

### 3. Run Sanity Check

First, verify your setup with a quick test:

```bash
# Start vLLM server
docker-compose up -d vllm

# Wait for server to be ready (check logs)
docker-compose logs -f vllm

# In another terminal, run sanity check
docker-compose run --rm bench-runner \
  python /app/core/bench_runner.py --config /configs/sanity_check.yaml
```

### 4. Run Full Benchmark

```bash
# Run context window sweep
docker-compose --profile benchmark up

# Or specify a different config
docker-compose run --rm bench-runner \
  python /app/core/bench_runner.py --config /configs/llama3_70b_sweep.yaml
```

### 5. Collect Results

```bash
# Results are in the ./results directory
ls -lh results/

# Copy to local machine for analysis
rsync -avz server:/path/to/InferenceServer/results/ ./results/
```

## 📁 Project Structure

```
.
├── README.md                    # This file
├── PRD.md                       # Product requirements document
├── docker-compose.yml           # Service orchestration
├── Dockerfile                   # Benchmark runner container
├── .env.example                 # Environment template
├── configs/                     # Benchmark configurations
│   ├── llama3_70b_sweep.yaml   # Context window sweep
│   ├── multitenancy_test.yaml  # Multi-tenant scenarios
│   └── sanity_check.yaml       # Quick validation test
├── core/                        # Python modules
│   ├── __init__.py
│   ├── bench_runner.py         # Main benchmark orchestrator
│   ├── telemetry.py            # GPU monitoring
│   └── utils.py                # Helpers and formatters
├── results/                     # Output directory (gitignored)
│   ├── *_detailed.json         # Per-request metrics
│   ├── *_summary.csv           # Aggregated results
│   └── *_telemetry.json        # GPU telemetry
└── notebooks/                   # Analysis notebooks
```

## 🔧 Configuration Guide

### YAML Config Structure

Each benchmark config includes:

```yaml
benchmark:
  name: "Benchmark Name"
  description: "What this test measures"
  output_prefix: "output_filename_prefix"

model:
  name: "meta-llama/Llama-3.1-70B-Instruct"
  quantization: "fp8"  # or "awq", "none"
  tensor_parallel_size: 1
  enable_prefix_caching: true

context_lengths:
  - 8192
  - 32768
  - 65536

requests:
  num_requests: 100
  concurrent_requests: 10
  arrival_pattern: "poisson"  # or "burst", "constant"
  rate_per_second: 2.0
  prompt_tokens_min: 512
  prompt_tokens_max: 2048
  completion_tokens: 512

telemetry:
  sample_interval_sec: 1.0
  collect_gpu_stats: true
```

### Available Benchmarks

1. **sanity_check.yaml**: Fast validation (10 requests, 8B model)
2. **llama3_70b_sweep.yaml**: Context window sweep (8k-128k)
3. **multitenancy_test.yaml**: Multi-tenant scenarios (LoRA + partitioned VRAM)

## 📊 Running Benchmarks

### Basic Usage

```bash
# Start vLLM server only
docker-compose up -d vllm

# Run specific benchmark
docker-compose run --rm bench-runner \
  python /app/core/bench_runner.py --config /configs/<config-file>.yaml

# Run with GPU monitoring
docker-compose --profile monitoring up -d nvitop
```

### Custom Model

```bash
# Override model via environment
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
MAX_MODEL_LEN=16384 \
docker-compose up -d vllm

# Run benchmark
docker-compose run --rm bench-runner \
  python /app/core/bench_runner.py --config /configs/sanity_check.yaml
```

### Multi-Tenancy Test

For partitioned VRAM scenarios, you'll need to modify `docker-compose.yml` to run multiple vLLM instances with different `CUDA_VISIBLE_DEVICES` settings (future enhancement).

## 📈 Analyzing Results

### Output Files

Each benchmark run generates:
- `<prefix>_<timestamp>_detailed.json`: Per-request metrics
- `<prefix>_<timestamp>_summary.csv`: Aggregated statistics  
- `<prefix>_<timestamp>_telemetry.json`: GPU monitoring data

### Key Metrics

- **TTFT (Time to First Token)**: Latency until first token appears
- **ITL (Inter-Token Latency)**: Average time between tokens
- **Throughput**: Tokens generated per second
- **VRAM Usage**: Peak memory consumption
- **Power Draw**: GPU power consumption (watts)

### Local Analysis

```bash
# Copy results to local machine
rsync -avz server:/path/to/InferenceServer/results/ ./results/

# Use Jupyter notebooks for visualization
cd notebooks
jupyter notebook
```

## 🐳 Docker Commands Reference

```bash
# Start services
docker-compose up -d                    # Start vLLM only
docker-compose --profile benchmark up   # Start with benchmark runner
docker-compose --profile monitoring up  # Start with GPU monitor

# View logs
docker-compose logs -f vllm            # vLLM server logs
docker-compose logs bench-runner       # Benchmark runner logs

# Stop services
docker-compose down                     # Stop all services
docker-compose down -v                  # Stop and remove volumes

# Rebuild containers
docker-compose build                    # Rebuild benchmark runner
docker-compose pull vllm               # Update vLLM image

# Shell access
docker-compose exec vllm bash          # Access vLLM container
docker-compose run --rm bench-runner bash  # Debug bench runner
```

## 🔍 Troubleshooting

### vLLM Server Won't Start

```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Check vLLM logs
docker-compose logs vllm
```

### Out of Memory Errors

Reduce `GPU_MEMORY_UTIL` in `.env`:
```bash
GPU_MEMORY_UTIL=0.85  # Default is 0.95
```

Or use smaller model/context length in config.

### Benchmark Runner Can't Connect

```bash
# Verify vLLM health
curl http://localhost:8000/health

# Check network
docker-compose exec bench-runner curl http://vllm:8000/health
```

### Slow Performance

- Enable FP8 quantization for Blackwell: `quantization: "fp8"`
- Increase `tensor_parallel_size` if using multiple GPUs
- Enable prefix caching: `enable_prefix_caching: true`

## 🧪 Development Workflow

### Local Modifications

The `core/` directory is mounted as a volume, so code changes are reflected immediately:

```bash
# Edit Python code locally
nano core/bench_runner.py

# Re-run benchmark (no rebuild needed)
docker-compose run --rm bench-runner \
  python /app/core/bench_runner.py --config /configs/sanity_check.yaml
```

### Adding New Benchmarks

1. Create new YAML config in `configs/`
2. Define benchmark parameters
3. Run: `docker-compose run --rm bench-runner python /app/core/bench_runner.py --config /configs/<new-config>.yaml`

## 🎓 Architecture Details

### Control Plane (Local)
- Source code management (Git)
- Configuration authoring (YAML)
- Results analysis (Jupyter/Pandas)

### Data Plane (Server)
- vLLM inference engine (Docker)
- Benchmark runner (Docker)
- Telemetry collectors (Docker)
- Results storage (`./results` volume)

### Communication
- Git for code sync
- rsync/SSH for results retrieval
- Docker networks for inter-container communication

## 📝 Best Practices

1. **Always run sanity check first** to validate setup
2. **Start with small models** before testing 70B variants
3. **Monitor GPU temperature** during long benchmarks
4. **Use prefix caching** for RAG-like workloads
5. **Collect telemetry** to correlate performance with hardware stats
6. **Version your configs** in Git for reproducibility

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Test with sanity check
4. Submit PR with results

## 📄 License

[Specify your license]

## 🔗 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

---

**Questions?** Open an issue or contact the maintainers.
