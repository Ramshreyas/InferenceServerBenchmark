# PRD: Blackwell Private Inference Testbench

## 1. Project Overview
Establish a professional, containerized benchmarking suite to evaluate LLM performance on an **NVIDIA RTX PRO 6000 (96GB Blackwell)**. The goal is to provide data-driven insights for model selection, VRAM allocation, and multi-tenant hosting strategies while maintaining a "zero-cruft" host environment.

---

## 2. Hardware Context (Target Environment)
* **GPU:** NVIDIA RTX PRO 6000 (Blackwell Architecture)
* **VRAM:** 96GB GDDR7
* **Driver/CUDA:** 580.105.08 / CUDA 13.0
* **Host OS:** Linux (Minimal packages installed)

---

## 3. System Architecture
The project follows a **Decoupled Controller-Executor** model:
* **Control Plane (Local Machine):** Source code, YAML configurations, and Jupyter/Streamlit analysis.
* **Data Plane (Server):** Dockerized `vLLM` engine and telemetry collectors.
* **Bridge:** Git for code synchronization; `rsync` or SSH for pulling result logs.

---

## 4. Functional Requirements

### FR1: Automated Performance Benchmarking
* **Engine:** `vllm/vllm-openai:latest` (optimized for Blackwell/FP8).
* **Metrics:** Capture Time to First Token (TTFT), Inter-Token Latency (ITL), and total tokens/sec.
* **Workload Simulation:** Support for synthetic request distributions (Poisson/Burst).

### FR2: Context Window & KV Cache Analysis
* Perform sweeps across context lengths: `8k, 32k, 64k, 96k, 128k`.
* Measure VRAM "Base Floor" (model weight size) vs. "Active Ceiling" (KV cache growth).
* Benchmark `enable-prefix-caching` for RAG performance evaluation.

### FR3: Multi-Tenancy Simulation
* **Scenario A (Shared Engine):** Single large model (e.g., Llama-3 70B) with Multi-LoRA adapters.
* **Scenario B (Partitioned):** Concurrent Docker containers with hard-capped VRAM limits (e.g., 1x 48GB model + 2x 24GB models).

### FR4: Telemetry & Monitoring
* Collect GPU power draw (W), thermal data, and memory clock speeds during peak load.
* Export all telemetry to timestamped JSON/CSV files in a mounted `./results` volume.

---

## 5. Recommended Directory Structure

```text
.
├── README.md               # Setup instructions & Makefile shortcuts
├── docker-compose.yml      # Orchestrates vLLM & Prometheus/nvitop
├── configs/                # Test definitions (YAML)
│   ├── llama3_70b_sweep.yaml
│   └── rmultitenancy_test.yaml
├── core/                   # Python logic
│   ├── bench_runner.py     # Main entry (runs inside container)
│   ├── telemetry.py        # NVIDIA-SMI wrapper for stats
│   └── utils.py            # Formatting & logging
├── results/                # .gitignore-d; stores JSON/CSV logs
└── notebooks/              # Local analysis (Pandas/Plotly)

```

## 6. Execution Workflow

1. Configure: Define the model, quantization (FP8/AWQ), and context window in configs/.

2. Sync: Push the repo to the server.

3. Run: Execute docker-compose up. The bench_runner.py script starts automatically, hits the vLLM API, and logs results.

4. Tear Down: docker-compose down leaves the server host clean.

5. Analyze: Pull the results/ folder to your local machine for visualization.
