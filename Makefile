# ==============================================================================
# Blackwell Inference Testbench — Makefile
# ==============================================================================
#
# PRIMARY WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
#  1. Edit models.yaml — the only file you need to touch.
#  2. Run one of the sweep targets below.
#
#   make sweep                   Context window sweep across ALL models
#   make kv-analysis             KV cache analysis (prefix caching on & off)
#   make sanity                  Quick 10-request check across all models
#   make sanity LABEL=mistral-7b Quick check for one specific model
#
# ONE-OFF / MANUAL
# ─────────────────────────────────────────────────────────────────────────────
#   make serve LABEL=mistral-7b  Boot vLLM for one model (then bench manually)
#   make bench-sanity            Run sanity bench against whatever is up
#   make bench-context-sweep     Run context sweep against whatever is up
#   make bench-kv-analysis       Run KV cache bench against whatever is up
#
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
#   make logs                    Tail vLLM server logs
#   make status                  Show containers + GPU state
#   make stop                    Stop all containers
#   make results                 List result files (most recent first)
#   make gpu-monitor             One-shot GPU snapshot
# ==============================================================================

DC       := docker compose
LABEL    ?=   # optional: filter to a single model label from models.yaml

# ==============================================================================
# PRIMARY  —  models.yaml-driven sweeps
# ==============================================================================

sweep:
	python core/sweep.py --bench context-sweep $(if $(LABEL),--label $(LABEL),)

kv-analysis:
	python core/sweep.py --bench kv-analysis $(if $(LABEL),--label $(LABEL),)

sanity:
	python core/sweep.py --bench sanity $(if $(LABEL),--label $(LABEL),)

# Boot a single model without running a benchmark (useful for exploratory runs)
serve:
	@if [ -z "$(LABEL)" ]; then \
		echo "Usage: make serve LABEL=<model-label>"; \
		echo "Labels are defined in models.yaml"; \
		exit 1; \
	fi
	python core/sweep.py --serve-only --label $(LABEL)

# ==============================================================================
# BENCH-ONLY  —  run against whatever vLLM server is currently up
# ==============================================================================

bench-sanity:
	$(DC) run --rm bench-runner \
		python /app/core/bench_runner.py --config /configs/sanity_check.yaml

bench-context-sweep:
	$(DC) run --rm bench-runner \
		python /app/core/bench_runner.py --config /configs/context_sweep.yaml

bench-kv-analysis:
	$(DC) run --rm bench-runner \
		python /app/core/bench_runner.py --config /configs/kv_cache.yaml

# ==============================================================================
# UTILITIES
# ==============================================================================

logs:
	$(DC) logs -f vllm

status:
	@echo "=== Containers ==="
	$(DC) ps
	@echo ""
	@echo "=== GPU State ==="
	@nvidia-smi \
		--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
		--format=csv,noheader,nounits

stop:
	$(DC) down

results:
	@echo "=== Result Files (most recent first) ==="
	@ls -lht results/ | head -40

gpu-monitor:
	$(DC) --profile monitoring run --rm nvitop

.PHONY: \
	sweep kv-analysis sanity serve \
	bench-sanity bench-context-sweep bench-kv-analysis \
	logs status stop results gpu-monitor
