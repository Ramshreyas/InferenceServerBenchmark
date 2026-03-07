#!/bin/bash
# =============================================================================
# vLLM entrypoint wrapper — installs extra dependencies before starting vLLM
# =============================================================================
# Some models (e.g., Voxtral) require packages not shipped in the base vLLM
# image.  This script installs them at container startup, then delegates to
# the standard vLLM serve entrypoint.
# =============================================================================
set -e

# Install audio processing libraries (required by Voxtral / mistral-common tokenizer)
pip install --quiet --no-cache-dir soxr librosa soundfile 'mistral-common[audio]' 2>/dev/null || true

# Upgrade transformers to v5 (Voxtral model card recommends it to suppress warnings)
pip install --quiet --no-cache-dir --upgrade transformers 2>/dev/null || true

# Hand off to vLLM serve with all original arguments
exec vllm serve "$@"
