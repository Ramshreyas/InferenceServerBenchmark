#!/bin/bash
# =============================================================================
# vLLM entrypoint wrapper — installs extra dependencies before starting vLLM
# =============================================================================
# Some models (e.g., Voxtral) require packages not shipped in the base vLLM
# image.  This script installs them at container startup, then delegates to
# the standard vLLM serve entrypoint.
# =============================================================================
set -e

# Install vLLM audio extras (required by Cohere Transcribe, Whisper, etc.)
pip install --quiet --no-cache-dir 'vllm[audio]' 2>/dev/null || true

# Install audio processing libraries (required by Voxtral / mistral-common tokenizer)
pip install --quiet --no-cache-dir soxr librosa soundfile 'mistral-common[audio]' 2>/dev/null || true

# NOTE: Do NOT upgrade transformers here. The vLLM nightly ships a compatible
# version; upgrading pulls in huggingface_hub with strict dataclass validation
# that breaks vLLM's internal WhisperConfig remapping for Voxtral
# (max_source_positions=None → TypeError).

# Hand off to vLLM serve with all original arguments
exec vllm serve "$@"
