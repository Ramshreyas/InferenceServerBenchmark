#!/bin/bash
# =============================================================================
# patch-sm120-fp4.sh — Patch vLLM + FlashInfer for native FP4 on SM120
# =============================================================================
# SM 12.0 (RTX PRO 6000 Blackwell, RTX 5090) shares FP4 tensor-core ISA with
# SM 10.0 (B100/B200), but vLLM and FlashInfer don't recognize it yet.
#
# This script extracts, patches, and mounts two files:
#   1. vllm mxfp4.py — treat device_capability_family(120) like family(100)
#   2. flashinfer compilation_context.py — accept major 12 when 10 is requested
#
# The CUTLASS FP4 MoE path is used (not TRT-LLM, which has a hard SM100 assert).
#
# Usage (called automatically by sweep.py for models with sm120_fp4_patches: true):
#   bash scripts/patch-sm120-fp4.sh <docker-image> <output-dir>
#
# Outputs:
#   <output-dir>/mxfp4.py
#   <output-dir>/compilation_context.py
# =============================================================================
set -euo pipefail

IMAGE="${1:?Usage: $0 <docker-image> <output-dir>}"
OUTDIR="${2:?Usage: $0 <docker-image> <output-dir>}"

mkdir -p "$OUTDIR"

MXFP4_PATH="$OUTDIR/mxfp4.py"
CC_PATH="$OUTDIR/compilation_context.py"

# Skip if patches already exist (idempotent)
if [[ -f "$MXFP4_PATH" && -f "$CC_PATH" ]]; then
    echo "[patch-sm120-fp4] Patches already exist in $OUTDIR — skipping extraction"
    exit 0
fi

echo "[patch-sm120-fp4] Extracting files from $IMAGE …"

# Extract source files from the Docker image
CONTAINER_ID=$(docker create "$IMAGE" /bin/true)
trap "docker rm -f $CONTAINER_ID >/dev/null 2>&1" EXIT

docker cp "$CONTAINER_ID:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py" "$MXFP4_PATH"
docker cp "$CONTAINER_ID:/usr/local/lib/python3.12/dist-packages/flashinfer/compilation_context.py" "$CC_PATH"

# ── Patch 1: vLLM mxfp4.py — recognize SM120 as SM100-compatible ──────────
echo "[patch-sm120-fp4] Patching mxfp4.py …"

# Replace is_device_capability_family(100) with (family(100) or family(120))
sed -i 's/current_platform\.is_device_capability_family(100)/( current_platform.is_device_capability_family(100) or current_platform.is_device_capability_family(120) )/g' "$MXFP4_PATH"

# Widen the Triton guard from <(11,0) to <(13,0) so SM120 can use Triton fallback too
sed -i 's/(9, 0) <= current_platform.get_device_capability() < (11, 0)/(9, 0) <= current_platform.get_device_capability() < (13, 0)/g' "$MXFP4_PATH"

# ── Patch 2: FlashInfer compilation_context.py — accept major 12 for JIT ──
echo "[patch-sm120-fp4] Patching compilation_context.py …"

python3 -c "
import sys

text = open('$CC_PATH').read()
old = '''if supported_major_versions:
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in supported_major_versions
            ]'''
new = '''if supported_major_versions:
            # SM120 (workstation Blackwell) shares FP4 tensor-core ISA with SM100
            _extended = set(supported_major_versions)
            if 10 in _extended:
                _extended.add(12)
            supported_cuda_archs = [
                major_minor_tuple
                for major_minor_tuple in self.TARGET_CUDA_ARCHS
                if major_minor_tuple[0] in _extended
            ]'''
result = text.replace(old, new)
if result == text:
    print('[patch-sm120-fp4] WARNING: compilation_context.py patch target not found — may already be patched or format changed', file=sys.stderr)
else:
    open('$CC_PATH', 'w').write(result)
    print('[patch-sm120-fp4] compilation_context.py patched OK')
"

echo "[patch-sm120-fp4] Patches written to $OUTDIR"
