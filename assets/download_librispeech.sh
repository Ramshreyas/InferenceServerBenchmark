#!/usr/bin/env bash
# =============================================================================
# Download LibriSpeech test-clean dataset for STT benchmarking
# =============================================================================
# Downloads the standard test-clean subset (~346 MB compressed, ~1.2 GB
# extracted) with reference transcripts.  This is the de-facto standard
# for WER evaluation.
#
# Usage:
#   bash assets/download_librispeech.sh          # download + extract
#   bash assets/download_librispeech.sh --check  # verify files exist
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR"
DATASET_DIR="$ASSETS_DIR/librispeech-test-clean"
TAR_FILE="$ASSETS_DIR/test-clean.tar.gz"
URL="https://www.openslr.org/resources/12/test-clean.tar.gz"

# ── Check mode ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--check" ]]; then
    if [[ -d "$DATASET_DIR/LibriSpeech/test-clean" ]]; then
        n_flac=$(find "$DATASET_DIR/LibriSpeech/test-clean" -name '*.flac' | wc -l)
        n_trans=$(find "$DATASET_DIR/LibriSpeech/test-clean" -name '*.trans.txt' | wc -l)
        echo "✓ LibriSpeech test-clean: $n_flac FLAC files, $n_trans transcript files"
        exit 0
    else
        echo "✗ LibriSpeech test-clean not found at $DATASET_DIR"
        echo "  Run: bash assets/download_librispeech.sh"
        exit 1
    fi
fi

# ── Download ────────────────────────────────────────────────────────────────
if [[ -d "$DATASET_DIR/LibriSpeech/test-clean" ]]; then
    echo "LibriSpeech test-clean already downloaded at $DATASET_DIR"
    exit 0
fi

echo "Downloading LibriSpeech test-clean (~346 MB)..."
mkdir -p "$DATASET_DIR"

if command -v wget &>/dev/null; then
    wget -c -O "$TAR_FILE" "$URL"
elif command -v curl &>/dev/null; then
    curl -L -C - -o "$TAR_FILE" "$URL"
else
    echo "ERROR: Neither wget nor curl found. Install one and retry."
    exit 1
fi

# ── Extract ─────────────────────────────────────────────────────────────────
echo "Extracting..."
tar -xzf "$TAR_FILE" -C "$DATASET_DIR"
rm -f "$TAR_FILE"

# ── Verify ──────────────────────────────────────────────────────────────────
n_flac=$(find "$DATASET_DIR/LibriSpeech/test-clean" -name '*.flac' | wc -l)
n_trans=$(find "$DATASET_DIR/LibriSpeech/test-clean" -name '*.trans.txt' | wc -l)
echo "✓ Downloaded: $n_flac FLAC files, $n_trans transcript files"
echo "  Location: $DATASET_DIR/LibriSpeech/test-clean"
