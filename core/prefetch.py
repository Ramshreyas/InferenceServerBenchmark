#!/usr/bin/env python3
"""
Pre-download all models in models.yaml to the HuggingFace cache.

Run from repo root:
    python core/prefetch.py
    make prefetch
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_models(models_file: str = "models.yaml") -> list:
    with open(models_file) as f:
        data = yaml.safe_load(f)
    return data["models"]


def main():
    models = load_models()
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Pre-fetching {len(models)} model(s) to HuggingFace cache...\n")

    failed = []
    for i, model in enumerate(models, 1):
        name = model["name"]
        print(f"[{i}/{len(models)}] Downloading: {name}")
        cmd = ["huggingface-cli", "download", name]
        if hf_token:
            cmd += ["--token", hf_token]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  *** FAILED: {name} (exit {result.returncode})")
            failed.append(name)
        else:
            print(f"  ✓ Done: {name}\n")

    print("\n" + "=" * 60)
    if failed:
        print(f"COMPLETED WITH ERRORS — {len(failed)} model(s) failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"All {len(models)} model(s) cached successfully.")


if __name__ == "__main__":
    main()
