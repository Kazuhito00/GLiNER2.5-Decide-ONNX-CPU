# -*- coding: utf-8 -*-
"""Put a runnable model under models/.

Every repository in this family exposes the same entry point; what it does
differs. GLiNER2.5-Decide has a published ONNX export, so the default path is a
download. The original checkpoint can also be re-exported here, which is what
you want if you fine-tune it.

    python export/prepare_model.py --list
    python export/prepare_model.py                   # download fp32
    python export/prepare_model.py --variant q4f16   # smaller, slower on CPU
    python export/prepare_model.py --self-export     # export from safetensors

No git and no git-lfs: files come from Hugging Face's resolve endpoint using
only the standard library. Self-export additionally needs
requirements-export.txt (torch, gliner2, onnx, onnxscript).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

ONNX_REPO = "onnx-community/GLiNER2.5-Decide-ONNX"
HF_REPO = "fastino/GLiNER2.5-Decide"
ONNX_DIR = ROOT / "models" / "onnx"
TORCH_DIR = ROOT / "models" / "torch"
SELF_DIR = ROOT / "models" / "onnx-selfexport"

VARIANTS = {
    "fp32": "reference precision (1.7 GB). 13/13 argmax, |dp| 4.4e-07. Fastest on CPU.",
    "fp16": "half the size, same answers",
    "q4f16": "smallest (0.5 GB) but SLOWER on CPU; 12/13 argmax",
    "q4": "as q4f16, fp32 elsewhere",
}

#: Everything the runtime reads, beside the graph itself.
SUPPORT_FILES = ["tokenizer.json", "config.json", "special_tokens_map.json",
                 "tokenizer_config.json"]

#: The checkpoint files the self-export reads.
CHECKPOINT_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "model.safetensors", "encoder_config/config.json",
]


def run(argv: list[str]) -> None:
    print("$ %s" % " ".join(str(a) for a in argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT)


def suffix(variant: str) -> str:
    return "" if variant == "fp32" else "_" + variant


def download(variant: str) -> None:
    name = "model%s" % suffix(variant)
    paths = SUPPORT_FILES + ["onnx/%s.onnx" % name, "onnx/%s.onnx_data" % name]
    print("%s (%s) -> %s" % (ONNX_REPO, variant, ONNX_DIR), flush=True)
    fetch_files(ONNX_REPO, paths, ONNX_DIR)


def self_export() -> None:
    print("%s -> %s" % (HF_REPO, TORCH_DIR), flush=True)
    fetch_files(HF_REPO, CHECKPOINT_FILES, TORCH_DIR)
    run([sys.executable, "export/export_onnx.py", "dynamo"])
    print("\nself-export written to %s" % SELF_DIR)
    print("it reproduces the published graph byte for byte; see artifacts/report.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the variants")
    parser.add_argument("--variant", default="fp32", choices=sorted(VARIANTS))
    parser.add_argument("--self-export", action="store_true",
                        help="export from the original checkpoint instead of downloading")
    args = parser.parse_args()

    if args.list:
        print("variants:\n")
        for name, note in VARIANTS.items():
            print("  --variant %-7s %s" % (name, note))
        print("\nOn CPU prefer fp32: 4-bit is a WebGPU optimisation and is")
        print("more than twice as slow here (470 ms vs 212 ms).")
        print("\n  --self-export   rebuild the graph from %s" % HF_REPO)
        print("                  (needed if you fine-tune the checkpoint)")
        return 0

    if args.self_export:
        self_export()
    else:
        download(args.variant)
        print("\nready: %s" % ONNX_DIR)

    print("run:    python demo_inference_text.py")
    print("verify: python verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
