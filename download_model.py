# -*- coding: utf-8 -*-
"""Download the model for demo_inference_text.py (this file and the demo are all you need to run it).

    python download_model.py                    # fp32 from this repository's GitHub release (1.7 GB, SHA-256 checked)
    python download_model.py --source hf        # the same fp32 file from Hugging Face
    python download_model.py --variant q4f16    # smaller, slower on CPU (Hugging Face)
    python download_model.py --self-export      # export from the original checkpoint (needs the `export` group)
    python download_model.py --list

Plain HTTPS from the standard library: no git, no git-lfs, no huggingface_hub. Interrupted downloads resume.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional

ROOT = pathlib.Path(__file__).resolve().parent

RELEASE_URL = "https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU/releases/download/v0.0.1/{name}"
#: The two files of the fp32 graph on the release, with their SHA-256.
RELEASE_FILES = {
    "model.onnx": "37af0b3a55a4199a798674cdf7786b88d4764df3e9575ebfcec3673cde0838be",
    "model.onnx_data": "f5aef45f7892cf3acf277516e225d5e653d6b4946918d728b3c3ab0882ac8d58",
}
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



ENDPOINT = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "gliner-decide-onnx/0.1 (urllib)"


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024
    return "%.1f GB" % size


def _remote_size(url: str) -> Optional[int]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError):
        return None


def fetch(url: str, dest: pathlib.Path, label: str = "") -> pathlib.Path:
    """Download ``url`` to ``dest``, resuming a partial file if one is there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    total = _remote_size(url)

    if dest.exists() and total is not None and dest.stat().st_size == total:
        print("  have  %-44s %s" % (label or dest.name, _human(total)))
        return dest

    have = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if have and total is not None and have < total:
        headers["Range"] = "bytes=%d-" % have
    elif have:
        have = 0
        partial.unlink()

    request = urllib.request.Request(url, headers=headers)
    started = time.time()
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 206):
            raise RuntimeError("HTTP %s for %s" % (response.status, url))
        if response.status == 200:
            have = 0  # server ignored the Range; start over
        mode = "ab" if have else "wb"
        with open(partial, mode) as handle:
            done = have
            last = 0.0
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                now = time.time()
                if total and (now - last > 0.5):
                    last = now
                    rate = done / max(now - started, 1e-6)
                    sys.stdout.write(
                        "\r  %-44s %5.1f%%  %s/s   " %
                        (label or dest.name, 100 * done / total, _human(rate))
                    )
                    sys.stdout.flush()

    if total is not None and partial.stat().st_size != total:
        raise RuntimeError(
            "%s: got %d bytes, expected %d" % (dest.name, partial.stat().st_size, total)
        )
    shutil.move(str(partial), str(dest))
    elapsed = time.time() - started
    sys.stdout.write("\r  got   %-44s %s in %.0f s\n"
                     % (label or dest.name, _human(dest.stat().st_size), elapsed))
    sys.stdout.flush()
    return dest


def fetch_files(
    repo: str,
    paths: Iterable[str],
    dest_dir: pathlib.Path,
    revision: str = "main",
    optional: Iterable[str] = (),
) -> None:
    """Fetch each path in ``paths`` from ``repo`` into ``dest_dir``.

    Paths in ``optional`` are skipped if the repository does not have them,
    which keeps one file list usable across model variants.
    """
    optional = set(optional)
    for path in paths:
        url = ENDPOINT.format(repo=repo, revision=revision, path=path)
        target = dest_dir / path
        try:
            fetch(url, target, label=path)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if path in optional:
                print("  skip  %-44s (not in this repository)" % path)
                continue
            raise SystemExit("failed to download %s: %s" % (url, exc)) from exc


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fetch_verified(url: str, dest: pathlib.Path, sha256: str, label: str = "") -> pathlib.Path:
    """``fetch`` plus a SHA-256 check. A file that fails the check is deleted, so the next run starts clean."""
    fetch(url, dest, label=label)
    actual = sha256_of(dest)
    if actual != sha256:
        dest.unlink()
        raise SystemExit("%s: sha256 mismatch (got %s, expected %s); the file was removed" % (dest.name, actual, sha256))
    print("  sha256 ok  %s" % (label or dest.name))
    return dest


def run(argv: list[str]) -> None:
    print("$ %s" % " ".join(str(a) for a in argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT)


def suffix(variant: str) -> str:
    return "" if variant == "fp32" else "_" + variant


def missing_support() -> list[str]:
    return [f for f in SUPPORT_FILES if not (ONNX_DIR / f).exists()]


def download_release() -> None:
    """fp32 from the GitHub release. The small support files are in the repository; fetch them only if absent."""
    if missing_support():
        fetch_files(ONNX_REPO, missing_support(), ONNX_DIR)
    print("GitHub release v0.0.1 (fp32) -> %s" % (ONNX_DIR / "onnx"), flush=True)
    for name, digest in RELEASE_FILES.items():
        fetch_verified(RELEASE_URL.format(name=name), ONNX_DIR / "onnx" / name, digest, label="onnx/" + name)


def download_hf(variant: str) -> None:
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
    parser.add_argument("--source", choices=["release", "hf"], default="release",
                        help="where fp32 comes from (default: this repository's GitHub release); other variants are always from hf")
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
        if args.variant == "fp32" and args.source == "release":
            download_release()
        else:
            download_hf(args.variant)
        print("\nready: %s" % ONNX_DIR)

    print("run:    python demo_inference_text.py")
    print("verify: python verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
