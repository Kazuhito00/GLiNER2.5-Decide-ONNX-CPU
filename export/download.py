# -*- coding: utf-8 -*-
"""Fetch files from a Hugging Face repository over plain HTTPS.

No git, no git-lfs, no huggingface_hub: the resolve endpoint serves the real
bytes (it redirects to a CDN, which urllib follows), so the standard library is
enough. Interrupted downloads resume with a Range request.

    from export.download import fetch_files
    fetch_files("onnx-community/GLiNER2.5-Decide-ONNX", ["onnx/model.onnx"], dest)
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional

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
