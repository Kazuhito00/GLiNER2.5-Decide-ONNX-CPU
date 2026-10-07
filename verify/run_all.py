# -*- coding: utf-8 -*-
"""Run every gate this repository has, in order, and summarise.

Gates that need only the runtime dependencies always run. Gates that need an
oracle (the Rust `tokenizers`, or torch) are skipped with a reason rather than
failing, so this command is meaningful in the minimal environment too.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (script, argv, modules it needs beyond the runtime, one-line purpose)
GATES = [
    ("check_prompt.py", [], [],
     "input_ids and marker positions vs the reference implementation"),
    ("verify.py", [], [],
     "labels and probabilities vs the Python gliner2 library"),
    ("check_limits.py", [], [],
     "512-token boundary and degenerate inputs"),
    ("check_tokenizer_fuzz.py", ["20000"], ["tokenizers"],
     "pure-Python tokenizer vs the Rust one"),
]

#: import name -> what it is, for the dependency report
RUNTIME = {"onnxruntime": "inference", "numpy": "arrays"}
OPTIONAL = {}   # this model is text only
FORBIDDEN = ["torch", "transformers", "tokenizers", "jinja2", "onnx", "regex"]


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def report_dependencies() -> None:
    present = [m for m in RUNTIME if have(m)]
    missing = [m for m in RUNTIME if not have(m)]
    extra = [m for m in OPTIONAL if have(m)]
    heavy = [m for m in FORBIDDEN if have(m)]
    print("  %-22s %-5s runtime: %s%s"
          % ("dependencies", "ok " if not missing else "XX ",
             ", ".join(present) or "none",
             ("  + " + ", ".join(extra)) if extra else ""))
    if missing:
        print("      missing runtime dependency: %s" % ", ".join(missing))
    if heavy:
        print("      note: %s present -- fine for development, not needed to run"
              % ", ".join(heavy))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to run the gates with")
    parser.add_argument("--full-fuzz", action="store_true",
                        help="run the tokenizer fuzz at full size (slow)")
    args = parser.parse_args()

    print("%s\n" % ROOT.name)
    report_dependencies()

    failed = skipped = passed = 0
    for script, argv, needs, purpose in GATES:
        path = ROOT / "verify" / script
        if not path.exists():
            continue
        absent = [m for m in needs if not have(m)]
        if absent:
            print("  %-22s skip  needs %s -- %s" % (script[:-3], ", ".join(absent), purpose))
            skipped += 1
            continue
        if script == "check_tokenizer_fuzz.py" and args.full_fuzz:
            argv = ["150000"]

        started = time.perf_counter()
        result = subprocess.run(
            [args.python, str(path), *argv], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        elapsed = time.perf_counter() - started
        ok = result.returncode == 0
        failed += not ok
        passed += ok
        summary = _summarise(result.stdout)
        print("  %-22s %-5s %-46s %5.1fs" % (script[:-3], "ok " if ok else "XX ",
                                             summary or purpose, elapsed))
        if not ok:
            tail = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
            for line in tail[-6:]:
                print("      %s" % line[:110])

    print("\n%d passed, %d failed, %d skipped" % (passed, failed, skipped))
    return 1 if failed else 0


def _summarise(stdout: str) -> str:
    """Pull the one line a gate prints as its verdict."""
    keys = ("failures:", "decisions ", "mismatches:", "worst ", "all edge cases")
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if any(key in line for key in keys):
            return line[:46]
    return lines[-1][:46] if lines else ""


if __name__ == "__main__":
    raise SystemExit(main())
