"""Latency of the torch-free runtime: cold start and per-call."""
import argparse, json, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="fp32")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--runs", type=int, default=40)
a = ap.parse_args()

t0 = time.perf_counter()
from gliner_decide.runtime import Decider
t_import = time.perf_counter() - t0

t0 = time.perf_counter()
d = Decider(ROOT / "models/onnx", variant=a.variant, intra_op_threads=a.threads)
t_load = time.perf_counter() - t0

cases = json.load(open(ROOT / "artifacts/golden.json", encoding="utf-8"))
text, qs = cases[2]["text"], [tuple(q) for q in cases[2]["questions"]]

t0 = time.perf_counter(); d.decide(text, qs); t_first = time.perf_counter() - t0
times = []
for _ in range(a.runs):
    t0 = time.perf_counter(); d.decide(text, qs); times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(json.dumps({
    "variant": a.variant, "threads": a.threads, "runs": a.runs,
    "import_s": round(t_import, 3), "session_load_s": round(t_load, 3),
    "first_call_s": round(t_first, 3),
    "median_ms": round(statistics.median(times), 1),
    "p95_ms": round(times[int(len(times) * 0.95) - 1], 1),
    "cold_total_s": round(t_import + t_load + t_first, 3),
}, indent=2))
