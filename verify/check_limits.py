"""Edge cases the reference set doesn't cover: 512-token boundary, degenerate inputs."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gliner_decide.protocol import DecideEncoder
from gliner_decide.runtime import Decider

ROOT = pathlib.Path(__file__).resolve().parents[1]
enc = DecideEncoder(ROOT / "models/onnx/tokenizer.json")
d = Decider(ROOT / "models/onnx", variant="fp32", intra_op_threads=4)

Q = [("sentiment", ["negative", "neutral", "positive"], None)]
fails = 0

def report(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

# 1. way over 512 tokens -> truncated, still runs
long_text = ("The deployment failed again and nobody is answering the pager. " * 120)
e = enc.encode(long_text, Q)
report("long text truncated to max_len", len(e["input_ids"]) == 512, f"len={len(e['input_ids'])}")
a = d.decide(long_text, Q)[0]
report("long text still infers", abs(sum(a["probabilities"].values()) - 1.0) < 1e-6,
       f"{a['label']} {a['confidence']:.3f}")

# 2. just under / around the boundary
for n in (30, 60, 120):
    t = "the export button crashes in safari but works in chrome. " * n
    e = enc.encode(t, Q)
    ok = len(e["input_ids"]) <= 512 and max(e["marker_positions"]) < len(e["input_ids"])
    report(f"boundary n={n}", ok, f"len={len(e['input_ids'])}")

# 3. schema alone over the limit must raise, not silently drop labels
try:
    enc.encode("hi", [("t", [f"label_number_{i}_with_padding_words" for i in range(400)], None)])
    report("oversized schema raises", False)
except ValueError as ex:
    report("oversized schema raises", True, type(ex).__name__)

# 4. degenerate inputs
e = enc.encode("", Q)
report("empty text -> '.'", len(e["input_ids"]) > 0, f"len={len(e['input_ids'])}")
a = d.decide("great job", [("ok", ["yes"], None)])[0]
report("single label -> prob 1.0", abs(a["confidence"] - 1.0) < 1e-9, f"{a['confidence']:.6f}")
a = d.decide("ありがとう、完璧に動きました", [("sentiment", ["negative", "neutral", "positive"], None)])[0]
report("non-English runs (out of model scope)", True, f"{a['label']} {a['confidence']:.3f}")

print(f"\n{'all edge cases passed' if not fails else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
