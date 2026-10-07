"""Step 3: minimal runtime vs the Python library's reference outputs.

Runs in the torch-free env. reference.json carries gliner2 v2.0.0's own
probabilities, so no torch import is needed to compare against it.
"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gliner_decide.runtime import Decider

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="fp32")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

ref = json.load(open(ROOT / "artifacts/golden.json", encoding="utf-8"))
d = Decider(ROOT / "models/onnx", variant=a.variant, intra_op_threads=a.threads)

worst_p = 0.0
agree = total = 0
for c in ref:
    questions = [tuple(q) for q in c["questions"]]
    answers = d.decide(c["text"], questions)
    for ans, (task, labels, _), py_probs in zip(answers, questions, c["probabilities"]):
        got = np.array([ans["probabilities"][l] for l in labels])
        exp = np.array(py_probs)
        dp = float(np.abs(got - exp).max())
        worst_p = max(worst_p, dp)
        py_best = labels[int(exp.argmax())]
        same = ans["label"] == py_best
        agree += same
        total += 1
        flag = "  " if same else "!!"
        print(f"{flag} {task[:34]:34s} onnx={ans['label'][:28]:>28s} {ans['confidence']:.4f} "
              f"| py={py_best[:28]:>28s} {exp.max():.4f} | max|dp|={dp:.2e}")

print(f"\nvariant={a.variant}  argmax agreement {agree}/{total}  worst |dp| {worst_p:.2e}")
