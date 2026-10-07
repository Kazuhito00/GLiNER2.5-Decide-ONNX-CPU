# -*- coding: utf-8 -*-
"""Text classification with GLiNER2.5-Decide, on onnxruntime + numpy.

    python demo_inference_text.py
    python demo_inference_text.py --state "..." --choice "intent=refund,order_status"

One prefill answers every question: the schema is laid out in the prompt with a
[L] marker per label, and the logits at those markers score the labels directly.
Nothing is generated, so the answer is always one of the labels supplied.

This model is text only; there is no demo_inference_image.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gliner_decide import Decider  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

EXAMPLE_STATE = (
    "Hi, I was charged twice for my Pro subscription this month. "
    "Please refund the duplicate payment."
)
EXAMPLE_QUESTIONS = [
    ("team", ["billing", "technical support", "sales"],
     {"billing": "charges, refunds, invoices",
      "technical support": "bugs, outages",
      "sales": "pricing, upgrades"}),
    ("intent", ["refund_request", "order_status", "cancel_subscription",
                "login_problem", "other"], None),
    ("angry", ["yes", "no"], None),
]


def parse_questions(args):
    questions = []
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        labels = [v.strip() for v in rest.split(",") if v.strip()]
        questions.append((name.strip(), labels, None))
    return questions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="omit to run the built-in example")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--model", default="models/onnx")
    parser.add_argument("--variant", default="fp32")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.state is None:
        state, questions = EXAMPLE_STATE, EXAMPLE_QUESTIONS
    else:
        state = args.state
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice")

    started = time.perf_counter()
    decider = Decider(ROOT / args.model, variant=args.variant, intra_op_threads=args.threads)
    load = time.perf_counter() - started

    started = time.perf_counter()
    answers = decider.decide(state, questions)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0

    print("state:\n  %s\n" % state[:160])
    for answer in answers:
        spread = "  ".join(
            "%s=%.3f" % kv
            for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
        )
        print("%-10s %-30s %.4f\n           %s"
              % (answer["task"], answer["label"], answer["confidence"], spread))
    print("\n%d questions in one forward pass" % len(answers))
    print("load %.1f s, inference %.3f s (%s)" % (load, elapsed, args.variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
