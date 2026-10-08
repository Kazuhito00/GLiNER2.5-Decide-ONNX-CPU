"""Minimal torch-free runtime for the GLiNER2.5-Decide classification path.

Dependencies: onnxruntime, numpy. Nothing else.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from .protocol import DecideEncoder

Question = Tuple[str, Sequence[str], Optional[Dict[str, str]]]

VARIANTS = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
}


GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Decider:
    """Loads one ONNX variant and answers typed questions about a text."""

    def __init__(
        self,
        model_dir: Union[str, pathlib.Path],
        variant: str = "fp32",
        intra_op_threads: Optional[int] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        model_dir = pathlib.Path(model_dir)
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; pick one of {sorted(VARIANTS)}")
        path = model_dir / "onnx" / VARIANTS[variant]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (was it pulled from LFS?)")

        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        self.variant = variant
        self.session = ort.InferenceSession(
            str(path), sess_options=opts,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.encoder = DecideEncoder(model_dir / "tokenizer.json")

    def logits(self, text: str, questions: Sequence[Question]) -> Tuple[np.ndarray, List[List[int]]]:
        enc = self.encoder.encode(text, questions)
        feed = {
            "input_ids": np.array([enc["input_ids"]], dtype=np.int64),
            "attention_mask": np.array([enc["attention_mask"]], dtype=np.int64),
            "marker_positions": np.array([enc["marker_positions"]], dtype=np.int64),
        }
        out = self.session.run(["logits"], feed)[0][0]
        return out, enc["groups"]

    def decide(
        self, text: str, questions: Sequence[Question]
    ) -> List[Dict[str, object]]:
        """One forward pass; a probability distribution per question.

        Softmax is taken within each question's own markers, matching the
        Python library's per-question normalization.
        """
        logits, groups = self.logits(text, questions)
        answers = []
        for (task, labels, _), g in zip(questions, groups):
            probs = softmax(logits[np.array(g)].astype(np.float64))
            best = int(probs.argmax())
            answers.append(
                {
                    "task": task,
                    "label": labels[best],
                    "confidence": float(probs[best]),
                    "probabilities": {l: float(p) for l, p in zip(labels, probs)},
                }
            )
        return answers

    def classify_text(
        self, text: str, schema: Dict[str, Sequence[str]]
    ) -> Dict[str, str]:
        """gliner2-compatible shape: {"intent": [...]} -> {"intent": "label"}."""
        questions = [(task, labels, None) for task, labels in schema.items()]
        return {a["task"]: a["label"] for a in self.decide(text, questions)}
