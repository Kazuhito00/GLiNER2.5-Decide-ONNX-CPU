"""GLiNER2.5-Decide classification layout, reproduced without transformers.

Ports the parts of ``gliner2.processor.Processor`` that the classification
(decide) path needs:

  ``( [P] prompt ( [L] l1 [L] l2 ... ) ) [SEP_STRUCT] ( [P] ... ) [SEP_TEXT] w w ... .``

Only ``tokenizers`` (the Rust fast tokenizer) is required; the shipped
``tokenizer.json`` already contains the ten GLiNER2 special tokens, so the
runtime ``add_special_tokens`` call the Python library makes is a no-op here.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .tokenizer import UnigramTokenizer

SEP_STRUCT = "[SEP_STRUCT]"
SEP_TEXT = "[SEP_TEXT]"
P_TOKEN = "[P]"
L_TOKEN = "[L]"
DESC_TOKEN = "[DESCRIPTION]"

# gliner2/processing/word_splitter.py :: WhitespaceTokenSplitter
_WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)

MAX_LEN = 512


def split_words(text: str, lower: bool = True) -> List[str]:
    """Word-level split used before subword tokenization."""
    return [
        (m.group().lower() if lower else m.group()) for m in _WORD_PATTERN.finditer(text)
    ]


def normalize_text(text: str) -> str:
    """gliner2/processor.py :: Processor._normalize_text."""
    if not text:
        return "."
    if not text.endswith((".", "!", "?")):
        return text + "."
    return text


class DecideEncoder:
    """Turns (text, questions) into the tensors the ONNX graph expects."""

    def __init__(self, tokenizer_path: str, max_len: int = MAX_LEN):
        self._tok = UnigramTokenizer(str(tokenizer_path))
        self.max_len = max_len
        # tokenize() is deterministic per string, so caching is byte-exact.
        self._ids_of = lru_cache(maxsize=50_000)(self._encode_word)

    def _encode_word(self, word: str) -> Tuple[int, ...]:
        return tuple(self._tok.encode(word))

    def schema_tokens(
        self,
        task: str,
        labels: Sequence[str],
        label_descriptions: Optional[Dict[str, str]] = None,
        prompt: Optional[str] = None,
    ) -> List[str]:
        """gliner2/processor.py :: Processor._transform_schema (inference mode)."""
        prompt_str = f"{task}: {prompt}" if prompt else task
        if label_descriptions:
            for label, desc in label_descriptions.items():
                if label in labels:
                    prompt_str += f" {DESC_TOKEN} {label}: {desc}"

        tokens = ["(", P_TOKEN, prompt_str, "("]
        for label in labels:
            tokens.extend([L_TOKEN, label])
        tokens.extend([")", ")"])
        return tokens

    def encode(
        self,
        text: str,
        questions: Sequence[Tuple[str, Sequence[str], Optional[Dict[str, str]]]],
    ) -> Dict[str, object]:
        """Build input_ids / attention_mask / marker_positions and label groups.

        ``marker_positions`` is the flat index of every ``[L]`` token, in
        question order -- the convention the exported graph expects.
        """
        schema_tokens_list = [
            self.schema_tokens(task, labels, descs) for task, labels, descs in questions
        ]

        words: List[str] = []
        for i, struct in enumerate(schema_tokens_list):
            if i:
                words.append(SEP_STRUCT)
            words.extend(struct)
        words.append(SEP_TEXT)
        words.extend(split_words(normalize_text(text), lower=True))

        input_ids: List[int] = []
        marker_positions: List[int] = []
        groups: List[List[int]] = [[] for _ in questions]

        q_idx = -1
        for word in words:
            start = len(input_ids)
            input_ids.extend(self._ids_of(word))
            if word == P_TOKEN:
                q_idx += 1
            elif word == L_TOKEN:
                groups[q_idx].append(len(marker_positions))
                marker_positions.append(start)

        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            kept = sum(1 for p in marker_positions if p < self.max_len)
            if kept != len(marker_positions):
                raise ValueError(
                    f"schema alone exceeds the {self.max_len}-token context "
                    f"({len(marker_positions) - kept} label markers cut off)"
                )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "marker_positions": marker_positions,
            "groups": groups,
        }
