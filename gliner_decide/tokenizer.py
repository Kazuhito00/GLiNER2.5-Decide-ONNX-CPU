# -*- coding: utf-8 -*-
"""SentencePiece Unigram tokenizer in pure Python.

Reads a HuggingFace ``tokenizer.json`` and reproduces what
``tokenizers.Tokenizer.encode(text, add_special_tokens=False)`` returns, so the
``tokenizers`` (Rust) dependency can be dropped. Only the pipeline this
checkpoint actually declares is implemented::

    added tokens -> Replace(2+ spaces / CR / LF / TAB -> " ") -> NFC -> Strip(right)
                 -> Metaspace(replacement=U+2581, prepend_scheme=always, split=true)
                 -> Unigram Viterbi (unk_id=3, byte_fallback=false, fuse_unk=true)

The post-processor ([CLS]/[SEP]) is deliberately not applied: the GLiNER2
classification layout adds no sentence markers.

Anything else a tokenizer.json might declare (BPE/WordPiece, a precompiled
charsmap, byte_fallback) is rejected at load time rather than silently ignored.

Verified against the Rust tokenizer on 312,706 inputs; see scripts/fuzz_tokenizer.py.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

__all__ = ["UnigramTokenizer"]

SPACE = "▁"  # the block character SentencePiece substitutes for a space
UNK_PENALTY = 10.0  # tokenizers/src/models/unigram/model.rs :: K_UNK_PENALTY
NEG_INF = float("-inf")

# Rust decides what is whitespace by the Unicode White_Space property, both in
# the regex crate's \s and in char::is_whitespace (which Strip uses). Python's
# re.UNICODE \s and str.rstrip() additionally treat U+001C..U+001F as space, so
# the set is spelled out here rather than relying on the Python defaults.
WHITE_SPACE = (
    "\t\n\x0b\x0c\r \x85\xa0 "
    "           "
    "    　"
)
_WS_CLASS = "[" + "".join("\\u%04x" % ord(c) for c in WHITE_SPACE) + "]"


class UnigramTokenizer:
    """Encode text to token ids the way the Rust tokenizer would."""

    def __init__(self, tokenizer_json: str):
        with open(tokenizer_json, encoding="utf-8") as fh:
            spec = json.load(fh)

        model = spec["model"]
        if model.get("type") != "Unigram":
            raise ValueError("expected a Unigram model, got %r" % (model.get("type"),))
        if model.get("byte_fallback"):
            raise ValueError("byte_fallback is not implemented")

        self.unk_id: int = model["unk_id"]
        self._vocab: Dict[str, Tuple[int, float]] = {}
        max_len_by_head: Dict[str, int] = {}
        min_score = 0.0
        for tid, (piece, score) in enumerate(model["vocab"]):
            self._vocab[piece] = (tid, score)
            if score < min_score:
                min_score = score
            head = piece[0]
            if len(piece) > max_len_by_head.get(head, 0):
                max_len_by_head[head] = len(piece)
        self._max_len_by_head = max_len_by_head
        self._unk_score = min_score - UNK_PENALTY

        self._check_normalizer(spec.get("normalizer"))
        self._check_pre_tokenizer(spec.get("pre_tokenizer"))

        # AddedVocabulary::extract_and_normalize runs in two passes: tokens
        # flagged normalized=false are split out of the RAW text first, and each
        # surviving piece is normalized on its own; tokens flagged
        # normalized=true are only matched afterwards, inside an already
        # normalized piece. The distinction is observable -- Strip(right) runs
        # per raw piece in the first pass but only once before the second -- so
        # the two groups are kept apart here. Longest-first within each group so
        # a longer token wins over one that prefixes it.
        added = sorted(spec.get("added_tokens", []), key=lambda t: -len(t["content"]))
        self._added: Dict[str, int] = {t["content"]: t["id"] for t in added}
        self._raw_re = self._alternation(t for t in added if not t.get("normalized"))
        self._norm_re = self._alternation(t for t in added if t.get("normalized"))

        self._ws_re = re.compile(_WS_CLASS + "{2,}|[\n\r\t]")

    # --------------------------------------------------------- spec guards

    @staticmethod
    def _alternation(tokens):
        pattern = "|".join(re.escape(t["content"]) for t in tokens)
        return re.compile(pattern) if pattern else None

    @staticmethod
    def _check_normalizer(norm) -> None:
        steps = norm["normalizers"] if norm and norm.get("type") == "Sequence" else [norm]
        got = [s.get("type") for s in steps if s]
        if got != ["Replace", "NFC", "Strip"]:
            raise ValueError("unsupported normalizer sequence: %r" % (got,))
        if steps[0]["pattern"].get("Regex") != r"\s{2,}|[\n\r\t]" or steps[0]["content"] != " ":
            raise ValueError("unexpected Replace step: %r" % (steps[0],))
        if steps[2].get("strip_left") or not steps[2].get("strip_right"):
            raise ValueError("unexpected Strip step: %r" % (steps[2],))

    @staticmethod
    def _check_pre_tokenizer(pre) -> None:
        steps = pre["pretokenizers"] if pre and pre.get("type") == "Sequence" else [pre]
        if [s.get("type") for s in steps if s] != ["Metaspace"]:
            raise ValueError("unsupported pre_tokenizer: %r" % (pre,))
        ms = steps[0]
        if (
            ms.get("replacement") != SPACE
            or ms.get("prepend_scheme") != "always"
            or not ms.get("split")
        ):
            raise ValueError("unexpected Metaspace config: %r" % (ms,))

    # ----------------------------------------------------------- normalize

    def _normalize(self, text: str) -> str:
        text = self._ws_re.sub(" ", text)
        text = unicodedata.normalize("NFC", text)
        return text.rstrip(WHITE_SPACE)

    # ------------------------------------------------------------- viterbi

    def _viterbi(self, chunk: str) -> List[int]:
        """Best-scoring segmentation of one metaspace chunk.

        Mirrors the Rust Lattice: candidates for an end position are visited in
        increasing begin position, and ties keep the first one -- which is what
        the Rust ``score > best_score`` comparison does.
        """
        n = len(chunk)
        best = [NEG_INF] * (n + 1)
        back: List[Tuple[int, int]] = [(0, 0)] * (n + 1)
        best[0] = 0.0
        vocab = self._vocab
        head_len = self._max_len_by_head

        for i in range(n):
            base = best[i]
            if base == NEG_INF:
                continue
            has_single = False
            limit = min(head_len.get(chunk[i], 0), n - i)
            for length in range(1, limit + 1):
                entry = vocab.get(chunk[i : i + length])
                if entry is None:
                    continue
                tid, score = entry
                if length == 1:
                    has_single = True
                end = i + length
                cand = base + score
                if cand > best[end]:
                    best[end] = cand
                    back[end] = (i, tid)
            if not has_single:
                cand = base + self._unk_score
                if cand > best[i + 1]:
                    best[i + 1] = cand
                    back[i + 1] = (i, self.unk_id)

        if n and best[n] == NEG_INF:  # unk always yields a path, so this is a bug
            raise RuntimeError("no segmentation found for %r" % (chunk,))

        out: List[int] = []
        pos = n
        while pos > 0:
            prev, tid = back[pos]
            # Unigram::from hardcodes fuse_unk = true, so a run of unknown
            # pieces collapses into a single [UNK] instead of one per character.
            if not (tid == self.unk_id and out and out[-1] == self.unk_id):
                out.append(tid)
            pos = prev
        out.reverse()
        return out

    # -------------------------------------------------------------- encode

    def _metaspace(self, text: str) -> List[int]:
        """Pre-tokenize one already-normalized piece and run Viterbi on it."""
        if not text:
            return []
        text = text.replace(" ", SPACE)
        if not text.startswith(SPACE):
            text = SPACE + text

        # Metaspace split, MergedWithNext: every chunk begins at a SPACE, and
        # Viterbi runs per chunk -- so no piece can ever span a space boundary.
        starts = [i for i, ch in enumerate(text) if ch == SPACE]
        out: List[int] = []
        for k, start in enumerate(starts):
            end = starts[k + 1] if k + 1 < len(starts) else len(text)
            out.extend(self._viterbi(text[start:end]))
        return out

    def _split_on(self, pattern, text: str):
        """Yield (piece, None) and (added_token, id) alternating."""
        if pattern is None:
            yield text, None
            return
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                yield text[pos : m.start()], None
            yield m.group(), self._added[m.group()]
            pos = m.end()
        if pos < len(text):
            yield text[pos:], None

    def _encode_normalized(self, text: str) -> List[int]:
        out: List[int] = []
        for piece, tid in self._split_on(self._norm_re, text):
            if tid is not None:
                out.append(tid)
            else:
                out.extend(self._metaspace(piece))
        return out

    def encode(self, text: str) -> List[int]:
        """Token ids for ``text``. No [CLS]/[SEP] are added."""
        if not text:
            return []
        out: List[int] = []
        for piece, tid in self._split_on(self._raw_re, text):
            if tid is not None:
                out.append(tid)
            else:
                out.extend(self._encode_normalized(self._normalize(piece)))
        return out

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]
