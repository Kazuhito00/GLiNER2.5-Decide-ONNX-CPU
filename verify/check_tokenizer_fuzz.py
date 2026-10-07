# -*- coding: utf-8 -*-
"""Oracle: pure-Python tokenizer vs the Rust `tokenizers` library.

Run in an env that still HAS tokenizers (.venv-onnx). The corpus deliberately
covers the shapes DecideEncoder actually feeds the tokenizer, plus the Unicode
classes where Python's `re` and Rust's regex are known to disagree.
"""
import json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gliner_decide.tokenizer import UnigramTokenizer
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKJSON = ROOT / "models/onnx/tokenizer.json"

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]",
            "[P]", "[C]", "[E]", "[R]", "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"]

WHITESPACE = ["\u00a0", "\u2028", "\u2029", "\u3000", "\x0b", "\x0c", "\x1c", "\x1d",
              "\x1e", "\x1f", "\u200b", "\u2007", "\u202f", "\ufeff"]

BASE = [
    "", " ", "  ", "\t", "\n", "a", "(", ")", ",", "|", ":", ".", "-", "_",
    "team", "payments", "account", "other", "urgent", "yes", "no", "billing team",
    "technical support", "degraded but there is a workaround", "three or more",
    "Which team should handle this?", "How many separate requests are in this message?",
    "billing team: charges, refunds, invoices",
    "Which team should handle this? [DESCRIPTION] billing team: charges, refunds, invoices"
    " [DESCRIPTION] technical support: bugs, outages",
    "URGENT: production database is down, all customers affected",
    "CamelCase", "ALLCAPS", "snake_case_name", "kebab-case-name", "mixed123digits",
    "0", "42", "3.14", "1,000,000", "2026-09-27", "v2.0.0",
    "user@example.com", "https://example.com/a/b?c=d&e=f", "www.example.co.jp",
    "@handle", "#hashtag", "50%", "$1,234.56", "¥5,400", "€99", "100°C",
    "naïve café résumé", "Straße Zürich", "İstanbul", "ﬁreﬂy", "ǅungla",
    "日本語のテキストです", "これはテストです。", "漢字かなカナ", "全角　スペース",
    "한국어 텍스트", "Русский текст", "العربية", "עברית", "ไทย", "हिन्दी",
    "emoji 🎉 test", "👨‍👩‍👧‍👦 family", "🇯🇵 flag", "a🎉b",
    "<0x41>", "<0xFF>", "\n literal", "a\u0301 combining", "\uFB01x",
    "e\u0301galite\u0301", "ﬀ ligature", "ⅷ roman", "㍿ square", "①②③",
    "a" * 40, "ab" * 30, "x" * 100,
    "the quick brown fox jumps over the lazy dog",
    "The export button crashes in Safari but works in Chrome. Not blocking.",
]
BASE += SPECIALS
BASE += [f"a{w}b" for w in WHITESPACE]
BASE += [f"{w}" for w in WHITESPACE]
BASE += [f"a{w}{w}b" for w in WHITESPACE[:6]]
BASE += [f"x [P] y {s} z" for s in SPECIALS[5:]]

ALPHABET = list(
    "abcdeABCDE 0123(),.:;-_|?!@#$%/" + chr(92) + chr(39) + chr(34) + "`~^*+=<>[]{}"
    "\u2581\u00e9\u00fc\u00f1\u00e7\u0130\u0131"
    "\u65e5\u672c\u8a9e\u6f22\u5b57\u3072\u3089\u304c\u306a\u30ab\u30bf"
    "\ud55c\uae00\u0420\u0443\u0441\u0627\u0644\u0639\u0e44\u0e17"
    "\U0001f389\U0001f44d\U0001f1ef\U0001f1f5\u200d\ufe0f"
    "\u00a0\u3000\u2009\u202f\u200b\ufeff\x0b\x0c\x1c\x85"
    "\t\n\r "
    "\u2460\u2177\u337f\ufb01\u01c5\u0301"
)


def random_cases(n, rng):
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(k)))
    # the shape prompt_str actually takes: words glued around a special token
    for _ in range(n // 4):
        w = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 12)))
        out.append(f"{w} {rng.choice(SPECIALS)} {w}: {w}")
    return out


def main():
    rng = random.Random(20260927)
    mine = UnigramTokenizer(TOKJSON)
    ref = Tokenizer.from_file(str(TOKJSON))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    cases = list(BASE) + random_cases(n, rng)
    # every word the real encoder would ever emit for the reference set
    refjson = json.load(open(ROOT / "artifacts/golden.json", encoding="utf-8"))
    for c in refjson:
        cases.append(c["text"])
        for task, labels, descs in c["questions"]:
            cases.append(task); cases.extend(labels)
            if descs:
                for k, v in descs.items():
                    cases += [k, v, f"{k}: {v}"]

    bad = []
    for t in cases:
        got = list(mine.encode(t))
        exp = ref.encode(t, add_special_tokens=False).ids
        if got != exp:
            bad.append((t, got, exp))

    print(f"cases: {len(cases)}   mismatches: {len(bad)}")
    for t, got, exp in bad[:15]:
        print(f"\n  input : {t!r}")
        print(f"  got   : {got}")
        print(f"  expect: {exp}")
        print(f"  gotstr: {[ref.id_to_token(i) for i in got]}")
        print(f"  expstr: {[ref.id_to_token(i) for i in exp]}")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
