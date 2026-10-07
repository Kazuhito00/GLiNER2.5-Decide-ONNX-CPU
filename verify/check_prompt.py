"""Step 3-1: reproduce gliner2's input_ids / marker_positions with tokenizers only."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gliner_decide.protocol import DecideEncoder

ROOT = pathlib.Path(__file__).resolve().parents[1]
ref = json.load(open(ROOT / "artifacts/golden.json", encoding="utf-8"))
enc = DecideEncoder(ROOT / "models/onnx/tokenizer.json")

ok = bad = 0
for i, c in enumerate(ref):
    out = enc.encode(c["text"], [tuple(q) for q in c["questions"]])
    id_match = out["input_ids"] == c["input_ids"]
    mp_match = out["marker_positions"] == c["marker_positions"]
    if id_match and mp_match:
        ok += 1
        continue
    bad += 1
    print(f"[case {i}] ids={'OK' if id_match else 'MISMATCH'} markers={'OK' if mp_match else 'MISMATCH'}")
    print(f"  text: {c['text'][:70]}")
    if not id_match:
        g, e = out["input_ids"], c["input_ids"]
        print(f"  len got={len(g)} exp={len(e)}")
        for k in range(min(len(g), len(e))):
            if g[k] != e[k]:
                print(f"  first diff at {k}: got={g[k]} exp={e[k]}")
                print(f"    got ctx {g[max(0,k-4):k+5]}")
                print(f"    exp ctx {e[max(0,k-4):k+5]}")
                break
    if not mp_match:
        print(f"  markers got={out['marker_positions']} exp={c['marker_positions']}")

print(f"\ninput_ids + marker_positions exact match: {ok}/{ok+bad} cases")
sys.exit(1 if bad else 0)
