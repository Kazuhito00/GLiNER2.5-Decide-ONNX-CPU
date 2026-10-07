"""Step 5: re-export the classification path ourselves, from local weights.

Mirrors the published conversion/export_onnx.py but reads the local checkpoint
and writes next to it, so the result can be diffed against the community export.
"""
import json, pathlib, sys, time
import torch, torch.nn as nn
from gliner2 import AutoExtractor

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "models/torch"
OUT = ROOT / "models/onnx-selfexport"
(OUT / "onnx").mkdir(parents=True, exist_ok=True)

m = AutoExtractor.from_pretrained(str(SRC)); m.eval()
enc, clf, proc = m.encoder, m.classifier, m.processor
print("special tokens:", proc.SPECIAL_TOKENS, flush=True)
cfg = json.loads(enc.config.to_json_string()); cfg["architectures"] = ["DebertaV2Model"]
json.dump(cfg, open(OUT / "config.json", "w"), indent=2)
proc.tokenizer.save_pretrained(str(OUT))

class DecideGraph(nn.Module):
    def __init__(self, e, c):
        super().__init__(); self.encoder, self.classifier = e, c
    def forward(self, input_ids, attention_mask, marker_positions):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        g = torch.gather(h, 1, marker_positions.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        return self.classifier(g).squeeze(-1)

graph = DecideGraph(enc, clf).eval()

# Reference logits on a real case, so parity is checked on real input, not a stub.
ref = json.load(open(ROOT / "artifacts/golden.json", encoding="utf-8"))[0]
ids = torch.tensor([ref["input_ids"]]); mask = torch.ones_like(ids)
pos = torch.tensor([ref["marker_positions"]])
with torch.no_grad():
    logits = graph(ids, mask, pos)
json.dump({"input_ids": ref["input_ids"], "marker_positions": ref["marker_positions"],
           "logits": logits[0].tolist()}, open(OUT / "torch_ref.json", "w"), indent=1)
print("torch logits:", [round(v, 5) for v in logits[0].tolist()], flush=True)

t0 = time.time()
mode = sys.argv[1] if len(sys.argv) > 1 else "dynamo"
common = dict(input_names=["input_ids", "attention_mask", "marker_positions"],
              output_names=["logits"], opset_version=17)
if mode == "dynamo":
    torch.onnx.export(graph, (ids, mask, pos), str(OUT / "onnx/model.onnx"),
        dynamic_shapes={"input_ids": {0: "batch", 1: "sequence"},
                        "attention_mask": {0: "batch", 1: "sequence"},
                        "marker_positions": {0: "batch", 1: "markers"}},
        dynamo=True, external_data=True, optimize=True, **common)
else:
    torch.onnx.export(graph, (ids, mask, pos), str(OUT / "onnx/model.onnx"),
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "marker_positions": {0: "batch", 1: "markers"},
                      "logits": {0: "batch", 1: "markers"}},
        do_constant_folding=False, dynamo=False, **common)
print("exported", mode, "in", round(time.time() - t0, 1), "s", flush=True)
