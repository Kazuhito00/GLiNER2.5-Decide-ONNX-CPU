---
license: apache-2.0
base_model: fastino/GLiNER2.5-Decide
library_name: transformers.js
pipeline_tag: text-classification
tags:
  - onnx
  - webgpu
  - typed-decisions
  - gliner2
  - open-jev
  - system-one
language:
  - en
---

# GLiNER2.5-Decide, ONNX for the browser

ONNX export of the **classification path** of [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), packaged for [Transformers.js](https://huggingface.co/docs/transformers.js) and the [open-jev](https://github.com/nico-martin/open-jev) typed-decision library. It runs on WebGPU (WASM fallback) with nothing leaving the tab.

GLiNER2.5-Decide is a DeBERTa-v3-large encoder with a trained label head, made by Fastino for typed operational decisions: routing, sentiment, yes/no gates, ordinal scores. Its extraction heads (span, boundary, record) are not part of this export.

## Use with open-jev

```ts
import { OpenJev, choice, noul, score } from "open-jev";

const jev = await OpenJev.load({ model: "gliner2-decide" }); // or "onnx-community/GLiNER2.5-Decide-ONNX"

const [team, angry, severity] = await jev.decide(
  "The export button crashes in Safari but works in Chrome. Not blocking.",
  [
    choice("Which team should handle this?", ["billing", "technical support", "sales"], {
      billing: "charges, refunds, invoices",
      "technical support": "bugs, outages",
    }),
    noul("Is the customer angry?"),
    score("How severe is this bug?", ["cosmetic", "degraded but there is a workaround", "blocking"]),
  ],
);
```

## Files

| File | Size | Notes |
| --- | --- | --- |
| `onnx/model.onnx` + `_data` | 1.74 GB | fp32, reference |
| `onnx/model_fp16.onnx` + `_data` | 872 MB | default on WebGPU with `shader-f16` |
| `onnx/model_q4.onnx` + `_data` | 888 MB | 4-bit MatMul weights, fp32 elsewhere (the 128k-token embedding table dominates) |
| `onnx/model_q4f16.onnx` + `_data` | 523 MB | 4-bit MatMul weights, fp16 elsewhere; smallest download |

## Graph

Inputs: `input_ids` `[batch, sequence]`, `attention_mask` `[batch, sequence]`, `marker_positions` `[batch, markers]` (int64, index of every `[L]` token).
Output: `logits` `[batch, markers]`, one logit per label from the model's 1024→2048→1 head. Softmax within each question's markers gives the distribution.

The sequence is the GLiNER2 processor's classification layout, which the caller must reproduce exactly (open-jev does):

```
( [P] prompt ( [L] label_1 [L] label_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word … .
```

- The prompt is the task text, with per-label descriptions appended as ` [DESCRIPTION] label: description`.
- The prompt and the labels keep their case and are tokenized as whole strings.
- The state is lowercased, split with the processor's word regex, given a terminal `.` if it has none, and tokenized one word at a time with no special tokens.
- `(` and `)` are tokenized as standalone words. No `[CLS]`/`[SEP]`.

## Fidelity

Compared with the Python `gliner2` library (v2.0.0) on 6 states and 13 questions:

| Variant | Same argmax | Worst probability difference |
| --- | --- | --- |
| fp32 | 13/13 | 5.8e-7 |
| fp16 | 13/13 | 2.5e-4 |
| q4 | 12/13 | 7.2e-2 |
| q4f16 | 12/13 | 7.1e-2 |

The q4 flip is a three-way call at 0.50 / 0.42 probability. Scripts to reproduce the export and these numbers are in `conversion/`.

## Limits

- 512-token context. Long states are cut at the end by open-jev.
- Trained on English operational text; not trained on public benchmarks (see the base model card).
- This export covers classification only. Entity, relation and structured extraction need the Python library.

## Credits

Model by [Fastino](https://huggingface.co/fastino) (Apache-2.0). Export and open-jev integration by Shreyas Karnik. Not affiliated with Fastino or TypeSafe.
