[[Japanese](README.md)/[English](README_EN.md)]

# GLiNER2.5-Decide-ONNX-CPU

A minimal-dependency inference implementation that runs the classification path of [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (typed decisions) on CPU with only onnxruntime and numpy.<br>
It uses no torch, no transformers and no tokenizers.

# Features
- Minimal dependencies: only two packages are needed at run time, onnxruntime and numpy (86 MB). `gliner2[local]` is 33 packages and 661 MB
- Two scripts are all you need to run it: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference)
- Decisions are practically identical to the official Python implementation (argmax 13/13, max probability difference 4.4e-07)
- Faster than the torch version on CPU (3 questions, about 100 tokens: 199 ms with 4 threads vs 465 ms for torch)
- The ONNX is identical to the fp32 of [onnx-community/GLiNER2.5-Decide-ONNX](https://huggingface.co/onnx-community/GLiNER2.5-Decide-ONNX) and is distributed in [Releases](https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU/releases/tag/v0.0.1)
- The official preprocessing (tokenization, prompt, marker positions) is ported to pure Python, with checks against the official implementation included

# Purpose of This Repository
This repository is for verifying the following.
- How far run-time dependencies can be reduced by cutting only the classification path out into ONNX
- Whether the official preprocessing can be ported to pure Python and produce the same inputs
- How speed and accuracy compare with the torch version on CPU

See [artifacts/report.md](artifacts/report.md) for the measurements and the reasoning behind the port.

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
```
These two are all that is needed to run. Downloading the model uses only the standard library.<br>
Environments are managed with [uv](https://docs.astral.sh/uv/).<br>
Re-exporting the ONNX yourself and some of the checks need the `export` group (torch, transformers, onnx, etc.).

# Installation

```bash
git clone https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU
cd GLiNER2.5-Decide-ONNX-CPU

# Install only the run-time dependencies (onnxruntime and numpy)
uv sync
```

# Download Model
The ONNX files are in [Releases](https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU/releases/tag/v0.0.1) (they are not in the repository).<br>
The script below fetches them into `models/onnx/onnx/`. It uses plain HTTPS from the standard library; neither git nor git-lfs is needed.
```bash
uv run download_model.py
```
- Downloads `model.onnx` and `model.onnx_data` (about 1.7 GB in total) and checks them against the SHA-256 pinned in the script (on a mismatch the file is deleted and the script stops)
- Does nothing if the files are already there, and resumes an interrupted download when run again
- The tokenizer and config files are in the repository
- To download by hand, put the two files from Releases into `models/onnx/onnx/`

Other ways to get a model.
```bash
uv run download_model.py --list                        # list the variants
uv run download_model.py --source hf                   # fp32 from Hugging Face
uv run download_model.py --variant q4f16               # smaller but slow on CPU (Hugging Face)
uv run --group export download_model.py --self-export  # export from the original checkpoint
```

| variant | size | argmax | CPU median |
|---|---:|---:|---:|
| fp32 | 1663 MB | 13/13 | 212 ms |
| q4f16 | 499 MB | 12/13 | 470 ms |

4-bit is an optimisation for WebGPU; on CPU the dequantization cost outweighs the gain, so fp32 is the fastest.<br>
`--self-export` writes the graph yourself from `fastino/GLiNER2.5-Decide`. The output is byte-for-byte identical to the published one, so the same procedure applies to fine-tuned checkpoints.

# Usage

### Example (Python)
```bash
uv run demo_inference_text.py
```

```python
from gliner_decide import Decider

d = Decider("models/onnx", variant="fp32")

d.classify_text("I was charged twice and support never replied.",
                {"intent": ["refund_request", "order_status", "other"]})
# -> {"intent": "refund_request"}

d.decide("The export button crashes in Safari but works in Chrome.", [
    ("severity", ["cosmetic", "degraded", "blocking"], None),
    ("team", ["billing", "technical support"],
     {"billing": "charges, refunds", "technical support": "bugs, outages"}),
])
```
All questions are answered in a single forward pass. The schema is laid out in the prompt, and the logits at the `[L]` marker placed on each label are used directly as that label's score.<br>
This model is text only, so there is no image demo.

### Input layout
```text
( [P] prompt ( [L] label_1 [L] label_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word … .
```
- `marker_positions` are the positions of the `[L]` tokens, flattened across questions
- The prompt and labels are tokenized as they are; the body text is lower-cased and tokenized word by word
- `(` and `)` are separate words, and no `[CLS]` / `[SEP]` is added
- Special token IDs: `[SEP_STRUCT]`=128001 `[SEP_TEXT]`=128002 `[P]`=128003 `[L]`=128007

# Verification
```bash
uv run verify/run_all.py                                 # all gates
uv run --with tokenizers verify/run_all.py --full-fuzz   # exhaustive comparison with the Rust tokenizers
```

| Gate | What it checks | Needs |
|---|---|---|
| check_prompt | `input_ids` and marker positions match gliner2 in 6/6 cases | run-time dependencies only |
| verify | labels and probabilities match the Python implementation (argmax 13/13) | run-time dependencies only |
| check_limits | 9 items such as the 512-token boundary, 400 labels, empty text | run-time dependencies only |
| check_tokenizer_fuzz | comparison with the Rust `tokenizers` (about 5,200 cases by default, 0 mismatches; 312,706 cases with `--full-fuzz`) | tokenizers |

### Performance (Core i7-12800H, CPU only)
3 questions, about 100 tokens. Cold start is 3.3 s, and even one thread takes 581 ms, so it is usable where cores cannot be spared.

| threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| median | 581 ms | 336 ms | 199 ms | 160 ms |

| | packages | size |
|---|---:|---:|
| gliner2[local] | 33 | 661 MB |
| this implementation | 2 | 86 MB |

# Limitations
- Classification only. NER, relation extraction and structured extraction are not in the ONNX
- 512-token limit (the tail is cut off beyond it)
- The model was trained on English operational text
- Constrained decoding (beam / exact) is not ported; only independent per-question softmax
- `tokenizer.py` is specific to this checkpoint and rejects any other tokenizer.json with an exception

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (two at run time; the export group is for exporting and the verification oracle)
uv.lock                  # uv lock file
download_model.py        # model download (Releases / Hugging Face) and self-export
demo_inference_text.py   # inference demo
gliner_decide/           # inference implementation (light dependencies only; does not import export/)
  runtime.py             #   ORT session, per-question softmax, classify_text-compatible API
  tokenizer.py           #   pure-Python SentencePiece Unigram
  protocol.py            #   GLiNER2 classification layout (prompt + marker positions)
export/
  export_onnx.py         # encoder + classification MLP in one graph (gather built in)
verify/                  # verification gates (run all with run_all.py)
artifacts/               # golden.json (reference data for verification), report.md
models/                  # ONNX (weights are gitignored), tokenizer and config files
```

# Author
Kazuhito Takahashi (https://x.com/KzhtTkhs)

# License
GLiNER2.5-Decide-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
The model weights follow the license of the original model ([fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)).
