[[Japanese](README.md)/[English](README_EN.md)]

# GLiNER2.5-Decide-ONNX-CPU

[fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)を、onnxruntime と numpy だけでCPU推論する最小依存の推論実装です。<br>
torch も transformers も tokenizers は未使用。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ（86 MB）。`gliner2[local]` は33パッケージ・661 MB
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ
- 判定は公式のPython実装と実質同一（argmax 13/13、最大確率差 4.4e-07）
- CPUではtorch版より速い（3問・約100トークンで 4スレッド 199 ms、torch版は 465 ms）
- ONNXは[onnx-community/GLiNER2.5-Decide-ONNX](https://huggingface.co/onnx-community/GLiNER2.5-Decide-ONNX)のfp32と同一のものを[Releases](https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU/releases/tag/v0.0.1)で配布
- 公式の前処理（トークナイズ・プロンプト・marker位置）を純Pythonで移植し、公式実装との突き合わせ検証を同梱

# Purpose of This Repository
以下の検証を目的としています。
- 分類パスだけをONNXに切り出し、実行時の依存をどこまで減らせるか
- 公式の前処理を純Pythonに移植して、公式実装と同じ入力が作れるか
- CPU推論でtorch版と比べて速度・精度がどうなるか

実測値と移植の根拠は[artifacts/report.md](artifacts/report.md)を参照してください。

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
```
実行に必要なのはこの2つだけです。モデルの取得も標準ライブラリのみで動きます。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
ONNXの自前エクスポートと一部の検証には、`export` グループ（torch、transformers、onnx など）が必要です。

# Installation

```bash
git clone https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU
cd GLiNER2.5-Decide-ONNX-CPU

# 実行時の依存（onnxruntime と numpy）のみインストール
uv sync
```

# Download Model
ONNXファイルは[Releases](https://github.com/Kazuhito00/GLiNER2.5-Decide-ONNX-CPU/releases/tag/v0.0.1)に置いています（リポジトリには含めていません）。<br>
以下のスクリプトで `models/onnx/onnx/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py
```
- `model.onnx` と `model.onnx_data`（合計 約1.7 GB）を取得し、スクリプトに固定したSHA-256と照合します（不一致ならファイルを削除して停止）
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します
- トークナイザと設定ファイルはリポジトリに含まれています
- 手動で取得する場合は、Releasesの2ファイルを `models/onnx/onnx/` に置いてください

他の取得方法です。
```bash
uv run download_model.py --list                        # variant一覧
uv run download_model.py --source hf                   # fp32をHugging Faceから取得
uv run download_model.py --variant q4f16               # 小さいがCPUでは遅い（Hugging Face）
uv run --group export download_model.py --self-export  # checkpointから自前でエクスポート
```

| variant | サイズ | argmax | CPU median |
|---|---:|---:|---:|
| fp32 | 1663 MB | 13/13 | 212 ms |
| q4f16 | 499 MB | 12/13 | 470 ms |

4bitはWebGPU向けの最適化のため、CPUでは逆量子化のコストが勝ち、fp32が最速です。<br>
`--self-export` は `fastino/GLiNER2.5-Decide` から自分で書き出します。出力は公開版とバイト単位で同一になることを確認済みで、finetune版にも同じ手順を適用できます。

# Usage

### 実行例(Python)
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
全問が1回の順伝播で決まります。スキーマがプロンプトに展開され、ラベルごとに置かれた `[L]` マーカー位置のlogitsが、そのまま各ラベルのスコアになります。<br>
このモデルはテキスト専用のため、画像のデモはありません。

### 入力レイアウト
```text
( [P] prompt ( [L] label_1 [L] label_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word … .
```
- `marker_positions` は各 `[L]` トークンの位置を、問をまたいでフラットに並べたもの
- promptとラベルは原文のまま丸ごとトークナイズし、本文は小文字化して単語ごとにトークナイズ
- `(` `)` は独立した単語で、`[CLS]` / `[SEP]` は付けない
- 特殊トークンID: `[SEP_STRUCT]`=128001 `[SEP_TEXT]`=128002 `[P]`=128003 `[L]`=128007

### GPUで実行する（任意）
`--gpu` を付けると、CUDA（または DirectML）で推論します。このonnxruntimeビルドにGPUプロバイダが無い場合は、警告を出してCPUで動きます。
```bash
uv run --no-sync demo_inference_text.py --gpu
uv run --no-sync verify/bench.py --gpu
```
```python
from gliner_decide import Decider
from gliner_decide.runtime import gpu_providers

d = Decider("models/onnx", variant="fp32", providers=gpu_providers())
```

必要なものと注意点です。
- `onnxruntime` の代わりに `onnxruntime-gpu`（Windowsなら `onnxruntime-directml` も可）を入れます。両者は同じ `onnxruntime/` ディレクトリにファイルを書くため、併存させないでください。入れ替えたときは `uv pip install --force-reinstall --no-deps onnxruntime-gpu` で入れ直します
- `uv pip install onnxruntime-gpu` は一度入れても、`uv run` が `pyproject.toml` に合わせて `onnxruntime`（CPU版）を再インストールし、GPUが使えなくなります。`uv run --no-sync` を使ってください（`UV_NO_SYNC=1` でも同じです）
- 実際に使われたプロバイダは、デモの最終行と `verify/bench.py` の出力（`provider`）で確認できます
- 起動は遅くなります（セッション生成に約10秒、初回推論に約0.5秒）。常駐して何度も推論する用途で効果があります

# Verification
```bash
uv run verify/run_all.py                                 # 全ゲート一括
uv run --with tokenizers verify/run_all.py --full-fuzz   # Rust tokenizersとの全件突き合わせ
```

| ゲート | 内容 | 必要なもの |
|---|---|---|
| check_prompt | `input_ids` とmarker位置が gliner2 と6/6一致 | 実行時依存のみ |
| verify | ラベルと確率がPython実装と一致（argmax 13/13） | 実行時依存のみ |
| check_limits | 512トークン境界・400ラベル・空文字など9項目 | 実行時依存のみ |
| check_tokenizer_fuzz | Rust `tokenizers` との突合（既定は約5,200件で不一致0、`--full-fuzz` で312,706ケース） | tokenizers |

### 性能（Core i7-12800H）
3問・約100トークンのケースです。コールドスタートは3.3秒で、1スレッドでも581 msなので、コアを割けない環境でも使えます。

| threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| median | 581 ms | 336 ms | 199 ms | 160 ms |

GPU（`--gpu`、NVIDIA GeForce RTX 3050 Ti Laptop GPU、CUDA 13.4）で同じケースを測った結果です。ウォームアップ後は速いものの、起動は遅くなります。

| | median | p95 | セッション生成 | 初回推論 |
|---|---:|---:|---:|---:|
| CPU（4スレッド、同条件で再測定） | 245 ms | 289 ms | 3.4 s | 0.28 s |
| GPU（CUDA） | 24.5 ms | 32.8 ms | 10.3 s | 0.49 s |

| | パッケージ数 | サイズ |
|---|---:|---:|
| gliner2[local] | 33 | 661 MB |
| この実装 | 2 | 86 MB |

# Limitations
- 分類のみです。NER・関係抽出・構造化抽出はONNXに含まれません
- 512トークン上限です（超過分は末尾を打ち切り）
- 英語の運用テキストで学習されたモデルです
- 制約付きデコード（beam / exact）は未実装です。公式の `gliner2` では、問をまたぐ制約（`implies`、`excludes`、件数、順序など）を満たす最大効用の組み合わせを選べますが、この実装は問ごとの独立softmaxで1ラベルを選ぶだけです（公式の `independent` デコーダ相当）
- 複数ラベル選択の問（sigmoid）は未実装です。全問を排他的な単一選択として扱います
- `tokenizer.py` はこのチェックポイント専用で、他のtokenizer.jsonは例外で弾きます

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は2つ。exportグループは書き出しと検証のoracle用）
uv.lock                  # uvのロックファイル
download_model.py        # モデル取得（Releases / Hugging Face）、自前エクスポート
demo_inference_text.py   # 推論デモ
gliner_decide/           # 推論実装（軽い依存のみ。export/ をimportしない）
  runtime.py             #   ORTセッション、問ごとのsoftmax、classify_text互換API
  tokenizer.py           #   SentencePiece Unigramの純Python実装
  protocol.py            #   GLiNER2の分類レイアウト（プロンプト + marker位置）
export/
  export_onnx.py         # encoder + 分類MLPを1グラフに（gather内蔵）
verify/                  # 検証ゲート（run_all.pyで一括）
artifacts/               # golden.json（検証の正解データ）、report.md
models/                  # ONNX（重みはgitignore）、トークナイザ・設定ファイル
```

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
GLiNER2.5-Decide-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)）のライセンスに従います。
