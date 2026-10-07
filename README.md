# GLiNER2.5-Decide-ONNX-CPU

[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)（型付き判定の分類パス）を、
**`onnxruntime` + `numpy` だけで CPU 推論**する最小ランタイム。torch も transformers も tokenizers も要らない。

- 判定は公式の Python 実装と実質同一（**argmax 13/13、最大確率差 4.4e-07**）。
- **CPU では torch 版より速い**（3 問・約 100 トークンで 4 スレッド 199 ms、torch 版は 465 ms）。
- 実行時に入るのは 2 パッケージ・86 MB（`gliner2[local]` は 33 パッケージ・661 MB）。
- ONNX は [`onnx-community/GLiNER2.5-Decide-ONNX`](https://huggingface.co/onnx-community/GLiNER2.5-Decide-ONNX)
  の公開版を使う（重みはリポジトリに含めない。`export/prepare_model.py` が取得する）。ここで足しているのは、
  **公式の前処理（トークナイズ・プロンプト・marker 位置）の純 Python 移植**と、それを突き合わせる検証。

実測値と移植の根拠は **[artifacts/report.md](artifacts/report.md)**。

## クイックスタート（uv）

```powershell
uv sync                                   # onnxruntime + numpy だけを入れる
uv run python export\prepare_model.py     # 公開 ONNX を取得（HTTPS のみ、git も git-lfs も不要）
uv run python demo_inference_text.py
uv run python verify\run_all.py           # 検証ゲートを一括実行
```

取得は Hugging Face の resolve エンドポイントからの素の HTTPS（標準ライブラリだけ）で、
中断しても同じコマンドで途中から再開する。このモデルは**テキスト専用**なので画像のデモは無い。

## 使い方

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

**全問が 1 回の順伝播で決まる。** スキーマがプロンプトに展開され、ラベルごとに置かれた `[L]` マーカー位置の
logits がそのまま各ラベルのスコアになる。

## モデルの選び方

```powershell
uv run python export\prepare_model.py --list
uv run python export\prepare_model.py --variant q4f16                  # 小さいが CPU では遅い
uv run --group export python export\prepare_model.py --self-export     # checkpoint から自前で書き出す
```

| variant | サイズ | argmax | CPU median |
|---|---:|---:|---:|
| `fp32` | 1663 MB | **13/13** | **212 ms** |
| `q4f16` | 499 MB | 12/13 | 470 ms |

**CPU では fp32 が最速。** 4bit は WebGPU 向けの最適化で、CPU では逆量子化コストが勝つ。
`--self-export` は `fastino/GLiNER2.5-Decide` から自分で書き出す（torch 等が要るので `--group export`）。
出力は公開版と**バイト単位で同一**になることを確認済みなので、finetune 版にも同じ手順を適用できる。

## 構成

```
gliner_decide/        ランタイム（軽い依存のみ。export/ を import しない）
  runtime.py            ORT セッション、問ごとの softmax、classify_text 互換 API
  tokenizer.py          SentencePiece Unigram の純 Python 実装
  protocol.py           GLiNER2 の分類レイアウト（プロンプト + marker 位置）
export/
  prepare_model.py      入口: 公開 ONNX の取得 / 自前エクスポート
  export_onnx.py        encoder + 分類 MLP を 1 グラフに（gather 内蔵）
verify/
  run_all.py            全ゲート一括
artifacts/            golden.json（検証の正解データ）、report.md
models/               ONNX と checkpoint（gitignore。ショートカットではなく実体のファイルを置く）
pyproject.toml        実行時依存は 2 つ。`export` グループは書き出しと検証の oracle 用
```

## 入力レイアウト（移植の核心）

```
( [P] prompt ( [L] label_1 [L] label_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word … .
```

- `marker_positions` = 各 `[L]` トークンの位置を問をまたいでフラットに並べたもの
- prompt とラベルは原文のまま丸ごとトークナイズ、本文は小文字化して単語ごと
- `(` `)` は独立した単語。`[CLS]` / `[SEP]` は付けない
- 特殊トークン ID: `[SEP_STRUCT]`=128001 `[SEP_TEXT]`=128002 `[P]`=128003 `[L]`=128007

## 検証

```powershell
uv run python verify\run_all.py
uv run --with tokenizers python verify\run_all.py --full-fuzz     # Rust tokenizers との全件突き合わせ
```

| ゲート | 内容 | 必要なもの |
|---|---|---|
| `check_prompt` | `input_ids` と marker 位置が gliner2 と 6/6 一致 | 実行時依存のみ |
| `verify` | ラベルと確率が Python 実装と一致（argmax 13/13） | 実行時依存のみ |
| `check_limits` | 512 トークン境界・400 ラベル・空文字など 9 項目 | 実行時依存のみ |
| `check_tokenizer_fuzz` | Rust `tokenizers` との突合（既定 20,000＋基本ケース＝約 5,200 件で不一致 0、`--full-fuzz` で 312,706 ケース） | + tokenizers |

## 性能（Core i7-12800H, CPU のみ）

| threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| median | 581 ms | 336 ms | 199 ms | **160 ms** |

3 問・約 100 トークンのケース。コールドスタート 3.3 秒。**1 スレッドでも 581 ms** なのでコアを割けない環境でも使える。
参考までに torch 版は同条件で 465 ms。

| | パッケージ数 | サイズ |
|---|---:|---:|
| `gliner2[local]` | 33 | 661 MB |
| **この実装** | **2** | **86 MB** |

## 制約

- **分類のみ。** NER・関係抽出・構造化抽出は ONNX に含まれない
- 512 トークン上限（超過分は末尾を打ち切り）
- 英語の運用テキストで学習されたモデル
- 制約付きデコード（beam / exact）は未移植。問ごとの独立 softmax のみ
- `tokenizer.py` はこのチェックポイント専用。他の tokenizer.json は例外で弾く

## ライセンス

Apache License 2.0（[LICENSE](LICENSE)）。モデルの重みは、元のモデル
（[fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)）のライセンスに従う。
