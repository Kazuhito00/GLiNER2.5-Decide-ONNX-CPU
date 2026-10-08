# GLiNER2.5-Decide の ONNX 化で依存パッケージは減らせるか — 検証結果

計測: 2026-09-27 / Windows 11 / Python 3.12 (uv) / CPU 推論 4 スレッド

## 結論

**Yes。分類（decide）用途に限れば、実行時依存は `gliner2[local]` の 33 パッケージ
661MB から、`onnxruntime` + `numpy` の 2 パッケージ 86MB になる。**
精度は torch 版と実質同一（fp32 で argmax 13/13・最大確率差 4.4e-07）、
CPU レイテンシは約 2.2 倍速くなった。

`transformers` だけでなく **`tokenizers`（Rust 拡張）も落とした**。
SentencePiece Unigram トークナイザを純 Python に移植し、
Rust 実装と **312,706 ケースでトークン ID が完全一致**することを確認している。

代償は **機能が分類のみに縮むこと** の 1 点。
NER・関係抽出・構造化抽出は引き続き `gliner2` の Python 実装が必要になる。

補足:
- **`onnx` パッケージは実行時には不要**。必要なのは `onnxruntime` と `numpy` だけで、
  依頼された 3 つよりさらに 1 つ少ない。`onnx` は自前エクスポート時のみ使う
- **2 パッケージ / 86MB は `--no-deps` で入れた場合**。通常の `pip install` では
  推移的依存が付く。どちらでも全検証ケースが通る
- **q4f16（499MB）は CPU では fp32 より 2 倍以上遅い。** 容量削減はブラウザ／エッジ向け

## 1. 依存パッケージ

それぞれ独立した uv venv を作り、実サイズを実測。

| 環境 | 構成 | パッケージ数 | サイズ |
|---|---|---:|---:|
| `.venv-torch` | `gliner2[local]` | **33** | **661 MB** |
| `.venv-onnx` | `onnxruntime tokenizers numpy`（通常インストール） | 21 | 114 MB |
| `.venv-min` | 同上を `--no-deps` | 3 | 94 MB |
| `.venv-bare` | **`onnxruntime numpy` を `--no-deps`** | **2** | **86 MB** |

`.venv-torch` は `gliner2[local]` のみを入れた素の状態で計測した値。
（その後この環境にはエクスポート用の `onnx` / `onnxscript` を追加したため、
いま同じコマンドで数えると 38 パッケージ 765MB になる。比較には素の 33 / 661MB を使う。）

`.venv-bare` で不在を確認済み:

```
torch         ModuleNotFoundError: No module named 'torch'
tokenizers    ModuleNotFoundError: No module named 'tokenizers'
transformers  ModuleNotFoundError: No module named 'transformers'
onnx          ModuleNotFoundError: No module named 'onnx'
```

**torch 環境からのみ消えるもの**:
`torch` `transformers` `tokenizers` `peft` `accelerate` `safetensors` `sympy`
`networkx` `mpmath` `jinja2` `markupsafe` `regex` `pydantic` `pydantic-core`
`requests` `urllib3` `psutil` `huggingface-hub` `setuptools` ほか。

## 2. モデルファイル

| ファイル | サイズ |
|---|---:|
| `model.safetensors`（torch, fp32） | 1855.7 MB |
| ONNX fp32 (`model.onnx` + `_data`) | 1663.6 MB |
| ONNX q4f16 (`model_q4f16.onnx` + `_data`) | **499.2 MB** |
| `tokenizer.json` | 8.4 MB |

ONNX fp32 が safetensors より小さいのは、分類パスに不要な
span / relation / count ヘッドが落ちているため。

## 3. 精度（Python `gliner2` v2.0.0 との一致）

検証データは ONNX リポジトリ同梱の `conversion/reference.json`
（6 テキスト・13 問、gliner2 本体が出した確率つき）。
**`onnxruntime` と `numpy` しか入っていない環境で実行**。

| variant | argmax 一致 | 最大確率差 |
|---|---:|---:|
| fp32 | **13/13** | 4.40e-07 |
| q4f16 | 12/13 | 6.82e-02 |

q4f16 で外れた 1 問は「メッセージ中の要求は何件か」で、
`two` 0.4010 vs `three or more` 0.4022 という実質互角のケース。

さらに、**ローカルの safetensors から自前で計算した生 logits** とも照合した:

```
torch : [-0.128, -1.45656, -1.43879, 0.22624, -0.16246]
onnx  : [-0.128, -1.45656, -1.43879, 0.22624, -0.16246]
max|Δlogit| = 7.153e-07
```

### 境界・縮退ケース

`verify/check_limits.py` で全 9 項目 PASS。
512 トークン超の打ち切り、512 ちょうど、スキーマだけで 512 超（400 ラベル → `ValueError`）、
空文字、ラベル 1 個、日本語。

## 4. トークナイザの純 Python 移植

`tokenizers` を落とすため、`gliner_decide/tokenizer.py` に
SentencePiece Unigram を移植した。実装したのはこのチェックポイントが
実際に宣言しているパイプラインのみで、それ以外（BPE / WordPiece /
precompiled charsmap / byte_fallback）はロード時に明示的に拒否する。

```
added tokens → Replace(2連続以上の空白・CR・LF・TAB → " ") → NFC → Strip(right)
             → Metaspace(U+2581, prepend_scheme=always, split=true)
             → Unigram Viterbi (unk_id=3, byte_fallback=false, fuse_unk=true)
```

### 一致検証

`verify/check_tokenizer_fuzz.py` で Rust 実装と突き合わせ:

```
cases: 312706   mismatches: 0
```

コーパスには、`DecideEncoder` が実際に投げる形（単語・`(`・`)`・複数語ラベル・
`[DESCRIPTION]` を含む prompt・15 個の特殊トークン）に加え、
CJK・絵文字（ZWJ 連結・国旗）・アクセント付きラテン・URL/メール・
結合文字・合字・丸数字、および Unicode 空白 14 種を含めた。

### 移植で踏んだ 2 つの落とし穴

いずれもリファレンス 6 ケースでは表面化せず、ファズで初めて出た。

1. **`fuse_unk` は JSON に出てこないが true。**
   Rust の `Unigram::from` がハードコードしており、連続する未知ピースは
   1 個の `[UNK]` に融合される。素直に実装すると文字数ぶん `[UNK]` が出る

2. **added token の処理は 2 パス。**
   `normalized: false` のトークン（`[P]` `[L]` など 14 個）は**生テキスト**から
   先に切り出され、残った各片が個別に正規化される。
   `normalized: true` のトークン（`[UNK]`）は**正規化後**に初めて切り出される。
   `Strip(right)` の適用範囲が変わるため、`[UNK]` を含むテキストで
   `▁` が 1 個ずれる（312,706 ケース中 1012 件がこれ）

3. 加えて **Python の `\s` / `str.rstrip()` は `U+001C`〜`U+001F` を空白扱いする**が、
   Rust は Unicode `White_Space` 準拠でこれらを含まない。
   `WHITE_SPACE` を明示列挙して回避した

### 速度への影響

| 処理 | 純 Python | Rust |
|---|---:|---:|
| 語彙ロード | 186.1 ms | 161.8 ms |
| 参照セット全 270 語（キャッシュなし） | 2.04 ms | 1.52 ms |
| 長い説明文 1 本 | 0.173 ms | 0.039 ms |

`DecideEncoder` の LRU キャッシュが温まった状態の `encode()` 全体は **0.019 ms**。
ONNX の前向き計算が 200ms 超なので、**トークナイズは全体の 0.01% 未満**。
実質的な速度低下はない。

## 5. レイテンシ（CPU, 4 スレッド, 40 回, OS キャッシュ温）

3 問・約 100 トークンのケース。

| バックエンド | コールドスタート合計 | median | p95 |
|---|---:|---:|---:|
| **ONNX fp32** | **3.32 s** | **212.4 ms** | **242.5 ms** |
| ONNX q4f16 | 3.58 s | 469.8 ms | 488.3 ms |
| gliner2 + torch | 9.09 s | 464.7 ms | 507.2 ms |

内訳（import / モデルロード / 初回推論）:
- ONNX fp32: 0.51 s / 2.57 s / 0.24 s
- gliner2 + torch: 3.65 s / 4.96 s / 0.47 s

計測機: **Intel Core i7-12800H**（14 コア / 20 スレッド）、RAM 32 GB。GPU は不使用。

### スレッド数の効き（ONNX fp32, 25 回）

| threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| median ms | 580.8 | 336.2 | **198.6** | **160.3** |
| p95 ms | 604.0 | 386.5 | 232.9 | 195.5 |

4 スレッドまでほぼ線形に効き、8 スレッドでもまだ改善する。
**1 スレッドでも 581 ms** なので、コアを割けない環境でも実用範囲に収まる。

**ONNX fp32 は torch より定常時で約 2.2 倍、コールドスタートで約 2.7 倍速い。**
なお OS のファイルキャッシュが冷えた初回は torch 側が 21.9 秒かかった
（safetensors 1.9GB の読み込み）。上表は両者ともキャッシュ温の条件。

**q4f16 は CPU では fp32 の 2 倍以上遅い。** 4bit の `MatMulNBits` は WebGPU 向けの
最適化で、CPU では逆量子化コストが勝つ。

## 6. 自前エクスポートの再現性

`export/export_onnx.py` で、ローカルの `models/torch` から自分で書き出した。

- torch 2.14.0+cpu / `dynamo=True` / opset 17 / 所要 **95.7 秒**
- 出力サイズ **1744089088 バイト — 配布版とバイト単位で同一**
- 生 logits: torch との差 **7.153e-07**（配布版と同値）
- gliner2 との一致: **argmax 13/13, 最大確率差 4.40e-07**（配布版と同値）

最大リスクと見ていた DeBERTa-v3 の disentangled attention は問題にならなかった。
`gliner2` 自身が `attn_implementation='sdpa'` を拒否して `eager` にフォールバックする。

→ **配布版に依存せず再現できる。** 将来 finetune した checkpoint にも同じ手順を適用できる。

エクスポート時のみ必要な追加依存: `torch` `gliner2[local]` `onnx` `onnxscript`。
Windows の cp932 コンソールでは `PYTHONUTF8=1` が必要（gliner2 が絵文字を print するため）。

## 7. 実装

`gliner_decide/` — 実行時依存は `onnxruntime` / `numpy` のみ。

| ファイル | 役割 |
|---|---|
| `tokenizer.py` | SentencePiece Unigram の純 Python 実装（正規化・Metaspace・Viterbi・added token 2 パス） |
| `protocol.py` | `gliner2.processor` の分類レイアウトを移植（word 分割・末尾ピリオド付与・`[P]`/`[L]` 配置・marker 位置算出・512 打ち切り） |
| `runtime.py` | ORT セッション、問ごとの softmax、`classify_text` 互換 API |
| `demo_inference_text.py` | CLI デモ |

```python
from gliner_decide import Decider
d = Decider("models/onnx", variant="fp32")
d.classify_text("I was charged twice and support never replied.",
                {"intent": ["refund_request", "order_status", "other"]})
# -> {"intent": "refund_request"}
```

### 入力レイアウト（移植の核心）

```
( [P] prompt ( [L] label_1 [L] label_2 … ) ) [SEP_STRUCT] ( [P] … ) [SEP_TEXT] word word … .
```

- `marker_positions` = 各 `[L]` トークンの位置を、問をまたいでフラットに並べたもの
- prompt とラベルは原文のまま丸ごとトークナイズ。本文は小文字化し単語ごとにトークナイズ
- `(` `)` は独立した単語として扱う。`[CLS]` / `[SEP]` は付けない
- ラベル説明は prompt 末尾に ` [DESCRIPTION] label: desc` として連結
- 特殊トークン ID: `[SEP_STRUCT]`=128001 `[SEP_TEXT]`=128002 `[P]`=128003 `[L]`=128007

## 8. 制約

- **分類のみ。** NER・関係抽出・構造化抽出は ONNX に含まれない
- **512 トークン上限。** 超過分は末尾を打ち切り
- 英語の運用テキストで学習されたモデル。日本語は学習範囲外（変換の問題ではない）
- 制約付きデコード（beam / exact）は未実装。問ごとの独立 softmax のみ（公式の `independent` デコーダ相当）
- 複数ラベル選択の問（sigmoid）は未実装。全問を排他的な単一選択として扱う
- バッチ推論は未実装（グラフは `[batch, …]` 対応済みなので拡張は容易）
- `tokenizer.py` はこのチェックポイントのパイプライン専用。
  他の tokenizer.json を渡すとロード時に `ValueError` で弾く（黙って誤動作しない）

## 9. 再現手順

```powershell
uv venv .venv-bare --python 3.12
uv pip install --python .venv-bare/Scripts/python.exe --no-deps onnxruntime numpy

.venv-bare\Scripts\python.exe verify\check_prompt.py            # 6/6 バイト一致
.venv-bare\Scripts\python.exe verify\verify.py --variant fp32   # 13/13, 4.4e-07
.venv-bare\Scripts\python.exe verify\check_limits.py            # 境界 9 項目
.venv-bare\Scripts\python.exe verify\bench.py --variant fp32
.venv-bare\Scripts\python.exe -c "import tokenizers"            # ModuleNotFoundError になること

# トークナイザの一致検証だけは tokenizers が要る（比較対象として）
.venv-onnx\Scripts\python.exe verify\check_tokenizer_fuzz.py 250000  # mismatches: 0
```
