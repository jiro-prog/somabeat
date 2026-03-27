# 指示書: 視覚モジュール統合 — Phase A実行

**日付:** 2026-03-25
**対象:** somabeat（integrated-system）
**親文書:** docs/vision_integration_taskflow.md
**スコープ:** 事前確認（A-0）+ Phase A（ベースライン計測と予備実験）

---

## 背景

vision_integration_taskflow.md に基づき、FieldEncoderのマルチモーダル拡張を開始する。本指示書は、タスクフロー作成後のシステム変更（SleepyJeanリファクタリング、fine-tuning停止、モデル変更等）との差分を確認した上でPhase Aを実行する手順。

Phase B以降はPhase Aのゲート判定結果に基づいて進行する。

---

## A-0. 事前確認（Phase A開始前に必須）

### A-0a. FieldEncoderの現在の状態を確認

以下を調査し、結果を報告すること。

1. FieldEncoder の具象クラスがどのファイルに実装されているか
2. 使用しているモデルが `intfloat/multilingual-e5-small`（384次元）であることを確認
3. `encode()` メソッドの入出力（入力: テキスト、出力: 384次元ベクトル）を確認
4. `query:` / `passage:` プレフィックスの処理がどこで行われているか（FieldEncoder内部 or ChromaDBバックエンド内部）
5. FieldEncoder Protocol の定義場所（`shared_state/interface.py` 想定）

**確認理由:** タスクフローは「既存のe5-small（384次元）」を前提としている。SleepyJeanリファクタリングやfine-tuning事故の過程でFieldEncoderに変更が入っていないことを確認する。

### A-0b. ChromaDBの現在の状態を確認

1. 場のChromaDBコレクションの次元数（384想定）
2. 現在の信号数（`snapshot()`で取得可能）
3. distance_fn（cosine想定）

**確認理由:** Phase Cでコレクション再構築が必要になる。現在の規模を把握しておく。

### A-0c. VRAMの余裕を確認

1. 現在Ollamaが使用しているVRAM量（`nvidia-smi`で確認）
2. Qwen3-8Bのロード状態でのVRAM残量
3. SigLIP ViT-B（86Mパラメータ）を追加ロードした場合の推定VRAM

**確認理由:** SigLIPのembedding事前計算（A-2b）でGPUを使う場合、Qwen3-8Bと共存できるかを確認。CPUでも実行可能だが時間がかかる。共存不可の場合はOllamaを一時停止してSigLIPのembedding計算を先に済ませる手順にする。

### A-0d. ストレージの余裕を確認

1. GameplayCaptions（6,020枚）のダウンロードサイズ推定
2. embedding キャッシュ（6020×384×4≈9MB + 6020×768×4≈18MB）のストレージ
3. Phase B用のVideoGameBunny-Dataset（185,259枚）の推定サイズ（Phase Aでは不要だが、Phase B移行時のストレージ計画のため）

### A-0e. 既存テスト数の確認

現在のテスト数を確認すること。タスクフロー作成時は160件だったが、現在はintegrated-system側179件 + SleepyJean側132件 = 311件に増加している。Phase CのテストパスラインをこのI数値に更新する必要がある。

---

## A-1. 環境構築

vision_integration_taskflow.md セクション3.2 の A-1 をそのまま実行する。

### A-1a. 依存パッケージのインストール

```bash
pip install transformers sentence-transformers torch datasets
```

既にインストール済みのパッケージはスキップ。chromadbは既に使用中のため不要のはず。

### A-1b. GameplayCaptionsのダウンロード

```python
from datasets import load_dataset
ds = load_dataset("asgaardlab/GameplayCaptions")
```

6,020枚。Phase Aの予備実験用。

### A-1c. モデルのダウンロード

```python
# e5-small（既にシステムで使用中の場合はスキップ）
from sentence_transformers import SentenceTransformer
e5 = SentenceTransformer("intfloat/multilingual-e5-small")

# SigLIP ViT-B
from transformers import AutoModel, AutoProcessor
siglip = AutoModel.from_pretrained("google/siglip2-base-patch16-256")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
```

SigLIPは画像エンコーダ部分のみ使用。テキストエンコーダは不要（e5が担う）。

---

## A-2. embedding事前計算

vision_integration_taskflow.md セクション3.2 の A-2 をそのまま実行する。

### A-2a. テキストembedding

GameplayCaptionsの全キャプションをe5でエンコード。

```python
# プレフィックス: "passage: {caption}"
text_embeddings = e5.encode(["passage: " + cap for cap in captions])
np.save("text_embeddings.npy", text_embeddings)  # 6020×384
```

**注意:** e5のプレフィックス処理（query: / passage:）がFieldEncoder内部に隠蔽されている場合、ここでは隠蔽されていない素のe5モデルを直接使う。A-0aで確認したプレフィックスの扱いに合わせること。

### A-2b. 画像embedding

GameplayCaptionsの全画像をSigLIP ViTでエンコード。

A-0cの結果に基づき:
- VRAM余裕あり → GPU使用
- VRAM余裕なし → Ollama一時停止してGPU使用、またはCPUで実行

```python
# SigLIPの画像前処理パイプラインを使用
img_embeddings = []
for img in images:
    inputs = processor(images=img, return_tensors="pt")
    outputs = siglip.get_image_features(**inputs)
    img_embeddings.append(outputs.detach().numpy())
np.save("img_embeddings.npy", np.vstack(img_embeddings))  # 6020×768
```

### A-2c. キャッシュの検証

- テキスト: e5空間内でテキスト同士のコサイン類似度が妥当か抜き取り確認（類似キャプション同士が高類似度）
- 画像: SigLIP空間内で視覚的に類似した画像が近いか抜き取り確認

---

## A-3. 空間間の構造分析

vision_integration_taskflow.md セクション3.2 の A-3 をそのまま実行する。

### A-3a. 線形CKA

e5空間（384次元）とSigLIP空間（768次元）のペア間の線形的な構造対応度を計算。

CKAが高い → 線形projectionで十分な可能性
CKAが低い → MLPが必要

### A-3b. Procrustes分析

直交変換で最もアラインした場合の残差を計算。残差が小さいほど射影が容易。

### A-3c. 次元ごとの分散分析

両空間のPCAを実行。有効次元数を比較。

**この結果が共通空間の次元D（初期値256）の判断材料になる。**

- SigLIPの有効次元数が256未満 → D=256で十分
- e5の有効次元数が256を大きく超える → D=384（e5と同次元）を検討。テキスト側の情報圧縮を避ける
- 両空間とも有効次元数が低い → D=128も選択肢

---

## A-4. 線形射影ベースライン

vision_integration_taskflow.md セクション3.2 の A-4 をそのまま実行する。

### A-4a. データ分割

train: 5,000枚 / val: 520枚 / test: 500枚

### A-4b. 線形射影の学習

img_embeddings × W = projected_img（768→384）
損失: cosine embedding loss
パラメータ数: 768×384 = 294,912

### A-4c. ベースライン精度

Cross-modal retrieval: Recall@1, @5, @10（画像→テキスト、テキスト→画像）

---

## A-5. ゲート判定

### 判定基準（タスクフローから変更なし）

**PASS条件:**
- 線形ベースラインのRecall@10 ≥ 0.3
- CKAとProcrustes残差からMLP層数が決定できている
- 共通空間の次元Dがデータに基づいて決定されている

**FAIL時:**
- SigLIPモデルの変更（ViT-L等）を検討
- または方式自体の再検討（共通空間ではなく、テキスト/画像を別コレクションに保持する案等）

### Phase Bへの引き継ぎ情報

PASSの場合、以下をPhase Bに引き継ぐ:
- 共通空間の次元D
- MLP projection headの層数
- CKA/Procrustes/PCAの数値レポート
- 線形ベースラインのRecall@1, @5, @10

---

## タスクフローとの差分まとめ

| 項目 | タスクフロー記載 | 現在の状態 | 対応 |
|------|----------------|-----------|------|
| FieldEncoder | e5-small 384次元 | A-0aで確認 | 変更なければそのまま |
| ChromaDB次元数 | 384想定 | A-0bで確認 | Phase Cで再構築 |
| 既存テスト数 | 160件 | 311件（179+132） | Phase Cの合格条件を更新 |
| VRAM | 記載なし | Qwen3-8Bが常駐 | A-0cで共存確認 |
| SleepyJeanのモデル | fine-tuned 4B | qwen3:8b（fine-tuning停止） | 影響なし（FieldEncoderは独立） |
| 場の信号数 | 記載なし | ~672件 | Phase Cの移行コスト把握 |

---

## 成果物

| ステップ | 成果物 |
|---------|--------|
| A-0 | システム状態確認レポート |
| A-2 | text_embeddings.npy, img_embeddings.npy |
| A-3 | CKA, Procrustes残差, PCA結果レポート |
| A-4 | 線形射影ベースライン精度レポート |
| A-5 | ゲート判定結果（PASS/FAIL + Phase B引き継ぎ情報） |
