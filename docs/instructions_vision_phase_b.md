# 指示書: 視覚モジュール統合 — Phase B実行

**日付:** 2026-03-25
**対象:** somabeat（integrated-system）
**親文書:** docs/vision_integration_taskflow.md
**前提:** Phase A完了（ゲートPASS）。CKA=0.28, D=384, MLP 2+層

---

## 設計方針

### タスクフローからの変更点

| 項目 | タスクフロー原案 | 本指示書 |
|------|----------------|---------|
| 共通空間次元D | 256 | **384**（Phase A PCA結果に基づく） |
| projection設計 | 両側学習 | **両側学習 + identity初期化 + α探索** |
| 学習データ | VideoGameBunny 185K枚（ゲーム特化） | **多ドメイン混合 200K枚** |
| 構造保存正則化α | 0.5固定 | **α探索（1.0, 0.5, 0.1）** |

### 設計判断の根拠

**両側学習を採用する理由:**
- 場の信号に優劣を持ち込まない。視覚信号をテキスト空間の二級市民にしない
- 設計原則2（無指向性の伝播）の精神との整合。場ではテキストと視覚が対等に共存すべき
- CKA=0.28（構造対応が弱い）。片側だけで差を埋めるのはMLPにとって重い。両側から歩み寄る方が到達精度が高い

**identity初期化:**
- projection_textの最終層をidentity matrix + ゼロバイアスで初期化
- 学習開始時点ではテキスト空間からの乖離がゼロ。構成1と同じ出発点
- 学習が進むにつれて対照学習の圧力でわずかに変形するが、構造保存正則化がブレーキをかける
- テキスト空間からの乖離量をαで連続的に制御できる

**多ドメインデータ:**
- ゲーム画面だけでなくPC画面、実写、Web画像を処理する汎用用途
- ドメイン多様性がprojection headの過適合を防ぎ、汎化性能を担保する

---

## B-1. データ準備

### B-1a. データセットのダウンロード

3つのソースから混合データを構築する。

| ソース | 枚数 | ドメイン | 取得方法 |
|--------|------|---------|---------|
| VideoGameBunny | 50,000 | ゲーム画面 | HuggingFace: asgaardlab/VideoGameBunny-Dataset からサブサンプリング |
| COCO Captions | 50,000 | 自然画像・実写 | HuggingFace: HuggingFaceM4/COCO からサブサンプリング |
| CC3M | 100,000 | Web画像全般 | HuggingFace: conceptual_captions からサブサンプリング |
| **合計** | **200,000** | **多ドメイン混合** | |

サブサンプリングは`seed=42`で固定。再現性を確保。

### B-1b. キャプション品質のサンプリング確認

各ソースから50件ずつ（計150件）を目視でキャプション品質を確認。

確認観点:
- 画像の内容を適切に記述しているか
- 極端に短い or 無意味なキャプションがないか
- 言語（英語統一か、混在していないか）

CC3Mは自動生成キャプションのため品質にばらつきがある。明らかに低品質な件（「a photo of a photo」等）はフィルタリングする。フィルタ後に100K件を下回る場合はサンプリング数を増やして補充。

### B-1c. データ分割

| 分割 | 枚数 | 用途 |
|------|------|------|
| train | 180,000 | projection head学習 |
| val | 10,000 | early stopping |
| test | 10,000 | 最終評価 |

testには各ソースから均等に含める（VideoGameBunny 2,500 + COCO 2,500 + CC3M 5,000）。ドメインごとの汎化性能を評価するため。

---

## B-2. 大規模embedding事前計算

### B-2a. テキストembedding

全キャプションをe5-smallでエンコード。

```python
# プレフィックス処理はFieldEncoderの実装に合わせる（A-0aで確認済み: "passage: "）
text_embeddings = e5.encode(["passage: " + cap for cap in captions])
np.save("text_cache.npy", text_embeddings)  # (200000, 384)
```

メモリ見積もり: 200,000 × 384 × 4bytes ≈ 290MB
推定時間: ~1分（CPU、Phase Aのスケーリングから）

### B-2b. 画像embedding

全画像をSigLIP ViT-Bでエンコード。

```python
# GPU使用。Ollamaを一時停止してVRAMを確保（A-0cの方針）
img_embeddings = encode_images_batch(images, siglip, processor, batch_size=64)
np.save("img_cache.npy", img_embeddings)  # (200000, 768)
```

メモリ見積もり: 200,000 × 768 × 4bytes ≈ 580MB
推定時間: ~73分（GPU、Phase Aの22ms/枚からスケーリング）

HuggingFaceからのストリーミングダウンロードとembedding計算をパイプライン化し、画像を手元に全件保持しなくてよい設計にする。ストレージの節約。

### B-2c. キャッシュの検証と永続化

- テキスト/画像の各空間内で類似サンプル同士のコサイン類似度を抜き取り確認
- .npy形式で永続化。再実験時に再計算を避ける
- メタデータ（ソース、キャプション、分割ラベル）をJSONで保存

---

## B-3. Projection Headの設計

### B-3a. アーキテクチャ

```python
# projection_text: 384 → 512 → 384
projection_text = nn.Sequential(
    nn.Linear(384, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Linear(512, 384),
)

# projection_img: 768 → 512 → 384
projection_img = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Linear(512, 384),
)
```

### B-3b. identity初期化（projection_textのみ）

```python
# projection_textの最終層をidentity初期化
# 学習開始時点でprojection_textは恒等写像に近い
with torch.no_grad():
    # 最終Linear層（512→384）をidentityに近づける
    # 384×512の行列の左上384×384ブロックをidentityに
    projection_text[-1].weight.zero_()
    projection_text[-1].weight[:384, :384].copy_(torch.eye(384))
    projection_text[-1].bias.zero_()
```

**注意:** 中間層（384→512）は通常のXavier初期化。最終層のみidentity初期化。
中間層を通るので完全な恒等写像にはならないが、学習初期のテキスト空間からの乖離を最小化する。

projection_imgは通常のXavier初期化（identity初期化の意味がない。入力768次元と出力384次元が異なるため）。

### B-3c. パラメータ数

| コンポーネント | パラメータ数 |
|-------------|------------|
| projection_text | 384×512 + 512 + 512×384 + 384 ≈ 394K |
| projection_img | 768×512 + 512 + 512×384 + 384 ≈ 590K |
| 温度パラメータτ | 1 |
| **合計** | **≈ 984K** |

CPUで十分学習可能。

---

## B-4. 学習

### B-4a. 損失関数

```python
def compute_loss(text_emb, img_emb, alpha, tau):
    # 対照学習損失（InfoNCE）
    text_proj = projection_text(text_emb)  # (N, 384)
    img_proj = projection_img(img_emb)     # (N, 384)

    # コサイン類似度行列
    text_proj_norm = F.normalize(text_proj, dim=-1)
    img_proj_norm = F.normalize(img_proj, dim=-1)
    logits = (text_proj_norm @ img_proj_norm.T) / tau

    # InfoNCE（対角要素が正例）
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss_contrastive = (loss_i2t + loss_t2i) / 2

    # 構造保存正則化
    # テキスト間のコサイン類似度行列を、e5元空間と共通空間で比較
    with torch.no_grad():
        original_sim = F.normalize(text_emb, dim=-1) @ F.normalize(text_emb, dim=-1).T
    projected_sim = text_proj_norm @ text_proj_norm.T

    # Pearson相関を最大化（= 1 - 相関を最小化）
    loss_structure = 1.0 - pearson_corrcoef(
        original_sim.flatten(), projected_sim.flatten()
    )

    # 統合損失
    loss_total = loss_contrastive + alpha * loss_structure
    return loss_total, loss_contrastive, loss_structure
```

### B-4b. α探索戦略

3つのα値で個別に学習を実行する。

| α | 意味 | 期待される挙動 |
|---|------|-------------|
| 1.0 | 構造保存最優先 | テキスト検索ほぼ劣化なし。cross-modal精度は控えめ |
| 0.5 | バランス | タスクフロー初期値。中間 |
| 0.1 | 対照学習優位 | cross-modal精度最大化。テキスト検索やや劣化リスク |

各αで独立に学習→評価し、両方の合格基準を満たすαの範囲を特定する。

**合格基準:**
- Cross-modal Recall@10 ≥ 0.5（img→text, text→img の両方）
- テキスト-テキスト Spearman順位相関 ≥ 0.85

### B-4c. 共通ハイパーパラメータ（3つのα実験で共通）

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| バッチサイズ | 256 | キャッシュベースなのでメモリ制約は緩い |
| 学習率 | 1e-3 | AdamW |
| エポック数 | 最大50 | early stopping: val lossが5エポック改善なしで停止 |
| 温度τ初期値 | 0.07 | 学習可能パラメータ |
| デバイス | CPU or GPU | embeddingキャッシュベースなのでCPUでも高速 |

### B-4d. 学習の実行

各α値について:

1. projection_text, projection_imgを初期化（B-3b: identity初期化 + Xavier初期化）
2. 学習ループ実行
3. val_recall@10のbestでチェックポイント保存
4. 学習カーブを保存（loss_contrastive, loss_structure, val_recall@10の推移）

**成果物:**
```
data/vision_phase_b/
├── alpha_1.0/
│   ├── projection_text.pt
│   ├── projection_img.pt
│   ├── training_curves.json
│   └── checkpoint_best.pt
├── alpha_0.5/
│   └── ...
└── alpha_0.1/
    └── ...
```

### B-4e. 学習が不安定な場合の対応策

| 症状 | 対応 |
|------|------|
| loss_contrastiveが下がらない | バッチサイズ増加（512, 1024）。負例が増えて対照学習が効きやすくなる |
| loss_structureが増大 | α増加。または学習率をprojection_textのみ小さくする |
| val_recall@10が振動 | 学習率スケジューラ追加（cosine annealing） |
| α=0.1でテキスト検索が大幅劣化 | α=0.1は不採用。α=0.3を追加試行 |

---

## B-5. 品質検証

### B-5a. Cross-modal retrieval精度（testセット）

各α値について:

| 指標 | 計測 |
|------|------|
| img→text Recall@1, @5, @10 | 画像embeddingから最近傍テキストを検索 |
| text→img Recall@1, @5, @10 | テキストembeddingから最近傍画像を検索 |

Phase Aの線形ベースライン（img→text R@10=0.404）との比較。MLPによる改善幅を確認。

### B-5b. テキスト-テキスト検索品質の検証（最重要）

各α値について:

1. testセットのテキスト間コサイン類似度の順位相関を計測
2. e5元空間での順序 vs 共通空間projection後の順序
3. **Spearman順位相関 ≥ 0.85が合格基準**

追加検証:
- 既存のSleepyJean RAGクエリを数十件用意し、e5元空間と共通空間で検索結果の上位5件を比較
- llamarcute-liveのsenseクエリ（「最近の自分の調子」等）で同様の比較

### B-5c. ドメインごとの汎化検証

testセットをソース別に分割し、各ドメインでのRecall@10を計測。

| ドメイン | testサイズ | 期待 |
|---------|----------|------|
| VideoGameBunny | 2,500 | trainに同ドメインあり→高精度 |
| COCO | 2,500 | trainに同ドメインあり→高精度 |
| CC3M | 5,000 | Web画像全般→汎化性能の指標 |

特定ドメインのRecall@10が他と大きく乖離していないかを確認。乖離が大きければ、そのドメインのtrain枚数を増やして再学習。

### B-5d. 場のシミュレーション検証

最良α値のprojection headを使い、場のsenseをシミュレーション:

1. テキスト信号と視覚信号をChromaDBに投入（テスト用コレクション）
2. テキストクエリで視覚信号がヒットするか
3. 視覚信号とテキスト信号が混在するFieldReadingが妥当か
4. 視覚信号同士のコサイン距離が視覚的類似度を反映しているか（視床フィルタの前提確認）

---

## B-6. α選定と最終判定

### α選定

3つのα値の結果を比較し、最良αを選定する。

```
α=1.0: Recall@10=?  Spearman=?
α=0.5: Recall@10=?  Spearman=?
α=0.1: Recall@10=?  Spearman=?
```

**選定基準:**
1. Spearman ≥ 0.85 を満たすα値のみ候補
2. 候補の中でRecall@10が最も高いαを選定
3. 全αでSpearman < 0.85 → identity初期化の重みを強化（最終層だけでなく中間層も拘束）して再学習
4. 全αでRecall@10 < 0.5 → SigLIPモデルの変更（ViT-L）を検討

### Phase B合格条件

- [ ] 最良α値でcross-modal Recall@10 ≥ 0.5（img→text, text→img 両方）
- [ ] 最良α値でテキスト-テキスト Spearman ≥ 0.85
- [ ] 3ドメインでのRecall@10低下がtrain全体の20%以内
- [ ] 場のシミュレーションで視覚信号がsenseで正しくヒット

---

## Phase Cへの引き継ぎ情報

PASSの場合:

| 項目 | 値 |
|------|-----|
| 最良α値 | B-6で決定 |
| projection_text.pt | 学習済み重み |
| projection_img.pt | 学習済み重み |
| 共通空間次元D | 384 |
| テストパスライン | 292件（189 + 103）。cuteness修正等で増えている場合は最新値 |
| 既存信号の移行 | 721件のテキスト信号をre-encode（projection_textを通す）。推定1秒未満 |

---

## 成果物一覧

| ステップ | 成果物 |
|---------|--------|
| B-1 | 200K件のキャプション + メタデータ |
| B-2 | text_cache.npy (200K×384), img_cache.npy (200K×768) |
| B-4 | α=1.0/0.5/0.1 の各projection head + 学習カーブ |
| B-5 | 各αのRecall, Spearman, ドメイン別精度レポート |
| B-6 | 最良α選定結果、Phase C引き継ぎ情報 |

---

## 推定工数

| ステップ | 時間 | ボトルネック |
|---------|------|-----------|
| B-1 データDL+確認 | 1-2時間 | ダウンロード速度 |
| B-2 embedding計算 | ~75分 | SigLIP画像エンコード（GPU） |
| B-3 実装 | 1時間 | コーディング |
| B-4 学習（α×3） | 各10-20分 × 3 = 30-60分 | キャッシュベースなので高速 |
| B-5 検証 | 1-2時間 | シミュレーション含む |
| B-6 判定 | 30分 | 結果の比較と判断 |
| **合計** | **5-8時間** | 大部分はDLと計算の待ち |
