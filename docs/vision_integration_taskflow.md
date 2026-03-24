# 視覚モジュール統合 — 共通embedding空間構築タスクフロー

- **親文書**: `bio_ai_architecture.md` セクション6（感覚系）, `shared_field_design.md` セクション5（共通エンコーダ）
- **作成日**: 2026-03-23
- **ステータス**: 設計草案
- **スコープ**: FieldEncoderのマルチモーダル拡張（レベル3：新規共通空間の構築）

---

## 1. 目的と方針

### 1.1 やること

既存のテキスト専用FieldEncoder（multilingual-e5-small, 384次元）を、画像とテキストの両方を扱えるマルチモーダルFieldEncoderに拡張する。テキスト用・画像用それぞれのエンコーダを保持したまま、対照学習によって新規の共通embedding空間（D次元）を構築する。

### 1.2 アーキテクチャ

```
テキスト → e5-small(frozen) → 384次元 → projection_text(MLP) ─┐
                                                                 ├→ 共通D次元空間
画像    → SigLIP ViT-B(frozen) → 768次元 → projection_img(MLP) ─┘
```

両エンコーダはfrozenとし、projection head（MLP 2〜3層）のみを学習する。共通空間の次元Dは256を基本とする（Phase Aでの予備実験結果により調整）。

### 1.3 設計原則との整合

| 設計原則 | 整合性 |
|---------|--------|
| 認知機能の三系統分離 | 視覚モジュールは三系統の外側の「感覚系」として位置づけ。既存系統に影響しない |
| 共有状態を介した間接協調 | 視覚モジュールはemitのみ行い、他系統を直接呼ばない。無指向性の原則を維持 |
| 自己同一性 | 視覚入力が増えても外界への人格は変わらない |
| 睡眠は全身状態 | 睡眠中は視覚モジュールもemitを停止する |

### 1.4 使用リソース

- **データセット**: VideoGameBunny-Dataset（185,259枚、413タイトル）+ GameplayCaptions（6,020枚、予備実験用）
- **エンコーダ**: intfloat/multilingual-e5-small（384次元）, google/siglip2-base-patch16-256（768次元, ViT-B 86Mパラメータ）
- **学習**: projection headのみ。事前計算したembeddingキャッシュを使用し、CPUでも実行可能

---

## 2. フェーズ概要

| フェーズ | 目的 | 期間目安 | 成果物 |
|---------|------|---------|--------|
| Phase A | ベースライン計測と予備実験 | 1〜2日 | 空間間の対応関係の定量評価、線形射影のベースライン精度 |
| Phase B | 共通空間の学習と品質検証 | 2〜3日 | 学習済みprojection head、cross-modal / text-text 検索精度レポート |
| Phase C | FieldEncoder差し替えとシステム統合 | 1日 | 新FieldEncoder実装、ChromaDBコレクション再構築、テスト通過 |

---

## 3. Phase A — ベースライン計測と予備実験

### 3.1 目的

2つのembedding空間（e5, SigLIP ViT）の間にどの程度の構造的対応があるかを定量的に把握する。共通空間構築の難易度を見極め、Phase Bの学習設計に必要な情報を集める。

### 3.2 タスクフロー

```
A-1. 環境構築
  │
  ├─ A-1a. 依存パッケージのインストール
  │         - transformers, sentence-transformers, torch, chromadb
  │         - datasets（HuggingFace）
  │
  ├─ A-1b. GameplayCaptionsのダウンロード（6,020枚、予備実験用）
  │         - HuggingFace: asgaardlab/GameplayCaptions
  │
  └─ A-1c. モデルのダウンロード
            - intfloat/multilingual-e5-small
            - google/siglip2-base-patch16-256（ViT-Bの画像エンコーダのみ使用）
  │
  ▼
A-2. embedding事前計算
  │
  ├─ A-2a. GameplayCaptionsの全キャプションをe5でエンコード → text_embeddings.npy（6020×384）
  │         - プレフィックス: "passage: {caption}"
  │
  ├─ A-2b. GameplayCaptionsの全画像をSigLIP ViTでエンコード → img_embeddings.npy（6020×768）
  │         - SigLIPの画像前処理パイプラインを使用
  │
  └─ A-2c. キャッシュの検証
            - テキスト: e5空間内でテキスト同士のコサイン類似度が妥当か抜き取り確認
            - 画像: SigLIP空間内で視覚的に類似した画像が近いか抜き取り確認
  │
  ▼
A-3. 空間間の構造分析
  │
  ├─ A-3a. 線形CKA（Centered Kernel Alignment）の計算
  │         - e5空間とSigLIP空間のペア間の線形的な構造対応度を測る
  │         - CKAが高ければ線形projectionで十分な可能性、低ければMLPが必要
  │
  ├─ A-3b. Procrustes分析
  │         - 直交変換（回転+反転）で最もアラインした場合の残差を計算
  │         - 残差が小さいほど、2つの空間が回転関係に近い（射影が容易）
  │
  └─ A-3c. 次元ごとの分散分析
            - 両空間の主成分分析（PCA）を実行
            - 有効次元数を比較（共通空間Dの選定材料）
  │
  ▼
A-4. 線形射影ベースライン
  │
  ├─ A-4a. データ分割
  │         - train: 5,000枚 / val: 520枚 / test: 500枚
  │
  ├─ A-4b. 線形射影の学習
  │         - img_embeddings × W = projected_img（768→384）
  │         - 損失: MSE(projected_img, text_embeddings) または cosine embedding loss
  │         - W のみ学習（パラメータ数: 768×384 = 294,912）
  │
  └─ A-4c. ベースライン精度の計測
            - Cross-modal retrieval: Recall@1, @5, @10（画像→テキスト、テキスト→画像）
            - テスト500件での評価
  │
  ▼
A-5. Phase A 判定
  │
  └─ 判定基準:
     - CKAとProcrustes残差から、MLP projection headの層数を決定
     - 線形ベースラインのRecall@10が 0.3 以上なら、空間間に十分な対応がある
     - 0.3 未満の場合、SigLIPモデルの再検討またはデータ拡張を検討
```

### 3.3 Phase A の合格条件

- [ ] e5とSigLIPの事前計算embeddingが正しく生成されている
- [ ] CKAスコアとProcrustes残差が算出されている
- [ ] 線形射影ベースラインのRecall@1, @5, @10が計測されている
- [ ] 共通空間の次元D、MLP層数の方針がデータに基づいて決定されている

---

## 4. Phase B — 共通空間の学習と品質検証

### 4.1 目的

対照学習により、テキストと画像が共存する共通embedding空間を構築する。テキスト-テキスト検索品質の維持を構造保存正則化で保証する。

### 4.2 タスクフロー

```
B-1. VideoGameBunny-Datasetのダウンロードと前処理
  │
  ├─ B-1a. データセットのダウンロード
  │         - HuggingFace: asgaardlab/VideoGameBunny-Dataset
  │         - 185,259枚。ストレージ要件を事前に確認
  │
  ├─ B-1b. キャプション品質のサンプリング確認
  │         - 100件程度を目視でキャプション品質を確認
  │         - ゲーム画面の内容を適切に記述しているか
  │         - 英語キャプション → e5はmultilingualなので英語テキストも対応可
  │
  └─ B-1c. データ分割
            - train: 170,000枚 / val: 10,000枚 / test: 5,259枚
            - ゲームタイトルごとに分割し、testに未見ゲームを含める
              （汎化性能の評価のため）
  │
  ▼
B-2. 大規模embedding事前計算
  │
  ├─ B-2a. 全キャプションのe5エンコード → text_cache（185,259×384）
  │         - バッチ処理、進捗表示つき
  │         - メモリ見積もり: 185,259 × 384 × 4bytes ≈ 270MB
  │
  ├─ B-2b. 全画像のSigLIP ViTエンコード → img_cache（185,259×768）
  │         - GPU使用推奨（CPUでも可能だが時間がかかる）
  │         - メモリ見積もり: 185,259 × 768 × 4bytes ≈ 540MB
  │
  └─ B-2c. キャッシュの永続化（.npyまたは.safetensors）
            - 再実験時に再計算を避ける
  │
  ▼
B-3. Projection Headの設計
  │
  ├─ B-3a. アーキテクチャ決定（Phase Aの結果に基づく）
  │         - 基本構成: MLP 2層（入力次元 → 512 → D）+ LayerNorm + GELU
  │         - projection_text: 384 → 512 → D
  │         - projection_img:  768 → 512 → D
  │         - D = 256（Phase AのPCA結果により調整）
  │
  └─ B-3b. パラメータ数の見積もり
            - projection_text: 384×512 + 512×256 ≈ 328K
            - projection_img:  768×512 + 512×256 ≈ 524K
            - 合計: ≈ 850K（CPUで十分学習可能）
  │
  ▼
B-4. 学習
  │
  ├─ B-4a. 損失関数の実装
  │
  │   対照学習損失（メイン）:
  │     - ミニバッチ内のN個の画像-テキストペアに対して
  │     - 正例（対角要素）のコサイン類似度を最大化
  │     - 負例（非対角要素）のコサイン類似度を最小化
  │     - InfoNCE loss（温度パラメータτは学習可能）
  │
  │   構造保存正則化:
  │     - ミニバッチ内テキスト間のコサイン類似度行列を、e5元空間と共通空間で計算
  │     - 両行列の相関（Pearson相関 or MSE）を最大化
  │     - loss_total = loss_contrastive + α × loss_structure_preservation
  │     - α = 0.5 を初期値とし、テキスト検索品質に応じて調整
  │
  ├─ B-4b. 学習ハイパーパラメータ
  │         - バッチサイズ: 256（キャッシュベースなのでメモリ制約は緩い）
  │         - 学習率: 1e-3（AdamW）
  │         - エポック数: 20〜50（early stopping: val lossが5エポック改善しなければ停止）
  │         - 温度τの初期値: 0.07
  │
  ├─ B-4c. 学習の実行
  │         - 学習カーブの監視: loss_contrastive, loss_structure_preservation, val_recall@10
  │         - チェックポイントの保存（val_recall@10のbest）
  │
  └─ B-4d. 学習が不安定な場合の対応策
            - αの増減（テキスト検索劣化 → α増加、cross-modal精度不足 → α減少）
            - 学習率スケジューラの追加（cosine annealing）
            - バッチサイズの増減
  │
  ▼
B-5. 品質検証
  │
  ├─ B-5a. Cross-modal retrieval精度（testセット）
  │         - 画像→テキスト: Recall@1, @5, @10
  │         - テキスト→画像: Recall@1, @5, @10
  │         - Phase Aの線形ベースラインとの比較
  │
  ├─ B-5b. テキスト-テキスト検索品質の検証（最重要）
  │         - testセットのテキスト間コサイン類似度の順位相関
  │         - e5元空間での順序 vs 共通空間projection後の順序
  │         - Spearmanの順位相関係数 ≥ 0.85 を合格基準とする
  │         - 既存のSleepyJean RAGクエリを数十件用意し、検索結果の上位5件を比較
  │
  ├─ B-5c. 未見ゲームタイトルでの汎化検証
  │         - testセット（未見ゲーム）でのRecall@10がtrainセットと大きく乖離しないか
  │         - 乖離が大きければ、ゲーム特化のoverfittingが発生している
  │
  └─ B-5d. 場のシミュレーション検証
            - 共通空間のベクトルをChromaDBに投入し、senseのシミュレーションを実行
            - テキストクエリで視覚信号がヒットするか
            - 視覚信号とテキスト信号が混在するFieldReadingが妥当か
  │
  ▼
B-6. Phase B 判定
  │
  └─ 判定基準:
     - Cross-modal Recall@10 ≥ 0.5（最低ライン。場の減衰が補える範囲）
     - テキスト-テキスト Spearman順位相関 ≥ 0.85（既存機能の維持）
     - 未見ゲームでのRecall@10低下が trainの20%以内
     - 上記を満たさない場合: αの再調整、D次元の変更、SigLIPモデルの変更を検討
```

### 4.3 Phase B の合格条件

- [ ] projection_text, projection_img が学習済み
- [ ] Cross-modal retrieval Recall@10 ≥ 0.5
- [ ] テキスト-テキスト Spearman順位相関 ≥ 0.85
- [ ] 未見ゲームでの汎化性能が許容範囲内
- [ ] 場のシミュレーションで視覚信号がsenseで正しくヒットする

---

## 5. Phase C — FieldEncoder差し替えとシステム統合

### 5.1 目的

学習済みprojection headをFieldEncoderに組み込み、既存システムに統合する。既存テストスイートの通過を確認する。

### 5.2 タスクフロー

```
C-1. MultimodalFieldEncoderの実装
  │
  ├─ C-1a. FieldEncoder Protocolの拡張
  │         class FieldEncoder(Protocol):
  │             def encode(self, text: str) -> NDArray[np.float32]: ...
  │             def encode_image(self, image: NDArray) -> NDArray[np.float32]: ...
  │             def dimensionality(self) -> int: ...
  │
  ├─ C-1b. 具象クラスの実装
  │         class MultimodalFieldEncoder:
  │             - self.text_encoder = e5-small（frozen）
  │             - self.image_encoder = SigLIP ViT-B（frozen）
  │             - self.projection_text = 学習済みMLP（Phase B成果物）
  │             - self.projection_img = 学習済みMLP（Phase B成果物）
  │
  │             def encode(self, text: str) -> NDArray:
  │                 raw = self.text_encoder.encode(text)     # 384次元
  │                 return self.projection_text(raw)          # D次元
  │
  │             def encode_image(self, image: NDArray) -> NDArray:
  │                 raw = self.image_encoder.encode(image)   # 768次元
  │                 return self.projection_img(raw)           # D次元
  │
  │             def dimensionality(self) -> int:
  │                 return D
  │
  └─ C-1c. query:/passage:プレフィックスの扱い
            - e5のプレフィックス処理は MultimodalFieldEncoder 内部に隠蔽
            - 場のインターフェースには影響しない（既存設計を踏襲）
  │
  ▼
C-2. ChromaDBコレクションの再構築
  │
  ├─ C-2a. 既存コレクションのバックアップ
  │         - snapshot() で場の現状を保存
  │         - SQLiteの対話ログ、knowledge_index.jsonは影響なし（場の外）
  │
  ├─ C-2b. 新コレクションの作成
  │         - 次元数をDに変更
  │         - distance_fn: cosine（既存と同じ）
  │
  └─ C-2c. 既存データの移行（任意）
            - 過去のテキスト信号を新FieldEncoderでre-encodeして再投入
            - または、空の状態から開始（Phase 1の暫定データなので許容可能）
  │
  ▼
C-3. 視覚モジュールの基本実装
  │
  ├─ C-3a. SensoryVisionモジュールのスケルトン実装
  │         class SensoryVision:
  │             def capture(self) -> NDArray:
  │                 """ゲーム画面をキャプチャしてndarrayで返す"""
  │                 ...
  │
  │             def emit_visual_signal(self, field: SharedField, encoder: FieldEncoder):
  │                 image = self.capture()
  │                 embedding = encoder.encode_image(image)
  │                 field.emit(
  │                     embedding=embedding,
  │                     origin=SignalOrigin(system="sensory:vision", context="game_frame"),
  │                     trace="[自動生成] ゲーム画面の視覚信号"
  │                 )
  │
  ├─ C-3b. 「視床」フィルタの実装
  │         - 前フレームのembeddingとの距離を計算
  │         - 閾値を超えた場合のみemit（重複抑制）
  │         - 閾値の初期値: コサイン距離 0.1（Phase B検証データから調整）
  │
  └─ C-3c. SignalOriginの命名規約
            - system: "sensory:vision"
            - context: "game_frame", "scene_change", "ui_state" 等
  │
  ▼
C-4. 既存システムとの統合テスト
  │
  ├─ C-4a. 既存テストスイートの通過確認
  │         - FieldEncoderの差し替えにより既存テストが壊れないか
  │         - 160件の既存テストが全て pass すること
  │
  ├─ C-4b. 新規テストの追加
  │         - encode_image が正しい次元のベクトルを返すか
  │         - encode と encode_image の出力が同一空間にいるか（コサイン類似度が [0,1] 範囲）
  │         - 視覚信号が emit → sense で正しく検索されるか
  │         - 視床フィルタが重複フレームを抑制するか
  │
  ├─ C-4c. 統合動作確認
  │         - llamarcute-live のsenseで視覚信号がFieldReadingに含まれるか
  │         - 睡眠サイクル中に視覚モジュールが停止しているか
  │         - FieldObserverのログに視覚emitが記録されるか
  │
  └─ C-4d. パフォーマンス計測
            - encode_image の推論レイテンシ
            - 視覚emit込みの対話1回あたりのレイテンシ増分
            - 目安: 対話フロー全体で元の500ms目安を大幅に超過しないこと
  │
  ▼
C-5. Phase C 判定
  │
  └─ 判定基準:
     - 既存テスト160件 + 新規テスト全件が pass
     - llamarcute-live がsenseで視覚信号を正しく取得できる
     - レイテンシが許容範囲内
```

### 5.3 Phase C の合格条件

- [ ] MultimodalFieldEncoder が FieldEncoder Protocol を満たしている
- [ ] ChromaDBコレクションが新次元数で再構築されている
- [ ] 既存テスト160件が全て pass
- [ ] 新規テスト（encode_image, 視覚sense, 視床フィルタ）が全て pass
- [ ] 視覚信号が llamarcute-live の対話フローで取得可能
- [ ] レイテンシが許容範囲内

---

## 6. 全体のリスクと対応策

| リスク | 影響 | 検知タイミング | 対応策 |
|-------|------|-------------|--------|
| e5とSigLIP空間の構造的対応が弱い | 共通空間の品質が低い | Phase A (CKA, Procrustes) | SigLIPモデルの変更（ViT-L等）、または方式Bへの撤退 |
| テキスト-テキスト検索品質の劣化 | SleepyJean RAG・llamarcute-live senseへの悪影響 | Phase B (Spearman相関) | αの増加、projection_textの層数増加、正則化手法の変更 |
| VideoGameBunnyのキャプション品質が低い | 対照学習の精度低下 | Phase B (B-1b) | キャプションをLLMで再生成、またはCC3M等の汎用データで補強 |
| ゲーム特化データへのoverfit | 一般画像への汎化失敗 | Phase B (B-5c) | 汎用データセット（CC3M）の混合、data augmentation |
| encode_imageのレイテンシ超過 | 対話応答速度の劣化 | Phase C (C-4d) | SigLIP ViTの量子化、推論バッチの非同期化 |

---

## 7. 成果物一覧

| フェーズ | 成果物 | 形式 |
|---------|--------|------|
| Phase A | embedding事前計算キャッシュ | .npy |
| Phase A | 空間分析レポート（CKA, Procrustes, PCA） | markdown or notebook |
| Phase A | 線形射影ベースライン精度 | 数値レポート |
| Phase B | 学習済み projection_text, projection_img | .pt (PyTorch state_dict) |
| Phase B | 学習カーブ・品質検証レポート | markdown or notebook |
| Phase C | MultimodalFieldEncoder 実装 | Python module |
| Phase C | SensoryVision 基本実装 | Python module |
| Phase C | 新規テスト | pytest |

---

## 8. スコープ外

以下は本タスクフローのスコープ外とし、Phase C完了後に別途検討する。

- 視覚モジュールの高度な画面理解（物体検出、シーン分類等）
- 推論モジュール（前フレームからの差分検出→行動決定）の設計・実装
- 動作モジュール（ゲーム操作）の設計・実装
- 視覚信号のtraceフィールドへの自動キャプション生成（LLMベース）
- SleepyJeanにおける視覚記憶の定着・忘却メカニズム
- 睡眠サイクルにおける視覚経験の統合
- 視覚信号の減衰パラメータの最適化（運用データに基づいて別途調整）

---

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-03-23 | 初版作成 |

---

*本文書はタスクフローの草案であり、各フェーズの実績に基づいて継続的に更新される。*
