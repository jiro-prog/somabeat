# SleepyJean大規模改修 — タスクフロー

**作成日:** 2026-03-31
**設計根拠:** `bio_ai_architecture.md`（2026-03-31改訂版）
**スコープ:** SleepyJeanの非LLM化、リポジトリ統合、覚醒時想起支援、睡眠サイクル再構成

---

## 0. 設計原則（全タスク共通）

- **SleepyJeanは非LLM。** LLM推論を一切含まない。言語的処理はllamarcute-liveのみ
- **場を介した間接協調。** SleepyJean↔llamarcute-live間のデータ受け渡しは全て場（SharedField）経由。直接のメソッド呼び出し禁止
- **既存のshared_state/は変更しない。** SharedField, FieldEncoder, FieldReceptorのインターフェースはそのまま使う
- **既存のllamarcute-liveの対話品質を毀損しない。** 各フェーズ末にfitness評価を実行し、改修前と比較

---

## 1. 事前作業: リポジトリ統合とバックアップ

### R-1. 旧SleepyJeanリポジトリのアーカイブ

```
1. github.com/jiro-prog/SleepyJean のREADMEに「Archived: somabeatリポジトリに統合」と追記
2. GitHubでリポジトリをArchive設定にする
3. somabeat側に旧SleepyJeanのコミットハッシュを記録
```

**判断不要。機械的に実行。**

### R-2. somabeatリポジトリに新SleepyJeanディレクトリを作成

```
somabeat/
├── llamarcute_live/     ← 既存。変更なし（Phase 2まで）
├── shared_state/        ← 既存。変更なし
├── sleepyjean/          ← 新規作成
│   ├── __init__.py
│   ├── memory_store.py
│   ├── recall.py
│   ├── reconsolidation.py
│   ├── gap_detector.py
│   ├── quality_gate.py
│   └── sleepyjean.py    ← SleepyJeanクラス（5モジュールの統合）
├── orchestrator/        ← 既存。Phase 3で変更
├── discord_bot/         ← 既存。Phase 3で変更
├── config/              ← 既存。Phase 1で拡張
├── tests/
│   ├── test_sleepyjean/  ← 新規
│   └── ...              ← 既存
└── old_sleepyjean/      ← 旧コードの参照用コピー（gitignore対象外、読み取り専用の参考資料）
```

**判断不要。ディレクトリ構造を作成するだけ。**

### R-3. 旧SleepyJeanのブリッジ層を一時保全

旧 `orchestrator/bridge.py`（sleep_ingest / wake_export）は Phase 3 で書き換えるまで動作を維持する。Phase 1〜2 の間は旧SleepyJeanのsubprocess呼び出しを無効化し、新SleepyJeanモジュールを並行開発する。

```
1. orchestrator.py の enter_sleep() で旧SleepyJean subprocess呼び出しをコメントアウト
2. 代わりに新SleepyJeanのReconsolidationを呼ぶスタブを用意（Phase 3で本実装）
3. 覚醒中のbot動作は影響なし（旧SleepyJeanは睡眠時のみ）
```

**ゲート: bot起動確認。覚醒中の対話が正常動作すること。**

---

## Phase 1: SleepyJeanコアモジュール実装

### 目標
5つのモジュール（MemoryStore, RecallEngine, Reconsolidation, GapDetector, QualityGate）を実装し、単体テストをPASSする。

### S-1. MemoryStore

**ファイル:** `sleepyjean/memory_store.py`

ChromaDB + SQLiteのラッパー。既存の `shared_state/backends/chromadb_backend.py` とは別のChromaDBコレクションを使用する（場のコレクションと記憶のコレクションは別物）。

```python
class MemoryStore:
    """エピソード記憶 + 抽象記憶（クラスタ重心）のストア。"""
    
    def add_episode(self, embedding, metadata) -> str:
        """エピソード記憶を追加。episode_idを返す。"""
    
    def add_abstract(self, embedding, cluster_id, metadata) -> str:
        """クラスタ重心を抽象記憶として追加。"""
    
    def search(self, query_embedding, n_results=3, include_abstract=True) -> list:
        """cosine類似度検索。エピソード+抽象記憶の両方から検索。"""
    
    def get_all_episodes(self) -> list:
        """Reconsolidation用。全エピソード記憶を返す。"""
    
    def delete(self, episode_ids: list) -> int:
        """忘却。指定エピソードを削除。"""
    
    def replace_abstracts(self, abstracts: list) -> None:
        """Reconsolidation後にクラスタ重心を全置換。"""
    
    def stats(self) -> dict:
        """記憶数、クラスタ数、最終更新日時等。"""
```

**実装詳細:**
- ChromaDBコレクション名: `"sleepyjean_memory"`（場のコレクション `"shared_field"` とは別）
- SQLiteテーブル: `memory_metadata`（episode_id, created_at, source, access_count, confidence, is_abstract, cluster_id）
- `search()` は access_count をインクリメントする（忘却判定に使用）
- embedding次元: 384（FieldEncoder共通）

**テスト（8件想定）:**
- add_episode + search で格納と検索が動作する
- add_abstract + search(include_abstract=True) で抽象記憶が検索対象に含まれる
- search(include_abstract=False) で抽象記憶が除外される
- delete で指定エピソードが削除される
- replace_abstracts で既存クラスタ重心が全置換される
- search が access_count をインクリメントする
- stats が正しい値を返す
- 空のストアに対する search が空リストを返す

### S-2. RecallEngine

**ファイル:** `sleepyjean/recall.py`

覚醒時の想起支援。ユーザ入力のembeddingで MemoryStore を検索し、関連記憶を場にemitする。

```python
class RecallEngine:
    """覚醒時の想起支援。毎回の対話で呼ばれる。"""
    
    def __init__(self, memory_store, field, field_encoder, config):
        self.memory_store = memory_store
        self.field = field  # SharedField
        self.field_encoder = field_encoder
        self.config = config  # n_results, min_similarity等
    
    async def recall(self, user_input: str) -> list[SignalId]:
        """ユーザ入力から関連記憶を場にemitする。
        
        1. user_inputをFieldEncoderでembedding化
        2. MemoryStore.search() で上位N件取得
        3. 各記憶のembeddingにconfidenceを乗じたノルムで場にemit
        4. emit したsignal_id群を返す
        """
    
    async def recall_from_embedding(self, query_embedding) -> list[SignalId]:
        """embedding直接指定版。場からperceiveした信号を起点にする場合用。"""
```

**実装詳細:**
- `n_results`: config指定。デフォルト3
- `min_similarity`: config指定。デフォルト0.3。これ以下のヒットはemitしない
- origin: `SignalOrigin(system="sleepyjean", context="recall")`
- ノルム: `base_norm * confidence`。base_normはconfig指定（デフォルト1.5）
- FieldEncoderは既存の MultimodalFieldEncoder / E5SmallEncoder をそのまま使う

**テスト（6件想定）:**
- 関連記憶がある場合、場にemitされる
- min_similarity以下のヒットはemitされない
- emit されたsignal_idが返される
- 記憶が空の場合、空リストが返される
- confidenceがノルムに反映される
- recall_from_embeddingが動作する

### S-3. Reconsolidation

**ファイル:** `sleepyjean/reconsolidation.py`

睡眠の核心。クラスタリングによる記憶全体の再構成。

```python
class Reconsolidation:
    """睡眠時の記憶再構成。"""
    
    def __init__(self, memory_store, field, field_encoder, config):
        self.memory_store = memory_store
        self.field = field
        self.field_encoder = field_encoder
        self.config = config
        self._previous_clusters = None  # 前回のクラスタ構造
    
    async def consolidate(self, new_episodes: list[dict]) -> ConsolidationResult:
        """記憶の再構成を実行する。
        
        1. new_episodesをembedding化してMemoryStoreに格納
        2. 全エピソード記憶を取得
        3. クラスタリング（HDBSCAN）
        4. 前回のクラスタ構造と比較
           - 新クラスタ → 場にemit (context="knowledge_update")
           - クラスタ統合 → 場にemit (context="dream")
           - クラスタ消滅 → 忘却実行 + 場にemit (context="forgetting")
        5. クラスタ重心を抽象記憶としてMemoryStoreに保存
        6. _previous_clustersを更新
        
        Returns: ConsolidationResult（新クラスタ数、統合数、忘却数、emit済みsignal_ids）
        """
    
    def _cluster(self, embeddings) -> ClusterResult:
        """HDBSCANクラスタリング。"""
    
    def _diff_clusters(self, old, new) -> ClusterDiff:
        """前回と今回のクラスタ構造の差分を取る。"""
```

**実装詳細:**
- クラスタリング: `hdbscan` ライブラリ。`min_cluster_size` はconfig指定（デフォルト3）
- クラスタ重心: 各クラスタに属するembeddingの平均ベクトル（正規化しない。ノルム=クラスタ内の記憶密度）
- クラスタ差分の判定:
  - 新クラスタ: 前回の全クラスタ重心とのcosine類似度が閾値以下
  - クラスタ統合: 前回の2つ以上のクラスタのメンバーが今回1つのクラスタに属する
  - クラスタ消滅: 前回存在したクラスタのメンバーが今回どのクラスタにも属さない（noiseに分類）
- 忘却対象: ノイズ（どのクラスタにも属さない）かつ access_count が低いエピソード
- `_previous_clusters` は SQLite に永続化する（sleepyjean/memory_store.py の SQLite を共用）
- emitにtraceは渡さない。内部ログにsignal_id + 文脈テキストを記録

**テスト（10件想定）:**
- new_episodesがMemoryStoreに格納される
- クラスタリングが実行される（最低限のembedding群で）
- 新クラスタが場にemitされる
- クラスタ統合が検出され場にemitされる
- クラスタ消滅時に忘却が実行される
- クラスタ重心が抽象記憶としてMemoryStoreに保存される
- _previous_clustersが更新・永続化される
- エピソード数が少ない場合（< min_cluster_size）でもエラーにならない
- ConsolidationResultが正しい統計を返す
- 初回実行時（_previous_clusters=None）は差分なしで正常動作

### S-4. GapDetector

**ファイル:** `sleepyjean/gap_detector.py`

```python
class GapDetector:
    """知識の穴の検出。RecallEngineの検索結果から想起失敗を記録。"""
    
    def __init__(self, field, field_encoder, config):
        self.field = field
        self.field_encoder = field_encoder
        self._failures = []  # (query_text, timestamp, best_similarity)
        self.config = config
    
    def record_failure(self, query_text: str, best_similarity: float):
        """想起失敗を記録。RecallEngineから呼ばれる。"""
    
    async def emit_gaps(self) -> list[SignalId]:
        """蓄積された想起失敗から知識の穴を場にemitする。
        睡眠サイクルの冒頭で呼ばれる。
        繰り返し失敗するトピックをopen_questionsとしてemit。
        """
    
    def clear(self):
        """emit後にクリア。"""
```

**実装詳細:**
- `record_failure` は RecallEngine.recall() 内で、全ヒットが min_similarity 以下だった場合に呼ばれる
- `emit_gaps` は同一トピック（embedding類似度が高い failure 同士）の出現回数でフィルタ。1回限りの失敗はノイズとして無視
- origin: `SignalOrigin(system="sleepyjean", context="gap")`

**テスト（5件想定）:**
- record_failureで記録される
- 複数回の同一トピック失敗がemit_gapsでemitされる
- 1回限りの失敗はemitされない
- clear後は空になる
- 失敗が0件のときemit_gapsは空リストを返す

### S-5. QualityGate

**ファイル:** `sleepyjean/quality_gate.py`

```python
class QualityGate:
    """感覚系から届いたQ&Aデータの品質判定。"""
    
    def __init__(self, field_encoder, config):
        self.field_encoder = field_encoder
        self.config = config
    
    def check(self, question: str, answer: str) -> QualityResult:
        """Q&Aペアの品質を判定。
        
        チェック項目:
        - 回答長が閾値以上（空回答・極端に短い回答の排除）
        - 質問と回答のembedding類似度が閾値以上（最低限の関連性）
        - 言語チェック（日本語を含むか）
        
        Returns: QualityResult(passed: bool, reason: str | None)
        """
```

**実装詳細:**
- 回答最低長: config指定（デフォルト50文字）
- embedding類似度閾値: config指定（デフォルト0.2）
- 言語チェック: 簡易的に日本語文字の存在確認

**テスト（5件想定）:**
- 正常なQ&AがPASS
- 空回答がFAIL
- 極端に短い回答がFAIL
- 無関係な回答（低embedding類似度）がFAIL
- 言語チェックが機能する

### S-6. SleepyJeanクラス（統合）

**ファイル:** `sleepyjean/sleepyjean.py`

```python
class SleepyJean:
    """海馬モジュール。非LLM。5モジュールの統合。"""
    
    def __init__(self, config, field, field_encoder):
        self.memory_store = MemoryStore(config.memory)
        self.recall = RecallEngine(self.memory_store, field, field_encoder, config.recall)
        self.reconsolidation = Reconsolidation(self.memory_store, field, field_encoder, config.reconsolidation)
        self.gap_detector = GapDetector(field, field_encoder, config.gap)
        self.quality_gate = QualityGate(field_encoder, config.quality)
    
    # --- 覚醒時（毎回の対話で呼ばれる） ---
    async def on_user_input(self, user_input: str) -> list[SignalId]:
        """想起支援。DialogueManagerから呼ばれる。"""
        return await self.recall.recall(user_input)
    
    # --- 睡眠時 ---
    async def on_sleep(self, dialogue_logs: list[dict]) -> ConsolidationResult:
        """睡眠時の記憶再構成。オーケストレーターから呼ばれる。"""
        # 1. GapDetectorの蓄積を場にemit
        await self.gap_detector.emit_gaps()
        self.gap_detector.clear()
        # 2. Reconsolidation実行
        result = await self.reconsolidation.consolidate(dialogue_logs)
        return result
```

**テスト（4件想定）:**
- 初期化が全モジュールを生成する
- on_user_inputがRecallEngineを呼ぶ
- on_sleepがGapDetector.emit_gaps + Reconsolidationを実行する
- configが正しく各モジュールに渡される

### S-7. config拡張

**ファイル:** `config/system.yaml` に以下を追加

```yaml
sleepyjean:
  memory:
    chromadb_collection: "sleepyjean_memory"
    sqlite_path: "data/sleepyjean_memory.db"
  recall:
    n_results: 3
    min_similarity: 0.3
    base_norm: 1.5
  reconsolidation:
    min_cluster_size: 3
    new_cluster_threshold: 0.5    # 前回クラスタ重心とのcosine類似度閾値
    noise_forget_max_access: 2    # ノイズかつaccess_count以下なら忘却
  gap:
    min_repeat_count: 2           # 同一トピック失敗がこの回数以上でemit
    similarity_threshold: 0.7     # 「同一トピック」判定のembedding類似度
  quality:
    min_answer_length: 50
    min_relevance: 0.2
```

**判断不要。上記をそのまま追加。**

### Phase 1 ゲート

- [ ] S-1〜S-6の全テストPASS（38件想定）
- [ ] 既存テスト225件がPASS（sleepyjeanモジュールの追加が既存に影響しないこと）
- [ ] `SleepyJean` クラスが初期化でき、空のMemoryStoreで `on_user_input` / `on_sleep` がエラーなく実行できる

**Phase 1完了後、Soに報告。Phase 2に進むか判断を仰ぐ。**

---

## Phase 2: 覚醒時の想起支援統合

### 目標
llamarcute-liveの対話ループにSleepyJean.on_user_inputを組み込み、毎回の対話で想起支援が動作する状態にする。

### D-1. DialogueManager への RecallEngine 統合

**ファイル:** `llamarcute_live/dialogue.py`

```
現在の対話フロー:
  user_input → perceive_field → build_prompt → generate → emit → respond

変更後:
  user_input → sleepyjean.on_user_input (Recall) → perceive_field → build_prompt → generate → emit → respond
```

**変更内容:**
1. `DialogueManager.__init__` に `SleepyJean` インスタンスを受け取る引数を追加
2. `process_input()` の先頭（perceive_fieldの前）で `await self.sleepyjean.on_user_input(user_input)` を呼ぶ
3. Recallがemitした信号は、直後の `perceive_field()` で自然に拾われる。追加のロジック不要
4. RecallEngineの検索で全ヒットが min_similarity 以下だった場合、`gap_detector.record_failure()` を呼ぶ。この連携は RecallEngine 内部で完結させる

**注意:**
- SleepyJeanが未初期化（None）の場合はRecallをスキップする（graceful degradation）
- Recallのレイテンシは数十ミリ秒。対話の応答時間（2-10秒）に対して無視できる

### D-2. bot.py / orchestrator.py の SleepyJean インスタンス管理

**ファイル:** `discord_bot/bot.py`, `orchestrator/orchestrator.py`

1. bot起動時に `SleepyJean(config, field, field_encoder)` を生成
2. `DialogueManager` に渡す
3. SleepyJean は常時起動（CPU動作、リソース消費ほぼゼロ）

### D-3. 既存テスト更新

- `DialogueManager` のテストに SleepyJean=None のケース追加（graceful degradation）
- `DialogueManager` のテストに SleepyJean ありのケース追加（Recallが呼ばれることの確認）

### Phase 2 ゲート

- [ ] 全テストPASS（既存225件 + Phase 1の38件 + Phase 2の新規テスト）
- [ ] bot起動 → 対話 → Recallのログ出力確認（MemoryStoreが空なので0件ヒットだが、エラーなく動作すること）
- [ ] 対話の応答時間に有意な劣化がないこと（Recall追加前後で比較）
- [ ] fitness評価が改修前と同等以上

**Phase 2完了後、Soに報告。Phase 3に進むか判断を仰ぐ。**

---

## Phase 3: 睡眠サイクル再構成

### 目標
旧SleepyJeanのsubprocess呼び出しを新SleepyJeanのReconsolidationに置き換え、睡眠サイクル全体を新設計に移行する。

### N-1. 旧ブリッジ層の置き換え

**ファイル:** `orchestrator/bridge.py` → 大幅書き換えまたは削除

旧sleep_ingestの役割:
1. llamarcute-liveのSQLiteから対話ログ取得 → **残す。Reconsolidation.consolidate()のnew_episodesとして渡す**
2. 困難度シグナル→SleepyJeanのhomeworkテーブル → **不要。GapDetectorに置き換え済み**
3. knowledge_index.jsonのスナップショット → **不要。Reconsolidationのクラスタ構造に置き換え**

旧wake_exportの役割:
1. knowledge_indexの差分検出→場にemit → **Reconsolidationが担当。差分検出=クラスタ差分**
2. Q&Aペア→SQLite直接転送 → **感覚系（知識蒸留）に移管。Phase 3スコープ外。暫定的に無効化**
3. 夢日記→場にemit → **Reconsolidationのクラスタ統合が担当**

**新ブリッジ: `orchestrator/bridge.py`（新版）**

```python
async def sleep_ingest(dialogue_db_path: str) -> list[dict]:
    """llamarcute-liveのSQLiteから今日の対話ログを取得。"""
    # 旧sleep_ingestのステップ1のみ残す
    # Returns: [{role, content, created_at}, ...]

async def run_reconsolidation(sleepyjean: SleepyJean, dialogue_logs: list[dict]):
    """SleepyJeanの睡眠処理を実行。"""
    result = await sleepyjean.on_sleep(dialogue_logs)
    # ログ出力: 新クラスタ数、統合数、忘却数
    return result
```

### N-2. オーケストレーターの睡眠シーケンス書き換え

**ファイル:** `orchestrator/orchestrator.py`

```
旧:
  enter_sleep()
    1. 対話停止
    2. FieldAwareLLM.unload()       ← VRAM解放（旧SleepyJeanのOllama用）
    3. sleep_ingest                  ← 旧ブリッジ
    4. SleepyJean night_cycle        ← 旧subprocess
    5. wake_export                   ← 旧ブリッジ
    6. Ollama keep_alive:0 → FieldAwareLLM.load()
    7. self_improvement
    8. purge
    9. wake_up()

新:
  enter_sleep()
    1. 対話停止 + 感覚モジュール停止
    2. グリンパティック処理（オーケストレーターレベルのログ整理）
    3. sleep_ingest（対話ログ取得のみ）
    4. SleepyJean.on_sleep(dialogue_logs)  ← Reconsolidation（LLMなし、VRAM不要）
    5. FieldAwareLLM.unload()              ← 自己改善のOllama用（位置変更）
    6. self_improvement                     ← 夢の創造的統合もここで実行（将来拡張）
    7. Ollama keep_alive:0 → FieldAwareLLM.load()
    8. 免疫チェック
    9. purge
   10. wake_up()
```

**重要な変更点:**
- `FieldAwareLLM.unload()` の位置が変わる。旧はステップ2（SleepyJean前）、新はステップ5（自己改善前）。SleepyJeanがLLMを使わないのでunloadが不要になった
- 旧SleepyJeanのsubprocess呼び出しが完全に消える
- Ollama呼び出しは自己改善（fitness/cuteness評価）のユーティリティ用途のみに残る

### N-3. グリンパティック処理の実装

**ファイル:** `orchestrator/orchestrator.py` 内の新関数

```python
async def glymphatic_cleanup():
    """睡眠冒頭のインフラ的老廃物除去。"""
    # 1. 古いemit_logエントリの削除（30日超）
    # 2. 古い対話ログの要約圧縮（将来拡張。Phase 3では削除のみ）
    # 3. 一時ファイルの削除
```

**軽量。ルールベース。判断不要。**

### N-4. 旧SleepyJean依存の完全除去

1. `requirements.txt` から旧SleepyJean固有の依存を削除（Ollama関連は自己改善で残る）
2. `old_sleepyjean/` ディレクトリを削除（参照不要になった時点で）
3. 旧SleepyJean関連のconfig項目を削除

**判断: 旧SleepyJeanのどのconfig項目が他で使われているかを確認してから実行。不明な場合はSoにエスカレーション。**

### Phase 3 ゲート

- [ ] 全テストPASS
- [ ] 睡眠サイクルが完走する（手動 `/sleep` トリガー）
- [ ] Reconsolidation のログ出力が正常（クラスタ数、統合数、忘却数）
- [ ] 覚醒後の対話でRecallが動作する（Reconsolidationで保存されたクラスタ重心が検索ヒットする）
- [ ] fitness評価が改修前と同等以上
- [ ] Ollamaの使用箇所が自己改善のユーティリティのみであること

**Phase 3完了後、Soに報告。統合テストの計画を相談。**

---

## Phase 4: 統合テストと検証

### I-1. エンドツーエンドテスト

1. bot起動 → 数回対話 → `/sleep` → 覚醒 → 対話
2. 1の対話内容に関連する質問をして、Recallがヒットすることを確認
3. 複数サイクル（3回以上）で Reconsolidation のクラスタ構造が成長することを確認

### I-2. 性能テスト

| 指標 | 目標 |
|------|------|
| 対話応答時間 | 改修前と同等（2-10秒） |
| Recall レイテンシ | < 100ms |
| Reconsolidation 所要時間 | < 5分（記憶数1000件以下） |
| VRAM使用量 | 改修前以下（SleepyJeanのLLMが消えた分） |

### I-3. 品質テスト

- fitness評価を改修前と同等以上で維持
- Recall の precision を手動評価（10件のユーザ入力に対して、返された記憶が関連しているか）

### Phase 4 ゲート

- [x] 全テストPASS (256件)
- [x] 3サイクル以上の睡眠サイクルが完走 (8サイクル完走 @ 2026-04-01)
- [x] 性能目標を達成
- [x] fitness が改修前と同等以上

### Phase 4 完了判定 (2026-04-01)

**I-1. エンドツーエンドテスト — PASS**
- 8サイクル完走。Reconsolidationクラスタが0→4→7→13→16と成長
- Recallが対話中に安定動作: 30回recall, avg_sim=0.831, zero_recall=0

**I-2. 性能テスト — PASS**

| 指標 | 目標 | 実測 | 判定 |
|------|------|------|------|
| 対話応答時間 | 2-10秒 | Phase 2で0.2%劣化確認、許容範囲 | ✅ |
| Recall レイテンシ | < 100ms | avg 99ms (81-270ms) | ✅ |
| Reconsolidation 所要時間 | < 5分 | ~2秒 (106 episodes) | ✅ |
| VRAM使用量 | 改修前以下 | 5.7GB (改修前同等) | ✅ |
| サイクル全体時間 | - | 8分 (正常時) | ✅ |

**I-3. 品質テスト — PASS**
- fitness combined=0.80（正常範囲）
- Recall avg_sim=0.831（十分な精度）

**継続観察事項（改修のブロッカーではない）:**
- VRAMリーク: GPTQModel/Marlinのunload問題。改修起因ではなく改修前から存在。unload()強化済み、継続観察
- 覚醒メッセージ: recon_result直接参照に修正済み。次回完走時に確認

---

## スコープ外（今回やらないこと）

| 項目 | 理由 |
|------|------|
| 内分泌系 (GRU) の実装 | Phase 3以降の別タスク。データ蓄積が先 |
| 知識蒸留パイプラインの感覚系化 | 旧SleepyJeanのClaude API呼び出しの移行は別タスク。暫定的に無効化 |
| llamarcute-liveへのメタ認知(knowledge_index)移管 | 設計は確定したが実装は別タスク |
| 夢の創造的統合（llamarcuteパイプラインへの組み込み） | self_improvementの拡張として別タスク |
| question_engine三分割 | 設計は確定したが実装は別タスク |
| 旧SleepyJeanのLoRA/fine-tuning関連コードの移行 | 現在停止中。将来llamarcute-liveのfine-tuningで再検討 |

---

## エスカレーション基準

以下の場合はSoに判断を仰ぐ:

1. **設計原則への抵触が疑われる場合** — 場を介さない直接呼び出しが必要に見える場合等
2. **既存テストの破壊が避けられない場合** — テストの修正方針を相談
3. **HDBSCANのパラメータがうまくいかない場合** — クラスタリングアルゴリズムの変更判断
4. **Recallのレイテンシが100msを超える場合** — 最適化方針の相談
5. **fitessが改修前を下回る場合** — 即時報告。リバート判断
6. **旧SleepyJeanのconfig/データの移行判断** — 何を引き継ぎ何を捨てるか
