# KG導入・Ollama除去・テキスト経路追加 — タスクフロー

**作成日:** 2026-04-01
**前提:** SleepyJean非LLM改修完了（Phase 1-4）。テスト255件PASS。
**設計根拠:** `bio_ai_architecture.md`（2026-03-31改訂版）、本日の設計議論

---

## 0. 設計原則（全タスク共通）

- **場のembedding注入は不可侵。** FieldAwareLLM.generate_with_field()は維持。場は影響の媒体として機能し続ける
- **テキスト経路は内容伝達。** KGから取得したトリプルはプロンプトにテキスト注入する。場を迂回するが、設計原則2の精緻化（場＝影響、テキスト＝内容）に基づく
- **LLMはFieldAwareLLM一本。** Ollama依存を完全に除去する
- **覚醒時は非LLM。** SleepyJeanのRecallはKGグラフ探索＋形態素解析のみ。睡眠時のトリプル抽出のみLLM使用
- **既存の対話品質を毀損しない。** 各Phase末にfitness評価

---

## Phase 1: generate_bare実装 + Ollama除去

### 目標
FieldAwareLLMにfield注入なしの推論モードを追加し、自己改善の全LLM呼び出しをOllamaからgenerate_bareに移行し、Ollama依存を完全除去する。

### G-1. FieldAwareLLM.generate_bare 実装

**ファイル:** `llamarcute_live/llm_inference.py`

```python
async def generate_bare(
    self, 
    system_prompt: str, 
    user_input: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    repetition_penalty: float = 1.3,
) -> str:
    """場の信号なしで推論。睡眠時のトリプル抽出・自己改善用。
    
    generate_with_field()のfield注入部分をスキップ。
    input_idsで普通にmodel.generate()を呼ぶ。
    後処理（<think>除去等）はgenerate_with_fieldと共通化。
    """
```

**実装詳細:**
- apply_chat_templateでinput_idsを生成（generate_with_fieldのトークン化ロジックと共通）
- inputs_embedsではなくinput_idsでmodel.generate()を呼ぶ
- 後処理は共通メソッドに抽出して共有
- `/no_think`の付与は呼び出し側が制御（引数で指定）

**テスト（4件）:**
- generate_bareが応答を返す
- system_promptが反映される（異なるsystem_promptで異なる応答）
- generate_with_fieldとgenerate_bareが同一モデルで交互に呼べる（KVキャッシュの干渉がない）
- 後処理（<think>除去）が動作する

### G-2. 自己改善のOllama→generate_bare移行

**ファイル:** `llamarcute_live/self_improve.py`, `llamarcute_live/fitness.py`, `llamarcute_live/cuteness.py`

現在これらはollama_client経由でLLM推論している。全呼び出しをFieldAwareLLM.generate_bareに置き換える。

**変更方針:**
- 各モジュールがOllamaClient依存の代わりにFieldAwareLLMインスタンスを受け取る
- `ollama_client.generate()` → `field_aware_llm.generate_bare()` に1対1置換
- system_prompt, user_input, temperature等のパラメータはそのまま維持
- `/no_think`の処理が必要な箇所はgenerate_bareに明示的に渡す

**注意:**
- 自己改善は睡眠中に実行される。覚醒中のgenerate_with_fieldとは排他（睡眠中は対話停止済み）
- unload/reloadが不要になる。モデルは常駐のまま

### G-3. 睡眠シーケンスの簡素化

**ファイル:** `orchestrator/orchestrator.py`

```
変更前:
  1. 対話停止
  2. グリンパティック処理
  3. sleep_ingest
  4. SleepyJean.on_sleep()
  5. FieldAwareLLM.unload()      ← 削除
  6. self_improvement (Ollama)    ← generate_bareに変更
  7. Ollama keep_alive:0          ← 削除
  8. FieldAwareLLM.load()         ← 削除
  9. 免疫チェック
  10. purge
  11. wake_up()

変更後:
  1. 対話停止
  2. グリンパティック処理
  3. sleep_ingest
  4. SleepyJean.on_sleep()
  5. self_improvement (generate_bare)
  6. 免疫チェック
  7. purge
  8. wake_up()
```

unload/reload/Ollama関連の全ステップが消える。

### G-4. Ollama依存の完全除去

1. `llamarcute_live/ollama_client.py` を削除
2. `requirements.txt` からOllama関連パッケージを削除（あれば）
3. `config/system.yaml` からOllama関連設定を削除
4. orchestrator.pyからOllamaインポート・呼び出しを削除
5. Ollama関連テストを削除または書き換え

**判断: ollama_client.pyを参照しているファイルを全検索してから実行。不明な依存があればSoにエスカレーション。**

### Phase 1 ゲート

- [ ] 全テストPASS
- [ ] 手動 `/sleep` → 自己改善がgenerate_bareで完走
- [ ] fitness評価が改修前（0.80-0.85）と同等
- [ ] 睡眠サイクル時間が短縮されている（Ollamaロード/アンロード分）
- [ ] VRAMリーク（unload関連）が発生しない（unload自体が不要になったため）
- [ ] `grep -r "ollama" --include="*.py"` がテスト・コメント以外でヒットしない

**Phase 1完了後、Soに報告。**

---

## Phase 2: KnowledgeGraph + 睡眠時トリプル抽出

### 目標
KnowledgeGraphモジュールを実装し、睡眠時にgenerate_bareで対話ログからトリプルを抽出してKGに格納する。ReconsolidationをクラスタリングベースからKGベースに差し替える。

### K-1. KnowledgeGraphモジュール実装

**ファイル:** `sleepyjean/knowledge_graph.py`

```python
@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_episode_id: str  # 抽出元の対話ログID
    created_at: datetime
    access_count: int = 0

class KnowledgeGraph:
    """SleepyJeanの知識グラフ。NetworkX + SQLite永続化。"""
    
    def __init__(self, sqlite_path: str):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._sqlite_path = sqlite_path
    
    def add_triples(self, triples: list[Triple]) -> AddResult:
        """トリプル群をグラフに追加。エンティティ解決含む。
        
        エンティティ解決: 
        - 完全一致: そのまま既存ノードに接続
        - 表記揺れ: embedding類似度 + 文字列類似度（Levenshtein）で判定
          閾値はconfig指定。判定に迷ったら別ノードとして保持（保守的）
        """
    
    def query(self, keywords: list[str], max_hops: int = 2) -> list[Triple]:
        """キーワードからノードを検索し、N-hopのサブグラフのトリプル群を返す。
        
        1. keywordsの各文字列でノード名を部分一致検索
        2. ヒットしたノードからmax_hopsまでエッジを辿る
        3. 辿ったトリプルをaccess_countでソートして返す
        4. 各ヒットノードのaccess_countをインクリメント
        """
    
    def pagerank(self) -> dict[str, float]:
        """各ノードの重要度。nx.pagerank()のラッパー。"""
    
    def diff(self, previous_snapshot: dict) -> GraphDiff:
        """前回スナップショットとの差分。
        
        Returns: GraphDiff(
            new_nodes: list[str],
            new_edges: list[tuple],
            merged_subgraphs: list[tuple[set, set]],  # (旧サブグラフ群, 新統合サブグラフ)
            lost_nodes: list[str],  # 前回あったが今回ないノード
        )
        """
    
    def prune(self, min_pagerank: float, min_access_count: int) -> list[str]:
        """重要度が低く参照もされていない孤立ノードを剪定。"""
    
    def snapshot(self) -> dict:
        """現在のグラフ構造のスナップショット。diff()の入力用。"""
    
    def stats(self) -> dict:
        """ノード数、エッジ数、連結成分数等。"""
    
    def save(self) -> None:
        """SQLiteに永続化。"""
    
    def load(self) -> None:
        """SQLiteから復元。"""
```

**実装詳細:**
- NetworkXの有向グラフ。ノード=エンティティ、エッジ=リレーション
- エッジ属性にTripleのメタデータを格納
- SQLite永続化: `kg_nodes`テーブル（name, created_at, access_count, pagerank_score）、`kg_edges`テーブル（subject, relation, object, source_episode_id, created_at）
- エンティティ解決にFieldEncoderのembedding類似度を使用（閾値はconfig）

**テスト（10件）:**
- add_triplesでノードとエッジが追加される
- add_triplesでエンティティ解決が動作（同一名ノードが重複しない）
- queryがキーワードからN-hopのトリプルを返す
- queryがaccess_countをインクリメントする
- query結果が空の場合は空リスト
- pagerankが正しく計算される
- diffが新ノード・新エッジ・消滅ノードを検出する
- pruneが低重要度ノードを削除する
- save/loadのラウンドトリップ
- 空グラフに対する全操作がエラーにならない

### K-2. トリプル抽出モジュール実装

**ファイル:** `sleepyjean/triple_extractor.py`

```python
class TripleExtractor:
    """対話ログからトリプルを抽出する。睡眠時にLLM(generate_bare)を使用。"""
    
    def __init__(self, llm: FieldAwareLLM, config):
        self.llm = llm
        self.config = config
    
    async def extract(self, dialogue_logs: list[dict]) -> list[Triple]:
        """対話ログ群からトリプルを抽出。
        
        1. 対話ログを適切なチャンクに分割（1-3往復単位）
        2. 各チャンクに対してgenerate_bareでトリプル抽出プロンプトを実行
        3. LLM出力をパースしてTripleオブジェクトに変換
        4. パース失敗行はスキップ（ログ出力）
        """
    
    def _build_extraction_prompt(self, dialogue_chunk: str) -> tuple[str, str]:
        """トリプル抽出用のsystem_promptとuser_inputを生成。"""
    
    def _parse_triples(self, llm_output: str, source_episode_id: str) -> list[Triple]:
        """LLM出力をパースしてTripleリストに変換。
        
        期待形式: '主語 | 関係 | 目的語' の改行区切り
        パース失敗行はスキップ。
        """
```

**抽出プロンプト:**
```
system: あなたは対話から事実情報を抽出するアシスタントです。
以下の対話から、事実のみをトリプル形式で抽出してください。
感想・主観・挨拶は除外してください。

出力形式（1行1トリプル）:
主語 | 関係 | 目的語

user: [対話ログ]
```

**テスト（6件）:**
- 事実を含む対話からトリプルが抽出される
- 雑談のみの対話から空リストが返る（事実なし）
- パース失敗行がスキップされる
- 複数チャンクが正しく処理される
- source_episode_idが各トリプルに付与される
- LLMが空応答を返した場合にエラーにならない

### K-3. ReconsolidationのKGベース差し替え

**ファイル:** `sleepyjean/reconsolidation.py` を書き換え

```python
class Reconsolidation:
    """睡眠時の記憶再構成。KGベース。"""
    
    def __init__(self, knowledge_graph, memory_store, field, 
                 triple_extractor, field_encoder, config):
        self.kg = knowledge_graph
        self.memory_store = memory_store  # フォールバック用に残す
        self.field = field
        self.triple_extractor = triple_extractor
        self.field_encoder = field_encoder
        self.config = config
    
    async def consolidate(self, dialogue_logs: list[dict]) -> ConsolidationResult:
        """記憶の再構成を実行。
        
        1. 対話ログをMemoryStoreに仮格納（既存と同じ。フォールバック用）
        2. TripleExtractor.extract()でトリプル抽出（generate_bare使用）
        3. KnowledgeGraph.add_triples()でKGに統合
        4. 前回のsnapshotとdiff
           - 新ノード → 場にemit (context="knowledge_update")
           - サブグラフ統合 → 場にemit (context="dream")
           - prune実行 → 場にemit (context="forgetting")
        5. snapshotを保存
        
        Returns: ConsolidationResult（抽出トリプル数、新ノード数、統合数、忘却数）
        """
```

**既存のHDBSCANクラスタリングコードは削除する。** KGに完全置き換え。MemoryStore（ChromaDB）はフォールバック検索用に残す。

**テスト（8件）:**
- 対話ログからトリプルが抽出されKGに格納される
- 前回との差分が正しく検出される
- 新ノードが場にemitされる
- サブグラフ統合が場にemitされる
- pruneが実行され場にemitされる
- ConsolidationResultが正しい統計を返す
- 対話ログが空の場合にエラーにならない
- トリプル抽出が0件の場合にエラーにならない（雑談のみの日）

### K-4. SleepyJeanクラスの更新

**ファイル:** `sleepyjean/sleepyjean.py`

```python
class SleepyJean:
    def __init__(self, config, field, field_encoder, llm: FieldAwareLLM):
        self.memory_store = MemoryStore(config.memory)
        self.knowledge_graph = KnowledgeGraph(config.kg.sqlite_path)
        self.triple_extractor = TripleExtractor(llm, config.extraction)
        self.recall = RecallEngine(...)  # Phase 3で更新
        self.reconsolidation = Reconsolidation(
            self.knowledge_graph, self.memory_store, field,
            self.triple_extractor, field_encoder, config.reconsolidation
        )
        # ...
```

**注意:** SleepyJeanの__init__にFieldAwareLLMを渡す。覚醒時のRecallではLLMを使わない（KGグラフ探索のみ）。LLMはon_sleep()内のReconsolidation→TripleExtractorでのみ使用される。

### K-5. config拡張

```yaml
sleepyjean:
  kg:
    sqlite_path: "data/sleepyjean_kg.db"
    entity_resolution:
      embedding_threshold: 0.85    # embedding類似度がこれ以上なら同一エンティティ
      levenshtein_threshold: 0.8   # 文字列類似度がこれ以上なら同一エンティティ候補
    prune:
      min_pagerank: 0.001
      min_access_count: 0
      protect_recent_days: 7       # 直近N日のノードはprune対象外
  extraction:
    chunk_size: 3                  # 1チャンクあたりの対話往復数
    max_new_tokens: 512
    temperature: 0.3               # 抽出は低temperatureで正確に
```

### Phase 2 ゲート

- [ ] 全テストPASS
- [ ] 手動 `/sleep` → トリプル抽出 + KG格納が完走
- [ ] KGにノードとエッジが格納されている（手動確認）
- [ ] Reconsolidationのログ出力が正常（抽出トリプル数、新ノード数）
- [ ] fitness評価が改修前と同等
- [ ] 睡眠サイクル時間が許容範囲内（トリプル抽出の追加コスト確認）

**Phase 2完了後、Soに報告。**

---

## Phase 3: Recallのテキスト経路化

### 目標
RecallEngineをKGベースに差し替え、検索結果をテキスト（トリプル列挙）としてllamarcute-liveのプロンプトに注入する。

### T-1. RecallEngineのKGベース差し替え

**ファイル:** `sleepyjean/recall.py` を書き換え

```python
class RecallEngine:
    """覚醒時の想起支援。KGグラフ探索ベース。非LLM。"""
    
    def __init__(self, knowledge_graph, memory_store, field, 
                 field_encoder, config):
        self.kg = knowledge_graph
        self.memory_store = memory_store  # フォールバック
        self.field_encoder = field_encoder
        self.field = field
        self.config = config
        self._tokenizer = None  # MeCab/fugashi/GiNZA
    
    def recall(self, user_input: str) -> RecallResult:
        """ユーザ入力から関連記憶を取得。
        
        1. 形態素解析でキーワード抽出（名詞・固有名詞）
        2. KnowledgeGraph.query(keywords, max_hops=2)
        3. トリプル群をテキスト化
        4. KGが空またはヒットなしの場合、MemoryStoreにフォールバック
        
        Returns: RecallResult(
            triples_text: str | None,  # プロンプト注入用テキスト
            signal_ids: list[SignalId],  # 場にemitしたsignal_id（任意）
            source: "kg" | "embedding" | "none",
        )
        """
    
    def _extract_keywords(self, text: str) -> list[str]:
        """形態素解析でキーワード抽出。名詞・固有名詞をフィルタ。"""
    
    def _format_triples(self, triples: list[Triple]) -> str:
        """トリプル群をプロンプト注入用テキストに変換。
        
        例:
        'あなたの記憶:
        - フォレトスはむし・はがねタイプ
        - フォレトスは防御力が高い
        - フォレトスはだいばくはつを覚える'
        """
```

**実装詳細:**
- 形態素解析: fugashi (MeCab) またはspaCy+GiNZA。config切り替え
- キーワード抽出: 品詞フィルタ（名詞、固有名詞、形容詞語幹）
- KGヒットなし時はMemoryStore（ChromaDB）にフォールバックしてembedding検索。この場合はテキストではなく従来のembedding経路で場にemit
- トリプルのテキスト化フォーマットは「あなたの記憶:」で始め、箇条書き。おうむ返し防止のため完成文ではなくキーワード的な短文
- Recallは同期メソッド（asyncではない）。覚醒時のレイテンシを最小化

**テスト（8件）:**
- キーワード抽出が名詞・固有名詞を返す
- KGヒットありの場合、triples_textが返される
- KGヒットなしの場合、MemoryStoreフォールバックが動作
- KGもMemoryStoreもヒットなしの場合、source="none"
- _format_triplesが正しいテキストを生成する
- access_countがインクリメントされる
- 空文字入力でエラーにならない
- 形態素解析が日本語テキストで動作する

### T-2. DialogueManagerへのテキスト経路追加

**ファイル:** `llamarcute_live/dialogue.py`

```
変更前の対話フロー:
  user_input 
    → sleepyjean.on_user_input (embedding emit) 
    → perceive_field 
    → build_prompt 
    → generate_with_field 
    → emit → respond

変更後:
  user_input
    → sleepyjean.recall(user_input)        ← テキスト取得
    → perceive_field                        ← 場のembedding（影響）
    → build_prompt(recall_text=...)         ← テキスト経路を追加
    → generate_with_field                   ← field embedding注入（影響）
    → emit → respond
```

**build_prompt の変更:**

```python
def build_prompt(self, user_input, recall_text=None):
    prompt_parts = []
    prompt_parts.append(self._invariant_constraints)
    prompt_parts.append(self._personality_rules)
    if recall_text:
        prompt_parts.append(recall_text)    # ← NEW: 想起テキスト
    prompt_parts.append(self._conversation_buffer)
    return "\n\n".join(prompt_parts)
```

想起テキストは行動規範の後、会話バッファの前に配置。「自分の過去の経験」として行動規範に続く自然な位置。

**テスト（4件）:**
- recall_textありのbuild_promptにテキストが含まれる
- recall_textなし（None）のbuild_promptが従来通り動作
- 想起テキストが行動規範と会話バッファの間に配置される
- トークン上限500に想起テキスト分が加算される（バッファが圧迫される）

### T-3. on_user_inputの更新

**ファイル:** `sleepyjean/sleepyjean.py`

```python
# 変更前
async def on_user_input(self, user_input: str) -> list[SignalId]:
    return await self.recall.recall(user_input)

# 変更後
def on_user_input(self, user_input: str) -> RecallResult:
    return self.recall.recall(user_input)
```

戻り値がRecallResult（テキスト + signal_ids + source）に変更。DialogueManagerがrecall_result.triples_textをbuild_promptに渡す。

### Phase 3 ゲート

- [ ] 全テストPASS
- [ ] 対話でKGの知識が応答に反映される（手動テスト: 教えた事実について質問）
- [ ] おうむ返しではなく自分の言葉で再構成されている
- [ ] Recallレイテンシ < 100ms（形態素解析 + KGクエリ）
- [ ] KGが空の場合にフォールバックが動作する
- [ ] 対話応答時間が許容範囲内
- [ ] fitness評価が改修前と同等

**Phase 3完了後、Soに報告。**

---

## Phase 4: 統合テスト + 品質検証

### I-1. エンドツーエンドテスト

1. bot起動 → 事実を含む対話を数回（「フォレトスはむし・はがねタイプだよ」等）
2. `/sleep` → トリプル抽出 + KG格納を確認
3. 覚醒後 → 教えた事実について質問 → KGから想起してテキスト経路で応答に反映されることを確認
4. 3サイクル以上で KG成長・prune・diff が正常動作

### I-2. 性能テスト

| 指標 | 目標 |
|------|------|
| 対話応答時間 | 改修前と同等（2-10秒） |
| Recall レイテンシ | < 100ms（形態素解析 + KGクエリ） |
| トリプル抽出（睡眠時） | 10対話分 < 3分（generate_bare @ 26 tok/s） |
| 睡眠サイクル全体 | < 30分（Ollama除去分の短縮を見込む） |
| VRAM使用量 | 5.7GB（unloadが不要、常駐のまま） |

### I-3. 品質テスト

- fitness評価: 改修前（0.80-0.85）と同等以上
- 事実想起テスト: 10件の事実を教え、翌サイクル後に質問。正答率を計測
- おうむ返し率: 想起テキストの文言がそのまま応答に出現する割合。低いほど良い

### Phase 4 ゲート

- [ ] 全テストPASS
- [ ] 3サイクル以上完走
- [ ] 性能目標達成
- [ ] 事実想起の正答率 > 50%（10件中5件以上）
- [ ] fitness維持

---

## スコープ外

| 項目 | 理由 |
|------|------|
| モデル変更（Qwen3.5-9B等） | 独立タスク。KG+Ollama除去が安定してから |
| 内分泌系 (GRU) | 別タスク |
| 場のembedding経路でのRecall（従来方式） | フォールバックとして残すが、主経路はテキスト |
| エンティティ解決の高度化（LLMベース） | まずはembedding+文字列類似度で。不十分なら後で拡張 |
| 設計原則2の文書更新 | 実装が安定してからbio_ai_architecture.mdに反映 |

---

## エスカレーション基準

1. **generate_bareの品質がOllamaと有意に異なる場合** — GPTQ量子化の影響か要調査
2. **トリプル抽出の品質が低い場合** — プロンプト改善で対応できるか、NLP前処理が必要か
3. **形態素解析ライブラリの選定** — fugashi(MeCab)とspaCy+GiNZAのどちらが適切か。日本語環境でのインストール容易性も考慮
4. **おうむ返し率が高い場合** — テキストフォーマットの粒度調整が必要。完全な文→キーワード列挙への変更等
5. **KGのノード数が爆発する場合** — prune閾値の調整、またはエンティティ解決の精度向上
6. **トークン上限500に想起テキストが収まらない場合** — 想起テキストの最大トークン数制限の設計判断

---

## 依存ライブラリ追加

| ライブラリ | 用途 | Phase |
|-----------|------|-------|
| networkx | KnowledgeGraph | 2 |
| fugashi または spacy+ginza | 形態素解析（Recall） | 3 |
