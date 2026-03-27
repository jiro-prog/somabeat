# 指示書: FieldReceptor統合改修タスクフロー

**日付:** 2026-03-26
**対象:** somabeat（integrated-system）
**親文書:** llamarcute_live_design.md（改訂版）、shared_field_design.md（改訂版）
**目的:** trace廃止 → perceive新設 → FieldReceptor導入 → embedding直接注入への移行

---

## 背景

llamarcute_live_design.md 3.4節でsense結果のtraceをテキストとしてLLMプロンプトに注入していた。これはbio_ai_architecture.md 5.3節「通信と可読性の完全分離」に違反。場のembeddingをFieldReceptor経由でLLMの入力embedding層に直接注入する設計に移行する。

---

## 依存関係

```
T1（インターフェース改修）
 ├─→ T2（内部ログ移行）──→ T3（ブリッジ層改修）
 └─→ T4（CKA/Procrustes計測）──→ T5（FieldReceptor実装）──→ T6（推論パス変更）──→ T7（対話フロー書き換え）
                                                                                        ↓
                                                                                      T8（統合テスト）
```

T2-T3系列とT4-T7系列は並列に進行可能。

---

## T1. 場のインターフェース改修

**依存:** なし（全タスクの起点）
**影響範囲:** shared_state/

### T1-a. Signal定義からtrace削除

`shared_state/interface.py`のSignal定義からtraceフィールドを完全に除去する。

```python
# Before
@dataclass(frozen=True)
class Signal:
    embedding: NDArray[np.float32]
    emitted_at: datetime
    origin: SignalOrigin
    trace: str                # ← 削除
    signal_id: SignalId

# After
@dataclass(frozen=True)
class Signal:
    embedding: NDArray[np.float32]
    emitted_at: datetime
    origin: SignalOrigin
    signal_id: SignalId
```

### T1-b. emitシグネチャからtrace引数削除

`SharedField.emit`のシグネチャからtrace引数を削除する。ChromaDB実装のemitもtrace関連の処理を除去。ChromaDBのdocumentsカラムには空文字列を設定する（ChromaDBの必須フィールドのため）。

### T1-c. perceive操作の新設

`SharedField`にperceive操作を追加する。PerceiveParams、PerceivedSignal、FieldPerceptionの型定義を追加。

```python
@dataclass
class PerceiveParams:
    decay_fn: DecayFunction = ExponentialDecay(half_life_hours=24.0)
    min_strength: float = 0.1
    max_signals: int = 50
    time_horizon: timedelta | None = None

@dataclass(frozen=True)
class PerceivedSignal:
    signal: Signal
    decay_factor: float
    strength: float       # decay_factor × ‖embedding‖

@dataclass(frozen=True)
class FieldPerception:
    signals: list[PerceivedSignal]   # strength降順
    perceived_at: datetime
```

ChromaDB実装: `collection.get()`で全件取得 → 読み取り時減衰・strength閾値の後処理。

### T1-d. senseにdeprecation warning追加

senseの呼び出し時にdeprecation warningを出力する。削除はしない（暫定残存）。

### T1-e. FieldReceptor Protocolの定義

`shared_state/interface.py`にFieldReceptor Protocolを追加する。

```python
class FieldReceptor(Protocol):
    def transduce(self, field_embeddings: list[NDArray[np.float32]],
                  strengths: list[float]) -> NDArray[np.float32]: ...
    def field_dimensionality(self) -> int: ...
    def agent_dimensionality(self) -> int: ...
```

### T1-f. FieldObserver更新

on_emitからtrace関連を除去。on_senseをon_perceiveに変更。signal_idベースのログ形式に変更。

### T1-g. 既存テストのtrace依存除去

trace引数を使っている全テストを修正。traceを参照しているアサーションを削除。emitの呼び出しからtrace引数を除去。

### 完了条件

- [ ] Signal定義にtraceフィールドが存在しない
- [ ] emitにtrace引数が存在しない
- [ ] perceiveが動作し、FieldPerceptionを返す
- [ ] senseがdeprecation warningを出力する
- [ ] FieldReceptor Protocolが定義されている
- [ ] 既存テスト全件PASS

---

## T2. emit内部ログへの移行

**依存:** T1
**影響範囲:** llamarcute-live/, sleepyjean/, orchestrator/

### T2-a. emit_logテーブルの追加

各系統のSQLiteにemit_logテーブルを追加する。

```sql
CREATE TABLE emit_log (
    signal_id TEXT PRIMARY KEY,
    context TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### T2-b. emit呼び出し箇所の全件修正

全てのemit呼び出しからtrace引数を削除し、emit後にsignal_idで内部ログを記録するパターンに変更する。

```python
# Before
signal_id = await field.emit(embedding=emb, origin=origin, trace="対話の要約テキスト")

# After
signal_id = await field.emit(embedding=emb, origin=origin)
self.db.insert_emit_log(signal_id=signal_id, context="対話の要約テキスト")
```

対象箇所の洗い出し:
- llamarcute-live: 対話経験emit、困難度シグナルemit、自己改善記録emit
- orchestrator/bridge: wake_exportの全emit箇所
- sensory/vision: SensoryVision.process_frameのemit

### 完了条件

- [ ] 全emit呼び出しからtrace引数が除去されている
- [ ] emit直後にsignal_idで内部ログを記録するパターンが全箇所に適用されている
- [ ] emit_logテーブルが各系統のSQLiteに存在する
- [ ] 既存テスト全件PASS

---

## T3. ブリッジ層の改修

**依存:** T1, T2
**影響範囲:** orchestrator/bridge/

### T3-a. sleep_ingestの改修

困難度シグナルの取得方法を変更する。

```python
# Before
reading = await field.sense(query=encode("今日の対話で困難だったこと"), ...)
for ws in reading.signals:
    homework_text = ws.signal.trace  # ← trace依存

# After
snap = await field.snapshot()
difficulty_signals = [s for s in snap.signals if s.origin.context == "difficulty"]
for signal in difficulty_signals:
    # signal_idでllamarcute-liveのemit_logを逆引き
    log_entry = llamarcute_db.get_emit_log(signal.signal_id)
    homework_text = log_entry.context
```

### T3-b. wake_exportの改修

1. 全emitからtrace引数を削除。内部ログにsignal_idで文脈を記録
2. Q&AペアをSQLite間直接転送に変更（場を経由しない）

```python
# Before
await field.emit(embedding=emb, origin=origin, trace=f"トピック {topic} を新たに学習した")

# After
signal_id = await field.emit(embedding=emb, origin=origin)
sleepyjean_db.insert_emit_log(signal_id=signal_id, context=f"トピック {topic} を新たに学習した")
```

```python
# Q&Aペアの転送（場を経由しない）
qa_pairs = sleepyjean_db.get_quality_checked_qa()
llamarcute_db.insert_rotation_tasks(qa_pairs)
```

### 完了条件

- [ ] sleep_ingestがsenseを使っていない（snapshot + contextフィルタに置き換え済み）
- [ ] sleep_ingestがtraceに依存していない（signal_idで内部ログを逆引き）
- [ ] wake_exportの全emitからtrace引数が除去されている
- [ ] Q&Aペアが場を経由せずSQLite間で直接転送されている
- [ ] 睡眠サイクルが完走する

---

## T4. CKA/Procrustes計測

**依存:** T1（FieldEncoder Protocolが確定していること）
**並列可能:** T2, T3と並列で進行可能
**影響範囲:** scripts/

### T4-a. 計測データの生成

既存の対話ログ（SQLite）のテキストをソースに、場のembeddingとLLMの内部表現のペアを生成する。

```
テキスト → FieldEncoder.encode_for_emit → 384次元ベクトル  ... X
テキスト → LLM tokenize → LLM内部表現 → ???              ... Y
```

LLM内部表現として3候補を取得する:
- **A. embedding層出力の平均プーリング**: 全トークンのembedding層出力の平均
- **B. 最終トークンのembedding層出力**: auto-regressiveモデルの最終位置
- **C. transformer第1層出力の平均プーリング**: 意味構造が形成される最初の層

### T4-b. CKA計測

X（場のembedding群）とY（LLM内部表現群）のCKA（Centered Kernel Alignment）を計算する。3候補それぞれに対して計測。

### T4-c. Procrustes計測

X→Yの最適線形変換後の残差を計測する。線形変換で十分な構造対応が取れるかを判断する。

### T4-d. 判断

| CKA | Procrustes残差 | 判断 |
|-----|---------------|------|
| 高い | 小さい | 線形変換で十分 |
| 高い | 大きい | 構造は対応するが線形では不十分。MLP |
| 低い | — | 空間の構造差が大きい。MLP + 十分な学習データ |

### 完了条件

- [ ] 3候補のCKA値とProcrustes残差が計測されている
- [ ] LLM側の表現形式（A/B/C）が決定されている
- [ ] FieldReceptorの変換層構造（線形 or MLP）が決定されている

---

## T5. FieldReceptor実装・学習

**依存:** T4
**影響範囲:** shared_state/

### T5-a. 変換層の実装

T4の結果に基づき、線形変換またはMLPを実装する。

```python
class FieldReceptorImpl:
    def __init__(self, field_dim: int, agent_dim: int):
        # T4の結果に基づく構造
        ...

    def transduce(self, field_embeddings, strengths):
        # strengthでスケーリング → 次元変換
        scaled = [emb * s for emb, s in zip(field_embeddings, strengths)]
        return self.transform(scaled)  # (K, agent_dim)
```

### T5-b. 学習データ生成

既存の対話ログテキストから、場のembeddingとLLMの内部表現のペアを生成する（T4-aで生成したデータを再利用可能）。

### T5-c. 学習

Phase Bのprojection head学習と同様のフレームワークで学習する。損失関数はT4で選定した表現形式に依存。

### T5-d. 検証

変換後のベクトルがLLMの入力空間に「馴染む」ことを定量検証する。同一テキストに対して、テキストトークン経由の表現と、場embedding → FieldReceptor変換後の表現の距離を計測。

### 完了条件

- [ ] FieldReceptorが学習済みで、FieldReceptor Protocolを満たす
- [ ] 変換後ベクトルとLLM内部表現の距離が許容範囲内
- [ ] 学習済み重みが永続化されている

---

## T6. LLM推論パスの変更

**依存:** T5
**影響範囲:** llamarcute-live/

### T6-a. 設計要求

場のembeddingをLLMの入力embedding層に直接注入するために、LLMの入力シーケンスをembeddingレベルで操作可能な推論パスが必要。

```
[system prompt tokens] [field signals] [user input tokens]
                        ↑
                    FieldReceptor出力をここに注入
```

### T6-b. 実現手段の調査・選定

設計が要求するものを先に定義した上で、それを実現する手段を探す。手段の候補:

- transformersライブラリのinputs_embeds引数を使った直接推論
- その他の手段があれば検討

**注記**: 手段が現時点で見つからない場合は「未実装」と明示し、妥協的な代替を設計に混入させない。

### T6-c. 推論パスの実装

選定した手段でLLM推論パスを実装する。以下が動作すること:

1. テキストをtokenize → embedding層を通す → テキスト部分のembedding列を取得
2. FieldReceptorの出力（K, hidden_dim）をテキストembedding列の適切な位置に挿入
3. 結合されたembedding列をtransformer層に通す
4. 応答テキストを生成

### T6-d. 既存パイプラインとの互換性確認

以下が新推論パスで動作すること:
- 通常の対話応答
- cuteness評価の候補間対話
- fitness評価のタスク回答
- 自己改善の候補生成

### 完了条件

- [ ] LLMの入力embedding層にFieldReceptor出力を注入して推論が実行できる
- [ ] テキストのみ（場の信号なし）での推論も引き続き動作する
- [ ] 応答品質が従来と同等（退行がない）

---

## T7. llamarcute-live対話フロー書き換え

**依存:** T5, T6
**影響範囲:** llamarcute-live/, orchestrator/

### T7-a. 覚醒中の対話フロー

sense × 2 + trace抽出 → perceive + FieldReceptor + embedding注入に全面変更する。

```python
# Before
reading1 = await field.sense(self_awareness_query, ...)
reading2 = await field.sense(user_context_query, ...)
prompt = build_prompt(
    behavioral_rules=rules,
    self_awareness=[ws.signal.trace for ws in reading1.signals],
    memory=[ws.signal.trace for ws in reading2.signals],
    user_input=user_text,
)
response = await ollama_chat(prompt)

# After
perception = await field.perceive(PerceiveParams(...))
field_embeddings = receptor.transduce(
    [ps.signal.embedding for ps in perception.signals],
    [ps.strength for ps in perception.signals],
)
response = await llm_inference(
    system_prompt=rules_text,
    field_embeddings=field_embeddings,
    user_input=user_text,
)
```

### T7-b. 自己改善（候補生成）

睡眠中の候補生成でも同じ方式を適用する。場のembedding注入により、LLMが直近の対話経験・記憶変化・困難度シグナルをembedding空間で受け取った上で候補を生成する。

### T7-c. 自己改善（cuteness評価）

候補間対話の各LLM推論にperceive + FieldReceptor + embedding注入を適用する。

### T7-d. 自己認識

3.3節の自己認識取得をperceive経由に変更する。専用のsenseクエリ（「自分の現在の知識状態」）を廃止し、LLMの注意機構が場の信号群から自己認識に関連する情報を選択的に処理する形に変更する。

### 完了条件

- [ ] 対話フローでsenseが一切使われていない
- [ ] 対話フローでtraceが一切使われていない
- [ ] perceive + FieldReceptor + embedding注入で対話が正常に動作する
- [ ] 自己改善パイプラインが新方式で動作する

---

## T8. 統合テスト・検証

**依存:** T3, T7（全タスク完了後）

### T8-a. 覚醒中の対話検証

- ユーザ入力に対して応答が生成される
- 応答品質が従来と同等
- 場への対話経験emitが正常に動作する（内部ログにsignal_id記録）
- 困難度シグナルが正常にemitされる

### T8-b. 睡眠サイクル検証

- 入眠シーケンスが正常に完了する
- sleep_ingestが対話ログをSleepyJeanに転送する
- sleep_ingestが困難度シグナルをhomeworkに変換する（snapshot + contextフィルタ + signal_id逆引き）
- SleepyJeanの夜間サイクルが正常に完走する
- wake_exportがSleepyJeanの出力を場にemitする（traceなし、内部ログ記録）
- Q&AペアがSQLite間で直接転送される
- 覚醒シーケンスが正常に完了する

### T8-c. perceive + FieldReceptor検証

- perceiveの結果がFieldReceptor経由でLLM推論に影響していることの定量検証
- 同一入力に対して、場にSleepyJeanの信号がある状態とない状態で応答が異なること
- 場が空のときでも対話が正常に動作すること（graceful degradation）

### T8-d. trace完全廃止の確認

- コードベース全体でtraceに依存する箇所がゼロであることのgrep確認
- emitにtrace引数を渡すコードが存在しないこと
- Signal定義にtraceフィールドが存在しないこと
- テスト全件PASS

### 完了条件

- [ ] 覚醒中の対話が正常動作
- [ ] 睡眠サイクルが完走
- [ ] perceive + FieldReceptorがLLM推論に影響している
- [ ] traceに依存するコードが一切存在しない
- [ ] テスト全件PASS

---

## 推定工数

| タスク | 推定 | 備考 |
|--------|------|------|
| T1 | 1-2日 | インターフェース変更 + 既存テスト修正 |
| T2 | 0.5-1日 | emit呼び出し箇所の全件修正 |
| T3 | 0.5-1日 | ブリッジ層のsense→snapshot置き換え |
| T4 | 1-2日 | 計測スクリプト作成 + 3候補計測 + 判断 |
| T5 | 1-3日 | T4結果依存。線形なら1日、MLPなら3日 |
| T6 | 2-3日 | 推論パスの大きな変更。調査含む |
| T7 | 1-2日 | 対話フロー + 自己改善パイプライン |
| T8 | 1-2日 | 統合テスト・検証 |
| **合計** | **8-16日** | T5の複雑さとT6の調査結果に大きく依存 |

**クリティカルパス**: T1 → T4 → T5 → T6 → T7 → T8（8-12日）
**並列パス**: T1 → T2 → T3（2-4日、クリティカルパスの裏で完了）
