# 共有状態の場 — Phase 0 インターフェース設計

- **親文書**: bio_ai_architecture.md セクション5
- **作成日**: 2026-03-14
- **ステータス**: 設計草案
- **スコープ**: バックエンド非依存の抽象インターフェース定義

---

## 1. 本文書の位置づけ

上位設計書（bio_ai_architecture.md）セクション5で定められた共有状態の場の設計思想を、実装可能な抽象インターフェースに落とす。本文書はPhase 0の成果物であり、Phase 1（ChromaDB暫定実装）およびそれ以降のバックエンド本実装の双方に対して安定した契約を提供することを目的とする。

**本文書が決めること**:
- 場に対する操作の抽象インターフェース
- 三特性（無指向性・濃度・減衰）のインターフェースレベルでの具体化
- 共通エンコーダの責務境界
- 人間向け可読性レイヤーの構造

**本文書が決めないこと**:
- バックエンドの選定・実装詳細（Phase 1以降）
- 共通エンベディングモデルの具体的な選定
- 減衰パラメータの具体値（運用データに基づいて決定）

---

## 2. 設計哲学 — 場はメッセージバスではない

共有状態の場を設計する上で最も重要な認識は、**場はメッセージバスでもイベントキューでもない**ということである。

従来のシステム間通信（REST、gRPC、pub/sub、メッセージキュー）はいずれも「送信者が受信者またはトピックを指定し、メッセージが消費される」というパラダイムに立つ。場はこれと根本的に異なる。

| 特性 | メッセージバス | 場 |
|---|---|---|
| 宛先 | 送信者が指定（宛先 or トピック） | なし。受信者が自ら選択的に取り込む |
| 消費 | メッセージは消費されると消える | 信号は消費されず、時間経過で減衰する |
| 表現 | 離散的（JSON、Protobuf等） | 連続量（ベクトル空間） |
| 蓄積 | キューに蓄積、順序保証 | 複数の信号が場に重畳し、濃度を形成する |

生体の血液に倣えば、場は**化学的媒体**である。ホルモンは宛先を持たず、血中に放出され、受容体を持つ細胞だけがそれを取り込む。同種のホルモンが複数の腺から放出されれば濃度が上がり、時間が経てば代謝されて濃度が下がる。場のインターフェースはこの性質を計算機上に忠実に再現する。

---

## 3. コア抽象

### 3.1 Signal（信号）

場に存在する情報の最小単位。生体における血中分子に相当する。

```python
@dataclass(frozen=True)
class Signal:
    """場に存在する一つの信号。不変オブジェクト。"""

    # === 本質的属性 ===
    embedding: NDArray[np.float32]   # 非正規化ベクトル。方向=意味、ノルム=濃度
    emitted_at: datetime             # 放出時刻

    # === 付帯情報（ルーティングには使われない） ===
    origin: SignalOrigin             # 放出元の系統識別（デバッグ・統計用）
    trace: str                       # 人間可読な自然言語記述（可読性レイヤー用）

    # === 場が付与 ===
    signal_id: SignalId              # 場が発行する一意識別子
```

**設計判断と根拠**:

- **embeddingは正規化しない**: 上位設計書5.5の方針に従い、ベクトルのノルムが濃度（緊急度・確信度）を担う。「異常を検知した」という信号と「軽微な違和感がある」という信号は方向が似ていてもノルムが異なる。
- **originはルーティングに使わない**: 無指向性の原則（設計原則2）を守るため、originは統計・デバッグ目的に限定する。読み取りAPIでorigin指定によるフィルタリングは提供しない。
- **traceは副産物**: 上位設計書5.3の「通信と可読性の分離」に従い、traceはエージェント間通信に関与しない。人間がデバッグ時に参照するための自然言語記述。

### 3.2 SignalOrigin（放出元情報）

```python
@dataclass(frozen=True)
class SignalOrigin:
    """信号の放出元を識別する付帯情報。"""
    system: str       # "llamarcute-live", "sleepyjean", "immune", "sensory:discord", etc.
    context: str      # 放出の文脈（任意の短い説明）
```

### 3.3 FieldReading（場の読み取り結果）

エージェントが場を感知した結果。個々の信号ではなく、「ある視点から場を観測したときに見える景色」を表す。

```python
@dataclass(frozen=True)
class WeightedSignal:
    """減衰と関連度が適用された信号。"""
    signal: Signal
    relevance: float     # クエリとの意味的関連度 [0, 1]
    decay_factor: float  # 時間減衰係数 (0, 1]
    effective_weight: float  # relevance × decay_factor × norm（最終的な重み）

@dataclass(frozen=True)
class FieldReading:
    """場の観測結果。"""
    signals: list[WeightedSignal]   # effective_weight降順
    observed_at: datetime            # 観測時刻
    query_embedding: NDArray[np.float32]  # どの視点から観測したか
```

**設計判断と根拠**:

`FieldReading`は生体における「血液検査の結果表」に相当する。場そのものの状態ではなく、ある瞬間にある視点から観測した射影である。同じ瞬間でもクエリが異なれば異なるReadingが返る。これは、同じ血中ホルモンでも受容体の種類によって結合するものが異なるのと同じである。

`effective_weight`の算出式: `relevance × decay_factor × ‖embedding‖`。三特性（無指向性→relevance、減衰→decay_factor、濃度→norm）が一つのスカラーに集約される。エージェントはこの重みを自身の処理に組み込むことも、個別の要素を分解して使うこともできる。

---

## 4. 場のインターフェース

### 4.1 概観

```python
class SharedField(Protocol):
    """共有状態の場の抽象インターフェース。"""

    # --- 書き込み ---
    def emit(self, embedding: NDArray[np.float32], origin: SignalOrigin,
             trace: str) -> SignalId: ...

    # --- 読み取り ---
    def sense(self, query: NDArray[np.float32],
              params: SenseParams | None = None) -> FieldReading: ...

    # --- 維持管理 ---
    def purge(self, criteria: PurgeCriteria) -> PurgeResult: ...
    def snapshot(self) -> FieldSnapshot: ...
```

操作は意図的に4つに絞っている。場は単純な媒体であり、複雑なロジックを持つべきではない。知性はエージェント側にある。

### 4.2 emit — 信号の放出

```python
def emit(self, embedding: NDArray[np.float32], origin: SignalOrigin,
         trace: str) -> SignalId
```

信号を場に放出する。宛先の指定は存在しない（無指向性）。

**事前条件**:
- `embedding`は共通エンベディングモデルの出力空間に属すること（セクション5参照）
- `embedding`は正規化しないこと（ノルムが濃度を担う）

**事後条件**:
- 信号は即座に場に存在し、以降の`sense`呼び出しで観測可能になる
- `emitted_at`は場が付与する（呼び出し元のクロックではなく場のクロック）
- 一意な`SignalId`が返される

**emit は fire-and-forget**: 放出元は信号がどのエージェントにどう解釈されるかを知らないし、知る必要がない。この「無関心」が無指向性の本質である。

### 4.3 sense — 場の感知

```python
@dataclass
class SenseParams:
    """感知のパラメータ。"""
    decay_fn: DecayFunction = ExponentialDecay(half_life_hours=24.0)
    max_signals: int = 50
    min_relevance: float = 0.0   # 関連度の下限閾値
    time_horizon: timedelta | None = None  # Noneなら全期間

def sense(self, query: NDArray[np.float32],
          params: SenseParams | None = None) -> FieldReading
```

場を感知し、クエリに関連する信号を減衰・濃度を考慮した重みと共に返す。

**設計上の核心**: senseは「メッセージを受信する」操作ではない。「血液検査をする」操作である。

- 信号はsenseによって消費されない。何度senseしても同じ信号が返る（減衰の進行を除く）。
- 返される信号の選択と重み付けは、クエリ（受容体）と信号（分子）の親和性、信号の濃度（ノルム）、そして時間減衰の三者によって決まる。
- 同一エージェントが異なるクエリでsenseすれば、異なるFieldReadingを得る。

**effective_weight の算出**:

```
effective_weight = relevance(query, signal.embedding) × decay_fn(now - signal.emitted_at) × ‖signal.embedding‖
```

ここで:
- `relevance`: クエリと信号のコサイン類似度。方向の一致度を測る。[0, 1]に正規化（負の類似度は0にクランプ）
- `decay_fn`: 時間減衰関数。放出からの経過時間に応じて (0, 1] の係数を返す
- `‖signal.embedding‖`: ベクトルのL2ノルム。信号の濃度に相当

この三項の積により、「意味的に関連が高く、最近放出され、強い確信で放出された信号」が最も大きな重みを持つ。

### 4.4 purge — 不要信号の除去

```python
@dataclass
class PurgeCriteria:
    """除去基準。複数指定時はOR結合。"""
    max_age: timedelta | None = None          # この期間より古い信号を除去
    below_effective_weight: float | None = None  # この重み未満の信号を除去（基準クエリ不要: decay × normのみで評価）
    custom: Callable[[Signal], bool] | None = None  # カスタム判定関数

@dataclass(frozen=True)
class PurgeResult:
    purged_count: int
    retained_count: int
    purged_signals: list[SignalId]  # 監査用

def purge(self, criteria: PurgeCriteria) -> PurgeResult
```

上位設計書3.2（排泄系）と連携する維持管理操作。SleepyJeanの夜間サイクルで呼び出されることを主な想定とするが、インターフェースレベルでは呼び出し元を制限しない。

`purge`は不可逆操作である。忘却の不可逆性は上位設計書で未解決の問いとして挙げられているが、インターフェースとしては「purgeされた信号は場から完全に消える」と定義する。アーカイブが必要な場合は、purge前にsnapshotを取る運用で対応する。

### 4.5 snapshot — 場の状態保存

```python
@dataclass(frozen=True)
class FieldSnapshot:
    """場の完全なスナップショット。"""
    signals: list[Signal]
    taken_at: datetime
    metadata: dict  # バックエンド固有の付加情報

def snapshot(self) -> FieldSnapshot
```

デバッグ、永続化、purge前のバックアップに使用する。場の全信号を減衰適用なしの生の状態で返す。

---

## 5. 共通エンコーダ — 場の外側の責務

上位設計書5.4（A案）で決定された共通エンベディングモデルは、場のインターフェースの**外側**に位置する。

```python
class FieldEncoder(Protocol):
    """共通エンベディング空間への変換器。場とエージェントの間に位置する。"""

    def encode(self, text: str) -> NDArray[np.float32]: ...
    def dimensionality(self) -> int: ...
```

**責務の分離**:

```
エージェントの内部表現
    ↓ [FieldEncoder.encode]
共通エンベディング空間のベクトル
    ↓ [SharedField.emit / sense]
場
```

- 場は「共通エンベディング空間のベクトル」だけを扱い、エンコーダの存在を知らない
- エージェントは自身の推論モデルとは独立に、FieldEncoderを通じて場と通信する
- FieldEncoderの具体実装（どのモデルを使うか）は差し替え可能。場のインターフェースには影響しない

**将来のアダプタ層への余地**: 上位設計書5.4で「非LLMエージェント加入時にアダプタ層を差し込める余地を残す」と記載されている。FieldEncoderをProtocolとして分離したことで、この余地は自然に確保される。非LLMモジュール（例: システムメトリクス収集）がField Encoderの代わりに独自のアダプタを使って場に接続できる。

---

## 6. 三特性の具体化

上位設計書5.1で定義された三特性が、本インターフェースでどう実現されるかを整理する。

### 6.1 無指向性の伝播

**原則**: 発信者は宛先を指定せず、受信者が選択的に取り込む。

**実現**:
- `emit`に宛先パラメータは存在しない
- `SignalOrigin`はデバッグ用メタデータであり、`sense`の検索条件には使えない
- `sense`の結果を決めるのはクエリ（＝受信者の関心）と信号の意味的関連度のみ

**意図的に排除したもの**:
- トピックベースのルーティング
- origin指定によるフィルタリング
- 特定エージェントへのダイレクトメッセージ

これらが必要に見える場面が出た場合、それは場ではなく別の通信手段（直接的なAPI呼び出し等）で解決すべきである。場に指向性を導入することは設計原則2の破壊に等しい。

### 6.2 濃度

**原則**: ON/OFFではなく、量によって質的に異なる応答を引き起こす。

**実現**:
- ベクトルのL2ノルムが濃度を担う（正規化禁止）
- `effective_weight`の算出にノルムが直接関与する
- 同種の信号（方向が近いベクトル）が複数存在すれば、senseの結果に複数の高重み信号が並ぶ。これは「濃度が高い」状態に相当する

**濃度の二つの側面**:

1. **単一信号の強度**: 一つの信号のノルムが大きい → その信号のeffective_weightが高い
2. **同種信号の蓄積**: 方向が近い信号が多数存在する → senseの結果に類似の高重み信号が複数並ぶ

生体では、この二つは区別されない（分子が多いことと、一つ一つの分子のシグナル強度が高いことは異なるメカニズムだが、受容体にとっての効果は累積的）。本インターフェースでは両者を個別に観測可能な形で返し、集約はエージェント側の責務とする。

> **設計メモ**: 将来的に、FieldReadingに「方向クラスタごとの累積濃度」のようなサマリを含めることは検討に値する。ただしPhase 0では過度な集約ロジックを場に持たせず、エージェント側の自由度を優先する。

### 6.3 時間的残留（減衰）

**原則**: 放出された信号が半減期に従い徐々に減衰し、近い過去の履歴を暗黙的にエンコードする。

**実現**:
- 信号は場に永続的に存在し続ける（purgeされるまで）
- 減衰は**読み取り時に適用**される（書き込み時ではない）
- decay_fnはSenseParamsで指定可能（エージェントごと・クエリごとに異なる減衰を適用できる）

**読み取り時減衰の根拠**:

書き込み時に信号を徐々に縮小する方式（信号のembeddingを定期的に縮退させる）も検討したが、以下の理由で読み取り時減衰を採用する。

- **可逆性**: 信号の原データが保存されるため、異なる減衰パラメータで再観測できる。パラメータ調整期（上位設計書5.5「段階的に調整する前提」）に不可逆な劣化が発生しない
- **エージェント固有の時間感覚**: 神経系（ミリ秒〜秒）と内分泌系（時間〜日）では「最近」の意味が異なる。読み取り時減衰なら、各エージェントが自身の時間スケールに合った減衰を適用できる
- **実装の単純さ**: 場のバックエンドは信号を保存するだけでよく、定期的な更新ジョブが不要

**減衰関数のインターフェース**:

```python
class DecayFunction(Protocol):
    """時間減衰関数。"""
    def __call__(self, elapsed: timedelta) -> float:
        """経過時間から減衰係数 (0, 1] を返す。"""
        ...

class ExponentialDecay:
    """指数減衰。上位設計書5.5で基本方針として記載。"""
    def __init__(self, half_life: timedelta):
        self.half_life = half_life

    def __call__(self, elapsed: timedelta) -> float:
        return 0.5 ** (elapsed / self.half_life)
```

DecayFunctionをProtocolとして定義したことで、指数減衰以外の減衰関数（ステップ減衰、ロジスティック減衰等）も将来的に導入可能。ただし上位設計書で「半減期に従い徐々に減衰」と記載されているため、デフォルトは指数減衰とする。

---

## 7. 人間向け可読性レイヤー

上位設計書5.3の「通信と可読性の分離」を実現する。

### 7.1 設計方針

可読性レイヤーは場のコア機能から完全に分離されたオブザーバーである。場の動作に一切影響を与えず、場を「外から覗き見る」だけの存在。

```python
class FieldObserver(Protocol):
    """場のイベントを外部から観測するオブザーバー。"""
    def on_emit(self, signal: Signal) -> None: ...
    def on_sense(self, query: NDArray[np.float32], reading: FieldReading) -> None: ...
    def on_purge(self, result: PurgeResult) -> None: ...
```

### 7.2 ログ出力例（想定）

```
[2026-03-14 03:22:15] EMIT  origin=sleepyjean context="nightly_consolidation"
  trace: "ユーザとの会話パターンに関する記憶を再構造化した。技術的な質問への応答精度に関する確信度が上昇。"
  ‖embedding‖=2.34  dim=768

[2026-03-14 03:22:16] SENSE  caller=llamarcute-live
  query_trace: "現在の記憶状態と最近の学習内容"
  results: 12 signals, top weight=1.87 (sleepyjean/nightly_consolidation, 1s ago)

[2026-03-14 03:30:00] PURGE  criteria={max_age: 7d}
  purged: 142 signals, retained: 891 signals
```

これが上位設計書で言う「医師の血液検査結果表」に相当する。エージェント間通信はembedding空間で行われるが、人間はこのログを見ることで場の状態を理解できる。

### 7.3 本番環境での抑制

上位設計書5.3に「本番では抑制可能」と記載されている。FieldObserverの接続はオプショナルであり、オブザーバーが未接続の場合、場は一切のログ出力を行わない。

---

## 8. インターフェースが意図的に定めないこと

Phase 0のインターフェースは以下を意図的にスコープ外とする。これらはPhase 1の運用データに基づいて判断すべき事項である。

| 項目 | 理由 |
|---|---|
| 信号の最大保持期間 | 運用データなしに決められない。purgeの運用方針で対応 |
| embeddingの次元数 | 共通エンベディングモデルの選定に依存 |
| 減衰パラメータの具体値 | 上位設計書5.5「段階的に調整する前提」 |
| 場の容量上限 | バックエンドに依存 |
| 同時アクセスの一貫性保証 | Phase 1（三系統の時間スケールが大きく異なるため、厳密な一貫性は不要と予想されるが、検証が必要） |
| senseの結果に対する集約・要約ロジック | エージェント側の責務。場は生データを返す |

---

## 9. Phase 1への橋渡し

ChromaDB暫定実装に向けた指針。

### 9.1 ChromaDBへのマッピング

| 場の概念 | ChromaDBの対応 |
|---|---|
| Signal | Documentとして格納。embeddingは直接保存、metadataにorigin/trace/emitted_at |
| emit | Collection.add |
| sense | Collection.query（cosine similarity） + 読み取り時減衰の後処理 |
| purge | Collection.delete（metadata条件） |

### 9.2 Phase 1で収集すべきデータ

Phase 1の目的は「共有状態の使用パターンを収集する」こと（上位設計書8章）。以下の観点でデータを収集し、本格的なバックエンド設計に活かす。

- 各系統のemit頻度・ベクトルの分布（次元ごとの分散、ノルムの分布）
- senseのクエリパターン（どの系統が何をどの頻度で問い合わせるか）
- 信号の有効寿命（purge時に実際に除去される信号の年齢分布）
- 同種信号の蓄積パターン（濃度の集約ロジックが必要になるか否か）
- 読み取り時減衰のパラメータ感度（異なるhalf_lifeでの結果の違い）

---

## 10. 未解決の問い

1. **senseのコスト問題**: llamarcute-liveがリアルタイム対話のたびにsenseを呼ぶ場合、ベクトル検索＋減衰計算のレイテンシが応答速度に影響する。キャッシュ層やpre-computationが必要か。これは代謝系（リソース管理）とも接点がある。

2. **場の「温度」**: 減衰パラメータを全信号に一律に適用するのか、信号の種類（origin.system等）によって異なる半減期を持たせるのか。生体では物質ごとに代謝速度が異なる。ただし、これはPhase 1の運用データを見てから判断すべきである。

3. **因果の追跡可能性**: 無指向性の原則を守りつつ、デバッグ時に「この応答はどの信号に影響されたか」を追跡したい場合がある。FieldObserverのon_senseログで部分的に対応可能だが、十分か。

4. **場の初期状態**: システム起動時、場は空である。これは生体の「出生」に相当するが、最初期にSleepyJeanやllamarcute-liveが何も感知できない「感覚遮断」状態が問題にならないか。初期信号のシーディングが必要か。

5. **濃度の集約**: セクション6.2で述べた「同種信号の蓄積」を場のレベルで集約すべきか、エージェント側に委ねるか。Phase 1のデータを見て判断する。

---

## 変更履歴

| 日付 | 変更内容 |
|---|---|
| 2026-03-14 | 初版作成 |

---

*本文書はPhase 0の設計草案であり、Phase 1の運用知見に基づいて継続的に更新される。*
