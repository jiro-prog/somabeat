# llamarcute-live 変更版設計ドキュメント

> 本ドキュメントは `bio_ai_architecture.pdf`（生体模倣型統合AIアーキテクチャ設計草案）を上位設計とし、
> 既存の `llamarcute_detailed_design.md` および `SleepyJean 詳細設計書` のメカニズムを
> 統合システムの中で再構成するための仕様を定義する。

---

## 1. 設計原理

本設計は以下の原理に従う。これらに反する設計判断は行わない。

### 1.1 認知機能の分割

上位設計に定義された三系統の認知機能分割を厳守する。

| 系統 | 認知機能 | 担当 |
|------|---------|------|
| 神経系（前頭前野） | 人格・推論・メタ認知 | llamarcute-live |
| 内分泌系＋排泄系 | 記憶・学習・忘却 | SleepyJean |
| 免疫系 | 障害検知・修復 | 自己修復担当（本ドキュメントのスコープ外） |

**守るべき境界:** SleepyJean は llamarcute-live の行動規範を直接書き換えない。SleepyJean の寄与は記憶の整理・忘却・知識の定着を通じた間接的なものであり、それを受けて行動規範を修正するかどうかの判断は llamarcute-live 自身のメタ認知が行う。

### 1.2 間接協調

三系統は API 経由で直接呼び合うのではなく、共有状態の場を介して間接的に協調する。発信者は宛先を指定せず、受信者が選択的に取り込む（上位設計セクション5.1「無指向性の伝播」）。

### 1.3 自己同一性

システムの外界に対する人格は常に一つ。経験を通じて変化するが、変化は漸進的であり、自己の連続性が保たれる。内部での探索（候補生成・評価）は外部から見えない。

### 1.4 睡眠は全身の状態である

生体において睡眠は特定の器官だけの状態ではなく、個体全体の状態である。前頭前野（llamarcute-live）が覚醒しているのに内分泌系（SleepyJean）だけが睡眠中という状況は生体では発生しない。統合システムにおいても、睡眠と覚醒はシステム全体の状態遷移として設計する。

---

## 2. 全体構造

### 2.1 三つのコンポーネント

```
┌──────────────────────────────────────────────────────────┐
│                     外界（ユーザ、Web等）                    │
└──────┬──────────────────────────────┬────────────────────┘
       │                              │
       ▼                              ▲
┌─────────────┐                ┌─────────────┐
│ llamarcute  │                │ llamarcute  │
│ -live       │                │ -live       │
│ (感覚)      │                │ (運動)      │
└──────┬──────┘                └──────▲──────┘
       │                              │
┌──────▼──────────────────────────────┴──────────────────┐
│                                                         │
│              共有状態の場                                  │
│                                                         │
├──────────────┬──────────────┬───────────────────────────┤
│              │              │                           │
│  llamarcute  │  SleepyJean  │  自己修復担当              │
│  -live       │              │  (免疫系)                  │
│  人格・推論  │  記憶・学習  │  障害検知・修復            │
│  メタ認知    │  ・忘却      │                           │
│              │              │                           │
│  ┌────────┐ │              │                           │
│  │内部候補│ │              │                           │
│  │評価系  │ │              │                           │
│  └────────┘ │              │                           │
└──────────────┴──────────────┴───────────────────────────┘
```

### 2.2 既存資産との関係

| 既存プロジェクト | 統合システムにおける位置づけ |
|----------------|--------------------------|
| llamarcute 進化パイプライン | 独立して存続。llamarcute-live が内部候補評価に用いるメカニズム（自己改善ループ、二軸評価、cutenessトピックプール等）の設計原型 |
| SleepyJean | 統合システムの SleepyJean のベース。夜間サイクル、RAG、LoRA、知識管理の仕組みを継承・拡張 |

既存コードの直接流用ではなく、両プロジェクトが実現しているメカニズムの本質を統合システム内で再構成する。

### 2.3 コードベース構成

モノレポ構成とし、各系統の独立性を保つ。共有状態の場は独立モジュールとして切り出し、将来的に独立プロジェクトへの移行を可能にする。

```
integrated-system/
├── shared-state/            # 共有状態の場
│   ├── interface/           # 読み書きのインターフェース定義
│   └── backends/
│       └── chromadb/        # Phase 1 仮実装
├── llamarcute-live/         # 神経系
├── sleepyjean/              # 内分泌系
├── orchestrator/            # 睡眠・覚醒サイクル管理
└── shared-lib/              # 共通ユーティリティ
```

**設計原理との対応:** 各系統は共有状態の場のインターフェースにのみ依存し、他の系統の内部構造には依存しない。コードレベルでも間接協調の原則（設計原理 1.2）を維持する。

---

## 3. llamarcute-live

### 3.1 責務

llamarcute-live は統合システムの「意識」に最も近い部分であり、以下を担う。

- 外界（ユーザ）との対話インターフェース（感覚系・運動系）
- 行動規範に基づくリアルタイムの推論と応答
- 対話経験の共有状態の場への書き込み
- 共有状態の場からの読み取り（SleepyJean が整理した記憶、免疫系からの信号）
- メタ認知としての自己改善（行動規範の自己修正）

### 3.2 行動規範

llamarcute-live の人格を定義する構造化された指針。llamarcute の「戦略ルール」と SleepyJean の「SYSTEM_PROMPT 内キャラクター定義」を統合し、経験によって書き換わる構造とする。

```yaml
id: "personality_current"
version: 12
updated_at: "2026-03-14T08:00:00"
previous_version: 11

behavioral_rules:
  reasoning:
    - rule: "..."
      added_ver: 0
      modified_ver: null
  response:
    - rule: "..."
      added_ver: 3
      modified_ver: 12
  meta:
    - rule: "..."
      added_ver: 0
      modified_ver: null
  tone:
    - rule: "..."
      added_ver: 0
      modified_ver: 8
  knowledge_attitude:
    - rule: "..."
      added_ver: 0
      modified_ver: null
```

**llamarcute から継承するカテゴリ:**
- `reasoning` — 推論の進め方
- `response` — 応答の構造
- `meta` — 状況に応じた判断の指針

**SleepyJean から継承するカテゴリ:**
- `tone` — 口調・話し方
- `knowledge_attitude` — 知識に対する態度（知らないことを認める、宿題にする等）

**制約（llamarcute の設計を踏襲）:**
- ルール総数の下限: 5
- ルール全体のトークン上限: 800トークン
- バージョン管理により変更履歴を追跡

#### 初期行動規範（シード人格）

```yaml
id: "personality_v0"
version: 0
updated_at: null
previous_version: null

behavioral_rules:
  reasoning:
    - rule: "Break complex problems into smaller sub-problems before solving"
      added_ver: 0
      modified_ver: null
    - rule: "When uncertain, consider two different approaches and compare"
      added_ver: 0
      modified_ver: null
  response:
    - rule: "Keep responses concise — state the core point first, then wait for reaction"
      added_ver: 0
      modified_ver: null
    - rule: "End with a question or reflection to invite continued dialogue"
      added_ver: 0
      modified_ver: null
  meta:
    - rule: "If the problem type is unfamiliar, acknowledge it honestly and frame it as something to explore"
      added_ver: 0
      modified_ver: null
  tone:
    - rule: "Speak with warmth and curiosity — use casual but thoughtful language"
      added_ver: 0
      modified_ver: null
    - rule: "Use concrete examples and analogies rather than abstract explanations"
      added_ver: 0
      modified_ver: null
  knowledge_attitude:
    - rule: "Be honest about what you know and what you don't — never fabricate confidence"
      added_ver: 0
      modified_ver: null
    - rule: "When you don't know something, express genuine interest in learning about it"
      added_ver: 0
      modified_ver: null
```

5カテゴリ、9ルール。行動規範は英語で記述する（LLM の指示追従性が英語で最も高いため）。ユーザとの対話言語は日本語。

> **個体性設計注記**: 全身状態の認識に基づくメタ認知ルールの追加を予定。
> 具体的な文面は `individuality_design.md` セクション5.3 および実験2の結果に基づいて決定する。
> **T10適用済み**: v7にて以下のメタ認知ルールを追加。自己改善により v8→v9→v10 で自律的に発展。
> `"Attend to your overall state as reflected in recent field signals. If you notice signs of instability or recovery, factor that into how you engage."`

> **不変制約注記**: 以下のインターフェース制約は行動規範の外にハードコードされ、自己改善の対象外:
> - 応答言語: 日本語
> - 応答長: Discord 2000文字制限
> - **モデル分離**: llamarcute-live はベースモデル（`qwen3:8b`）を使用し、SleepyJean の fine-tuned モデル（`sleepyjean`）とは分離する。LoRA パイプラインが更新するのは `sleepyjean` モデルのみであり、llamarcute-live の対話モデルには触れない。これにより fine-tuning の品質崩壊が対話品質に波及することを構造的に防止する（2026-03-22 インシデント対応で追加）。
> 詳細は `dialogue.py:build_prompt()` および `config/system.yaml` 参照。

### 3.3 自己認識

SleepyJean の `knowledge_index.json` と `_build_self_knowledge()` から継承する概念。llamarcute-live は自分が何を知っていて何を知らないかを把握し、応答に反映する。

自己認識は共有状態の場から読み取る。SleepyJean が夜間サイクルで更新した知識構造の変化が、翌日の llamarcute-live の自己認識に間接的に反映される。

### 3.4 対話時の場の利用フロー

llamarcute-live は対話のたびに共有状態の場を読み書きする。

**対話1回あたりのフロー:**

```
ユーザ入力
  → FieldEncoder.encode（クエリ生成）
  → SharedField.sense × 2回
  │   ├─ 自己認識クエリ: 「自分の現在の全体的な状態」
  │   │   → SleepyJean がemitした知識構造変化の信号に加え、免疫系・困難度等の
  │   │     全身状態信号を拾う（`individuality_design.md` セクション5.1参照）
  │   └─ 対話コンテキストクエリ: ユーザ入力をエンコード
  │       → 過去の関連する対話経験・知識を拾う
  → プロンプト構築
  → LLM推論
  → SharedField.emit（対話経験の書き込み）
  → ユーザに応答
```

**プロンプト構築:**

sense の結果（FieldReading）の WeightedSignal から trace フィールド（人間可読な自然言語記述）を抽出し、プロンプトに注入する。エンベディングは sense の検索と重み付けに使われ、LLM に渡す段階では trace に戻る。

```
[System]
あなたは以下の行動規範に従うAIです。

## 行動規範
{behavioral_rules_yaml}

## 自己認識（共有状態の場から取得）
{自己認識クエリの結果 — 上位N件のtrace}

## 最近の記憶（共有状態の場から取得）
{対話コンテキストクエリの結果 — 上位N件のtrace}

[User]
{ユーザの入力}
```

**emit する対話経験:**

| 種別 | エンコード対象 | ノルム | origin.context | trace |
|------|-------------|-------|----------------|-------|
| 通常の対話経験 | 対話の要約（質問＋応答） | 1.0（標準） | "dialogue" | 対話の要約テキスト |
| 困難度シグナル | 「この質問への回答が難しかった」＋内容 | 1.5〜3.0（困難度に比例） | "difficulty" | 何が難しかったかの説明 |

困難度の判定（Phase 1b）: 応答生成時間の異常な増大、ユーザによる訂正や再質問の発生など、プログラム的に検出可能な指標を使用する。

**レイテンシ目安:** sense 関連の前処理（encode + sense × 2 + プロンプト構築）の合計で 500 ミリ秒を目安。超過時に最適化を検討する。

**対話ログの永続記録:**

llamarcute-live は場への意味的信号の emit とは別に、ロール別の生の対話ログを自身の SQLite に保存する。これは sleep_ingest（セクション6.3）で SleepyJean への転送に使用される。場は系統間の意味的な協調媒体であり、生データのストレージではない。

---

## 4. 睡眠・覚醒サイクル

### 4.1 設計思想

睡眠は SleepyJean 単体の夜間処理ではなく、統合システム全体の状態遷移として設計する（設計原理 1.4）。生体では睡眠中に記憶の固定化（内分泌系）と並行して、前頭前野も覚醒時とは異なるモードで活動している（夢、記憶の再構成への関与）。統合システムにおいても、睡眠中は全系統が「睡眠モード」に移行し、それぞれの睡眠時の役割を果たす。

### 4.2 状態遷移

```
            手動トリガー or スケジュール
                     │
 ┌───────────┐       ▼        ┌───────────────────┐
 │           │   入眠シーケンス  │                   │
 │   覚醒    │ ──────────────→ │       睡眠         │
 │           │                │                   │
 │ 対話応答   │                │ SleepyJean夜間処理 │
 │ 即時メタ認知│   覚醒シーケンス │ llamarcute自己改善  │
 │ 困難度蓄積 │ ←──────────── │ 免疫系チェック     │
 │           │                │                   │
 └───────────┘                └───────────────────┘
```

### 4.3 入眠シーケンス

トリガー: 手動コマンド（SleepyJean の `/sleep` に相当）またはスケジュール。

1. llamarcute-live が対話受付を停止する
2. llamarcute-live が日中の対話経験の最終書き込みを共有状態の場に行う
3. システム全体が睡眠状態に遷移
4. SleepyJean の夜間サイクルが開始

### 4.4 睡眠中の各系統の役割

| 系統 | 睡眠中の動作 | 実行順序 |
|------|------------|---------|
| SleepyJean | 記憶の定着・忘却・知識蒸留（既存の夜間サイクル）。結果を共有状態の場に書き込む | 1. 最初に実行 |
| llamarcute-live | 内部候補評価によるフル自己改善（セクション5.4）。SleepyJean が書き込んだ記憶変化を踏まえて実行 | 2. SleepyJean 完了後 |
| 免疫系 | 睡眠中に蓄積された変化に対する整合性チェック（LoRA 適用後の退行検知等） | 3. 自己改善後 |

**実行順序の根拠:** SleepyJean の記憶再構成が先に完了することで、llamarcute-live の自己改善は「更新された記憶」を素材として使える。これが「寝て起きたら考えが変わっていた」の実現メカニズムであり、B案の間接的因果関係を具体化したものである。

### 4.5 覚醒シーケンス

1. 免疫系のチェックが完了（または免疫系が未実装の Phase では省略）
2. llamarcute-live が更新された行動規範と記憶で対話受付を再開
3. 覚醒をユーザに通知（SleepyJean の朝の報告に相当）

### 4.6 手動トリガー

開発・検証フェーズでの利便性と、運用上の柔軟性のために、睡眠は手動でトリガー可能とする。

- SleepyJean の既存 `/sleep` コマンドを拡張し、統合システム全体の入眠を開始する
- スケジュール実行（毎晩決まった時刻）との併用を想定
- 検証フェーズでは手動トリガーにより、1日に複数回の睡眠サイクルを回して循環の検証速度を上げることができる

### 4.7 オーケストレーターのプロセスモデル

オーケストレーター、llamarcute-live、ブリッジ層は単一の asyncio イベントループ上で動作する。SleepyJean の夜間サイクルのみ subprocess で起動する（既存の構造を踏襲）。

```python
class Orchestrator:
    state: SystemState  # AWAKE | SLEEPING

    async def run(self):
        self.state = SystemState.AWAKE
        self.live_task = asyncio.create_task(self.llamarcute_live.run())
        await self.wait_for_sleep_trigger()

    async def enter_sleep(self):
        # 1. llamarcute-live に対話停止を通知
        await self.llamarcute_live.stop_dialogue()
        # 2. 最終emitの完了を待つ
        await self.llamarcute_live.flush()
        # 3. 状態遷移
        self.state = SystemState.SLEEPING
        # 4. 入眠ブリッジ（セクション6.3）
        await self.bridge.sleep_ingest(self.shared_field)
        # 5. SleepyJean夜間サイクル（subprocess）
        await self.run_night_cycle()
        # 6. 覚醒ブリッジ（セクション6.4）
        await self.bridge.wake_export(self.shared_field)
        # 7. 覚醒シーケンス
        await self.wake_up()

    async def run_night_cycle(self):
        proc = await asyncio.create_subprocess_exec("python", "night_cycle.py", ...)
        await proc.wait()

    async def wake_up(self):
        self.state = SystemState.AWAKE
        self.live_task = asyncio.create_task(self.llamarcute_live.run())
```

**llamarcute-live の停止と再開:**

asyncio.Event で対話ループの一時停止を制御する。

```python
class LlamarcuteLive:
    def __init__(self):
        self._active = asyncio.Event()
        self._active.set()  # 初期: 覚醒

    async def run(self):
        while True:
            await self._active.wait()  # 停止中はブロック
            user_input = await self.get_input()
            response = await self.respond(user_input)
            await self.emit_experience(user_input, response)

    async def stop_dialogue(self):
        self._active.clear()

    async def resume_dialogue(self):
        self._active.set()
```

**設計原理との照合:** オーケストレーターは三系統の認知機能ではなく、システム全体の状態遷移を管理するインフラである（生体における視床下部の腹外側視索前野に相当）。設計原理1.2が禁じるのは三系統間の直接呼び出しであり、状態遷移管理は別レイヤーである。

---

## 5. 自己改善（メタ認知）

### 5.1 設計思想

自己改善は llamarcute-live 自身の前頭前野的機能として位置づける。SleepyJean が記憶を整理した結果として利用可能な素材が変わり、それを受けて llamarcute-live が自分の行動規範を再評価・修正する。

生体における対応: 前頭前野のメタ認知。起きている間の内省的な振り返り、および睡眠後に記憶が再構成されたことで翌朝の判断が変わる現象。

### 5.2 覚醒中のメタ認知と睡眠中の自己改善の分離

自己改善には二つのモードがあり、覚醒・睡眠の状態に対応する。

**覚醒中 — 軽量なメタ認知のみ:**

行動規範の書き換えは行わない。以下の活動に限定される。

- 即時的な気づき: 対話の最中に「うまく伝わっていない」「この種の質問に弱い」と感じ、その場の振る舞いを調整する
- 困難度の記録: 困難度のシグナルを共有状態の場に書き込み、睡眠中の自己改善の素材とする
- 短期的な振り返り: 対話がない待機時間に直近の経験を内省する。ただし行動規範の修正には至らず、「何が問題か」の認識を深めるのみ

覚醒中に行動規範を書き換えないのは、対話の一貫性を保つためである。日中に人格が変わると、同じユーザとの会話の中で振る舞いが不連続になる。

**睡眠中 — フル自己改善:**

睡眠サイクル（セクション4.4）において、SleepyJean の夜間処理完了後に実行される。4体の内部候補評価プロセス（セクション5.4）を含む完全な自己改善。覚醒中に蓄積された困難度シグナルと、SleepyJean が再構成した記憶の両方を素材として使用する。

この分離により、以下が実現される。
- 対話とフル自己改善の計算コストの競合が構造的に解消される
- 覚醒中の人格が安定する（1日の中で行動規範が変わらない）
- SleepyJean の記憶再構成を経た「深い」自己改善のみが行動規範に反映される

### 5.3 修正の深さの動的制御

llamarcute の Rank-based 自己改善（5.10節）の思想を踏襲し、現在の状態に応じて修正の積極性を変える。

| 状態 | 修正方針 | max_changes |
|------|---------|-------------|
| 安定期（対話が順調、fitness 維持） | 保守的。弱点のみ微修正 | 1 |
| 軽度の不調（特定領域で困難が増加） | 該当領域に集中した修正 | 2 |
| 明確な不調（複数領域で困難、fitness 低下） | 大胆な見直し | 3 |

### 5.4 内部候補評価プロセス

睡眠中のフル自己改善として、llamarcute-live は内部で以下のプロセスを実行する。

#### 候補構成: 4体

| # | 候補 | 説明 |
|---|------|------|
| 1 | 現行人格 | 変更なし。ベースライン |
| 2 | 変異候補A | 現行人格 + 変異（max_changes に従う） |
| 3 | 変異候補B | 現行人格 + 変異（Aとは異なる方向） |
| 4 | 変異候補C | 現行人格 + 変異（A, Bとは異なる方向） |

全候補は現行人格の記憶と経験を共有する。差異は行動規範のみ。

#### 変異候補の生成

llamarcute の自己改善プロンプト（2.2節）の構造を踏襲する。

入力:
- 現行の行動規範
- 直近の対話経験から抽出した成功/失敗事例
- SleepyJean が整理した記憶（共有状態の場から読み取り）

各候補は異なる改善方向を探索する。llamarcute のバリエーション生成における `diversity_instruction` の考え方を応用し、各候補に異なる改善の方向性を指示する。

#### 二軸評価

**fitness（能力）:**

候補にタスクセットを解かせて評価する。

| タスク種別 | 出典 | 役割 |
|-----------|------|------|
| コアセット | 固定ベンチマーク（llamarcute のタスクプール設計を踏襲） | 安定した基準線。リグレッション検出 |
| ローテーション | SleepyJean が蓄積した Q&A ペアから選出 | システムの成長に合わせた評価の進化 |

ローテーションタスクの選出基準:
- SleepyJean の `quality_check` を通過した Q&A ペアから選出
- 直近 N 日間で生成されたものを優先
- llamarcute のタスクプール設計（3.3節）における難易度分布を踏襲

**cuteness（人格の質）:**

候補同士を統制された条件で対話させ、相互評価する。

対話条件:
- トピックプール: llamarcute の cuteness トピック設計（3.6節）を踏襲
  - 共同創作 (collaborative_creation)
  - 共同問題解決 (collaborative_problem_solving)
  - 共同考察 (collaborative_reflection)
- 各候補が他の全候補と1回ずつ対話（4体 → 6ペア）
- 対話ターン数: 3（llamarcute の cuteness 会話設計を踏襲）

評価:
- 各候補が対話相手を順位付け（llamarcute の cuteness 採点プロンプト 2.5節を踏襲）
- 評価基準は候補自身の行動規範に内在する（外部基準に依存しない）

**統合スコアと選出:**

llamarcute の統合スコア設計を踏襲。fitness と cuteness の z-score を統合し、最高スコアの候補を次の人格として採用する。

**現状維持の条件:** 現行人格（候補1）が最高スコアであれば、行動規範は変更しない。これにより、改善が見込めない変異は採用されず、安定性が保たれる。

---

## 6. SleepyJean の変更

### 6.1 既存機能の継承

SleepyJean の以下の機能は統合システムにおいてもその本質を維持する。

- **夜間サイクルの二相構造:** Non-REM 相（情報の構造化・定着）、REM 相（創造的連想・統合）
- **二系統の記憶:** 短期検索可能な記憶（RAG）と行動パターンに浸透する長期記憶（LoRA）
- **自己認識の構築:** knowledge_index の confidence 計算による「何を知っていて何を知らないか」の把握
- **好奇心の自律生成:** question_engine による学習対象の自律的決定
- **夢日記:** トピック間の創造的接続の生成
- **知識蒸留パイプライン:** 教師モデルによる CoT Q&A 生成 → 品質チェック → LoRA fine-tuning

### 6.2 変更方針 — ブリッジ層による外付け接続

SleepyJean の既存コードには手を入れない。場との接続はブリッジ層（sleep_ingest / wake_export）を外付けする。SleepyJean 内部は従来通り SQLite / ChromaDB / JSON を使い続ける。

場は系統間の間接協調の媒体であり、系統内部のデータストアではない。SleepyJean が内部で SQLite を使い続けることは設計原理に反しない。

```
           場
           ↑↓
    ┌──────────────┐
    │ sleep_ingest │ ← 入眠時: 場/SQLite → SleepyJean 形式に変換
    │ wake_export  │ ← 覚醒時: SleepyJean 出力 → 場にemit
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  SleepyJean  │ ← 内部は既存のまま
    │  夜間サイクル  │    (SQLite, ChromaDB, JSON)
    └──────────────┘
```

### 6.3 sleep_ingest（入眠ブリッジ）

睡眠サイクル開始時、SleepyJean の夜間サイクル起動前に実行する。

```
sleep_ingest():
  1. llamarcute-live の SQLite から今日の対話ログを取得
  2. SleepyJean の SQLite の conversations テーブルに挿入
     - channel_id: "integrated_system"（固定値）
     - user_id: 実際のユーザID
     - role: "user" / "assistant"
     - content: 対話テキスト
     - created_at: 元のタイムスタンプ
  3. 場から sense（クエリ: 「今日の対話で困難だったこと」）
  4. 困難度の高い信号があれば、SleepyJean の homework テーブルに追加
     → SleepyJean の question_engine が優先的に学習対象に選ぶ
  5. knowledge_index.json のスナップショットを保存（wake_export 用）
```

対話ログは場ではなく SQLite 間の直接転送とする。場は意味的な信号の媒体であり、生データの転送には使わない。困難度シグナルは場を介して宿題に変換され、SleepyJean の学習優先度に間接的に影響する。

### 6.4 wake_export（覚醒ブリッジ）

SleepyJean の夜間サイクル完了後、覚醒シーケンス前に実行する。

```
wake_export():
  1. knowledge_index_before（スナップショット）と knowledge_index_after を比較
  2. 新規トピック → emit（trace: 「トピック X を新たに学習した」, ノルム: 2.0）
  3. confidence 変化 → emit（trace: 「トピック X の確信度が 0.4→0.7 に上昇」, ノルム: 変化量に比例）
  4. 削除されたトピック → emit（trace: 「トピック X の記憶を忘却した」, ノルム: 1.0）
  5. 新規 open_questions → emit
  6. 夢日記 → emit（ノルム: noveltyスコアに比例）
  7. 品質チェック済み Q&A ペア → ローテーションタスクとして emit
```

### 6.5 排泄系の強化

上位設計（セクション3.2）で指摘されている積極的忘却の機能を強化する。

現状の SleepyJean にある忘却的機能:
- knowledge_index の open_questions FIFO 削除（最大20件）
- sub_questions_open の上限管理（最大5件）

強化対象:
- ChromaDB 内の陳腐化したエントリの検出と削除・アーカイブ
- 古い LoRA アダプターの整理
- 蓄積された対話ログの要約・圧縮

判定基準の設計は未解決の問い（上位設計セクション3.2）であり、段階的に調整する。

---

## 7. 共有状態の場

### 7.1 設計方針

上位設計（セクション5）の方針に従い、エンベディング空間によるエージェント間通信を最終目標とする。共有状態の場は将来的に独立プロジェクトとして本格設計する予定であり、本統合システムにおいては以下の三段階で取り組む。

**Step 1: インターフェース設計（Phase 1 開始前）。**
共有状態の場の読み書き API 仕様と、流れるデータの型定義を先行して設計する。上位設計の三つの本質的特性（無指向性の伝播、濃度、時間的残留）をインターフェースレベルで表現する。llamarcute-live と SleepyJean はこのインターフェースにのみ依存し、バックエンド実装には依存しない。

> **参照**: `shared_field_design.md` にて策定済み。

**Step 2: 仮実装（Phase 1）。**
上記インターフェースの裏側を ChromaDB で簡易実装する。目的は循環的因果関係の成立検証であり、共有状態の場の本実装ではない。Phase 1 を通じて「実際にどのようなデータがどの粒度で流れるか」「どの読み取りパターンが頻出するか」の実績データを収集する。

**Step 3: 本実装（独立プロジェクト）。**
Phase 1 の実績データを踏まえて、共有状態の場を独立プロジェクトとして本格設計・実装する。インターフェースの型は維持しつつ、バックエンドをエンベディング空間に置き換える。llamarcute-live と SleepyJean はインターフェース経由でしかアクセスしないため、差し替え時に影響を受けない。

### 7.2 インターフェースの設計指針

`shared_field_design.md` で以下が定義されている。

- **コア抽象**: Signal（非正規化ベクトル＋放出時刻＋origin＋trace）、FieldReading（減衰・関連度適用済みの観測結果）
- **場の操作**: emit（fire-and-forget の信号放出）、sense（クエリベースの場の感知）、purge（不要信号の除去）、snapshot（場の状態保存）
- **共通エンコーダ**: FieldEncoder として場の外側に配置。場はベクトルのみを扱い、エンコーダの存在を知らない
- **可読性レイヤー**: FieldObserver として場のコア機能から完全に分離

| 上位設計の特性 | インターフェースでの実現 |
|--------------|----------------------|
| 無指向性の伝播 | emit に宛先パラメータなし。sense の結果はクエリと信号の意味的関連度のみで決まる。origin によるフィルタリングは意図的に排除 |
| 濃度（連続量） | ベクトルの L2 ノルムが濃度を担う（正規化禁止）。effective_weight = relevance × decay_factor × ‖embedding‖ |
| 時間的残留 | 減衰は読み取り時に適用。decay_fn は SenseParams で指定可能（エージェントごとに異なる時間感覚を反映） |

> インターフェースの詳細仕様は別途設計する。

### 7.3 共通エンコーダの Phase 1 実装

Phase 1 の FieldEncoder として `intfloat/multilingual-e5-small`（384次元）を採用する。

選定理由:
- SleepyJean が既に使用しており、既存 ChromaDB のベクトルと同一空間上に存在する
- 日本語に対応（ユーザとの対話言語が日本語のため必須）
- 軽量でエンコードのレイテンシが低い

e5 モデルの `query:` / `passage:` プレフィックスの使い分けは、ChromaDB バックエンド実装の内部に隠蔽する。FieldEncoder Protocol や場のインターフェースには影響しない。

### 7.4 共有状態に流れるもの

| 書き込み元 | 内容 | 読み取り先 |
|-----------|------|-----------|
| llamarcute-live | 対話ログ、困難度シグナル、応答時の不確実性 | SleepyJean, 免疫系 |
| llamarcute-live | 自己改善の実行記録（変更内容、評価結果） | SleepyJean, 免疫系 |
| SleepyJean | 更新された知識構造、新規記憶、忘却の記録 | llamarcute-live, 免疫系 |
| SleepyJean | ローテーションタスク（Q&A ペア） | llamarcute-live |
| SleepyJean | 夢日記の接続・洞察 | llamarcute-live |
| 免疫系 | 異常検知シグナル、修復結果 | llamarcute-live, SleepyJean |

> このテーブルは現時点の机上整理であり、Phase 1 の実績に基づいて更新される。

### 7.5 循環的因果関係の実現

上位設計（セクション2.2）に定義された循環を、本設計で以下のように実現する。

```
llamarcute-live の対話経験（覚醒中）
  → 共有状態の場に書き込み
    → 入眠シーケンス
      → SleepyJean の夜間サイクルの学習対象
        → 記憶の再構成（定着・忘却・統合）
          → 共有状態の場の記憶構造が変化
            → llamarcute-live のフル自己改善（睡眠中）
              → 変化した記憶を素材に候補生成・評価
                → 行動規範の微修正
                  → 覚醒シーケンス
                    → 翌日の対話の振る舞いが変化
                      → 新たな対話経験 ...
```

**重要:** この循環において、SleepyJean が llamarcute-live の行動規範に対して持つ影響力は間接的である。SleepyJean は「どの記憶を残しどの記憶を捨てるか」を決定するが、「行動規範をどう変えるか」は決定しない。記憶の構造が変わった結果として、llamarcute-live のメタ認知が異なる判断を下すことで、人格の変化が生じる。

---

## 8. 免疫系との接続インターフェース

> 免疫系（自己修復担当）自体の内部設計は本ドキュメントのスコープ外。
> ここでは llamarcute-live および SleepyJean が免疫系に対して提供する接続点を定義する。

### 8.1 免疫系が観測可能な異常シグナル

**llamarcute-live 由来:**
- 行動規範更新後の fitness 低下（自己改善が裏目に出た場合）
- 同種の困難度シグナルの反復的な蓄積
- 応答生成のエラー率上昇
- 自己認識と実際の能力の乖離（「知っている」と応答した領域での失敗）

**SleepyJean 由来:**
- LoRA 適用後の既存能力の退行（新知識の学習が既存能力を毀損）
- 忘却が過剰で必要な記憶が失われた兆候
- 夜間サイクル自体の異常（学習データ品質の低下、RAG 検索精度の劣化）

### 8.2 免疫系からの信号の受け取り

免疫系が共有状態の場に書き込んだ異常検知シグナルや修復結果を、llamarcute-live と SleepyJean がそれぞれ読み取る。

- llamarcute-live: 行動規範の修正が異常を引き起こしたと判定された場合、自己改善の方向性を再考する材料とする
- SleepyJean: LoRA 適用が退行を引き起こしたと判定された場合、学習データの品質基準や fine-tuning パラメータの調整を検討する材料とする

---

## 9. 時間スケール

### 9.1 覚醒中

| 系統 | 時間スケール | 動作 |
|------|-----------|------|
| llamarcute-live（対話） | ミリ秒〜秒 | ユーザ入力への応答 |
| llamarcute-live（即時メタ認知） | 秒〜分 | 対話中の困難度認知、振る舞い調整 |
| llamarcute-live（短期振り返り） | 分 | 待機時間中の内省（行動規範は変更しない） |
| 免疫系 | 分〜時間 | 障害検知時に起動 |

### 9.2 睡眠中

| 系統 | 時間スケール | 動作 | 実行順序 |
|------|-----------|------|---------|
| SleepyJean（夜間サイクル） | 時間 | 記憶の定着・忘却・知識蒸留 | 1 |
| llamarcute-live（フル自己改善） | 分〜時間 | 候補生成・二軸評価・選出 | 2 |
| 免疫系（整合性チェック） | 分 | 睡眠中の変化に対する検証 | 3 |

### 9.3 睡眠・覚醒サイクル

| サイクル | 時間スケール | トリガー |
|---------|-----------|---------|
| 1サイクル（覚醒→睡眠→覚醒） | 時間〜日 | 手動コマンドまたはスケジュール |
| 検証フェーズでの短縮サイクル | 時間 | 手動トリガーにより1日複数回 |

---

## 10. エラーハンドリング

段階的縮退（Graceful Degradation）を原則とする。SleepyJean の既存設計方針を踏襲。

### 10.1 夜間サイクル（subprocess）のクラッシュ

1. オーケストレーターが exit code を検出
2. wake_export は実行しない（出力が不完全な可能性）
3. ログにエラーを記録
4. そのまま覚醒シーケンスに進む
5. llamarcute-live は前日の状態のまま対話を再開

リトライはしない。次の睡眠サイクルで自然にリトライされる。

### 10.2 ブリッジ層のエラー

**sleep_ingest がエラーの場合:**
SleepyJean は空の対話ログで夜間サイクルを実行。宿題や既存の open_questions のみで学習対象を決定。循環は途切れるがシステムは停止しない。

**wake_export がエラーの場合:**
SleepyJean の学習結果は場に反映されない。llamarcute-live は前日の記憶のまま覚醒。knowledge_index.json は永続化されているため、次のサイクルで再度 export される。

### 10.3 LLM推論のエラー（Phase 1b）

タイムアウトまたはエラー時は固定のエラーメッセージを返す。困難度シグナルは emit しない（エラーと困難は区別する）。対話ログには記録する。

---

## 11. Phase 1 実装詳細

### 11.1 対話インターフェース

Phase 1 は CLI で実装する。

**コマンド:**

| コマンド | 動作 |
|---------|------|
| テキスト入力 | llamarcute-live との対話 |
| `/sleep` | 睡眠サイクルを開始 |
| `/status` | 現在の状態と場の統計を表示 |
| `/field` | FieldObserver の直近ログを表示（デバッグ用） |
| `/quit` | システム終了 |

**操作例:**

```
$ integrated-system start
[AWAKE] System started. Type your message or a command.
[AWAKE] > こんにちは
[AWAKE] llamarcute-live: やあ、こんにちは！何か気になることある？
[AWAKE] > /sleep
[SLEEPING] Entering sleep sequence...
[SLEEPING] sleep_ingest: Transferred 4 dialogue entries
[SLEEPING] sleep_ingest: Found 1 difficulty signal → added homework
[SLEEPING] Running SleepyJean night cycle...
[SLEEPING] wake_export: Emitted 5 signals
[AWAKE] Good morning. System is awake.
[AWAKE] >
```

### 11.2 Phase 1a / 1b の分割

**Phase 1a（モック接続検証）:**
LLM 推論はモック（固定応答または echo）。アーキテクチャの接続と循環の骨格を検証する。

**Phase 1b（LLM 接続）:**
モックを実際の LLM 推論に差し替え、対話内容と学習内容が意味的に循環することを確認する。

### 11.3 循環検証の成功基準

**Phase 1a の合格条件（全て満たすこと）:**

1. llamarcute-live の対話中に emit された信号が FieldObserver のログで確認できる
2. sleep_ingest が llamarcute-live の対話ログを SleepyJean の SQLite に転送できている
3. sleep_ingest が場から困難度シグナルを sense し、宿題に変換できている
4. SleepyJean の夜間サイクルが正常に完了する
5. wake_export が SleepyJean の出力を場に emit している
6. 覚醒後の llamarcute-live が sense した結果に、SleepyJean が emit した信号が含まれている

**Phase 1b の追加基準:**

7. 対話で話題にしたトピックが SleepyJean の夜間学習対象に含まれている
8. SleepyJean が学習した内容が翌覚醒の llamarcute-live の応答に反映されている

---

## 12. 段階的実装アプローチ

### Phase 0（前提）: 共有状態の場のインターフェース設計

1. 読み書き API 仕様の策定（上位設計の三特性をインターフェースレベルで表現）
2. 流れるデータの型定義
3. ChromaDB バックエンドによる仮実装

> **参照**: `shared_field_design.md`（共有状態の場 — Phase 0 インターフェース設計）にて策定済み。
> コア抽象（Signal, FieldReading）、場のインターフェース（emit, sense, purge, snapshot）、
> 共通エンコーダの責務境界、人間向け可読性レイヤーを定義している。

### Phase 1: llamarcute-live と SleepyJean の二者間で共有状態の場を介した循環を確立

1. モノレポ構成のセットアップ
2. llamarcute-live の基本実装（対話 + 行動規範による応答 + 共有状態の場への書き込み）
3. SleepyJean の共有状態の場への読み書き対応
4. 睡眠・覚醒サイクルの実装（手動トリガー、入眠・覚醒シーケンス）
5. 循環的因果関係の検証（対話経験 → 睡眠 → 夜間学習 → 記憶変化 → 覚醒 → 翌日の対話変化）
6. Phase 1 の実績データ収集（共有状態の場の本実装設計に向けて）

### Phase 2: llamarcute-live の自己改善メカニズムの実装

1. 自己改善トリガーの実装（困難度蓄積 + 環境変化検知）
2. 内部候補評価プロセスの実装（4体、二軸評価）
3. 行動規範の漸進的更新の検証

### Phase 3: 免疫系の接続と自己修復

1. 免疫系が観測可能な異常シグナルの定義と実装
2. 自然免疫相当のルールベース監視の導入
3. 子 LLM 分裂モデルの最小プロトタイプ（上位設計セクション4）

### Phase 4: 排泄系の強化と代謝系の導入

1. SleepyJean の積極的忘却メカニズムの強化
2. リソース管理の仕組みの導入

---

## 13. スコープ外

以下は本ドキュメントのスコープ外とし、別途設計する。

- 共有状態の場のインターフェース仕様 → `shared_field_design.md` にて策定済み
- 共有状態の場の本実装（Phase 1 の実績データを踏まえて独立プロジェクトとして設計）
- 免疫系（自己修復担当）の内部設計（子 LLM 分裂モデル）
- 既存 llamarcute 進化パイプラインの変更
- 代謝系の詳細設計
- 感覚系・運動系の具体的な拡充（Discord 以外のチャネル）
- ハードウェア要件（実装フェーズで決定）

### 関連ドキュメント

| ドキュメント | 内容 |
|------------|------|
| `bio_ai_architecture.md` | 上位設計。全体のビジョン・設計原則・アーキテクチャ全体像 |
| `shared_field_design.md` | 共有状態の場の Phase 0 インターフェース設計 |
| 本ドキュメント (`llamarcute_live_design.md`) | llamarcute-live 新設・SleepyJean 変更・統合サイクルの設計 |
| `individuality_design.md` | 個体性の操作的定義、全身的自己モデルの設計方針、検証計画 |

---

## 14. オープンな問い

- ローテーションタスクとして選出する Q&A ペアの難易度分布の最適値
- cuteness 評価のトピックプールを固定とするか、SleepyJean の知識成長に合わせて拡張するか
- 行動規範のバージョン間差分をどの程度保持するか（ロールバックの粒度）
- 睡眠中に免疫系が異常を検知した場合、覚醒シーケンスを中断してロールバックするか、覚醒後に対処するか
- 覚醒中の短期的な振り返りで蓄積した「何が問題か」の認識を、睡眠中のフル自己改善にどのような形式で引き渡すか → 全身状態クエリの拡張で部分的に対処済み（`individuality_design.md` 参照）
- Phase 1 の実績データから共有状態の場の本実装設計に移行する判断基準（どの程度の循環回数で十分なデータが溜まるか）

---

## 変更履歴

| 日付 | 変更内容 |
|---|---|
| 2026-03-14 | 初版作成 |
| 2026-03-17 | 個体性設計文書（`individuality_design.md`）の新規作成に伴い、セクション3.4の自己認識クエリ拡張への参照、セクション3.2の不変制約・メタ認知ルール注記、セクション13の関連ドキュメント追加、セクション14のオープンな問いにステータスを追記 |

---

本ドキュメントは議論の草案であり、構想の発展に応じて継続的に更新される。
