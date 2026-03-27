# 指示書: SleepyJean 夜間サイクル リファクタリング

**日付:** 2026-03-22
**対象:** SleepyJean 夜間学習パイプライン + integrated-system（cuteness対話ログ）
**目的:** fine-tuning停止、Claude API最小化、ローカルLLM中心への移行、depth廃止

---

## 背景

1. 統合システムにおいてSleepyJeanのfine-tunedモデルを推論に使う箇所がない。llamarcute-liveは別モデル(qwen3:8b)で対話しており、fine-tunedの重みは誰にも使われていない
2. 知識伝達はRAG（ChromaDB）と共有状態の場を介したテキストベースの経路で機能している
3. Claude APIのレート制限で夜間サイクル全体が失敗するインシデントが発生した。API依存を最小化し耐障害性を上げたい
4. depthの多段階Claude API呼び出しを廃止し、「Claude API 1回で模範的な要点整理を取得→ローカル8BでQ&A生成」に変更する

---

## 作業一覧

| # | 作業 | 影響範囲 |
|---|------|---------|
| 0 | 事前確認（実装前に必須） | 両方 |
| 1 | LoRA fine-tuningパイプラインの停止 | SleepyJean |
| 2 | SleepyJeanのモデルをqwen3:8bに変更 | SleepyJean |
| 3 | depth廃止→target_qa_countへの置き換え | SleepyJean |
| 4 | Q&A生成方式の変更（Claude要点整理+ローカル生成） | SleepyJean |
| 5 | Web探索結果の要約パイプライン構築 | SleepyJean |
| 6 | homework信号への対話コンテキスト追加 | integrated-system + SleepyJean |
| 7 | 夢日記のローカルLLM移行 | SleepyJean |
| 8 | curiosity生成のローカル移行 | SleepyJean |
| 9 | cuteness勝者の対話ログ保存 | integrated-system |

---

## 作業0（最初に実施）: 事前確認

以下を調査し、結果に応じて作業1-9の実装を調整すること。

### 0.1 SleepyJeanのOllamaクライアントの現状

SleepyJeanのコード内でOllama（ローカルLLM）を推論に使っている箇所を全て洗い出すこと。現在どのモデル名を指定しているか（`sleepyjean:latest`、`qwen3:8b`等）を確認し、一覧にすること。

### 0.2 Web探索結果の形式

SleepyJeanのWeb探索が返すデータの形式を確認すること。
- ソースごとにテキストが分かれているか、全部連結された1テキストか
- 各ソースにタイトル・URL・スニペット等のメタデータがあるか
- 1トピックあたりの典型的なテキスト量（文字数）

作業5（Web要約パイプライン）の実装に必要。

### 0.3 depthの現在の実装

`depth`が具体的にどの処理をN回繰り返す制御に使われているかを確認すること。
- Web検索を何回行うか
- Claude APIを何回呼ぶか
- Q&A生成を何ラウンド行うか
- その他の用途

作業3（depth廃止）の影響範囲を正確に把握するために必要。

### 0.4 RAG登録の形式

Q&AペアがChromaDBにどう登録されているか確認すること。
- instructionをキー、outputを値として登録しているか
- エンベディングはinstructionから生成しているか
- Q&Aの形式が変わった場合に検索精度に影響するか

### 0.5 Non-REM / REM相の構造

夜間サイクルのNon-REM相（情報の構造化）とREM相（夢日記）がコード上どう分離されているか確認すること。作業7（夢日記ローカル移行）が相構造を壊さないことを確認するため。

### 0.6 既存テストのClaude APIモック箇所

SleepyJean側のテストで、Claude APIをモックしているテストを洗い出すこと。API呼び出し箇所が変わるので、モックの差し替えが必要。

### 0.7 wake_exportへの影響確認

wake_exportがQ&Aペアを場にemitする処理を確認すること。Signal.extraに`instruction`と`output`を格納する形式が維持される限り影響なし。Q&A生成がローカルに変わっても出力形式が同じであることを確認。

---

## 作業1: LoRA fine-tuningパイプラインの停止

### 変更内容

`night_cycle.py`から以下の処理を無効化する:

- `lora_train.py`の呼び出し（`run_training()`）
- GGUF変換
- Modelfile更新
- Ollama create（モデル登録）

### 注意

- `lora_train.py`自体は**削除せず残す**（将来のllamarcute-live fine-tuning用に再利用の可能性）
- `night_cycle.py`の`_save_results()`で`fine_tuning`フェーズの結果を`"disabled"`として記録する

---

## 作業2: SleepyJeanのモデルをqwen3:8bに変更

### 変更内容

作業0.1の調査結果に基づき、SleepyJeanのローカルLLM呼び出しを全て`qwen3:8b`に統一する。

### 設定

config（SleepyJean側）にモデル名を設定項目として持つこと:

```yaml
sleepyjean:
  local_model: "qwen3:8b"
```

コード中でモデル名をハードコードせず、この設定値を参照する。

---

## 作業3: depth廃止→target_qa_countへの置き換え

### 変更内容

作業0.3の調査結果に基づき、depthに依存する全ての処理を特定し、廃止または置き換える。

depth（`[5, 4, 3, 3, 2]`の固定配分）を廃止し、`target_qa_count`を導入:

```yaml
sleepyjean:
  target_qa_count:
    homework: 5
    curiosity: 3
```

- depthに依存する多段階処理ループを削除
- Web探索は1回/トピックに統一
- Q&A生成はローカル8Bに「{target_qa_count}件生成してください」と渡す

---

## 作業4: Q&A生成方式の変更

### 新方式のフロー（トピックごとに実行）

```
ステップa: Claude API 1回 → 要点整理の取得
ステップb: ローカル8B 1回 → Q&Aペア生成
```

### ステップa: Claude API呼び出し

**プロンプト:**

```
あなたは学習素材を作成する専門家です。
以下のトピックについて、学習者がQ&Aで理解を深めるための要点を整理してください。

## トピック
{topic_name}

## 背景（ある場合）
{context_summary or "なし"}

## 参考情報
{web_summary}

## 出力形式
以下の3点を簡潔に書いてください:
1. このトピックの核心（一番大事なこと）
2. よくある誤解や混同しやすいポイント
3. 理解の前提となる関連概念
```

- `context_summary`: homework信号の対話コンテキスト要約（作業6）。curiosityの場合は「なし」
- `web_summary`: Web探索結果の要約、200-300字（作業5）

### ステップb: ローカル8BでQ&A生成

**プロンプト:**

```
以下の要点整理と参考情報をもとに、Q&Aペアを{target_qa_count}件生成してください。

## 要点整理（専門家による）
{claude_output}

## 参考情報（Web探索結果）
{web_results_detail}

## 出力形式
JSON配列で出力:
[{"instruction": "質問文", "output": "回答文"}, ...]

注意:
- 各Q&Aは独立して理解できること
- 「核心」「よくある誤解」「関連概念」のそれぞれからQ&Aを作ること
```

- `claude_output`: ステップaの出力
- `web_results_detail`: Web探索結果の切り詰め版（作業5のステップ1出力、合計2000字以内）
- `target_qa_count`: homework=5、curiosity=3

### フォールバック（トピックごと）

ステップaでClaude APIがエラーを返した場合:

```python
try:
    reference = await claude_api.generate_reference(topic, web_summary, context)
except Exception:
    logger.warning("Claude API failed for topic '%s', proceeding without reference", topic)
    reference = ""
```

referenceが空の場合、ステップbのプロンプトから「要点整理」セクションを省略し、Web探索結果のみでQ&A生成する。

5トピック中N番目で失敗した場合、1〜N-1は高品質（Claude要点整理あり）、N〜5は低品質（ローカルのみ）。サイクル全体は中断しない。

### 品質チェック

プログラム的な判定に置き換える:

- JSON配列としてパースできるか
- 各要素に`instruction`と`output`が存在し空でないか
- `output`が50文字以上か
- `instruction`の重複がないか
- パース失敗時、ローカル8Bに再生成を1回だけ依頼する

---

## 作業5: Web探索結果の要約パイプライン

### 2段階処理

```
Web探索の生テキスト
  → ステップ1: プログラム的に切り詰め（コード処理）
  → ステップ2: ローカル8Bで要約
```

### ステップ1: プログラム的切り詰め

作業0.2の調査結果に基づき実装。

- ソースごとの上限: 500字（先頭から切り詰め）
- 合計上限: 2000字（超過時は上位ソースを優先）
- ソースにタイトル・URLがあれば冒頭に付記

### ステップ2: ローカル8Bで要約

```
以下の情報を200字以内で要約してください。
重要な事実と数値を優先し、一般的な記述は省略してください。

{truncated_web_results}
```

### 用途の使い分け

| 用途 | 使うデータ | 文字数目安 |
|------|----------|-----------|
| Claude API（作業4ステップa） | ステップ2の要約 | 200-300字 |
| ローカル8B Q&A生成（作業4ステップb） | ステップ1の切り詰め版 | 最大2000字 |

---

## 作業6: homework信号への対話コンテキスト追加

### 変更箇所1: integrated-system — `bridge/sleep_ingest.py`

sleep_ingestがdifficulty信号をhomeworkに変換する際、対応する対話ログ（llamarcute_live.dbのconversationsテーブル）から直近5ターン分を取得し、ローカル8Bで100字以内に要約する。

```python
async def _ingest_difficulty_signals(self, ...):
    for signal in difficulty_signals:
        theme = extract_theme(signal)

        # NEW: 対話コンテキストの要約
        recent_turns = get_recent_dialogue(db, signal.emitted_at, n=5)
        if recent_turns:
            context = await local_llm.summarize(
                f"以下の対話を100字以内で要約してください:\n{recent_turns}",
                max_length=100
            )
        else:
            context = ""

        db.execute(
            "INSERT INTO homework (theme, context, ...) VALUES (?, ?, ...)",
            (theme, context, ...)
        )
```

homeworkテーブルに`context`カラムを追加する（TEXT, デフォルト空文字列）。

### 変更箇所2: SleepyJean — Q&A生成時

作業4のステップaで`context_summary`を、homeworkテーブルの`context`カラムから取得する。curiosityトピックの場合は「なし」。

### 注意

- 対話ログが存在しない場合は空文字列をセット
- コンテキスト要約の生成はsleep_ingest内（入眠時）で実行されるため、対話のレイテンシには影響しない

---

## 作業7: 夢日記のローカルLLM移行

### 変更内容

夢日記生成のClaude API呼び出しをローカル8B（Ollama qwen3:8b）に変更する。

- 既存の夢日記生成プロンプトをそのまま流用
- 呼び出し先のみClaude API → Ollamaに変更
- 後続処理（open_questions追加、ChromaDB RAG登録、wake_export）は変更しない

### 注意

- 夢日記は**廃止しない**。curiosity生成（作業8）が夢日記のopen_questionsに依存している
- 品質低下の可能性あり。dream_metricsのconnections_count、insights_count、avg_noveltyで追跡
- 作業0.5の確認結果に基づき、Non-REM/REM相の構造を崩さないこと

---

## 作業8: curiosity生成のローカル移行

### 変更内容

T2で実装した`_fallback_from_knowledge()`を正規のcuriosity生成ルートに昇格させる。

```python
# 変更前
curiosity_questions = await _generate_curiosity_from_claude(...)
if not curiosity_questions:
    curiosity_questions = _fallback_from_knowledge(...)

# 変更後
curiosity_questions = _generate_curiosity_from_knowledge(...)
```

ソースの優先度:
1. knowledge_indexのopen_questions（夢日記由来）→ 新しい順
2. 低confidenceトピックの深化 → confidence昇順
3. 上記で枠が埋まらない場合: ローカル8Bに「最近の学習内容から新しい疑問を生成して」と依頼

### 注意

- open_questionsの供給は夢日記（作業7）に依存。dream_metricsのopen_questions_addedがゼロに張り付いたら夢日記の品質を再検討

---

## 作業9: cuteness勝者の対話ログ保存

### 変更箇所: integrated-system — `llamarcute_live/self_improve.py`または`cuteness.py`

```sql
CREATE TABLE IF NOT EXISTS cuteness_dialogue_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id TEXT,
    winner_id TEXT,
    partner_id TEXT,
    topic TEXT,
    dialogue TEXT,  -- JSON: [{"role": "winner", "content": "..."}, ...]
    cuteness_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

保存タイミング: `select_and_update()`で勝者決定後、勝者が参加した全対話を保存。

### 注意

- **勝者の対話のみ**保存（データ量の抑制）
- 蓄積するだけで、読み出す処理は現時点で不要

---

## 変更後の夜間サイクルフロー

```
1. トピック選出（API不要）
   - homework: 最大2件（homeworkテーブル）
   - curiosity: 最低2件（knowledge_index → open_questions / 低confidence）

2. トピックごとに:
   a. Web探索（API不要）
   b. Web結果の切り詰め（プログラム的、500字/ソース、合計2000字上限）
   c. Web結果の要約（ローカル8B、200-300字）
   d. Claude API 1回: 要点整理（要約 + コンテキストを入力）
      → 失敗時: 要点整理なしでステップeに進む
   e. ローカル8B 1回: Q&A生成（要点整理 + Web結果詳細を入力）
   f. 品質チェック（プログラム的）
   g. RAG登録 + wake_export

3. 夢日記（ローカル8B、API不要）
   → connections / insights → open_questions → ChromaDB → wake_export

4. dream_metrics記録

（LoRA fine-tuning: 停止）
```

### Claude API呼び出し

| 変更前 | 変更後 |
|--------|--------|
| ~15-20回/サイクル | **5回/サイクル（トピック数と同数）** |
| 失敗でサイクル全体停止 | トピックごとにフォールバック |

---

## 確認事項

全作業完了後、以下を確認すること:

1. night_cycleがfine-tuningなしで正常完了すること
2. SleepyJeanの全LLM呼び出しが`qwen3:8b`を指していること
3. depthに依存する処理が全て廃止または置き換え済みであること
4. Web要約パイプライン（切り詰め→8B要約）が動作し、200-300字の要約を生成すること
5. Claude API要点整理が「核心」「誤解」「関連概念」の3点を返すこと
6. Claude API失敗時にフォールバック（要点整理なしでQ&A生成）が動作すること
7. ローカル8BがQ&Aを正しいJSON形式で生成すること（homework=5件、curiosity=3件）
8. 品質チェック（JSON形式、空チェック、最低50文字、重複チェック）が動作すること
9. 夢日記がローカル8Bで生成され、dream_metricsに記録されること
10. curiosity生成がknowledge_indexから動作すること
11. homeworkテーブルにcontextカラムが追加され、sleep_ingestがコンテキスト要約を保存すること
12. cuteness勝者の対話ログがllamarcute_live.dbに保存されること
13. wake_exportのrotation_task信号の形式（Signal.extraのinstruction/output）が維持されること
14. 既存テスト（SleepyJean側 + integrated-system側）が全パスすること
