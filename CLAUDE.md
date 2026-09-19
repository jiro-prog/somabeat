# CLAUDE.md

## 作業開始時の必須手順
1. [docs/design_journal.md](docs/design_journal.md) の直近エントリを確認する（保留事項・受容済み制約）
2. タスクフロー（docs/instructions/instructions_*.md）がある場合はそれに従う
3. 現在のテスト件数を確認する（ベースライン）

---

## architect サブエージェントへの委譲

設計判断は自分でしない。architect に聞く。

### 委譲すべき場面
- 新しいクラス/モジュールの設計
- 既存設計の変更
- 仕様の曖昧な部分の解釈
- 2つ以上の実装方針で迷ったとき
- 「この変更は設計原則に反するか？」が即答できないとき

### 委譲しなくてよい場面
- バグ修正（原因が明確で設計変更を伴わないもの）
- configパラメータの調整（閾値、トークン数、バッファサイズ等）
- テスト追加・修正
- ログ出力の追加
- リファクタリング（設計変更を伴わないもの）

### 委譲時に渡す情報
architect はコードを見ない。以下を自然言語で伝える:
- 何が起きているか（症状・状況）
- どの選択肢で迷っているか（A vs B）
- 関連する設計上の文脈（どの系統に関わるか、場との関係など）

architect の回答に「人間にエスカレーションすべき」とあれば、従うこと。

---

## エスカレーション基準

### 必ず人間に返す
- architect が「エスカレーションすべき」と返した場合
- スコープの拡大（タスクフローに記載されていない作業が必要）
- 不可逆な変更（データ削除、バックエンド変更、モデル差し替え）
- 設計文書（docs/ 配下）の変更

### エスカレーションの形式

```
## エスカレーション: [1行要約]

**状況:** [何が起きているか]
**選択肢:**
- A: [説明。利点と欠点]
- B: [説明。利点と欠点]
**architectの見解:** [サブエージェントの回答を要約]
**設計原則との関係:** [なぜ自律判断できないか]

指示をお願いします。
```

---

## デバッグ手法

### 症状→仮説→テスト→判定

```
1. 症状を正確に記録する（何が起きたか、再現条件は何か）
2. 仮説をリストアップする
3. 仮説を検証するテストを設計する（変数を1つずつ分離）
4. テストを実行し、結果を表形式で記録する
5. 結果から判定する（支持/棄却）
6. 判定に基づいて次のアクションを決める
```

### 切り分けのルール
- 本番コードを壊さない方法で介入する（configフラグ、一時的なオーバーライド）
- 介入ポイントは最小限
- テスト後に介入を確実に元に戻す
- 1回の観察で原因を断定しない。最低3回の再現を確認する

### 段階的修正
修正は段階的に適用し、各段階でテストする。一括適用して「全部直った」は受け入れない。どの修正がどの症状に効いたかを個別に確認する。

### 比較テストの記録

```markdown
| 条件 | 変数A | 変数B | 結果 | 備考 |
|------|-------|-------|------|------|
| ベースライン | X | Y | ... | ... |
| テスト1 | X' | Y | ... | ... |
| テスト2 | X | Y' | ... | ... |
```

---

## 設計ジャーナル・プロトコル

### design_journal.md への記録

```markdown
## [日付] [タスク名]

### 判断: [1行要約]
- **状況:** ...
- **選択肢:** A: ... / B: ...
- **決定:** [A or B]
- **根拠:** [architect の回答 or 自律判断の理由]
- **リスク:** [この判断で受け入れたリスク]

### 発見: [1行要約]
- **内容:** ...
- **対処:** [修正済み / 保留 / 受容]
- **保留理由:** [保留の場合のみ]
```

### 記録のタイミング
- architect に委譲するたびに、回答と採用した判断を記録する
- 「やらない」と決めたことも記録する
- テスト結果の比較表もジャーナルに含める

---

## プロジェクト概要

integrated-system: Somabeatの統合システム。Discord Bot (Sleepy Jean) を通じた対話、自己改善、夜間学習サイクルを統合。

## アーキテクチャ
- `llamarcute_live/` — 対話エンジン (DialogueManager, FieldAwareLLM, Personality)
- `discord_bot/` — Discord Bot エントリポイント
- `orchestrator/` — システム統合・ライフサイクル管理
- `shared_state/` — SharedField, FieldReceptor, TurboQuant, MultimodalEncoder
- `sleepyjean/` — 海馬モジュール (KnowledgeGraph, TripleExtractor, MemoryStore, RecallEngine, Reconsolidation, GapDetector, QualityGate。覚醒時は非LLM、睡眠時のトリプル抽出のみgenerate_bare使用)
- `config/system.yaml` — 全体設定

## LLM推論
- モデル: AlphaGaO/Qwen3-8B-GPTQ (GPTQ 4-bit + Marlin kernel, W4A16 fused GEMV)
- KVキャッシュ: FP16（TurboQuantは無効化、コードは保持）
- Field embedding注入: system tokens と user tokens の間に挿入
- /no_think: system content末尾に付与、skip_special_tokens=False で処理
- 性能: ~26 tok/s, ~8.5s応答, VRAM 5.7GB (RTX 3060 Ti)
- config切り替え: `llamarcute_live.llm.quantization` で nf4/gptq_marlin 選択可能

## 推論エンジンの使い分け
- 場のembeddingを受け取る認知機能 → FieldAwareLLM.generate_with_field()（inputs_embeds注入）
- 場を読まないユーティリティ（自己改善のfitness/cuteness評価、wake message） → FieldAwareLLM.generate_bare()（input_idsベース）
- SleepyJean → LLM不使用（HDBSCAN + FieldEncoder のみ）
- Ollama依存は完全除去済み。モデルは常駐、VRAM排他制御不要

## 重要な設計判断
- 行動規範は英語で記述（LLMの指示追従性が英語で最も高い）
- 不変制約はbuild_prompt()にハードコード（自己改善スクリプトからアクセス不能にする）
- GPTQ量子化でもNF4同様の軽微なトークン崩壊（低頻度外来語）は受容見込み
- FR出力ノルムはL2正規化不可、一律スケーリングで保持
- 覚醒メッセージはReconsolidation結果+dialogue_logsを直接generate_bareに渡す（場のsenseを経由しない。テキスト情報はテキスト空間で処理が正当）

## テスト
- `python -m pytest tests/` (277件)
- テスト失敗はスキップせず修正する

## 設計適合監査

二層構造で設計原則違反を防止する。

### 第1層: 構造チェック（自動）
- `scripts/check_design_invariants.py` — 既知の8種類の設計原則違反を検出
- Claude Code hook で shared_state/ bridge/ への編集時に自動実行（違反時はブロック）
- orchestrator が wake_up 時に4サイクルに1回実行（config: `design_check.interval_cycles`）

### 第2層: 意味的レビュー（手動）
- `/design-conformance` スキルで architect サブエージェント込みの深いレビュー
- **運用指針: 睡眠サイクル3〜5回に1回は実行すること**
- 第1層がカバーしない「新しい種類の違反」を発見する役割
- 新しい違反パターンを発見したら `check_design_invariants.py` にルールを追加して第1層に昇格させる

## debug設定 (config/system.yaml)
- `debug.empty_perceive`: perceive結果を空にする（field embedding切り分け用）
- `debug.disable_turboquant`: TurboQuant無効化（現在デフォルト無効。NF4使用時のみ関連）

---

## 変更の横展開ルール

パラメータ・関数シグネチャ・データ構造を変更するとき、**コードの変更前に**以下を実行する:

1. **config確認:** `config/system.yaml` で当該パラメータが設定されていないか確認する。コードのデフォルト値だけを見て安心しない
2. **全出現検索:** `grep -r 'パラメータ名' . --include='*.py' --include='*.yaml'` で全出現を洗い出す
3. **呼び出し元追跡:** 関数シグネチャを変えたら、全call siteを更新する。`grep -r '関数名' . --include='*.py'` で漏れなく確認
4. **変更リスト作成:** 変更が必要な全箇所をリスト化してから着手する。1箇所ずつ直しながら探すな

configとコードのデフォルト値が二重管理になっている場合、**configが真**。コードのデフォルト値はconfigが未設定時のフォールバックにすぎない。

---

## チェックリスト

### 開始時
- [ ] design_journal.md の直近エントリを確認したか
- [ ] テスト件数のベースラインを確認したか

### 判断時
- [ ] architect に委譲すべきか確認したか
- [ ] エスカレーション基準に該当しないか
- [ ] 変数は1つだけ変えているか
- [ ] 結果を表形式で記録したか

### 変更時
- [ ] config/system.yaml の実際の値を確認したか
- [ ] 全出現箇所を grep で洗い出したか
- [ ] 全call siteを更新したか

### 終了時
- [ ] 全テストがPASSしているか
- [ ] debug用の一時変更を元に戻したか
- [ ] design_journal.md に判断と発見を記録したか
- [ ] 保留事項があれば明記したか
