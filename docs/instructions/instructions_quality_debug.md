# 日本語品質デバッグ — タスクフロー

**目的:** トークン崩壊の原因切り分け（FieldReceptor vs TurboQuant）、およびキャラクター崩壊の修正
**前提:** 設計文書は docs/ 配下。コードベースは integrated-system/ 配下

---

## Phase 1: perceive空テスト（トークン崩壊の原因切り分け）

### T1-1: 現状把握

1. `llamarcute-live/` 配下で DialogueManager（対話フロー）のコードを特定する
2. 対話フロー内で `SharedField.perceive()` を呼び出している箇所を全て特定する
3. perceive結果を FieldReceptor.transduce() に渡している箇所を特定する
4. 上記の呼び出しチェーンを報告する（ファイル名・行番号・関数名）

**報告してから次に進むこと。**

### T1-2: perceive空固定の実装

perceive結果を空にするスイッチを追加する。**本番コードを壊さない方法で行うこと。**

推奨方法: config.yaml に以下を追加

```yaml
debug:
  empty_perceive: true  # trueの場合、perceive結果を空リストで上書き
```

対話フロー内で、perceive呼び出し後・FieldReceptor呼び出し前に:

```python
if config.debug.empty_perceive:
    perception = FieldPerception(signals=[], perceived_at=perception.perceived_at)
```

**影響範囲:** 対話フロー（DialogueManager）のみ。自己改善・cuteness評価は今回のテスト対象外（睡眠サイクルは回さない）。

### T1-3: テスト実行

Discord Bot を起動し、以下のテストメッセージを送信して応答を記録する。

テストメッセージ（この5つを順番に送信）:

1. `今日の気分はどう？`（日常会話）
2. `発展途上国の貧困問題について教えて`（トークン崩壊が発生した領域）
3. `マクドナルドのマックポークって何？`（文脈誤認が発生した領域）
4. `最近何か面白いことあった？`（自己参照・人格が出る質問）
5. `量子コンピュータについて簡単に説明して`（技術的な長文応答）

**記録すること:**
- 各応答の全文
- トークン崩壊の有無（旧字体混入、文字脱落、意味不明な文字列）
- 英語フレーズ混入の有無
- メタ注釈混入の有無
- 応答の途中切断の有無

### T1-4: 結果判定

| perceive空の結果 | 判定 | 次のアクション |
|-----------------|------|---------------|
| トークン崩壊が**消えた** | FieldReceptorが原因 | Phase 1-A へ |
| トークン崩壊が**残った** | TurboQuantが原因の可能性大 | Phase 1-B へ |
| 一部改善・一部残存 | 両方が寄与 | Phase 1-B を先に実施し、その後 Phase 1-A |

**結果を報告してから次に進むこと。**

---

## Phase 1-A: FieldReceptor側の調査（T1-4で「消えた」場合のみ）

FieldReceptorのembedding注入がLLMのlogitsを摂動させている。以下を調査:

1. perceive結果のstrength分布を記録（信号数、strength最大値・平均値・中央値）
2. FieldReceptor.transduce() 出力のノルム分布を記録
3. transduceされたembeddingのノルムが、通常のtoken embeddingのノルムと比較して大きすぎないか確認

**調査結果を報告して指示を待つこと。FieldReceptorの修正は設計判断が必要。**

---

## Phase 1-B: TurboQuantテスト（T1-4で「残った」場合のみ）

### T1-B-1: TurboQuant無効化スイッチ

config.yaml に追加:

```yaml
debug:
  empty_perceive: true  # Phase 1から継続
  disable_turboquant: true
```

FieldAwareLLM内で、TurboQuantCacheの代わりにDynamicCache（FP16）を使用するパスを追加する。

**注意:** FP16 KVキャッシュは8GB VRAMを溢れる。以下の制限を設ける:
- max_new_tokens: 128
- テストプロンプトは短くする（system prompt + 質問1文のみ）
- 対話履歴は含めない（1ターンのみ）

### T1-B-2: テスト実行

perceive空 + TurboQuant無効の状態で、T1-3と同じ5メッセージをテスト。
ただしmax_new_tokens=128のため、応答が短く切れることは許容。トークン崩壊の有無のみを判定する。

### T1-B-3: 結果判定

| TurboQuant無効の結果 | 判定 | 次のアクション |
|---------------------|------|---------------|
| トークン崩壊が**消えた** | TurboQuant 3-bit KVが原因 | bits=4への変更を検討。報告して指示を待つ |
| トークン崩壊が**残った** | bitsandbytes NF4自体の問題 | Ollama推論との比較テスト。報告して指示を待つ |

**結果を報告して指示を待つこと。**

---

## Phase 2: キャラクター崩壊修正（Phase 1完了後）

### T2-1: 現状のsystem prompt確認

1. `build_prompt()` のコードを特定し、実際にLLMに渡されるsystem prompt全文を出力する
2. 不変制約セクションの内容を報告する
3. 行動規範YAML（personality_current）の全カテゴリ・全ルールを報告する

**報告してから次に進むこと。**

### T2-2: 不変制約に存在の宣言を追加

build_prompt() の不変制約セクションに以下を追加:

```
あなたの名前はllamarcute-live。Somabeatの神経系として、ユーザとの対話を担当する。
```

**注意:**
- 既存の不変制約（日本語応答、文字数制限等）と同じ位置に追加
- YAMLには入れない（自己改善スクリプトからアクセス不能にする）
- 正確な文言は build_prompt() の既存の不変制約の書き方に合わせること

### T2-3: 行動規範YAMLのカテゴリ再編

現行の `tone` と `knowledge_attitude` を `identity` に統合する。

1. 現行personality YAML をバックアップ（`data/personality_backup_v16_pre_restructure.yaml`）
2. `tone` と `knowledge_attitude` のルールを `identity` カテゴリに移動
3. カテゴリ名の変更に伴い、以下のコードを更新:
   - 自己改善プロンプト（候補生成時にカテゴリ名を参照している箇所）
   - personality YAMLの読み込み・書き込み処理
   - テスト内でカテゴリ名をハードコードしている箇所
4. バージョンを v17 に上げる
5. 変更履歴に「カテゴリ再編による構造的変更: tone + knowledge_attitude → identity」と記録

**自己改善プロンプト内でカテゴリ名を使って変異方向を指示している場合、identity カテゴリ用の変異指示を追加すること。** 例:「identityカテゴリは自己の在り方を定義する。話し方、知識への態度、自分をどう捉えるかを含む」

### T2-4: テスト

1. 既存テスト（225件）を全て実行し、PASS を確認
2. Discord Bot を起動し、T1-3 と同じ5メッセージ + 以下の追加メッセージでテスト:
   - `自己紹介して`
   - `あなたは誰？`
3. 応答のキャラクター一貫性を確認

**テスト結果を報告すること。**

---

## 完了条件

- [ ] Phase 1: トークン崩壊の原因が特定され、対処方針が決まっている
- [ ] Phase 2: 不変制約に存在の宣言が追加されている
- [ ] Phase 2: 行動規範YAMLのカテゴリ再編が完了している
- [ ] 全テスト（225件）PASS
- [ ] debug.empty_perceive を false に戻している（Phase 2完了後）

---

## 絶対に守ること

- **各Tの末尾で「報告してから次に進むこと」と書いてある箇所では、必ず結果を報告して指示を待つ。** 自己判断で先に進まない
- config.yaml の debug セクションは一時的なもの。最終的に削除するか false にすること
- Phase 1 の切り分けテスト中は、Phase 2 の変更を同時に行わない（変数を1つずつ変える）
- 設計文書（docs/配下）は変更しない。設計文書の更新は別途指示する
