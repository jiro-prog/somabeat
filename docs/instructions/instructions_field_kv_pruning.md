# 軸A: field KV エントリ除去 — タスクフロー

**作成日:** 2026-03-28
**親文書:** performance_tuning_report.md, llamarcute_live_design.md 3.4節
**前提:** field=10, Gリバート後（0.9-1.7 tok/s）がベースライン

---

## 0. 目的

prefill完了後、decode開始前に、場のembeddingに対応するKVキャッシュエントリを除去することで、decode時のattentionコスト（現在21%）を削減する。

**設計原則との整合性:** 確認済み。perceive → FieldReceptor → inputs_embeds注入の経路は不変。場自体には変更なし。最適化はLLM内部のKVキャッシュ管理に閉じる。濃度情報はprefill時のattentionパターンを通じてテキストトークンのKV表現に焼き込まれる。生体対比: 受容体にホルモンが結合してシグナル伝達カスケードを起こした後、ホルモン分子自体を保持し続ける必要はない。

**ゲート条件:** A-1の計測結果に基づき、field位置へのdecode時attention weightが全体の **2%未満** であることをPASS条件とする。2%以上であれば本施策は中止し、軸B（Marlinカーネル）に移行する。

---

## A-0. 事前確認

| # | 確認事項 | 期待値 |
|---|----------|--------|
| 0-1 | テスト225件PASS | 全PASS |
| 0-2 | field=10, 固定スケーリング(0.0752)が有効 | config確認 |
| 0-3 | Gリバート済み（repeat_interleave使用） | コード確認 |

---

## A-1. decode時のfield位置attention weight計測

### 目的

decode（自己回帰生成）中に、各生成トークンがfield位置（system prompt直後の10トークン）にどの程度attendしているかを計測する。

### 方法

FieldAwareLLMのgenerate()を改修し、計測モードを追加する。

```python
# 計測用フック（本番コードには入れない。スクリプト単体で実行）
# generate()の各decode stepで:
#   1. output_attentions=True でattention weightを取得
#   2. 全layer・全headについて、生成トークン→field位置のattention weight合計を記録
#   3. 生成トークン→全位置のattention weight合計で割って比率を算出
```

### 入力条件

以下の4条件で計測（performance_tuning_report.md と同一の質問セット）:

| # | 質問 | 目的 |
|---|------|------|
| 1 | 「気分はどう？」 | 短い応答。自己言及的（場の信号と関連しやすい） |
| 2 | 「貧困問題について教えて」 | 長い応答。場との関連が薄い知識質問 |
| 3 | 「量子コンピュータとは」 | 長い応答。場との関連が薄い知識質問 |
| 4 | 「最近面白いことあった？」 | 自己言及的（場の信号と関連しやすい） |

### 出力

| 計測値 | 単位 |
|--------|------|
| field位置へのattention weight比率（layer平均, head平均） | % |
| field位置へのattention weight比率（layer別, head別） | % |
| 生成トークン位置ごとの推移（序盤 vs 終盤） | % |
| 質問ごとの比較 | % |

### 注意事項

- `output_attentions=True` はVRAMを大量消費する。max_new_tokens=30程度に制限して計測
- TurboQuantのfused attentionがoutput_attentionsと互換性があるか事前確認。非互換なら、計測時のみTurboQuantを無効にして計測（KVキャッシュ圧縮はattention weight分布に影響しないため、FP16キャッシュでの計測結果は有効）
- VRAMが足りない場合は、1層ずつ計測する方式に切り替え

### ゲート判定

| 条件 | 判定 | 次のステップ |
|------|------|-------------|
| 全質問でfield attention weight < 2%（layer・head平均） | PASS | A-2へ |
| 自己言及質問(#1,#4)で > 2%、知識質問(#2,#3)で < 2% | 要検討 | 条件付き除去の設計を検討 |
| 全質問で > 2% | FAIL | 軸A中止、軸Bへ |

---

## A-2. field KVエントリ除去の実装

### 前提

A-1でゲートPASS。

### 実装方針

FieldAwareLLMのgenerate()にフックを追加し、prefill完了後・decode開始前にKVキャッシュからfield位置のエントリを除去する。

```
prefill:
  input_embeds = [system_prompt] [field_1..field_K] [user_input]
  → model forward → past_key_values にK+N+M個のKVエントリ

field KV除去:
  → past_key_values の各layerについて:
      key[:, :, sys_len : sys_len+K, :] を削除（テンソルslice）
      value[:, :, sys_len : sys_len+K, :] を削除
  → past_key_values のシーケンス長がK個減少

decode:
  → field位置が存在しないため、attend対象が減少
  → repeat_interleaveの対象も減少（GQA比率×K本分）
```

### 設計上の位置づけ

- FieldAwareLLM内部の最適化。外部インターフェースに変更なし
- perceive, FieldReceptor, SharedFieldには一切変更なし
- configで有効/無効を切り替え可能にする: `field_kv_pruning: true/false`
- デフォルトは `false`（A-3のゲートPASSまで）

### TurboQuantとの整合

TurboQuantはKVキャッシュを圧縮形式で保存している。field位置のKVエントリ除去は、TurboQuantCacheのupdate()後に行う。具体的には:

- TurboQuantCacheの内部テンソル（量子化済みkey、FP16 value）からfield位置をslice
- `_seen_tokens` カウンタをK個減算
- prefill後の`_fp16_keys`解放と同じタイミングで実行

### 実装時の確認事項

| # | 確認 |
|---|------|
| 1 | past_key_valuesのslice後、position_idsの再計算が必要か |
| 2 | TurboQuantCacheの内部状態（codebook index等）がsliceに対応可能か |
| 3 | attention_maskの更新が必要か |

---

## A-3. 速度・品質検証

### 速度計測

performance_tuning_report.md と同一条件で計測:

| # | 質問 | 計測値 |
|---|------|--------|
| 1 | 気分（ウォームアップ） | tok/s, 応答時間 |
| 2 | 貧困問題 | tok/s, 応答時間 |
| 3 | 量子コンピュータ | tok/s, 応答時間 |
| 4 | 面白いこと | tok/s, 応答時間 |

ベースライン（field=10, Gなし）との比較表を作成。

### 品質検証

| # | 検証項目 | 方法 |
|---|----------|------|
| 1 | 応答の自然さ | 4質問の応答を目視比較（field KV除去あり vs なし） |
| 2 | 自己言及率 | 「最近の自分の調子」等の自己言及質問で、場の信号由来の情報が応答に反映されているか |
| 3 | 早期EOS | 3回以上テストし、不自然な打ち切りがないか |
| 4 | 既存テスト | 225件PASS |

### ゲート判定

| 条件 | 判定 |
|------|------|
| tok/s がベースライン以上 AND 品質劣化なし AND 225件PASS | PASS → デフォルト有効化 |
| tok/s 改善あり BUT 品質劣化あり | 要判断。劣化の程度による |
| tok/s 改善なし | 中止。除去のオーバーヘッドが利得を相殺している |

---

## A-4. 設計文書更新

A-3ゲートPASSの場合、以下を更新:

| ドキュメント | 更新内容 |
|------------|---------|
| llamarcute_live_design.md 3.4節 | 「prefill後のfield KVエントリ除去」を対話フローに追記。設計根拠（受容体→カスケードの対比）を記載 |
| performance_tuning_report.md | A-1計測結果、A-3速度・品質結果を追記 |
| somabeat_handoff.md | 現在の構成にfield KV pruning: trueを追記、tok/sを更新 |

---

## タスク依存関係

```
A-0（事前確認）
  → A-1（attention weight計測）
    → [ゲート判定]
      → PASS → A-2（実装）→ A-3（検証）→ [ゲート判定] → A-4（文書更新）
      → FAIL → 軸B（Marlinカーネル検証）へ移行
```

---

## リスクと緩和策

| リスク | 確率 | 緩和策 |
|--------|------|--------|
| output_attentionsがTurboQuantと非互換 | 中 | 計測時のみTurboQuant無効化。attention分布自体はKV圧縮の影響を受けにくい |
| field attention weightが質問によって大きく変動 | 中 | 質問4種で計測。自己言及質問で高い場合、条件付き除去を検討 |
| KVキャッシュsliceがTurboQuantの内部状態を破壊 | 低 | A-2実装時にユニットテストで確認。フォールバックとして除去無効化 |
| position_idsの不整合で生成品質劣化 | 低 | A-3の品質検証で検出。位置エンコーディングの再計算ロジックを確認 |

---

## 軸Bへの移行条件

A-1でゲートFAILの場合、または A-3で速度改善が見られない場合、軸B（Marlinカーネル検証）に移行する。軸Bのタスクフローは別文書で作成する。

軸Bの最小検証:
1. AlphaGaO/Qwen3-8B-GPTQ（Marlinフォーマット、4-bit、group_size 128）をロード
2. `model(inputs_embeds=...)` の疎通確認
3. TurboQuant無効状態でtok/s計測
4. TurboQuantとの共存検証
