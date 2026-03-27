# SleepyJean学習改善 + LoRA fine-tuning タスクフロー

- **親文書**: sleepyjean_learning_analysis.md
- **作成日**: 2026-03-21
- **最終更新**: 2026-03-21
- **スコープ**: homework/curiosity配分修正、LoRA fine-tuning実施、夢効果追跡

---

## 進捗サマリ

| タスク | ステータス | 完了日 | 備考 |
|--------|-----------|--------|------|
| T1: homework/curiosity配分修正 | ✅ 完了 | 03-21 | hw最大2, cur最低2。テスト4件追加 |
| T2: 空サイクル対処 | ✅ 完了 | 03-21 | knowledge_indexフォールバック。テスト5件追加 |
| T3: 夢効果追跡 | ✅ 完了 | 03-21 | dream_metricsテーブル。テスト9件追加 |
| T4: fine-tuningベースライン | ⏳ 待ち | — | T1-T3後数サイクル運用してから |
| T5: LoRA fine-tuning実施 | ✅ 完了(前倒し) | 03-21 | 528 QA、Qwen3-4B、パイプライン自動化 |
| T6: 効果検証 | ⏳ 待ち | — | 8B vs fine-tuned 4Bの比較 |
| T7: 運用観察 | ⏳ 待ち | — | fine-tuning後の数サイクル |
| T8: 総合評価 | ⏳ 待ち | — | 個体性T12と合流 |

**注記:** T5（LoRA fine-tuning）は13夜連続失敗の発見に伴い、パイプライン修正と合わせて前倒し実施。T4のベースライン記録は未実施のため、T6では8B（fine-tuning前のモデル）との比較で代替する。ベースモデルはVRAM制約によりQwen3-8B→Qwen3-4Bに変更。

---

## 全体構造

```
T1-T3: homework/curiosity配分修正 + 空サイクル対処 + 夢追跡
  │
  └─ 数サイクル運用（配分の効果確認）
       │
       T4-T7: LoRA fine-tuning（手動1回）
         │
         └─ 数サイクル運用（fine-tuning効果確認）
              │
              T8: 総合評価（T12中間評価と合流）
```

**依存関係:** T1-T3は独立して着手可能。T4はT1-T3完了後、数サイクルの運用データを確認してから実施。T8はT11/T12（個体性実験3）と合流。

---

## T1: homework/curiosity固定枠配分の実装

### 背景

MAX_TOPICS_PER_NIGHT=5の全枠がhomeworkに占有され、curiosity（自発的知識探索）がゼロになっている。学習の多様性が失われる構造的問題。

### 設計

案A（固定枠配分）を採用。

```
MAX_HOMEWORK_PER_NIGHT = 3   # homework枠の上限
MIN_CURIOSITY_PER_NIGHT = 2  # curiosity枠の下限（= 5 - 3）
```

- homeworkキューに5件以上あっても、1サイクルで処理するのは最大3件
- 残り2枠はcuriosityで埋める
- homeworkが0-2件の場合は、残り枠がcuriosityに回る（合計は常に5）
- 定数はconfig/system.yamlに外出しし、後から調整可能にする

### 変更箇所

**`sleepyjean/question_engine.py`** — トピック選出ロジック

現状の選出フロー（推定）:
```
1. homeworkキュー（priority=10）から全件取得
2. MAX_TOPICS_PER_NIGHT件まで選出
3. 枠が余ればcuriosityで埋める
```

変更後:
```
1. homeworkキューからMAX_HOMEWORK_PER_NIGHT件まで取得
2. 残り枠（5 - homework取得数）をcuriosityで埋める
```

### テスト

- homework 5件以上 → 3件のみ選出、残り2件がcuriosity
- homework 2件 → 2件homework + 3件curiosity
- homework 0件 → 5件全てcuriosity
- MAX_HOMEWORK_PER_NIGHTの設定値変更が反映されること

---

## T2: 空サイクル問題の対処

### 背景

対話がない日（conversationsテーブルに入力なし）はnight_cycleがスキップされる。T1のcuriosity最低保証があれば、対話がなくてもcuriosityで学習できるはず。

### 確認事項

night_cycleのスキップ判定を確認する。

- スキップ条件が「conversationsテーブルに新規入力がない」なら → **スキップ条件を変更**。homeworkキューまたはcuriosityソースがあれば実行する
- スキップ条件が「question_engineがトピックを0件返した」なら → T1の修正でcuriosityが常に返るので、**追加修正不要**

### 変更箇所

night_cycleのスキップ条件に依存。確認してから判断。最悪のケースでも、スキップ条件にcuriosityフォールバックを追加する1箇所の修正で済む。

### テスト

- 対話0件の状態でnight_cycleを実行 → curiosityトピックで学習が走ること

---

## T3: 夢効果追跡の計測追加

### 背景

夢日記は安定して生成されているが、llamarcute-liveの応答への影響が未追跡。T8のmetricsインフラに1カテゴリ追加する。

### 変更箇所

**`llamarcute_live/metrics.py`** — SIGNAL_CATEGORIESに`dream`を追加

```python
SIGNAL_CATEGORIES = {
    "immune":     lambda ws: ws.signal.origin.context == "immune",
    "sleepyjean": lambda ws: ws.signal.origin.system == "sleepyjean",
    "difficulty": lambda ws: ws.signal.origin.context == "difficulty",
    "dialogue":   lambda ws: ws.signal.origin.context == "dialogue",
    "dream":      lambda ws: ws.signal.origin.context == "dream",  # 追加
}
```

wake_exportで夢日記をemitする際のorigin.contextを事前に確認すること。`"dream"`でない場合は実際の値に合わせる。

**`llamarcute_live/metrics.py`** — metrics_senseテーブルにdream_count, dream_weight_sumカラム追加

既存のinit_metrics_db()がIF NOT EXISTSで冪等なら、ALTER TABLEまたはテーブル再作成が必要。

### テスト

- 夢信号を含むsense結果でdream_countが正しく記録されること

---

## T4: LoRA fine-tuning前のベースライン記録

### 背景

fine-tuningの効果を測るには、適用前の応答品質のベースラインが必要。T9（個体性ベースライン）とは別に、知識面での品質を記録する。

### 手順

1. 現行のsleepyjeanモデル（3/11版）で、以下の質問セットに回答させる:
   - homework由来の頻出トピック（SleepyJeanが学習済みの領域）から5問
   - curiosity由来のトピック（SleepyJeanが自発的に学んだ領域）から5問
   - 未学習の領域から5問（コントロール群）
2. 回答の品質を記録（正確性、詳細度、自信度の定性評価）

### 成果物

ベースライン質問セット（15問）と回答記録。fine-tuning後に同じ質問で再評価する。

---

## T5: LoRA fine-tuning実施

### 前提

- T1-T3が完了し、数サイクル運用済み
- T4のベースラインが記録済み

### 手順

SleepyJeanの既存fine-tuningパイプラインを使用して、蓄積済みのQAペアでLoRA fine-tuningを手動実行する。

1. 訓練データの確認: `data/training/`以下のQAペアの件数と内容を確認
2. データの品質チェック: SleepyJeanのquality_checkを通過しているか確認。通過していないデータがあればフィルタ
3. fine-tuning実行: 既存のスクリプト/手順に従って実行
4. 適用後のモデルサイズ・LoRAアダプタサイズを記録

### 注意

- fine-tuning前にsleepyjeanモデルのバックアップを取ること
- 免疫系が退行を検知する可能性がある。ヘルスチェックの結果を確認

---

## T6: fine-tuning後の効果検証

### 手順

1. T4と同じ質問セット（15問）をfine-tuning後のモデルで再評価
2. ベースラインと比較:
   - 学習済み領域（homework由来）: 品質が向上しているか
   - 学習済み領域（curiosity由来）: 同上
   - 未学習領域: 変化がないか（退行していないか）
3. 免疫系の反応を確認: ヘルスチェックの結果、conservative_modeの発動有無

### 判定

| 結果 | 判定 | 次のアクション |
|------|------|-------------|
| 学習済み領域で向上、未学習で退行なし | 成功 | 自動化を検討 |
| 学習済み領域で向上、未学習で退行あり | 部分成功 | 訓練データのフィルタ強化を検討 |
| 変化なし | 効果不十分 | データ量・品質・パラメータを再検討 |
| 全体的に劣化 | 失敗 | バックアップにロールバック、原因分析 |

---

## T7: 運用サイクルでの観察

fine-tuning適用後、数サイクル通常運用して以下を観察:

- llamarcute-liveの対話品質（ユーザー体感）
- rotation_scoreの変化（fine-tuning後のモデルでQ&A評価の精度が変わるか）
- SleepyJeanの次回夜間学習の品質（fine-tuned modelでのQ&A生成品質）
- 免疫系の反応

---

## T8: 総合評価

T1-T7の結果と、個体性実験3のT12中間評価を合わせて、以下を総合的に評価する。

| 観点 | データソース |
|------|-----------|
| 学習の多様性 | homework/curiosity比率の推移 |
| 知識の定着 | fine-tuning前後の応答品質比較 |
| 循環の完全性 | 対話→学習→fine-tuning→応答品質向上の経路が機能しているか |
| 個体性 | 自己言及率、フォールバック到達率（T12） |
| 自己改善 | 行動規範の進化軌跡（v7→v10以降） |

---

## タスク依存関係と工数

```
T1: homework/curiosity配分修正        ← 即着手可能
T2: 空サイクル対処                    ← T1と並行（確認作業）
T3: 夢効果追跡                       ← T1と並行（小作業）
  │
  └─ 数サイクル運用（3-5サイクル）
       │
       T4: fine-tuningベースライン     ← 運用データ確認後
       T5: LoRA fine-tuning実施       ← T4完了後
       T6: 効果検証                   ← T5完了後
       T7: 運用観察                   ← T6完了後（数サイクル）
         │
         T8: 総合評価                 ← T7 + T12合流
```

### 工数見積もり

| タスク | 見積もり | 備考 |
|--------|---------|------|
| T1 | 1-2時間 | question_engine.pyの修正 + テスト |
| T2 | 30分-1時間 | スキップ条件の確認と修正 |
| T3 | 30分 | metricsにカテゴリ追加 |
| T4 | 1時間 | 質問セット設計 + 回答記録 |
| T5 | 1-2時間 | 既存パイプラインの手動実行 |
| T6 | 1時間 | 同一質問での再評価 + 比較 |
| T7 | 数日（通常運用） | 追加作業なし |
| T8 | 2-3時間 | データ分析 + 評価レポート |

**T1-T3の完了まで:** 半日
**T4-T6の完了まで:** T1-T3後、数サイクル運用を挟んで半日
**T8まで:** 1-2週間（大部分は運用の待ち時間）

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| curiosity枠でWeb検索失敗が再発 | 成功率低下 | T1修正前の成功率66%は許容範囲。depth調整で改善可能 |
| fine-tuningで既存能力が退行 | 応答品質劣化 | バックアップからロールバック。免疫系が検知する設計 |
| 空サイクルのスキップ条件がSleepyJean内部に深く組み込まれている | T2の修正コストが予想より高い | T1のcuriosity保証だけでも部分的に解決 |
| 夢信号のorigin.contextが想定と異なる | T3のカテゴリ分類が動かない | wake_export.pyを確認してから実装 |

---

*本文書はタスクフローの計画であり、実行結果に応じてタスクの追加・変更がありうる。*
