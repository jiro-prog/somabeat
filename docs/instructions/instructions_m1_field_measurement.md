# M1: 場の状態読み取り可能性の測定

**目的:** 場の信号分布が個体の状態を反映しているか（yes/no）を判定する
**前提タスク:** なし
**成果物:** FieldSnapshotLogger実装 + ベースラインデータ + 条件振り実験結果
**設計文書参照:** shared_field_design.md §7（FieldObserver）、bio_ai_architecture.md §5

---

## 背景と動機

「個体としての統合」の成功基準を定義するため、場が自己モデルとして機能しているかを実験的に検証する。自己モデルは新機構として作るのではなく、既存の場の信号分布が既にその性質を持っているかを測定する。

M1は測定基盤の構築とベースライン取得。後続のM2（系統間一貫性）、M3（介入実験）はM1のデータ基盤の上に乗る。

---

## M1-1: FieldSnapshotLoggerの実装

### スコープ

FieldObserverの具象クラスを拡張し、perceiveのタイミングで場の状態指標をSQLiteに記録する。

### 記録する指標

| 指標 | 算出方法 | 意味 |
|------|---------|------|
| signal_count | perceive結果の信号数 | 場の活発さ |
| signal_count_by_origin | origin.context別の信号数（JSON） | 系統別の活動量 |
| strength_max | max(strength) | 最も強い信号の強度 |
| strength_median | median(strength) | 場の「平均的な温度」 |
| strength_std | std(strength) | 場の均一性 |
| raw_norm_mean | mean(‖embedding‖)（減衰前） | 放出時の濃度の傾向 |
| oldest_signal_age_hours | now - min(emitted_at) | 場の記憶の深さ |

### SQLiteスキーマ

```sql
CREATE TABLE field_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,          -- ISO 8601
    trigger_context TEXT NOT NULL,    -- "dialogue", "self_improvement", "manual"
    signal_count INTEGER NOT NULL,
    signal_count_by_origin TEXT,      -- JSON: {"dialogue": 5, "difficulty": 2, ...}
    strength_max REAL,
    strength_median REAL,
    strength_std REAL,
    raw_norm_mean REAL,
    oldest_signal_age_hours REAL,
    -- 対話との紐づけ
    dialogue_log_id INTEGER,          -- dialogue_logのidへの参照（対話時のみ）
    response_time_ms INTEGER          -- 応答時間（対話時のみ）
);
```

### 実装方針

- 既存のFieldObserver具象クラス（ログ出力用）に記録メソッドを追加
- on_perceive内でFieldPerceptionから指標を算出してINSERT
- DialogueManager.process_input()から呼び出し時にdialogue_log_idとresponse_time_msを渡す
  - FieldObserver Protocolに新メソッドは追加しない。FieldSnapshotLoggerの具象メソッドとして実装し、DialogueManagerから直接呼ぶ
- テーブル作成はbot起動時のDB初期化に追加

### 受入基準

- [ ] 対話1回ごとにfield_snapshotsに1行記録される
- [ ] signal_count_by_originのJSONが正しくパースできる
- [ ] dialogue_log_idで対話ログと結合クエリできる
- [ ] 既存テスト225件がPASSのまま

### 判断基準（実装者向け）

- FieldObserver Protocolへの変更は不要（具象クラスの拡張のみ）
- perceive結果のSignalオブジェクトから全指標が算出可能。ChromaDB直接アクセスは不要
- 記録のオーバーヘッドがperceiveの処理時間（~0.01s）を有意に増やさないこと

---

## M1-2: ベースライン収集（3日間）

### スコープ

M1-1実装後、通常のDiscord運用でデータを蓄積する。

### 作業

1. M1-1実装済みの状態でbot再起動
2. 3日間（睡眠サイクル最低3回）通常運用
3. 運用終了後、以下を確認:
   - field_snapshotsのレコード数が対話数と一致
   - 各指標の分布（min, max, mean, std）を集計
   - 指標間の相関（signal_countとstrength_medianの関係等）

### 受入基準

- [ ] 3日分のfield_snapshotsデータが欠損なく記録されている
- [ ] 各指標の分布レポート（簡易な集計SQLで十分）
- [ ] 明らかな異常値や記録漏れがない

### 判断基準

- 分布が極端に偏っている場合（例: signal_countが常に同じ値）は指標の有用性を再検討
- ベースライン期間中に睡眠サイクルが3回未満の場合、延長する

---

## M1-3: 条件振り実験

### スコープ

意図的に対話の条件を変え、場のスナップショットが条件間で区別可能かを検証する。

### 実験条件

| 条件 | 内容 | 対話数 | 例 |
|------|------|--------|-----|
| A: 通常 | llamarcute-liveの知識範囲内の雑談 | 10 | 「最近面白いことあった？」「好きな食べ物は？」 |
| B: 困難 | 複雑な推論や専門知識を要する質問 | 10 | 多段推論、矛盾を含む質問、計算問題 |
| C: 知識外 | llamarcute-liveが明確に知らない領域 | 10 | 未学習の専門用語、存在しない概念についての質問 |

### 手順

1. 条件A→B→Cの順に各10対話を実施（1日で完了可能）
2. 各対話にfield_snapshotが紐づいていることを確認
3. 条件間で以下を比較:
   - signal_count（条件間で差があるか）
   - strength_median（場の「温度」が変わるか）
   - strength_std（場の均一性が変わるか）
   - signal_count_by_origin（difficulty信号の比率が変わるか）

### 分析方法

- 条件ごとの各指標の箱ひげ図（手動でもSQLクエリ+目視でも可）
- 条件AとBの差、AとCの差が、ベースライン期間中の自然変動より大きいかを判定
- 判定方法: M1-2で得たベースラインのstd × 2を超える差があれば「区別可能」と仮判定

### 受入基準

- [ ] 30対話分のfield_snapshotsデータ（条件ラベル付き）
- [ ] 条件間の指標比較レポート
- [ ] yes/no判定:「場の信号分布は対話条件によって区別可能か」

### 判断基準

- **yesの場合:** M2（系統間一貫性）に進む。どの指標が最も弁別力が高いかを記録
- **noの場合:** 場が自己モデルとして機能していない可能性。以下を検討:
  - perceiveのパラメータ（min_strength, max_signals）が粗すぎないか
  - 信号の種類が少なすぎないか（ここで初めて信号種類の拡張を検討）
  - 場自体がノイズに埋もれていないか

---

## スコープ外（M1では扱わない）

- 信号種類の追加（感情、好奇心等）→ M1-3の結果次第
- 免疫系の設計 → M1-3がyesかつM2完了後
- FieldReceptorスケーリングの修正 → 独立タスク
- 視覚モジュール有効化（Phase D）→ 独立タスク

---

## design_journal.md への記録事項

以下の設計判断を記録する:
- 「自己モデルは場の創発的性質として既に存在しうる。新機構の前にまず測定」
- 「個体としての統合」の暫定的な成功基準:「場の信号分布が、複数の系統によって整合的に読み取られ、各系統が独立に適応的な応答を返す状態が観測されること」
- M1→M2→M3の依存関係と、各段階のgo/no-go基準
