# perceiveパラメータ調整（M1-2前提修正）

**目的:** perceive結果が場の信号分布を適切に反映するようにする
**前提タスク:** M1-1完了済み
**後続タスク:** M1-2ベースライン再収集（3日間）
**設計文書参照:** shared_field_design.md §4.3, §6.3 / bio_ai_architecture.md §5.5 / llamarcute_live_design.md §3.4

---

## 背景

M1-2のベースライン収集で、field_snapshotsの全44件がsignal_count=10、origin 100% difficultyという結果になった。調査の結果:

- 場には597信号が存在し、5種類のoriginが分布している（dialogue 146, knowledge_update 143, dream 104, difficulty 104, rotation_task 96）
- difficulty信号の平均ノルムが45.51（×2.0倍率による）で他の信号（25-27）を大幅に上回る
- perceiveのmax_signals=10 + strength=decay×normの線形計算により、高ノルム信号がtop-10を独占

2段階で修正し、M1-2のベースラインを再収集する。

---

## ステップ1: ×2.0廃止 + max_signals=30

### スコープ

difficulty信号のノルム倍率を削除し、perceiveのmax_signalsを拡大する。

### 変更箇所

**dialogue.py（emit_experience内）:**
- difficulty信号のembeddingに対するノルム×2.0の乗算を削除
- emitされるembeddingはFieldEncoder.encode_for_emit()の出力そのまま

**perceiveのデフォルトパラメータ:**
- max_signals: 10 → 30
- 変更箇所: PerceiveParamsのデフォルト値、またはDialogueManager/OrchestratorでのPerceiveParams生成箇所

**設計文書:**
- llamarcute_live_design.md §3.4: difficulty信号のノルム仕様を「FieldEncoder出力そのまま（倍率なし）」に更新
- llamarcute_live_design.md §3.4: max_signalsの記述を30に更新
- shared_field_design.md §4.3: PerceiveParamsのmax_signalsデフォルトを30に更新（仕様上のデフォルト50は維持してもよい。運用値として30を使う形でも可）

### 受入基準

- [ ] difficulty信号のemitにノルム倍率が適用されていないこと（emitログで確認）
- [ ] perceive結果にdifficulty以外のorigin（dialogue, dream, knowledge_update等）が出現すること
- [ ] 既存テスト全PASS
- [ ] field_snapshotsにorigin多様性のあるレコードが記録されること（1回の対話で確認可能）

### 判断基準（実装者向け）

- difficulty信号の×2.0は dialogue.py の emit_experience() 内にある。grep `* 2` や `norm` で特定
- max_signalsの変更はPerceiveParamsのデフォルト値またはインスタンス生成箇所。複数箇所で生成されている場合はconfig化を検討するが、スコープを広げすぎない
- 既存のfield_snapshotsデータ（44件）は削除しない。比較用に保持

### 確認手順

1. bot起動後、1回対話する
2. field_snapshotsの最新レコードのsignal_count_by_originを確認
3. difficulty以外のoriginが含まれていれば受入基準PASS
4. 含まれていなければ、snapshot()で場の全信号を確認し、perceiveのstrength計算が正しいか調査

---

## ステップ1確認運用（1日）

### スコープ

ステップ1適用後、1日間の通常運用でperceiveの結果を確認する。

### 確認項目

- [ ] origin分布: difficulty以外の信号がperceive結果に含まれるか
- [ ] signal_count: 30件上限に常に張り付いていないか（張り付いていたらmax_signalsをさらに上げるか、min_strengthの調整が必要）
- [ ] strength分布: 最大strengthと最小strengthの比率（ダイナミックレンジ）を記録

### go/no-go判定

- **ステップ2に進む条件:** origin分布が改善されたが、まだ偏りが残っている（例: difficulty + knowledge_updateが70%以上を占める）
- **ステップ2スキップ条件:** origin分布が十分に多様（5種類のoriginが全て出現し、特定originが50%を超えない）→ そのままM1-2ベースライン再収集に進む
- **追加調査条件:** 改善が見られない → perceive自体の実装を再確認

---

## ステップ2: べき乗圧縮（Weber-Fechner）

### スコープ

perceiveのstrength計算に非線形変換を導入し、高ノルム信号の支配を抑制する。

### 設計根拠

生体の受容体応答はWeber-Fechnerの法則に従い、刺激の対数（またはべき乗）に比例する。現状の線形strength（decay × norm）は生体的に不自然であり、高濃度信号が低濃度信号を完全に遮蔽する問題を引き起こす。べき乗変換により、濃度の序列は保存しつつダイナミックレンジを圧縮する。

**重要:** 場に保存されるembeddingは無変更。正規化禁止（bio_ai_architecture.md §5.5）に抵触しない。変換はperceive時の知覚計算にのみ適用される。

### 変更箇所

**shared_state/interface.py（PerceiveParams）:**
- `strength_exponent: float = 0.5` パラメータを追加
- 設計上のデフォルトとしてα=0.5。将来の調整用にパラメータ化

**shared_state/chromadb_backend.py（perceive実装）:**
- strength計算を変更:
  - 変更前: `strength = decay_factor × norm`
  - 変更後: `strength = decay_factor × norm^α`（α = strength_exponent）
- min_strengthの閾値はα適用後のstrengthに対して評価される

**shared_state/observer.py（FieldSnapshotLogger）:**
- strength_exponentの値をfield_snapshotsに記録（将来αを変更した際の比較用）
- field_snapshotsテーブルにstrength_exponent REAL カラムを追加

**設計文書:**
- shared_field_design.md §4.3: strength算出式を `decay × norm^α` に更新。Weber-Fechnerの根拠を記載
- shared_field_design.md §3.3: PerceivedSignalのstrengthフィールドの説明を更新
- llamarcute_live_design.md §3.4: perceiveの説明を更新

### 受入基準

- [ ] strength計算が `decay × norm^α` で行われていること（ユニットテストで確認）
- [ ] α=1.0のとき従来と同じ結果になること（後方互換性テスト）
- [ ] α=0.5のとき、ステップ1よりorigin分布が均一化されること
- [ ] PerceiveParamsにstrength_exponentが追加され、デフォルト0.5であること
- [ ] field_snapshotsにstrength_exponentが記録されること
- [ ] 既存テスト全PASS
- [ ] 設計文書3件が更新されていること

### 判断基準（実装者向け）

- α=0.5は初期値。M1-3の実験結果を見て調整する可能性がある
- min_strengthの閾値はα適用後の値で評価するため、現在の閾値(1.0)が適切かを確認すること。norm^0.5で計算するとstrengthの絶対値が変わるため、閾値の再調整が必要になる可能性が高い
  - 例: norm=25.5, decay=1.0 → 従来strength=25.5, 新strength=5.05
  - 現在のmin_strength=1.0は新計算でも概ね妥当だが、確認すること
- FieldReceptorのtransduce()には影響しない（FieldReceptorはperceive後のembeddingをそのまま受け取る。strengthはperceive内部のフィルタリングにのみ使用）

---

## ステップ2完了後 → M1-2ベースライン再収集

ステップ2の受入基準を満たしたら、M1-2のベースライン3日間の収集を再開する。

- field_snapshotsの既存データ（ステップ1以前の44件 + ステップ1運用中のデータ）は削除しない
- ベースライン期間のカウントはステップ2適用後のbot再起動時点から開始
- 3日間で睡眠サイクル最低3回を条件とする

---

## スコープ外

- difficulty信号の困難度比例ノルム（仕様の1.5〜3.0）の実装 → M1完了後に別タスクで検討
- perceiveのmin_strength閾値の最適化 → M1-3のデータから判断
- 他の信号種類の追加（感情、好奇心等）→ M1-3の結果次第
- difficulty signalからhomeworkへの変換の検索クエリ品質 → 別件として記録済み

---

## design_journal.md への記録事項

- difficulty信号×2.0の廃止理由: 設計根拠が薄い恣意的な定数。困難度比例の仕様すら未実装。perceive結果の偏りの直接原因
- max_signals 10→30: 10は下限として設定した値であり、場の信号種類の多様性を反映するには不十分だった
- Weber-Fechner圧縮の導入根拠: 生体の受容体応答は刺激の対数/べき乗に比例する。線形strengthは高濃度信号の支配を許し、場の情報多様性を損なう。embeddingの正規化禁止原則とは非抵触（perceive時の知覚計算にのみ適用、場のデータは無変更）
