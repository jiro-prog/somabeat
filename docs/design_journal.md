# 設計ジャーナル

---

## 2026-04-01 M1-2: ベースラインv1/v2分離 + 再収集開始

### 判断: KG統合後のベースライン再収集
- **状況:** M1-2ベースライン132件のうち、3/29-3/30（46件）はKG導入前、3/31-4/1（86件）はKG導入後。信号特性に断層あり（avg_signal: 10→30、strength_median: 46→5）
- **選択肢:** A: 既存データをそのまま使う / B: v1/v2でラベル分離し、KG統合後のv2で3日間再収集
- **決定:** B
- **根拠:** ベースラインは「このシステム構成での通常動作パターン」の記録。KG導入前のデータではM1-3の比較対象として不正確
- **リスク:** 3日間のコード凍結（全体）

### 実装: baseline_versionカラム追加
- field_snapshotsテーブルに `baseline_version TEXT NOT NULL DEFAULT 'v2'` を追加
- 既存132件を `v1` にUPDATE（比較クエリ用に保持、削除しない）
- マイグレーションはinit_db()のALTER TABLE + UPDATEで実装

### 安定判定基準（M1-2 → M1-3 go/no-go）
- day2-day3間のavg_strength_medianの変化が、day1-day2間の変化の50%以下 → 安定、M1-3へ
- 上記を満たさない → 1日延長して再判定
- v2収集期間中はコード変更凍結（全モジュール）

---

## 2026-04-01 KG導入・Ollama除去 Phase 1: generate_bare実装 + Ollama除去

### 判断: FieldAwareLLM一本化（Ollama除去）
- **状況:** fitness/cuteness評価、wake message生成、対話フォールバック等がOllama経由だった。VRAM排他制御（unload/reload）が複雑化の原因
- **決定:** generate_bare（input_idsベース、field注入なし）をFieldAwareLLMに追加し、全Ollama呼び出しを置換
- **根拠:** タスクフロー instructions_kg_ollama_removal.md Phase 1 に基づく。モデル常駐化によりunload/reloadが不要に
- **リスク:** generate_bareの品質がOllamaと異なる可能性（同一モデルのため影響は軽微と予想）

### 変更内容
- `llm_inference.py`: `_postprocess()` 共通メソッド抽出、`generate_bare()` 追加（input_idsベース）
- `fitness.py`: `ollama_client.chat()` → `llm.generate_bare()` に全置換。`ollama_model`/`ollama_parallel` → `llm`/`parallel`
- `cuteness.py`: `ollama_client.chat()`/`chat_messages()` → `llm.generate_bare()` に全置換
- `self_improve.py`: Ollamaフォールバック削除（llm=None時はスキップ）
- `dialogue.py`: Ollamaフォールバック削除
- `discord_bot/bot.py`: wake message生成を `orchestrator.llm.generate_bare()` に移行
- `bridge/sleep_ingest.py`: `_get_dialogue_context()` のLLM要約をテキスト切り詰めに簡素化
- `orchestrator.py`: `_unload_ollama()`/`_wait_for_ollama()` 削除、睡眠シーケンス簡素化（unload/reload不要）
- `ollama_client.py` 削除、`test_ollama_client.py` 削除
- config: `ollama_parallel` → `eval_parallel` にリネーム

### テスト
- 257件PASS（256元 + 6新規 - 5 ollama_client削除）
- generate_bare テスト6件: 応答返却、system_prompt反映、交互呼び出し、think除去、no_think制御

### 発見: generate_bareで入力トークンがストリップされていなかった（致命的バグ）
- **内容:** `model.generate(input_ids=...)` は入力+生成トークンの全シーケンスを返す。入力部分をストリップせず `_postprocess` に渡していたため、応答にプロンプト全文が混入。cuteness会話が指数膨張（373→8597トークン）、fitness core_accuracy 0.27に低下
- **対処:** 修正済み。`output_ids[:, n_input:]` で入力トークンを除外
- **影響:** 初回sleep中に発覚。fitnessは次回サイクルで再検証

### Phase 1 ゲート達成（2回目sleep）
- fitness: mutation_25_B **0.80** (目標0.80-0.85範囲内)
- 人格更新: v24→v25
- generate_bareでの自己改善正常動作を確認
- タイムアウト1800s→3600sに延長（GPU排他シリアル実行のため）
- eval_parallel: 2→1に変更（セマフォオーバーヘッド回避）

---

## 2026-04-01 KG導入・Ollama除去 Phase 2: KnowledgeGraph + 睡眠時トリプル抽出

### 判断: KGベースReconsolidation（HDBSCAN完全置換）
- **状況:** SleepyJeanのReconsolidationがHDBSCANクラスタリングベースだった。KGベースに置換
- **決定:** KnowledgeGraph (NetworkX+SQLite) + TripleExtractor (generate_bare) でReconsolidationを再実装
- **根拠:** タスクフロー K-1〜K-5。HDBSCANクラスタリングは削除、MemoryStoreはフォールバック保持
- **リスク:** トリプル抽出品質がLLMのプロンプト追従性に依存。chitchat日は0トリプル（正常動作）

### 変更内容
- `sleepyjean/knowledge_graph.py`: 新規。Triple, KnowledgeGraph (add_triples, query, pagerank, diff, prune, save/load)
- `sleepyjean/triple_extractor.py`: 新規。TripleExtractor (generate_bare, チャンク分割, パース)
- `sleepyjean/reconsolidation.py`: KGベースに全面書き換え。HDBSCAN依存を削除
- `sleepyjean/sleepyjean.py`: KnowledgeGraph, TripleExtractor統合。__init__にllm引数追加
- `orchestrator/orchestrator.py`: SleepyJean初期化にllm渡し
- `config/system.yaml`: kg, extraction設定ブロック追加

### テスト
- 270件PASS（+24件: KG 10, TripleExtractor 6, Reconsolidation 8）
- 旧HDBSCAN Reconsolidationテスト11件は書き換え

---

## 2026-04-01 KG導入・Ollama除去 Phase 3: Recallのテキスト経路化

### 判断: RecallEngine KGベース化 + テキスト経路追加
- **状況:** Recallがembedding emit経路のみだった。KGからテキスト取得→プロンプト注入経路を追加
- **決定:** 形態素解析(fugashi)でキーワード抽出→KG query→トリプルテキスト化→build_promptに注入
- **根拠:** タスクフロー T-1〜T-3。KGヒットなし時はMemoryStoreフォールバック
- **リスク:** 形態素解析の品質（UniDicに未登録の固有名詞は分割される）。KGが小さいうちはフォールバック頻度が高い

### 変更内容
- `sleepyjean/recall.py`: KGベースに全面書き換え。RecallResult導入。fugashiで形態素解析
- `sleepyjean/sleepyjean.py`: on_user_inputをsyncに変更、戻り値をRecallResultに
- `llamarcute_live/dialogue.py`: build_prompt(recall_text=)追加、recall呼び出しをsyncに
- テスト: 276件PASS（+6件）

### 保留事項
- dialogue.py/orchestrator.py に `ollama_model` パラメータが後方互換用に残存（使用されない）
- scripts/ 配下のOllama参照（run_pilot_t25.py等）は未修正（本番外）
- Phase 3ゲートの手動検証（対話でKG知識が応答に反映）は次回サイクルで実施

---

## 2026-04-01 sleep_ingest重複投入バグ修正 + クリーンアップ

### 発見: 対話ログが毎サイクル全件再投入されエピソードが重複蓄積
- **内容:** fetch_dialogue_logs()が「今日の全対話」を毎サイクル取得し、consolidate()で重複チェックなしにadd_episode()していた。96エピソード中ユニーク32個、最大6重複。recall(n_results=3)が同一内容のコピーを返し、教えた知識にリーチできない状態だった
- **対処:** 修正済み
- **根本原因:** 差分取得の仕組みがなかった

### 判断: ウォーターマーク方式で差分取得（architect判断: B）
- **状況:** 重複防止をconsolidate側(embedding比較)でやるか、fetch側(ID追跡)でやるか
- **選択肢:** A: consolidate()でembedding重複チェック / B: fetch_dialogue_logs()でウォーターマーク
- **決定:** B
- **根拠:** 根本原因は「既にingest済みのログを再取得している」こと。入力品質は入力側で保証すべき。consolidate側にフィルタを挟むのは責務の混在
- **リスク:** ウォーターマーク破損時は全件再取得にフォールバック（最悪=現状と同じ）

### 変更内容
- `bridge/sleep_ingest.py`: fetch_dialogue_logs()にウォーターマーク導入、save_ingest_watermark()追加
- `orchestrator/orchestrator.py`: sleepyjean_db_path渡し + consolidation成功後にウォーターマーク保存
- 既存重複64件をクリーンアップ（cosine sim>0.99、access_count高い方を保持）
- クラスタ再計算: 16→2クラスタ（重複によるアーティファクトが解消）

---

## 2026-04-01 SleepyJean大規模改修 完了

### 判断: Phase 4ゲート達成、改修完了
- **決定:** SleepyJean大規模改修（Phase 1-4）を完了とする
- **根拠:** 8サイクル完走、全指標が正常範囲内

### 改修内容
- SleepyJean非LLM化完了（ChromaDB + HDBSCAN + embedding検索）
- 覚醒時Recall導入（30回実行、avg_sim 0.831、99ms avg）
- 睡眠時Reconsolidation導入（クラスタ成長: 0→16、106 episodes）
- 同サイクル忘却保護バグ発見・修正
- VRAMリーク（既知、改修起因ではない）の修正強化、継続観察中

### 最終指標
| 指標 | 値 |
|------|-----|
| Reconsolidation clusters | 16 |
| Recall avg_sim | 0.831 |
| fitness combined | 0.80 |
| サイクル時間（正常時） | 8分 |
| テスト | 256件 PASS |

### 継続観察（改修のブロッカーではない）
- VRAMリーク: GPTQModel/Marlin unload問題。成功率53%、継続観察
- 覚醒メッセージ: recon_result直接参照に修正済み、次回完走時に確認

---

## 2026-04-01 設計適合監査 + バグチェック

### 判断: 設計適合監査 Phase 1+2 全項目OK
- **状況:** 定期監査（前回から数サイクル経過）
- **Phase 1:** check_design_invariants.py PASS（8種類の構造チェック）
- **Phase 2:** architect による意味的レビュー — テキスト不在、無指向性、場の接続、睡眠サイクル全てOK
- **改善提案（保留）:** immune health check が `_run_self_improvement()` 内部にあり、self_improvement のタイムアウト/例外で immune がスキップされる構造。独立ステップ化が望ましいが、immune が cycle_result を入力に使うため設計検討が必要。現時点では受容。

### 発見: wake_export の open questions カウント漏れ
- **内容:** `bridge/wake_export.py` で新しい open question の signal は正しく emit されていたが、summary dict にカウントキーがなく、ログにも出力されていなかった
- **対処:** 修正済み。`new_questions_emitted` キー追加、カウント実装、ログ出力・run_cycle.py の表示追加
- **影響:** 機能的影響なし（signal emit は正常動作）。テレメトリの欠落のみ

---

## 2026-04-01 VRAMリーク修正 + 覚醒メッセージ修正 + テストhook仕組化

### 修正1: FieldAwareLLM.unload() VRAMリーク（間欠的）

- **症状:** unload後のVRAMが16MiB（正常）にならず5819MiB（全リーク）になることが17回中8回。Ollamaが残り2.3GBで推論→サイクル8分→30分+に悪化
- **原因:** GPTQModel.load()のwrapperがローカル変数のまま放置→wrapper↔model間の循環参照がGCの気まぐれ次第。accelerateのdispatchフックも参照を保持
- **修正:**
  - `self._gptq_wrapper`でwrapperを明示保持、unload時に削除
  - `accelerate.hooks.remove_hook_from_submodules()`でフック除去
  - `gc.collect()` ×2回 + `torch.cuda.synchronize()` + VRAM検証ログ
- **リスク:** 修正後の実運用で再現するか確認必要（次回サイクルで検証）

### 修正2: _generate_wake_message CUDA OOM

#### 判断: encode_for_senseを廃止し、Reconsolidation結果を直接使用
- **状況:** 覚醒メッセージ生成が毎回CUDA OOMで失敗。Ollama占有中にGPUのtext encoderを呼んでいた。さらにsignal_id[:8]（UUID）を「学んだこと」として渡しており無意味
- **選択肢:** A: recon_result+dialogue_logsをOllamaに直接渡す / B: CPU フォールバック / C: 機能無効化
- **決定:** A
- **根拠（architect）:** 場のembeddingからテキスト情報を復元するBは設計と逆行。recon_resultとdialogue_logsはテキスト空間で利用可能な情報であり、テキストタスク（覚醒メッセージ）に直接使うのが正当
- **修正:**
  - `encode_for_sense`呼び出しを削除（GPU不使用、VRAM競合解消）
  - `recon_result`（クラスタ数/統合/忘却）と`dialogue_logs`（直近5件のユーザ発話）をOllamaプロンプトに埋め込み
  - `SenseParams` import削除

### 仕組化: pytest実行時のbot自動停止・再起動

- **状況:** CLAUDE.mdに「テスト前にbot停止」のルールがあるが、Claudeが忘れて違反
- **修正:** Claude Code hookで自動化
  - PreToolUse: pytestコマンド検出→bot_stop.sh→sentinel作成
  - PostToolUse: sentinel存在時→bot_start.sh→sentinel削除
  - botが動いていない場合は何もしない

### sleepyjean-diagnosis スキル刷新

- SKILL.md/diagnose.pyを旧SleepyJean→新アーキテクチャに全面書き換え
- 7チェック: Reconsolidation/Recall/MemoryStore/ChromaDB/VRAMリーク/Self-improvement/サイクル時間

### テスト結果
- 修正前: 256件PASS
- 修正後: 256件PASS

### 保留事項
- VRAMリーク修正の実運用検証（次回サイクルで53%→改善確認）
- 覚醒メッセージの実運用確認（wake_enabled: true で次回サイクル）

---

## 2026-03-31 SleepyJean大規模改修 Phase 1完了

### 実施内容
- R-2: sleepyjean/ ディレクトリ構造作成、old_sleepyjean/ 参照用コピー
- R-3: orchestrator.pyの旧SleepyJean subprocess呼び出しを無効化。覚醒時対話の正常動作を確認
- S-7: config/system.yaml にsleepyjean新設定（memory, recall, reconsolidation, gap, quality）追加
- S-1〜S-6: 5コアモジュール + 統合クラスを実装

### モジュール構成
| モジュール | ファイル | テスト数 |
|-----------|---------|---------|
| MemoryStore | sleepyjean/memory_store.py | 8 |
| RecallEngine | sleepyjean/recall.py | 6 |
| Reconsolidation | sleepyjean/reconsolidation.py | 10 |
| GapDetector | sleepyjean/gap_detector.py | 5 |
| QualityGate | sleepyjean/quality_gate.py | 5 |
| SleepyJean（統合） | sleepyjean/sleepyjean.py | 4 |

### テスト結果
- 修正前: 232件PASS
- 修正後: 270件PASS（+38件）

### 判断: R-3でFieldAwareLLM.unload()の早期実行も無効化
- **状況:** 旧SleepyJeanのOllama用にenter_sleep冒頭でunloadしていたが、新SleepyJeanはLLMを使わない
- **決定:** self_improvementが内部で自前のunload/load管理をするため、早期unloadは不要。無効化
- **根拠:** self_improve.py L543-545で mutation generation後にllm.unload()を実行。Ollama用VRAMの確保はself_improvement内部で完結
- **リスク:** なし。self_improvementのVRAM排他制御に変更なし

### 発見: hdbscanライブラリが未インストール
- **対処:** pip install hdbscan で追加。requirements.txtへの追加はPhase 3で整理

### 保留事項
- hdbscanのrequirements.txt追加 → Phase 3
- old_sleepyjean/ のgit管理方針 → Phase 3完了時に判断
- 旧config項目（root_path, night_cycle_script等）の除去 → Phase 3

## 2026-03-31 SleepyJean大規模改修 Phase 3完了

### 実施内容
- N-1: bridge/sleep_ingest.pyに新関数(fetch_dialogue_logs, run_reconsolidation)追加
- N-2: orchestrator.pyの睡眠シーケンスを新設計に書き換え（対話ログ取得→Reconsolidation→self_improvement→purge→wake_up）
- N-3: グリンパティック処理(orchestrator/glymphatic.py)を実装（emit_log/dialogue_log老廃物除去）
- N-4: 旧SleepyJean依存の完全除去（旧configキー、_run_night_cycle、cleaner、旧model_separationテスト）

### テスト結果
- 修正前: 273件PASS
- 修正後: 255件PASS（旧cleaner 15件 + 旧model_separation 4件 削除、reconsolidation_failure 1件追加）

### 睡眠サイクル完走確認（チェックポイント）
- N-1/N-2完了後に手動/sleepでReconsolidation単体完走を確認
- `fetch_dialogue_logs: 23 entries` → `Reconsolidation: 11 episodes, 0 clusters, 0 new, 0 merged, 11 forgotten`
- 初回のためクラスタ未形成→全noise→全忘却は想定通り。複数サイクルで蓄積すればクラスタ形成
- self_improvement以降も正常動作、FieldAwareLLM reload成功

### 判断: cleanerセクションとbridge/cleaner.pyの呼び出しを丸ごと削除
- **状況:** cleanerは旧SleepyJeanのChromaDB/LoRAデータを掃除していた。新SleepyJeanはこれらを使わない
- **決定:** A（完全削除）。Soの指示
- **根拠:** 旧データはもう生成されない。手動掃除が必要ならその時にスクリプトを書く。enabledフラグで残すのは死んだコード
- **注意:** 旧データ（/home/jiro/sleepyjean/data/chromadb、/home/jiro/sleepyjean/models/lora）のディレクトリ自体はファイルシステムに残存。コードからの参照のみ削除

### 判断: _unload_ollamaのフォールバックから旧sleepyjeanモデルを除去
- **状況:** /api/ps失敗時のフォールバックでsleepyjeanモデルをunloadしようとしていた
- **決定:** llamarcute-liveモデルのみに変更
- **根拠:** 新SleepyJeanはOllamaを使わない。sleepyjeanモデルのunloadは不要

### 保留事項
- 旧データの物理削除 → 別途判断
- scripts/run_cycle.py等の旧スクリプトは参照のみ残存（実行はされない）
- Phase 4（統合テスト・性能テスト）は別タスク

## 2026-03-31 バグチェック・設計監査・テスト整理

### バグ修正（6件）
1. **recall.py**: confidence < 1e-3のガード追加 + norm比較を`> 1e-8`に（architect判断B: 現行設計維持）
2. **sleepyjean.py**: `config.get("sleepyjean", config)` → `config.get("sleepyjean", {})` に修正
3. **reconsolidation.py**: `_load_previous_clusters`のexceptにjson.JSONDecodeError, KeyError追加
4. **reconsolidation.py**: whitespace-onlyコンテンツのスキップ追加
5. **quality_gate.py, gap_detector.py, reconsolidation.py**: `norm == 0` → `norm < 1e-8` に変更（3ファイル）

### 設計監査結果
- **PASS**: 10ファイル中8ファイル
- **recall.pyのノルム上書き**: architect判断でB（設計違反ではない）。RecallEngineがemitするのは「記憶想起」信号であり、濃度はbase_norm * confidenceで制御する正当な設計
- **bridge/sleep_ingest.pyのlegacy部分**: LLM呼び出しを含むが、新Phase 3パスからは呼ばれない。削除予定のlegacyコード

### バグ修正追加: MemoryStoreのChromaDBがEphemeralClientだった
- **状況:** configに`chromadb_persist_directory`が未設定 → EphemeralClientが使われ、プロセス再起動でChromaDBのデータが全消失。SQLiteのメタデータだけ残りChromaDB本体と不整合
- **修正:** config/system.yamlに`chromadb_persist_directory: "data/sleepyjean_chromadb"`を追加。PersistentClientが使われるようになった
- **横展開確認:** memory_store.pyは既にconfigからの分岐を実装済み。テストはEphemeralClientのままで正しい（揮発で問題ない）
- **データクリーンアップ:** 孤立したSQLiteメタデータとreconsolidation_stateを削除

### バグ修正追加: 同サイクル新規エピソードの即座忘却防止
- **状況:** Reconsolidation.consolidate()がステップ1でadd_episodeした直後、ステップ5の忘却判定でaccess_count=0のnoiseとして即削除。初回サイクルでMemoryStoreが永遠に空のまま
- **根因:** 忘却判定が「同サイクルで生まれたばかりのエピソード」を区別していない
- **修正:** ステップ1でadd_episodeしたIDをsetで保持し、ステップ5の忘却判定でスキップ。記憶は最低1サイクル生存する
- **テスト追加:** test_new_episodes_protected_from_same_cycle_forgetting（11件目）

### テストコード整理
- conftest.py に共通fixture（store, rand_emb, make_field, make_encoder等）を集約
- グローバル `_counter` によるコレクション名一意化 → `uuid.uuid4()` + `tmp_path` に置き換え
- 手動 `store.close()` → conftest.pyのfixture teardownに統一（一部はまだ手動。Reconsolidationテストは独自store生成のため）
- 重複ヘルパー関数の削除

---

## 2026-03-31 SleepyJean大規模改修 Phase 2完了

### 実施内容
- D-1: DialogueManager.process_input()の先頭にRecall呼び出しを追加（perceive_fieldの前）
- D-2: orchestrator.pyでSleepyJeanインスタンスを生成しDialogueManagerに渡す
- D-3: Phase 2テスト3件追加（graceful degradation, recall呼び出し確認, 失敗時非致死）

### テスト結果
- 修正前: 270件PASS
- 修正後: 273件PASS（+3件）

### 動作確認
- bot起動→Discord対話→Recallログ出力確認
- `[TIMING] recall: 0.042s (0 memories emitted)` — MemoryStore空で0件、42ms、エラーなし
- 応答時間19.1秒（Recall追加による劣化なし。42msは全体の0.2%）

### 設計上の注意点
- SleepyJean=Noneの場合はRecallスキップ（graceful degradation）
- Recall失敗時はwarningログのみで対話続行（non-fatal）
- Recallがemitした信号は直後のperceive_field()で自然に拾われる。追加ロジック不要

---

## 2026-03-30 perceiveパラメータ調整（M1-2前提修正）

### 判断: difficulty信号×2.0の廃止
- **状況:** perceive結果の全44件がsignal_count=10、origin 100% difficulty。difficulty信号のノルムが×2.0倍率で他の信号を支配
- **選択肢:** A: 倍率を下げる / B: 倍率を完全撤廃
- **決定:** B
- **根拠:** タスクフロー指示。×2.0は設計根拠が薄い恣意的な定数。困難度比例の仕様（1.5〜3.0）すら未実装。将来の困難度比例ノルムはM1完了後に別タスクで検討
- **リスク:** difficulty信号が他の信号と同等のノルムになることで、困難度の優先知覚が失われる。M1-3のデータで評価

### 判断: max_signals運用値の確認
- **状況:** max_signals=10（スクリプト）だった。orchestratorのデフォルトは既に30
- **決定:** 変更不要（orchestratorで30を使用済み）
- **根拠:** 10は信号種類の多様性を反映するには不十分。30は5種類のoriginを含む場で適切

### 判断: Weber-Fechner べき乗圧縮の導入（α=0.5）
- **状況:** difficulty×2.0の廃止後も、高ノルム信号が知覚を支配する構造的問題が残る可能性
- **選択肢:** A: 線形strengthのまま / B: べき乗圧縮（α=0.5）
- **決定:** B
- **根拠:** タスクフロー指示。生体の受容体応答はWeber-Fechnerの法則に従い、刺激のべき乗に比例する。embeddingの正規化禁止原則とは非抵触（perceive時の知覚計算にのみ適用、場のデータは無変更）
- **リスク:** α=0.5は初期値。M1-3の実験結果で調整が必要になる可能性がある。min_strength閾値（1.0）はnorm^0.5で計算後も概ね妥当だが、モニタリング必要

### テスト結果
- 修正前: 230件PASS
- 修正後: 232件PASS（+2件: strength_exponentの後方互換性テスト、圧縮比テスト）

### 設計文書更新
- shared_field_design.md §3.3, §4.3: strength算出式を`decay × norm^α`に更新、Weber-Fechnerの根拠を記載
- llamarcute_live_design.md §3.4: difficulty信号のノルム仕様を「FieldEncoder出力そのまま」に更新、max_signals運用値30を追記、strength計算式を更新

### 発見: 初回snapshot（id=45）でdialogue/dream/rotation_task不在
- **内容:** 修正後の初回snapshotでorigin=knowledge_update 8, difficulty 2。dialogue等が不在
- **対処:** 想定通り。半減期24h・2日経過でdecay=0.25、norm=25.5のdialogue信号のstrength≈1.26（閾値付近）。3日前は0.63で閾値以下。新しいdialogue信号がemitされればdecay≈1.0でstrength≈5.05になり出現する
- **確認運用の判定基準:** (1) 対話5回以上後のsnapshotでdialogue originが出現するか (2) 睡眠サイクル後にknowledge_update/dreamの新鮮な信号が出るか

### 発見: config/system.yamlのmax_signals=10がコードデフォルト30を上書きしていた
- **内容:** orchestratorのコードデフォルト30を確認してOKとしたが、config/system.yamlで10に上書きされていた。id=45〜49のsnapshot全てsignal_count=10だった原因
- **対処:** config修正(10→30)、横展開チェック実施、CLAUDE.mdに横展開ルール追加、PostToolUse hook追加

### 確認運用結果: PASS（id=51）
- signal_count=30、origin 5種類（knowledge_update 14, dialogue 6, dream 5, difficulty 4, immune:rollback 1）、max share 47%
- ステップ2スキップ条件（5種類全出現・特定origin 50%未満）を満たす
- **M1-2ベースライン再収集を2026-03-31から開始。3日間+睡眠サイクル3回で2026-04-03に判定**

### 保留事項
- difficulty信号の困難度比例ノルム（1.5〜3.0）→ M1完了後
- min_strength閾値の最適化 → M1-3のデータから判断

---

## 2026-03-30 寝言英語化 + レポートQA空欄の修正

### 発見: 寝言がQwen3の思考モード（英語）で出力される
- **内容:** SleepyJeanの寝言生成がQwen3の`thinking`出力（英語）をそのまま使用していた。Ollama APIでは`/no_think`（transformers用）ではなく`think: false`パラメータが必要。
- **対処:** 修正済み。sleepyjean/scripts/night/local_llm.py と llamarcute_live/ollama_client.py の両方に`think: false`を追加。
- **影響範囲:** Ollamaを使う全ての生成（寝言、QA生成、夢日記、wake message、fitness/cuteness評価）

### 発見: 朝のレポートファイルでQA欄が空
- **内容:** morning_report.pyが`question`/`answer`キーを期待しているが、data_generatorは`instruction`/`output`キーで生成。キー不一致でレポート上は空欄に見える。QA自体はSQLiteのrotation_tasksに正常に格納済み。
- **対処:** 修正済み。morning_report.pyで両方のキー名にフォールバック。

---

## 2026-03-30 仕様適合監査 — 設計原則違反の修正

### 判断: Signal dataclassをfrozen=Trueに変更、extraフィールドを削除
- **状況:** 監査でSignalがmutableであり、extra: dictフィールドがテキスト注入経路になることを発見。設計原則5.3「通信と可読性の完全分離」に構造的に反する
- **選択肢:** A: extraを残してドキュメントで「使うな」と書く / B: extraを削除して構造で守る
- **決定:** B
- **根拠:** 設計原則5.3は「テキストを読んでプロンプトに注入するという設計逸脱を構造的に不可能にする」と明言。ドキュメントベースの制約はドリフトする
- **移行:** wake_exportの_emit_qa_pairs（field経由Q&A転送）を削除。本番パスは既に_transfer_qa_pairs（SQLite直接転送）を使用済み。run_t28_cycles.pyもSQLite読み取りに移行
- **リスク:** 既存ChromaDBに保存済みのextra_json付きsignalは読み飛ばされる（影響なし、backendはextra_jsonを無視するようになった）

### 判断: PurgeCriteriaからorigin_system/origin_contextフィールドを削除
- **状況:** PurgeCriteriaにorigin系フィルタがあり、purgeでもorigin指定ができる状態。shared_field_design.md 6.1で「origin指定によるフィルタリング」は意図的に排除と明記されているが、実装では存在
- **選択肢:** A: purgeは維持管理なのでorigin filterを許容 / B: purgeでも非指向性を貫く
- **決定:** B
- **根拠:** 無指向性原則はperceive/senseだけでなく場の全操作に適用すべき。origin filterを使っていたimmune.pyのrollback時purgeは、高ノルムrollback信号の自然減衰で代替
- **リスク:** immune rollback後にstale self_improvement信号が残存するが、decay半減期24hで自然消滅。次のsleepサイクルでの通常purge（older_than=7d）でも最終的に除去される

### 判断: FieldObserverからon_senseメソッドを削除
- **状況:** senseは非推奨だがon_senseがFieldObserver Protocolに存在。on_senseはquery_text引数を受け取りログにテキスト出力しており、「テキスト不在」原則に反する
- **決定:** on_senseをProtocol・実装・呼び出し元から完全削除
- **根拠:** sense自体が非推奨であり、observerのProtocolに残す正当性がない。呼び出し箇所はchromadb_backendの1箇所のみ

### 判断: 設計文書を実装に合わせて3点更新
- **状況:** (1) 入眠シーケンスの「最終field書き込み」は実装に不在 (2) 3.2節のカテゴリ構造がv0のまま (3) 不変制約の配置先が未記載
- **決定:** (1) 入眠シーケンスから削除（対話ターンごとのemitでカバー済み） (2) v17のカテゴリ構造に更新 (3) build_promptハードコード方針を明記
- **根拠:** いずれも設計ジャーナル(2026-03-28)で承認済みの判断を文書に反映

### 発見: wake_exportのQ&A転送が二重経路で存在
- **内容:** _transfer_qa_pairs（SQLite直接転送、本番使用）と_emit_qa_pairs（field経由、スクリプトのみ使用）が併存していた。_emit_qa_pairsはSignal.extraにinstruction/outputを格納しており、設計原則5.3に反する
- **対処:** _emit_qa_pairsを削除。run_t28_cycles.pyをSQLite読み取りに移行。修正済み

### テスト結果
- 修正前: 225件PASS
- 修正後: 225件PASS（件数変動なし。テストの修正はassertionの更新のみ）

### 判断: FieldAwareLLM.unload()からto("cpu")を除去
- **状況:** 2回目以降のsleepサイクルでVRAM 2057MiB残留→システムRAM枯渇→Ollama起動不能→night cycle全段階失敗。to("cpu")がシステムRAMを消費する副作用が原因
- **選択肢:** A: to("cpu")除去、del + gc.collect() + empty_cache()のみに / B: プロセスメモリ削減 / C: Ollamaメモリ要件削減
- **決定:** A
- **根拠:** to("cpu")はスタックフレーム参照問題の回避策だったが、gc.collect()追加後は不要。del self._modelで属性参照切断→gc.collect()で循環参照回収→empty_cache()でCUDAページ返却、の3段階で十分
- **リスク:** GPTQラッパーの循環参照がgc.collect()で回収されない場合、VRAM残留が再発。その場合はB案に切り替え
- **検証:** 2回連続load→unloadテスト PASS。Round 1: 5810→8 MiB、Round 2: 5810→8 MiB。to("cpu")なしで完全解放を確認

### 判断: VRAM排他制御の網羅的修正（3件）
- **状況:** to("cpu")除去後、残存する排他制御の穴を網羅監査で3件検出
- **Finding 1 (CRITICAL):** finally句でllm.load()の前にOllamaをunloadしていない。self_improvementのタイムアウト/失敗時にOllama+LLMが共存
- **Finding 2 (HIGH):** bot.pyの_generate_wake_messageがLLMロード後にOllamaを呼ぶ。毎回の正常な覚醒で排他違反
- **Finding 3 (MEDIUM):** _unload_ollama()がqwen3:8bだけunloadしてsleepyjeanモデルを放置
- **修正:**
  1. finally句: wake message生成（Ollama使用）→ _unload_ollama() → llm.load() の順序に変更
  2. _generate_wake_message()の結果を_pending_wake_messageに事前格納、wake_up()では送信のみ
  3. _unload_ollama()を/api/psで全ロード済みモデルを検出して全てunload。fallbackで両モデル名を使用
- **検証:** 225件テストPASS

### 発見: difficulty signal→homework→検索クエリの品質問題
- **内容:** 宿題テーマ「この質問への回答が難しかった: 応答に23秒かかった質問: やっほー」がそのまま検索クエリになり、無関係なPDFしかヒットしない。difficulty signalの生テキストが検索に不適
- **対処:** 保留。Ollama起動不能の問題とは独立。M1-2ブロック要因ではないため後回し

### 判断: 設計適合監査の定期実行方式
- **状況:** 第1層（構造チェック・hook）は既知の違反を検出するが、新種の違反はカバーしない。第2層（意味的レビュー）を3-5サイクルに1回自動実行する方法を検討
- **選択肢:** A: Phase 1のみorchestratorに組み込み、Phase 2はCLAUDE.md運用指針 / B: 全体をorchestrator subprocess化 / C: CLAUDE.md指針のみ
- **決定:** A
- **根拠:** architect判断 — Phase 1は静的検証でインフラレベル（orchestratorに許容）。Phase 2はLLM認知判断であり免疫系の機能。orchestratorに認知的判断を埋め込むのは設計原理1（三系統分離）に反する
- **実装:** wake_up()内でcycle_count % interval_cycles == 0の時にcheck_design_invariants.pyを実行。config: design_check.interval_cycles=4
- **第2層の運用:** CLAUDE.mdに「3-5サイクルに1回 /design-conformance を実行」と明記。新しい違反パターン発見時はcheck_design_invariants.pyにルール追加して第1層に昇格

---

## 2026-03-28 Marlinカーネル検証 (軸B)

### 判断: 推論エンジンをbitsandbytes NF4からGPTQ+Marlinに切り替え
- **状況:** FFN層のmatmul（全体の79%）がbitsandbytes NF4のdequant→FP16 matmulで律速。Marlin (W4A16 fused GEMV) への差し替えを検証
- **計測結果:**

| 構成 | tok/s | 応答時間 | ピークVRAM |
|------|-------|----------|-----------|
| NF4 + TQ + pruning (従来) | 1.2 | 53s | ~6.1 GB |
| Marlin + TQ + pruning | 14.6 | 14.2s | 5.72 GB |
| Marlin + FP16 KV | 25.9 | 8.5s | 5.68 GB |

- **決定:** 構成B (Marlin + FP16 KV) を採用。約21倍の速度改善
- **根拠:** RTX 3060 Tiのメモリバンド幅制約下で、NF4のdequant読み出し回数削減の効果が劇的。TurboQuantはMarlin環境下では純粋なオーバーヘッド (14.6 vs 25.9)
- **リスク:** GPTQモデル(AlphaGaO/Qwen3-8B-GPTQ)への依存。NF4パスは_load_nf4()として保持

### 判断: TurboQuantを本番パスから無効化（コード・テストは保持）
- **状況:** Marlin + FP16 KVで5.68GB、8GB以内に余裕。TurboQuantのKVキャッシュ圧縮は不要
- **決定:** config `kv_cache_bits: 0` で無効化。コード・テスト24件は残す
- **根拠:** (1) 将来のハードウェア変更時に必要になる可能性 (2) 実装自体の学び（Lloyd-Maxコードブック等）に価値
- **リスク:** なし。無効化であり削除ではない

### 判断: field KV pruningを無効化
- **状況:** 構成Bの25.9 tok/sはpruning無しの数字。FP16 KVで10トークンのattentionオーバーヘッドは誤差
- **決定:** 無効化。pruningロジックはTurboQuantCacheに密結合しており、FP16 KV時には適用されない
- **根拠:** コードの複雑さ削減。Marlinの速度ではfield 10トークンの影響は無視可能

### 発見: difficulty signal閾値(10s)がMarlin環境で初めて有意に機能
- **内容:** 従来53s/応答で全対話がdifficulty扱いだった。Marlin移行後(2-10s)で閾値が有効化。ただし200トークン超の正常応答が10s付近になるため、閾値が実質的に「長い応答」を検出している
- **対処:** 保留。運用データを見てから判断。応答時間ではなくユーザの再質問率等の別指標も候補

### 発見: GPTQモデルのnative dtypeはbfloat16（NF4はfloat16）
- **内容:** Marlin kernelはbfloat16/float16両対応だが、GPTQModel.load()はbfloat16で推論。field embedding注入時のdtypeはsys_embeds.dtypeに自動一致（既存コードで対応済み）
- **対処:** 修正不要。既存のdtype合わせロジックが正常動作

---

## 2026-03-28 日本語品質デバッグ

### 判断: TurboQuant 3-bit → 4-bit に恒久変更
- **状況:** perceive空テスト（field=0）でトークン崩壊が残存 → TurboQuant無効（FP16 KV）で深刻な崩壊が消滅、軽微な崩壊のみ残存 → 4-bit TQで深刻・軽微とも消滅
- **選択肢:** A: 3-bitのまま別のアプローチ / B: 4-bitに変更
- **決定:** B
- **根拠:** 3条件比較でトークン崩壊がTQ bit数に単調依存することを確認。4-bitでピークVRAM 6.1GB allocated、8GB以内
- **リスク:** マージン約1.9GB。将来的にcontext長が伸びた場合にVRAMが不足する可能性

### 判断: NF4由来の低頻度外来語崩壊を受容
- **状況:** FP16 KVでも「コンピッタ」「クォーブit」が微妙に残存
- **決定:** 受容
- **根拠:** bitsandbytes NF4量子化の限界。ハードウェア制約内で解消不能。ユーザ体験としても致命的ではない

### 判断: マックポークのハルシネーションを量子化と分離して記録
- **状況:** 全条件（3-bit TQ, FP16, 4-bit TQ）でマックポークの事実誤認が残存
- **決定:** Qwen3-8Bの知識限界として記録のみ。量子化の問題ではない
- **根拠:** 条件を変えても事実誤認の内容は変わるがハルシネーション自体は消えない

### 判断: 存在の宣言を不変制約に配置、identityカテゴリは進化可能
- **状況:** system promptに人格定義がなく、Qwen3デフォルトの自己認識が出る
- **選択肢:** A: 全てを不変制約に / B: 存在の宣言（名前・役割）は不変制約、自己の在り方は進化可能
- **決定:** B
- **根拠:** 「私はllamarcute-liveである」は事実の記述で変異させる意味がない。「好奇心を持って接する」は人格の質であり進化可能であるべき
- **リスク:** カテゴリ再編（tone + knowledge_attitude → identity）でv16→v17の構造的変更。T11長期観察データとの連続性にノイズ

### 判断: 行動規範のカテゴリ再編（tone + knowledge_attitude → identity）
- **状況:** キャラクター崩壊修正のため、自己の在り方を統合的に扱うカテゴリが必要
- **決定:** 統合
- **根拠:** tone と knowledge_attitude は「自分をどう捉え、どう振る舞うか」という同一の責務。分離する設計上の理由がない

### 判断: 会話バッファをトークン数上限500に変更（回避策）
- **状況:** メッセージ数ベース(max=10)では入力長が予測不能。sys_embeds≧660で早期EOS発生。5往復は本番で容易に到達
- **選択肢:** A: トークン上限で入力長を制御（回避策） / B: FieldReceptorスケーリングの根本改修
- **決定:** A（回避策）。Bは近期の独立タスクとして切り出す
- **根拠:** 回避策で本番安定化が即座に得られる。根本修正は設計判断を含みスコープ外
- **リスク:** 会話履歴が~134トークン(2-3往復)に制限される。文脈保持能力が低下。FieldReceptorスケーリングのtext_norm/field_normがシーケンス長に依存する構造的脆弱性は未解決
- **実装:** build_prompt()でベースプロンプトのトークン数を計測し、残り予算で履歴を新しい方から詰める。LLMトークナイザ使用、未ロード時はUTF-8/3のfallback
- **検証:** バッファ充填状態(5往復)で「最近何か面白いことあった？」を3回 → sys=456-501, output=49-66, 早期EOS発生せず

### 発見: text_norm/field_normスケーリングのシーケンス長依存（根本原因）
- **内容:** 現在のスケーリング `scale = sys_embeds.norm(dim=-1).mean() / field_tensor.norm(dim=-1).mean()` はsys_embedsの長さに暗黙に依存。短い入力では正常(ratio=1.00)だが、長い入力でattentionパターンが不安定化し早期EOSを誘発
- **対処:** 保留。近期タスクとして独立調査を予定
- **候補:** α: 固定スケーリング係数 / β: field embeddingのノルムクランプ / γ: 注入位置の変更。いずれもFieldReceptor再学習は不要
- **設計原則との関係:** bio_ai_architecture.md 5.3節「場の信号の影響力はプロンプト長に依存すべきではない」。現状はこの原則に反している

### VRAM記録: 4-bit TurboQuant
- **モデルロード後(対話時):** allocated=6,122MiB (マージン2,070MiB)
- **睡眠サイクル(自己改善)ピーク:** 7,558MiB / 8,192MiB (マージン634MiB, 7.7%)
- **判定:** OOMなしで完走。ただしマージンが非常に薄い。自己改善の候補数(現在3)を増やす余地はない
- **回避策候補(将来):** 候補を3→2に減らす / 候補評価を逐次実行（並行ではなく1体ずつ）

### 発見: bot.logのバッファリング問題
- **内容:** `python -m discord_bot.bot > data/bot.log 2>&1` でstdoutがブロックバッファリングされ、ログがリアルタイムに書き込まれない。睡眠サイクル中のログが欠落
- **対処:** 保留。次回起動時に `python -u` で unbuffered にする
- **保留理由:** 品質デバッグのスコープ外

### 品質デバッグ完了サマリ
| 問題 | 原因 | 対処 | 状態 |
|------|------|------|------|
| トークン崩壊(深刻) | TurboQuant 3-bit KV | 4-bitに変更 | 根本解決 |
| トークン崩壊(軽微) | bitsandbytes NF4 | 受容 | 受容 |
| キャラクター崩壊 | 存在宣言の欠如 + カテゴリ構造 | 不変制約追加 + identity統合(v17) | 根本解決 |
| メタ注釈混入 | 不変制約の文言 | 命令形に修正 | 根本解決 |
| 英語フレーズ混入 | 行動規範YAML内の英語例文 | 例文削除 | 根本解決 |
| 途中切断 | max_new_tokens=256 | 512に変更 | 根本解決 |
| 早期EOS | field embedding + seq長の相互作用 | conv_buffer 500トークン上限 | 回避策 |
| ハルシネーション | Qwen3-8Bの知識限界 | 受容 | 受容 |

---

## 2026-03-28 応答速度改善（field KV pruning + perceive max_signals削減）

### 判断: perceive max_signals 30→10に削減
- **状況:** field=30で0.3-0.7 tok/s。field embeddingのdecodeコスト（GQA展開 × 全layer × 毎トークン）が支配的
- **選択肢:** A: min_strengthを上げて自然に絞る / B: max_signalsで制限
- **決定:** B。strength分布が17-39の狭い帯域に集中しており、min_strength調整では10本に絞れない
- **根拠:** field本数は設計パラメータであり設計原則ではない。perceiveのmax_signalsは「安全弁」として既に設計されている
- **結果:** field=10で0.9-1.7 tok/s（field=30比で約2倍改善）

### 判断: GQA展開排除（G）の試行と棄却
- **状況:** repeat_interleaveによるGQA展開がメモリバンド幅を浪費している仮説
- **決定:** 5次元テンソルreshape + broadcast matmulで実装したが、悪化（0.4-0.6 tok/s）。リバート
- **根拠:** PyTorchのcuBLAS最適化は4次元テンソルに対して有効。5次元は非最適パスに入る
- **教訓:** メモリアクセス理論と実測は一致しない。必ずベンチマークで検証

### 判断: field KV pruning の採用
- **状況:** A-1計測でdecode時のfield位置attention weightが全質問で2%未満（0.83-1.11%）
- **決定:** prefill完了後にfield位置のKVキャッシュエントリを除去
- **根拠:** 受容体→カスケード対比。濃度情報はprefillのattentionパターンでテキストKVに焼き込まれる
- **結果:** 応答時間122秒→53秒（2.3倍高速）。tok/s 0.8→1.2。品質劣化なし
- **見積もり誤差:** 当初7-10%と見積もったが実際は53%改善。KVエントリ除去がメモリバンド幅の共有リソース解放として増幅された

### A-1 計測データ: decode時field attention weight

| 質問 | 平均 | 序盤5step | 終盤5step |
|------|:---:|:---:|:---:|
| 気分はどう？(自己言及) | 1.11% | 2.10% | 0.60% |
| 貧困問題(知識) | 0.98% | 2.01% | 0.48% |
| 量子コンピュータ(知識) | 0.83% | 2.04% | 0.31% |
| 面白いこと(自己言及) | 1.01% | 2.36% | 0.50% |

### 速度改善サマリ

| 構成 | tok/s | 応答時間 | 状態 |
|------|:---:|:---:|------|
| field=30, 動的スケーリング | 0.3-0.7 | 83-140秒 | 品質デバッグ前 |
| field=10, 固定スケーリング | 0.9-1.7 | 61-92秒 | B単体 |
| **field=10, 固定スケーリング, KV pruning** | **1.2** | **53秒** | **現在の本番構成** |

### 発見: Marlinカーネル（軸B）の優先度低下
- **内容:** 目標範囲(50-100秒)に入ったため、軸Bは「運用不能→何とかする」から「さらに快適にする」に変化
- **対処:** 保留。本番運用とデータ蓄積を優先
- **保留理由:** field KV pruningで十分な速度を達成。軸Bのコスト（GPTQモデル検証、inputs_embeds互換性確認）に見合う緊急性がない

---

## テンプレート

```
### 判断: [1行要約]
- **状況:**
- **選択肢:** A: ... / B: ...
- **決定:**
- **根拠:**
- **リスク:**

### 発見: [1行要約]
- **内容:**
- **対処:** [修正済み / 保留 / 受容]
- **保留理由:**
```
