# 処理フロー概要

**最終更新:** 2026-03-28 (Marlinカーネル切り替え後)

---

## 1. 対話フロー（覚醒時）

ユーザーがDiscordでメッセージを送信してから応答が返るまでの一連の処理。

```
Discord Message
  │
  ▼
IntegratedBot.on_message()          discord_bot/bot.py
  │  ガード: bot自身 / チャンネルID / 状態(AWAKE)
  │
  ▼
DialogueManager.process_input()     llamarcute_live/dialogue.py
  │
  ├─ 1. save_log("user", ...)       SQLite dialogue_log に記録
  │
  ├─ 2. perceive_field()            場の知覚
  │     │
  │     ├─ ChromaDBField.perceive()  全信号を取得
  │     │    時間減衰: exp(-ln2/24h × 経過時間)
  │     │    強度フィルタ: decay × ||embedding|| ≥ 1.0
  │     │    上限: 10信号、7日以内
  │     │
  │     └─ FieldReceptor.transduce()  信号をLLM空間へ変換
  │          MLP: 384d → 512 → 4096d
  │          出力: (K, 4096) float32  K ≤ 10
  │
  ├─ 3. build_prompt()              システムプロンプト構築
  │     │
  │     ├─ 不変制約（ハードコード）
  │     │    名前、応答言語、文字数制限
  │     │
  │     ├─ 行動規範（personality_v0.yaml）
  │     │    5-15ルール、進化可能
  │     │
  │     └─ 直近の会話（ワーキングメモリ）
  │          トークン予算500以内で新しい方から詰める
  │
  ├─ 4. generate_response()          LLM推論
  │     │
  │     └─ FieldAwareLLM.generate_with_field()
  │          │
  │          ├─ トークン化
  │          │    system: apply_chat_template + "/no_think"
  │          │    user:   apply_chat_template + generation_prompt
  │          │
  │          ├─ embedding取得
  │          │    sys_embeds  = embed_layer(sys_ids)
  │          │    user_embeds = embed_layer(user_ids)
  │          │
  │          ├─ field embeddingスケーリング
  │          │    base_text_norm ≈ 1.5
  │          │    キャリブレーション: generate_with_field()初回呼び出し時に
  │          │    sys_embedsのノルムから1回だけ計測・固定。以降不変。
  │          │    ※load()時ではない（sys_embedsが必要なため）
  │          │    scale = base_text_norm / field_mean_norm
  │          │    FR出力ノルム(~22) → テキストノルム(~1.5)に合わせる
  │          │
  │          ├─ 入力結合
  │          │    [sys_embeds] [field_embeds] [user_embeds]
  │          │
  │          ├─ 生成 (GPTQ+Marlin, FP16 KVキャッシュ)
  │          │    ~26 tok/s, VRAM 5.7GB
  │          │
  │          └─ 後処理
  │               <think>タグ除去 → 特殊トークン除去 → strip
  │
  ├─ 5. emit_experience()            対話経験を場に放出
  │     要約テキスト → E5-small encode → ChromaDB emit
  │     ※応答時間 >10s なら difficulty signal も追加放出
  │       注: Marlin移行前(53s/応答)は全対話がdifficulty扱いだった。
  │       現在(2-10s)では10s閾値が初めて有意に機能する。
  │       200トークン超の正常応答が10s付近になるため、
  │       閾値再調整の要否は運用データを見て判断する。
  │
  ├─ 6. 会話バッファ更新              最大10エントリ（5往復）
  │
  └─ 7. save_log("assistant", ...)   SQLite dialogue_log に記録
```

### 所要時間の内訳（典型値）

| フェーズ | 時間 |
|----------|------|
| perceive_field | ~0.01s |
| generate (LLM推論) | 2-10s |
| emit_experience | ~0.01s |
| **合計** | **2-10s** |

---

## 2. 推論エンジン構成

```
FieldAwareLLM
  │
  ├─ 量子化: GPTQ 4-bit + Marlin kernel (W4A16 fused GEMV)
  │    モデル: AlphaGaO/Qwen3-8B-GPTQ
  │    ロード: gptqmodel.GPTQModel.load(backend="marlin")
  │    dtype: bfloat16
  │
  ├─ KVキャッシュ: FP16 (標準DynamicCache)
  │    TurboQuant: 無効 (kv_cache_bits=0)
  │    field KV pruning: 無効
  │
  └─ VRAM: 5.7GB (ピーク5.7GB / 8GB GPU)
```

### 推論エンジンの使い分け

| 用途 | エンジン | 理由 |
|------|----------|------|
| 対話（場の信号あり） | FieldAwareLLM | inputs_embeds注入が必要 |
| ユーティリティ（場を読まない） | Ollama | VRAM排他制御で共存 |

### VRAM排他制御

```
対話中:   FieldAwareLLM (5.7GB)  ←  Ollama は使用不可
睡眠中:   FieldAwareLLM.unload() → Ollama (night cycle + self-improvement)
起床時:   Ollama keep_alive:0 → FieldAwareLLM.load()
```

---

## 3. 睡眠サイクル（夜間処理）

トリガー: 手動 `/sleep` または定時（午前3時）

```
enter_sleep()                       orchestrator/orchestrator.py
  │
  ├─ 1. 対話停止 + 会話バッファクリア + 感覚モジュール停止
  │
  ├─ 2. FieldAwareLLM.unload()      VRAM解放 (Ollama用)
  │
  ├─ 3. sleep_ingest                 [timeout: 5分]
  │     今日の対話ログ → SleepyJean DB に転送
  │     difficulty信号 → 宿題として登録
  │
  ├─ 4. SleepyJean night_cycle      [timeout: 30分]
  │     外部リポジトリのサブプロセスとして実行
  │     LoRA学習 + 知識生成
  │
  ├─ 5. wake_export                  [timeout: 5分]
  │     新規知識 → 場に信号として放出
  │     知識インデックスの差分を反映
  │
  ├─ 6. Ollama keep_alive:0          VRAM解放
  │     FieldAwareLLM.load()         再ロード
  │
  ├─ 7. self_improvement             [timeout: 30分]
  │     │
  │     ├─ difficulty信号の蓄積量 → 変更数を決定
  │     ├─ 変異候補3つ生成（LLM）
  │     ├─ fitness評価（コアタスク + ローテーションタスク）
  │     ├─ cuteness評価（ピア会話 6ペア×3ターン）
  │     ├─ 選択: 75% fitness + 25% cuteness
  │     └─ 免疫チェック（閾値違反時にロールバック）
  │
  ├─ 8. purge                        7日超の古い信号を削除
  │
  └─ 9. wake_up()                    状態=AWAKE、対話再開、起床メッセージ
```

### 睡眠サイクルの所要時間（典型値）

| フェーズ | 時間 |
|----------|------|
| sleep_ingest | ~10s |
| night_cycle | 20-30分 |
| wake_export | ~10s |
| self_improvement | 20-30分 |
| purge | ~1s |
| **合計** | **~60-70分** |

---

## 4. 主要コンポーネントと責務

```
discord_bot/bot.py              Discordイベントハンドラ（メッセージ受信、スケジュール起動）
orchestrator/orchestrator.py    状態管理（AWAKE/SLEEPING）、睡眠シーケンス制御
llamarcute_live/
  dialogue.py                   対話ループ（perceive→prompt→generate→emit→log）
  llm_inference.py              FieldAwareLLM（GPTQ+Marlin / NF4 切替）
  personality.py                行動規範の読み書き
  self_improve.py               変異候補生成
  fitness.py                    タスク評価
  cuteness.py                   ピア会話評価
  selection.py                  候補選択
  immune.py                     健全性チェック・ロールバック
  ollama_client.py              Ollamaフォールバック
shared_state/
  backends/chromadb_backend.py  ChromaDB場（perceive/sense/emit）
  field_receptor.py             FieldReceptor（384d→4096d MLP）
  encoder.py                    E5-small / MultimodalFieldEncoder
  turboquant.py                 TurboQuant圧縮（現在無効）
  turboquant_fused_attn.py      Fused decode attention（現在無効）
  interface.py                  Signal, FieldPerception 等の型定義
config/system.yaml              全体設定
```

---

## 5. データの流れ

```
ユーザー入力 (テキスト)
    │
    ▼
E5-small encode → 384d embedding
    │                    │
    │                    ▼
    │              ChromaDB (場)
    │                    │
    │              perceive() → 上位10信号
    │                    │
    │              FieldReceptor.transduce()
    │                    │
    │                    ▼
    │              (K, 4096) field embeddings
    │                    │
    ▼                    ▼
Qwen3-8B tokenize → embed_layer → [sys][field][user] → generate
                                                           │
                                                           ▼
                                                      応答テキスト
                                                           │
                                    ┌──────────────────────┤
                                    ▼                      ▼
                              E5 encode → emit       Discord送信
                              (場に経験を蓄積)
```

---

## 6. 設定パラメータ（主要）

| パラメータ | 値 | 場所 |
|-----------|-----|------|
| 量子化方式 | gptq_marlin | config: llamarcute_live.llm.quantization |
| GPTQモデル | AlphaGaO/Qwen3-8B-GPTQ | config: llamarcute_live.llm.gptq_model |
| KVキャッシュ | FP16 (bits=0) | config: llamarcute_live.llm.kv_cache_bits |
| perceive上限 | 10信号 | config: llamarcute_live.perceive.max_signals |
| perceive強度閾値 | 1.0 | config: llamarcute_live.perceive.min_strength |
| 会話バッファ | 10エントリ | config: llamarcute_live.max_conversation_history |
| プロンプトトークン上限 | 500 | ハードコード (dialogue.py) |
| 生成トークン上限 | 512 | FieldAwareLLM default |
| temperature | 0.7 | FieldAwareLLM default |
| repetition_penalty | 1.3 | FieldAwareLLM default |
| fitness/cuteness重み | 0.75/0.25 | config: self_improvement |

---

## 7. 性能特性

| 指標 | 値 | 備考 |
|------|-----|------|
| 推論速度 | ~26 tok/s | GPTQ+Marlin, RTX 3060 Ti |
| 応答時間 | 2-10s | 出力トークン数依存 |
| VRAM使用量 | 5.7GB | モデル重み + KVキャッシュ |
| VRAM余裕 | ~2.3GB | 8GB GPU上 |
| テスト件数 | 225件 | 全PASS |
