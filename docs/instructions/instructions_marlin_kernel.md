# 軸B: Marlinカーネル検証 — タスクフロー

**作成日:** 2026-03-28
**親文書:** performance_tuning_report.md, llamarcute_live_design.md 3.4節
**前提:** field KV pruning有効、field=10、1.2 tok/s / 53秒がベースライン

---

## 0. 目的

FFN層のmatmul（全体の79%）を支配するbitsandbytes NF4のdequantコストを、MarlinカーネルのfusedなW4A16 GEMVに置き換えることで削減する。

**設計原則との整合性:** Marlinは重み量子化カーネルの差し替えであり、場・FieldReceptor・FieldEncoder・perceiveには一切変更がない。LLMの推論エンジン内部の最適化に閉じる。FieldReceptor再学習は不要（hidden_dim=4096は不変）。

**リスク:** Ampere世代consumer GPUでMarlin+GPTQの品質劣化報告あり（vllm issue #5793）。品質検証を厳格に行う。

---

## B-0. 事前確認

| # | 確認事項 | 期待値 |
|---|----------|--------|
| 0-1 | テスト225件PASS（field KV pruning有効状態） | 全PASS |
| 0-2 | 現在のFieldAwareLLMのモデルロード方式を確認 | bitsandbytes NF4 |
| 0-3 | GPTQModel or AutoGPTQのインストール可否 | pip install確認 |
| 0-4 | AlphaGaO/Qwen3-8B-GPTQのダウンロード | ~5GB、ディスク容量確認 |

---

## B-1. Marlin単体の疎通確認（TurboQuant・field KV pruning無効）

### 目的

GPTQモデル + Marlinカーネルで `model(inputs_embeds=...)` が動作することを確認する。TurboQuantやfield KV pruningとの共存は後回し。まず動くかどうかだけ見る。

### 手順

```python
# 最小検証スクリプト（本番コードには入れない）
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "AlphaGaO/Qwen3-8B-GPTQ"
gptq_config = GPTQConfig(bits=4, backend="marlin")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=gptq_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# inputs_embeds疎通テスト
import torch
dummy_embeds = torch.randn(1, 10, 4096, device="cuda", dtype=torch.float16)
output = model(inputs_embeds=dummy_embeds)
print("inputs_embeds OK:", output.logits.shape)

# 生成テスト（inputs_idsで）
inputs = tokenizer("こんにちは", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### 確認事項

| # | 確認 | 判定基準 |
|---|------|---------|
| 1 | モデルがVRAMに収まるか | nvidia-smiでピークVRAM確認。8GB以内 |
| 2 | `model(inputs_embeds=...)` がエラーなく動くか | logits.shapeが正しい |
| 3 | 日本語テキスト生成が正常か | 文字化け・ループ・ゴミなし |
| 4 | hidden_dimが4096であることの確認 | model.config.hidden_size == 4096 |

### ゲート判定

| 条件 | 判定 | 次のステップ |
|------|------|-------------|
| 全確認PASS | PASS | B-2へ |
| inputs_embedsが非対応 | FAIL | 軸B中止（Marlinカーネルの制約） |
| VRAMオーバー | FAIL | モデル候補の変更を検討（RedHatAI/Qwen3-8B-quantized.w4a16等） |
| 品質異常 | 要調査 | A10品質問題の再現か確認 |

---

## B-2. Marlin単体での速度・品質計測

### 目的

TurboQuant無効の状態で、Marlin単体のtok/sと品質を計測する。bitsandbytes NF4との比較ベースラインを確立する。

### 速度計測

同一質問セットで計測。**field KV pruningは無効**にして、NF4との公平な比較を行う。

| # | 質問 | 計測値 |
|---|------|--------|
| 1 | 気分（ウォームアップ） | tok/s, 応答時間, VRAM |
| 2 | 貧困問題 | tok/s, 応答時間 |
| 3 | 量子コンピュータ | tok/s, 応答時間 |
| 4 | 面白いこと | tok/s, 応答時間 |

比較表:
```
| 条件              | tok/s | 応答時間 |
|-------------------|-------|----------|
| NF4, field=10     | 0.8   | 122s     |  ← 既存ベースライン（pruningなし）
| Marlin, field=10  | ???   | ???      |  ← B-2計測
```

### 品質計測

| # | 検証項目 | 方法 |
|---|----------|------|
| 1 | 応答の自然さ | 4質問の応答を目視確認 |
| 2 | 日本語品質 | 文字化け、不自然な語順、NF4以上の劣化がないか |
| 3 | 早期EOS | 3回テスト |
| 4 | ゴミ出力 | A10品質問題の兆候（ループ、無関係な出力）がないか |

### ゲート判定

| 条件 | 判定 | 次のステップ |
|------|------|-------------|
| tok/s がNF4より改善 AND 品質正常 | PASS | B-3へ |
| tok/s 改善なし AND 品質正常 | FAIL | Marlinのメリットなし。軸B中止 |
| 品質異常 | FAIL | A10品質問題の再現。軸B中止 |

---

## B-3. TurboQuant + field KV pruningとの共存検証

### 目的

B-2でMarlin単体の効果を確認した上で、既存のTurboQuant（KVキャッシュ圧縮）とfield KV pruningを再有効化し、フルスタックでの動作を検証する。

### 技術的な確認ポイント

| # | 確認事項 | 懸念 |
|---|----------|------|
| 1 | TurboQuantのmonkey-patchがMarlinモデルに適用可能か | GPTQモデルの内部構造（attention層のクラス名等）がNF4と異なる可能性 |
| 2 | TurboQuantCacheがMarlinのattention出力と互換か | attention層の出力dtypeが異なる可能性 |
| 3 | field KV pruningのKVキャッシュsliceが正常に動くか | Marlinモデルのpast_key_valuesの構造が同一か確認 |
| 4 | Ollama↔FieldAwareLLMのVRAM排他制御 | GPTQモデルのロード・アンロード方式がNF4と同じか |
| 5 | FieldAwareLLMのunload()/load()がGPTQモデルで正常か | GPTQモデルの再ロード方式の確認 |

### 手順

1. FieldAwareLLMのモデルロードをGPTQ+Marlinに差し替え
2. TurboQuantのmonkey-patchを適用
3. field KV pruningを有効化
4. 4質問テストで速度・品質計測
5. 睡眠サイクル1回完走テスト（Ollama排他制御含む）

### 速度計測

```
| 条件                              | tok/s | 応答時間 |
|-----------------------------------|-------|----------|
| NF4 + TQ + pruning (現在の本番)   | 1.2   | 53s      |
| Marlin + TQ + pruning             | ???   | ???      |
```

### ゲート判定

| 条件 | 判定 |
|------|------|
| tok/s 改善 AND 品質正常 AND 睡眠サイクル完走 AND 225件PASS | PASS → B-4へ |
| TurboQuantとの共存不可 | Marlin + field KV pruningのみ（TurboQuant無効）で再評価 |
| 睡眠サイクル失敗 | Ollama排他制御の修正が必要。修正後再テスト |

---

## B-4. 本番切り替えと設計文書更新

B-3ゲートPASSの場合:

| 作業 | 内容 |
|------|------|
| config更新 | モデルIDとquantization設定をGPTQ+Marlinに変更 |
| somabeat_handoff.md更新 | 推論エンジンをMarlinに更新、tok/sを更新 |
| performance_tuning_report.md更新 | B-2, B-3の計測結果を追記 |
| llamarcute_live_design.md | 推論エンジンの記載を更新（必要に応じて） |
| テスト225件確認 | 全PASS |
| コミット | 本番切り替え |

---

## タスク依存関係

```
B-0（事前確認）
  → B-1（Marlin疎通）
    → [ゲート判定]
      → PASS → B-2（速度・品質計測）
        → [ゲート判定]
          → PASS → B-3（フルスタック共存）
            → [ゲート判定]
              → PASS → B-4（本番切り替え）
              → FAIL（TQ非互換）→ TQ無効でMarlin+pruningのみで再評価
          → FAIL → 軸B中止
      → FAIL → 軸B中止
```

---

## フォールバック戦略

軸Bが中止になった場合、現在の本番構成（NF4 + TurboQuant + field KV pruning, 1.2 tok/s）を維持する。これは目標範囲（50-100秒）に入っており、運用上の問題はない。

さらなる速度改善が必要な場合の代替案:
1. **AWQ + Marlin:** GPTQの代わりにAWQで量子化したモデルをMarlinで実行。品質がGPTQより良い報告あり
2. **Speculative Decoding:** ドラフトモデル（Qwen3-0.6B）で先読み。FFNの実行回数自体を削減
3. **ハードウェア変更:** Mac mini M5移行（2026年6月WWDC見込み）

---

## リスクと緩和策

| リスク | 確率 | 緩和策 |
|--------|------|--------|
| inputs_embedsが非対応 | 低 | Marlinは重み量子化。入力経路には影響しないはず |
| A10品質問題の再現（RTX 3060 Ti） | 中 | B-2で品質検証を厳格に実施。異常があれば即中止 |
| TurboQuantとの共存不可 | 中 | B-3でTQ無効のMarlin+pruningのみを代替評価 |
| GPTQモデルの品質がNF4に劣る | 中 | B-2で応答品質を目視比較。許容範囲外なら中止 |
| Marlinでの速度改善がRTX 3060 Tiでは限定的 | 中 | B-2の計測で判断。改善率が20%未満なら中止 |
| VRAM増加によるOOM | 低 | B-1でピークVRAM計測。8GB超なら候補モデル変更 |
