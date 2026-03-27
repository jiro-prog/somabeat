# TurboQuant KVキャッシュ圧縮 実装レポート

**日付:** 2026-03-27
**コミット:** ff9597b
**対象:** somabeat integrated-system
**論文:** Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", arXiv:2504.19874, ICLR 2026

---

## 1. 背景と目的

### 問題

llamarcute-live（Discord bot）の応答に10分かかっていた。原因調査の結果、以下が判明:

- Qwen3-8B 4-bit (bitsandbytes NF4) のモデル本体: 6.1GB
- KVキャッシュ: FP16で最大14.6GB（36層 x 8KVヘッド x 128次元 x シーケンス長）
- RTX 3060 Ti VRAM: 8GB
- **ピーク14.6GBがVRAMを溢れ、PyTorchがCPUへスワップ → 2 tok/s**

### 目標

TurboQuantによるKVキャッシュ圧縮で、ピークVRAMを8GB以内に収める。

---

## 2. 実装したコンポーネント

### 2.1 コアアルゴリズム (`shared_state/turboquant.py`, 450行)

| クラス | 役割 |
|--------|------|
| `LloydMaxCodebook` | Beta(alpha, alpha)分布に対する最適スカラー量子化器。Lloyd-Maxアルゴリズムで centroids/boundaries を計算。save/load対応 |
| `generate_rotation_matrix()` | Haar測度によるd×d直交行列生成（QR分解）。回転後の各座標がBeta分布に従い、座標ごとの独立量子化が最適になる |
| `QJLCorrector` | QJL 1-bit残差補正。Rademacher行列による射影 + sign圧縮。bit packing (8x圧縮) 対応 |
| `TurboQuantCompressor` | 統合クラス。compress/decompress/memory_bytes。全演算GPU上torch.matmulで実行 |

### 2.2 Fused Decode Attention (`shared_state/turboquant_fused_attn.py`, 224行)

| 関数 | 役割 |
|------|------|
| `turboquant_decode_attention()` | Pre-rotated query最適化によるfused attention。PyTorch GPU ops実装 |
| `reference_decode_attention()` | ループベースのリファレンス実装（正しさ検証用） |

**核心的な最適化:** `Q @ K_restored^T = ||k|| * (Q @ Pi^T) @ z_rec^T`

クエリを事前に逆回転すれば、キーの逆回転（行列積）が不要になる。codebook lookup + dot productだけで済む。

### 2.3 KVキャッシュ統合 (`llamarcute_live/kv_cache.py`, 287行)

| クラス/関数 | 役割 |
|-------------|------|
| `TurboQuantLayer` | transformers `CacheLayerMixin` 準拠。圧縮キー(indices+norms) + FP16値を保持 |
| `TurboQuantCache` | `DynamicCache` 継承。generate()にそのまま渡せる |
| `patch_model_for_turboquant()` | Qwen3の全attentionレイヤーをmonkey-patch。prefillは通常attention、decodeはfused attention |

### 2.4 推論エンジン統合 (`llamarcute_live/llm_inference.py`)

`FieldAwareLLM`に`kv_cache_bits`パラメータ追加。load時にcompressor初期化、generate時にcache生成+model patch。

### 2.5 周辺ツール

| ファイル | 役割 |
|----------|------|
| `scripts/turboquant/precompute_codebook.py` | コードブック事前計算・永続化 |
| `scripts/profile_turboquant.py` | CUDA event計測による速度内訳プロファイラ |
| `data/turboquant/codebook_d128_b{2,3,4}.npz` | 事前計算済みコードブック（確定値） |
| `tests/test_turboquant.py` | 24テスト |

---

## 3. 実装中に発生した問題と修正

### 3.1 数学の誤り: 回転方向 (ユーザーが発見)

**問題:** `turboquant_decode_attention()`で`q_rot = q @ rot`としていた。

**正しくは:** `q_rot = q @ rot.T`（逆回転 Pi^T）

**導出:**
```
圧縮: z = x_hat @ Pi^T
復元: x_hat = z_rec @ Pi   (= z_rec @ (Pi^T)^T)
内積: q @ k_rec^T = ||k|| * q @ (z_rec @ Pi)^T
     = ||k|| * q @ Pi^T @ z_rec^T
     = ||k|| * (q @ Pi^T) @ z_rec^T
```

Piの転置が必要。`reference_decode_attention()`も同様に修正。

### 3.2 MSEビット数の誤り (ユーザーが発見)

**問題:** `LloydMaxCodebook(head_dim, bits - 1)` — Prod変種用にQJL分の1bitを予約していた。

**正しくは:** MSE変種ではQJLを使わないため、全ビットをLloyd-Maxに使用: `LloydMaxCodebook(head_dim, bits)`

**影響:** 3-bitの場合、4レベル(2-bit MSE)→8レベル(3-bit MSE)。cosine similarity 0.155→0.987に改善。これが修正前にガーベッジ出力だった根本原因。

### 3.3 Python closure bug (monkey-patch)

**問題:** `patch_model_for_turboquant()`のforループで:

```python
for layer in model.model.layers:
    attn = layer.self_attn
    def patched_forward(...):
        # attn を参照 → 全36層が最後のattnを共有
```

**修正:** クロージャファクトリで明示的にキャプチャ:

```python
def make_patched_forward(orig_fwd, l_idx, attn_module):
    def patched_forward(...):
        # attn_module を参照 → 各層で独立
    return patched_forward
attn.forward = make_patched_forward(original_forward, layer_idx, attn)
```

### 3.4 dtype不整合 (複数箇所)

| 箇所 | 問題 | 修正 |
|------|------|------|
| QJL encode | float residual @ double S_torch | `residual.float()`, S_torchを`dtype=torch.float32`で生成 |
| patched forward | fused attn出力(float) → o_proj(bfloat16) | `attn_output.to(hidden_states.dtype)` |

### 3.5 unload時のリソースリーク (バグチェックで発見)

| 問題 | 修正 |
|------|------|
| `_model_patched`がリセットされない → 再ロード後patchされない | `unload()`に`self._model_patched = False`追加 |
| `_compressor`が解放されない → GPU tensorsリーク | `unload()`に`del self._compressor`追加 |

### 3.6 get_mask_sizes二重カウント (バグチェックで発見)

**問題:** `_seq_length`は`update()`内で加算済みなのに、`get_mask_sizes()`が`_seq_length + query_length`を返していた。

**修正:** `_seq_length`のみを返すように変更。現在のコードパスでは未発火（patched forwardがバイパス）だが、潜在バグとして修正。

---

## 4. 設計判断

### 4.1 MSE変種を採用、Prod変種は見送り

論文はMSE（再構成誤差最適化）とProd（内積推定のバイアス補正）の2変種を提案。Prodは1bitをQJL残差補正に割く。

**判断:** MSE変種を採用。理由:
- 3-bit MSEで cosine sim 0.987 が得られ、十分な精度
- Prodにすると実質2-bit MSE + 1-bit QJLとなり、MSE精度が大幅に低下
- QJLの実装は完了済み(`QJLCorrector`)だが、現時点では使用しない

### 4.2 Tritonカーネルは断念、PyTorch opsで実装

当初Triton JITカーネルを試みたが、ループ内のスカラーインデックス操作でTritonの制約に当たり断念。PyTorch GPU ops（torch.matmul, codebook indexing via `centroids[indices_long]`）で実装。

**影響:** プロファイリングの結果、fused attentionは全体の6.2%しか占めず、カーネル最適化の効果は限定的と判明。ボトルネックはFFN/LayerNorm（74%）。

### 4.3 Prefill通常attention + Decode fused attention

- Prefill（初回の複数トークン処理）: 通常のattention forward。FP16キーをそのまま使う
- Decode（1トークンずつの生成）: monkey-patchしたfused attention。圧縮キーを直接参照

**理由:** Prefillは一度しか実行されないため、精度を優先。Decodeは毎トークン実行されるため、VRAM効率を優先。

### 4.4 FP16キーのprefill後解放 (TQ-6c)

Prefill用のFP16キーをdecode開始時に解放。decode時のVRAMを大幅削減。

**初期実装の問題:** FP16キーを圧縮キーと並行保持していたため、VRAM削減効果がなかった（ピーク14,632MiB）。解放により6,324MiBに改善。

### 4.5 コードブックの事前計算・永続化 (TQ-3)

毎回Lloyd-Maxを計算すると浮動小数点の非決定性で微妙に異なるコードブックが生成されるリスクがある。圧縮時と復元時で同一のコードブックを使うことはTurboQuantの正しさの前提。

**実装:** `.npz`ファイルに事前計算・保存。`TurboQuantCompressor`はデフォルトパスから自動ロード。ファイルがない場合はwarning付きでon-the-fly計算にフォールバック。

---

## 5. 計測結果

### 5.1 VRAM

| 指標 | TurboQuantなし | TurboQuantあり (TQ-6c後) |
|------|---------------|--------------------------|
| Peak VRAM | 14,631 MiB | **6,324 MiB** |
| 削減率 | - | **57%** |
| 8GB以内 | No (CPUスワップ) | **Yes** |

### 5.2 速度

| 指標 | TurboQuantなし | TurboQuantあり |
|------|---------------|----------------|
| tok/s | ~2 (CPUスワップ) | **3.1-3.8** |
| 改善率 | - | **55-90%** |

### 5.3 速度内訳 (per token, 36層合計)

| コンポーネント | 時間/トークン | 割合 |
|----------------|--------------|------|
| OTHER (FFN, LayerNorm等) | 218ms | **74%** |
| cache_update (圧縮) | 36ms | 12% |
| qkv_rope | 19ms | 7% |
| fused_attn | 18ms | **6%** |
| o_proj | 2ms | 1% |

**結論:** fused attentionカーネル自体は十分速い（6%）。速度のボトルネックはモデル本体のFFN演算。TurboQuantの役割はVRAM削減（=CPUスワップ防止）であり、それは達成。

### 5.4 品質

TurboQuantあり/なしで同一入力に対する出力を比較:

- 標準出力: 「お久しぶりですね。」 → TQ出力: 同一
- 64トークン生成: 自然な日本語、文法的に正常

---

## 6. ファイル構成

```
shared_state/
  turboquant.py              # コアアルゴリズム (450行)
  turboquant_fused_attn.py        # Fused decode attention (224行)

llamarcute_live/
  kv_cache.py                 # DynamicCache統合 (287行)
  llm_inference.py            # FieldAwareLLM統合 (変更71行)

data/turboquant/
  codebook_d128_b2.npz        # 事前計算済み 2-bit コードブック
  codebook_d128_b3.npz        # 事前計算済み 3-bit コードブック (本番使用)
  codebook_d128_b4.npz        # 事前計算済み 4-bit コードブック

scripts/
  turboquant/
    precompute_codebook.py    # コードブック生成 (106行)
  profile_turboquant.py       # 速度プロファイラ (285行)

tests/
  test_turboquant.py          # 24テスト (300行)

docs/instructions/
  instructions_turboquant.md  # 指示書 (692行)
```

合計: 13ファイル, +2,458行

---

## 7. 残課題

| 項目 | 状態 | 備考 |
|------|------|------|
| cache_update最適化 | 未着手 | `torch.cat`による再アロケーションが12%。事前確保バッファで改善可能だが効果は限定的 |
| Tritonカーネル | 断念 | PyTorch opsで十分。fused attentionは全体の6% |
| Prod変種 (QJL補正) | 保留 | MSE変種で精度十分。QJLコードは実装済みだが未使用 |
| bot再起動+本番テスト | 未実施 | TurboQuant有効状態でのDiscord運用テスト |
| 速度改善 | 対象外 | ボトルネックはFFN (74%)。モデル変更またはHW変更の領域 |
