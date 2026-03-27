# 指示書: TurboQuant KVキャッシュ圧縮 — 実験的実装

**日付:** 2026-03-27
**対象:** somabeat（integrated-system）
**目的:** RTX 3060 Ti 8GBでのKVキャッシュVRAM溢れを解消し、推論速度を改善する
**根拠:** Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", arXiv:2504.19874, ICLR 2026
**ステータス:** 実験的。本番適用は検証結果を見て判断

---

## 背景

Qwen3-8B 4-bitの推論でKVキャッシュがVRAM 8GBを溢れ、CPUスワップにより2 tok/sまで低下している。モデル本体は6.1GBでVRAMに収まるが、KVキャッシュ+中間バッファがピーク14.6GBに達する。TurboQuantでKVキャッシュを3-4bitに圧縮すれば、VRAM内に収まる見込み。

---

## アルゴリズム概要（論文Section 3より）

### TurboQuant_MSE（Algorithm 1）

```
入力: x ∈ R^d（KVキャッシュの1ヘッド、d=head_dim）

1. ノルム抽出
   γ = ||x||₂
   x̂ = x / γ                     # 単位球上に射影

2. ランダム回転
   z = Π · x̂                     # Π: d×d直交行列（事前生成、固定）
   # 回転後の各座標 z_i ~ Beta(d/2-1/2, d/2-1/2) を [-1,1] にスケール
   # 高次元(d≥64)では z_i ≈ N(0, 1/d) に近似
   # 座標間はほぼ独立 → 座標ごとの独立量子化が最適に近い

3. 座標ごとLloyd-Max量子化
   for i in 0..d-1:
     idx_i = codebook.quantize(z_i)   # Beta分布の最適コードブック
   # b-bit → 2^b セントロイド

復元:
   z_i ≈ codebook.centroids[idx_i]
   x̂ ≈ Π^T · z_reconstructed
   x ≈ γ · x̂

格納: indices (b*d bits) + γ (16 bits)
```

### TurboQuant_Prod（Algorithm 2 — 内積用）

```
KVキャッシュのattention計算には内積精度が重要。
MSE最適量子化器は内積推定にバイアスを持つ（論文Theorem 3.3）。

4. 残差計算
   r = x̂ - dequant_mse(quant_mse(x̂))

5. QJL 1-bit補正
   signs = sign(S · r)            # S: m×d ランダム行列（事前生成、固定）
   # m = d が標準的。各残差座標を1-bitに圧縮
   # これにより内積推定が不偏になる

合計: (b-1)-bit MSE + 1-bit QJL = b-bit/座標
```

### コードブックの事前計算

Beta(α,α) 分布（α = (d-1)/2）に対するLloyd-Max最適量子化器を事前計算する。

```
Lloyd-Max アルゴリズム:
  1. 初期セントロイド c_1..c_K を等間隔に配置（K = 2^b）
  2. 反復:
     a. 境界点: t_j = (c_j + c_{j+1}) / 2
     b. セントロイド更新: c_j = E[X | t_{j-1} < X < t_j]
        = ∫_{t_{j-1}}^{t_j} x·f(x)dx / ∫_{t_{j-1}}^{t_j} f(x)dx
        f(x) = Beta(α, α) の密度関数
     c. 収束判定: セントロイドの変化 < ε
  3. 出力: centroids, boundaries
```

d=128（Qwen3-8Bのhead_dim）の場合、Beta分布はN(0, 1/128)に十分近い。コードブックは次元に依存するが、一度計算すれば固定。

---

## 実装タスク

### TQ-1. コアアルゴリズム実装

**新規ファイル:** `shared_state/turboquant.py`

#### TQ-1a. Lloyd-Maxコードブック計算

```python
class LloydMaxCodebook:
    """Beta分布に対する最適スカラー量子化コードブック。"""

    def __init__(self, dim: int, bits: int):
        """
        Args:
            dim: ベクトル次元（= head_dim、Qwen3-8Bでは128）
            bits: 量子化ビット数（2, 3, or 4）
        """
        self.n_levels = 2 ** bits
        self.alpha = (dim - 1) / 2  # Beta(α, α) のパラメータ
        self.centroids, self.boundaries = self._compute_codebook()

    def _compute_codebook(self) -> tuple[NDArray, NDArray]:
        """Lloyd-Maxアルゴリズムでコードブックを計算する。

        Beta(α, α) 分布を [-1, 1] にスケールして使用。
        scipy.stats.beta + scipy.integrate.quad で期待値を計算。
        """
        ...

    def quantize(self, values: NDArray) -> NDArray:
        """連続値をインデックスに変換。np.searchsortedで高速化。"""
        ...

    def dequantize(self, indices: NDArray) -> NDArray:
        """インデックスをセントロイド値に復元。"""
        ...
```

**注意:** コードブックは dim と bits の組み合わせで一意に決まる。一度計算したら .pt または .npy で永続化し、以降はロードするだけ。

#### TQ-1b. ランダム回転行列

```python
def generate_rotation_matrix(dim: int, seed: int = 42) -> NDArray:
    """d×d直交行列を生成する。

    方法: d×dガウスランダム行列のQR分解。
    Q行列が一様分布する直交行列（Haar測度）になる。

    Args:
        dim: ベクトル次元
        seed: 再現性のためのシード

    Returns:
        (dim, dim) の直交行列。det = +1 に正規化。
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(H)
    # QR分解のQ行列の符号を正規化
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)
```

**注意:** 回転行列は固定シードで生成し、量子化と復元で同じ行列を使う。head_dimごとに1つ。Qwen3-8Bの全attention headで同じ回転行列を共有する。

#### TQ-1c. QJL 1-bit残差補正

```python
def generate_qjl_matrix(dim: int, m: int | None = None,
                         seed: int = 43) -> NDArray:
    """QJLのランダム射影行列を生成する。

    S ∈ R^{m×d}、各要素は ±1/√m（Rademacher分布）。
    m = dim がデフォルト。

    Returns:
        (m, dim) のランダム射影行列
    """
    ...

class QJLCorrector:
    """QJL 1-bit残差補正。内積推定の不偏性を保証する。"""

    def __init__(self, dim: int, seed: int = 43):
        self.S = generate_qjl_matrix(dim, seed=seed)

    def encode(self, residual: NDArray) -> NDArray:
        """残差を1-bitに圧縮。

        Args:
            residual: MSE量子化後の残差ベクトル (d,) or (N, d)
        Returns:
            sign bits (m,) or (N, m)。bool配列。
        """
        projected = residual @ self.S.T  # (N, m)
        return projected > 0

    def decode_for_inner_product(self, signs: NDArray,
                                  residual_norm: float) -> NDArray:
        """内積計算のためのQJL補正項を返す。

        復元されたベクトルとの内積を計算する際に加算する。
        """
        ...
```

#### TQ-1d. 統合クラス

```python
class TurboQuantCompressor:
    """TurboQuantのKVキャッシュ用統合インターフェース。"""

    def __init__(self, head_dim: int, bits: int = 3, seed: int = 42):
        """
        Args:
            head_dim: attention headの次元（Qwen3-8Bでは128）
            bits: 量子化ビット数（3推奨。3-bitで約5x圧縮）
            seed: 回転行列・QJL行列のシード
        """
        self.head_dim = head_dim
        self.bits = bits
        self.codebook = LloydMaxCodebook(head_dim, bits - 1)  # MSE用は(b-1)-bit
        self.rotation = torch.from_numpy(
            generate_rotation_matrix(head_dim, seed=seed)
        )
        self.qjl = QJLCorrector(head_dim, seed=seed + 1)

    def compress(self, kv_tensor: torch.Tensor) -> CompressedKV:
        """KVキャッシュテンソルを圧縮する。

        全演算はtorch.matmul等でGPU上で実行する。NumPyに変換しない。
        回転行列・QJL行列はself.rotation.to(kv_tensor.device)で
        KVテンソルと同じデバイスに配置する（初回のみ）。

        生成ループの各ステップで呼ばれるため、レイテンシが重要。
        128×128の回転行列乗算はGPU上ならμs単位で完了する。

        Args:
            kv_tensor: shape (batch, n_heads, seq_len, head_dim)
                       FP16/BF16のKVキャッシュ

        Returns:
            CompressedKV: 圧縮されたKVデータ
        """
        ...

    def decompress(self, compressed: CompressedKV) -> torch.Tensor:
        """圧縮されたKVキャッシュを復元する。

        attention計算の直前に呼ばれる。GPU上で完結させること。

        Returns:
            shape (batch, n_heads, seq_len, head_dim) のFP16テンソル
        """
        ...

@dataclass
class CompressedKV:
    """圧縮されたKVキャッシュデータ。"""
    indices: torch.Tensor      # int8。shape (batch, n_heads, seq_len, head_dim)
    norms: torch.Tensor        # float16。shape (batch, n_heads, seq_len)
    qjl_signs: torch.Tensor   # uint8（bit packed）。shape (batch, n_heads, seq_len, head_dim // 8)
    residual_norms: torch.Tensor  # float16。shape (batch, n_heads, seq_len)

    # qjl_signsのbit packing:
    # PyTorchのboolは8bit/要素のため、128次元のboolは128bytesを消費する。
    # uint8にbit packすれば128次元 → 16bytes（8分の1）。
    # pack: signs_packed = np.packbits(signs_bool)
    # unpack: signs_bool = np.unpackbits(signs_packed)
    # torch版: カスタムのpack/unpack関数を用意する
```

### TQ-2. transformersのKVキャッシュへの統合

**新規ファイル:** `llamarcute_live/kv_cache.py`

transformersのgenerate()は内部でDynamicCacheオブジェクトを使ってKVキャッシュを管理する。このキャッシュを圧縮版に差し替える。

#### TQ-2-pre. DynamicCacheスモークテスト（TQ-2着手前に必ず実施）

**TQ-2の最大リスクはDynamicCache継承がgenerate()で受け入れられるかどうか。** コアアルゴリズム（TQ-1）を実装する前に、以下の最小テストで確認する。全体の実装可否がここで決まる。

```python
"""DynamicCache継承の受け入れテスト。TQ-1より先に実行。"""
from transformers import DynamicCache

class DummyCache(DynamicCache):
    """何もしない継承。generate()に受け入れられるかだけ確認。"""
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 親の実装をそのまま呼ぶ（圧縮なし）
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

# FieldAwareLLMでテスト
llm.load()
dummy_cache = DummyCache()
output = llm.model.generate(
    inputs_embeds=some_embeds,
    past_key_values=dummy_cache,
    max_new_tokens=10,
)
print(f"PASS: generate() accepted DummyCache, output length={output.shape[1]}")
```

**判断基準:**
- PASS → TQ-2aに進む
- FAIL（TypeError, NotImplementedError等）→ DynamicCacheの内部APIを調査。transformersのバージョン固有の問題か、根本的に継承不可かを判別
- 根本的に不可の場合 → generate()のステップを手動でループし、各ステップでKVキャッシュを圧縮する方式にフォールバック

**transformersバージョンの確認:**
```python
import transformers
print(transformers.__version__)
# DynamicCacheのソースを確認
import inspect
print(inspect.getfile(DynamicCache))
```

#### TQ-2a. TurboQuantCache クラス

```python
from transformers import DynamicCache

class TurboQuantCache(DynamicCache):
    """TurboQuantで圧縮されたKVキャッシュ。

    DynamicCacheを継承し、update()とget()をオーバーライドする。
    新しいKVペアが追加されるたびにTurboQuantで圧縮し、
    attention計算時に復元して返す。
    """

    def __init__(self, compressor: TurboQuantCompressor):
        super().__init__()
        self.compressor = compressor
        self._compressed_keys: list[CompressedKV] = []
        self._compressed_values: list[CompressedKV] = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """新しいKVペアを圧縮して格納する。"""
        # FP16のまま格納する代わりに、圧縮して格納
        compressed_k = self.compressor.compress(key_states)
        compressed_v = self.compressor.compress(value_states)
        # 圧縮データをリストに追加
        ...
        # attention計算のために一時的に復元して返す
        return self.compressor.decompress(compressed_k), \
               self.compressor.decompress(compressed_v)

    def get_seq_length(self, layer_idx=0):
        ...

    def get_usable_length(self, new_seq_length, layer_idx=0):
        ...
```

**注意:** transformersのDynamicCacheのAPIはバージョンで変わる。TQ-2-preのスモークテストで動作確認済みのバージョンをrequirements.txtに固定すること。update()のシグネチャ（特にcache_kwargsの扱い）はバージョン間で異なる場合がある。

#### TQ-2b. FieldAwareLLMへの統合

`llamarcute_live/llm_inference.py`のgenerate_with_field()を修正し、generate()呼び出し時にTurboQuantCacheを渡す。

```python
# Before
outputs = self.model.generate(
    inputs_embeds=combined_embeds,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    ...
)

# After
from llamarcute_live.kv_cache import TurboQuantCache
from shared_state.turboquant import TurboQuantCompressor

compressor = TurboQuantCompressor(head_dim=128, bits=3)
past_key_values = TurboQuantCache(compressor)

outputs = self.model.generate(
    inputs_embeds=combined_embeds,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    past_key_values=past_key_values,
    ...
)
```

### TQ-3. コードブックの事前計算と永続化

**スクリプト:** `scripts/turboquant/precompute_codebook.py`

```bash
python scripts/turboquant/precompute_codebook.py --dim 128 --bits 2,3,4
# → data/turboquant/codebook_d128_b2.npy
# → data/turboquant/codebook_d128_b3.npy
# → data/turboquant/codebook_d128_b4.npy
```

各ファイルにはcentroidsとboundariesが含まれる。TurboQuantCompressor初期化時にロード。

### TQ-4. テスト

**新規ファイル:** `tests/test_turboquant.py`

| テスト | 内容 |
|--------|------|
| test_codebook_convergence | Lloyd-Maxが収束すること。反復回数が妥当であること |
| test_codebook_symmetry | Beta(α,α)は対称分布。コードブックも原点対称であること |
| test_rotation_orthogonality | Π^T · Π = I であること |
| test_rotation_distribution | 回転後の座標がBeta分布に近いこと（KSテスト） |
| test_quantize_dequantize_mse | MSEが論文の理論値に近いこと（3-bit: ≈0.03） |
| test_qjl_unbiasedness | QJL補正後の内積推定が不偏であること（多数回の平均が真値に収束） |
| test_compress_decompress_shape | 圧縮→復元で元の形状が保たれること |
| test_memory_reduction | 圧縮後のメモリが元の1/5以下であること（3-bit、bit packing適用後） |
| test_bit_packing | QJL signsのuint8 pack/unpackが正確に往復すること |
| test_attention_fidelity | 圧縮KVでのattention scoreが元と高相関（cosine sim > 0.99） |

### TQ-5. 実機検証

#### TQ-5a. VRAM計測

```python
# TurboQuantなし
outputs_baseline = model.generate(...)
print(torch.cuda.memory_summary())  # ピークVRAM確認

# TurboQuantあり
past_kv = TurboQuantCache(compressor)
outputs_tq = model.generate(..., past_key_values=past_kv)
print(torch.cuda.memory_summary())  # ピークVRAM確認
```

期待: ピークVRAMが14.6GB → 8GB以下に収まる

#### TQ-5b. 速度計測

同じ入力で生成速度を比較:
- tok/s（TurboQuantなし vs あり）
- 期待: 2 tok/s → 15-30 tok/s（CPUスワップ解消による改善）

**圧縮/復元レイテンシの個別計測:**
```python
# compress単体の速度
kv_dummy = torch.randn(1, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
compressed = compressor.compress(kv_dummy)
end.record()
torch.cuda.synchronize()
print(f"compress: {start.elapsed_time(end):.3f}ms")  # 目標: <1ms/トークン
```

生成ループで毎トークン圧縮/復元が走る。圧縮+復元で1ms/トークン以内なら、生成速度への影響は無視できる（30 tok/sで33ms/トークンのうち1ms）。1ms以上ならボトルネックの可能性。

#### TQ-5c. 応答品質

同じプロンプトでTurboQuantあり/なしの応答を比較。3.5-bitで精度劣化ゼロが論文の主張。3-bitでも99.5%のattention fidelity（コミュニティ実装の検証結果）。

---

## 設計上の注意

### TurboQuantの配置

TurboQuantCompressorは`shared_state/`に配置する。理由:
- KVキャッシュの圧縮はLLM推論のインフラであり、llamarcute-liveの認知機能ではない
- 将来的に他の系統がLLM推論を行う場合にも再利用可能
- FieldReceptorやFieldEncoderと同じ「場とエージェントの接続インフラ」レイヤー

### FieldAwareLLMインターフェースへの影響

**影響なし。** generate_with_field()の外部インターフェースは変更しない。KVキャッシュの圧縮はgenerate()の内部で完結する。呼び出し側（DialogueManager、SelectionManager、CutenessEvaluator）は変更不要。

### 回転行列・QJL行列のメモリ

- 回転行列: 128×128×4bytes = 64KB
- QJL行列: 128×128×4bytes = 64KB
- コードブック: 数百bytes

合計128KB程度。無視できるサイズ。

### bits の選択

| bits | 圧縮率 | MSE | 推奨度 |
|------|--------|-----|--------|
| 2 | ~8x | 0.117 | 攻撃的。品質劣化あり |
| 3 | ~5x | 0.03 | **推奨。品質劣化ほぼゼロ** |
| 4 | ~4x | 0.009 | 安全。圧縮率は控えめ |

3-bitから開始。品質に問題があれば4-bitに切り替え。

### Tritonカーネルの設計判断

**MSE変体（TurboQuant_MSE）を使用する。Prod変体ではない。** Prod変体（MSE + QJL）は不偏内積推定を提供するが、2段階の復元が必要でカーネルが複雑になる。MSE変体は全ビットをLloyd-Max量子化に割り当て、dequantize-on-the-flyがコードブックlookup + 逆回転の1ステップで完結する。参照実装（DEJAN blog）も同じ判断。

**keyのみ圧縮。valueはFP16。** attention計算のQ@K^Tがメモリ帯域バウンドであり、ここがVRAMボトルネック。softmax@Vは計算バウンドなので圧縮の恩恵が小さく、実装複雑度が倍増する。

**QJL補正は使わない。** MSE変体のみ使用するため、TQ-1cのQJLCorrector実装は将来の拡張用に残すが、Tritonカーネルでは使用しない。

---

## 作業順序

```
TQ-2-pre: DynamicCacheスモークテスト（最優先。全体の実装可否を判断）
  ↓ PASSなら続行、FAILならフォールバック設計
TQ-1a: Lloyd-Maxコードブック計算（核心の数学）
  ↓
TQ-1b: ランダム回転行列
  ↓
TQ-1c: QJL 1-bit残差補正
  ↓
TQ-1d: 統合クラス（TurboQuantCompressor。全演算GPU上torch.matmulで実行）
  ↓
TQ-3: コードブック事前計算スクリプト
  ↓
TQ-4: コアアルゴリズムのテスト（論文の理論値と照合）
  ↓
TQ-2a: TurboQuantCache（DynamicCache継承）
  ↓
TQ-2b: FieldAwareLLMへの統合
  ↓
TQ-5: 実機検証（VRAM・速度・品質）
```

**TQ-2-preを最初に置く理由:** DynamicCacheの継承がgenerate()で受け入れられなければ、TQ-1のコア実装は正しくても統合できない。1時間以内に判明するリスクを最初に潰す。

TQ-4（テスト）をTQ-2aの前に置く理由: コアアルゴリズムが論文の理論値と一致することを確認してからKVキャッシュに統合する。コアが間違っていたら統合しても意味がない。

---

## TQ-6. Tritonカーネル: 圧縮KVのまま attention 計算

### 背景

TQ-2の実装で判明した問題: transformersのattention計算はFP16テンソルを前提としており、圧縮KVを毎トークン復元する方式ではVRAM節約と速度改善が両立しない。解決策は、圧縮形式のままattention内積を計算するTritonカーネルの実装。

参照: 0xSero/turboquant（Tritonカーネル3本）、DEJAN blog（fused実装の設計判断）

### 設計判断: keyのみ圧縮する

attention計算の2ステップ:
1. `scores = Q @ K^T` — メモリ帯域バウンド（KVキャッシュの読み出しがボトルネック）
2. `output = softmax(scores) @ V` — 計算バウンド

ステップ1がVRAMボトルネック。**keyのみ圧縮し、valueはFP16のまま保持する**。理由:
- keyの圧縮だけでKVキャッシュの半分を削減。8GBに収まる可能性が十分ある
- valueの圧縮は追加のTritonカーネルが必要で複雑さが倍増
- 参照実装（0xSero）も同じ判断

### TQ-6a. fused decode attention カーネル

圧縮keyのままQ@K^T内積を計算するTritonカーネル。decode phase（1トークンずつ生成）で使用。

```python
import triton
import triton.language as tl

@triton.jit
def turboquant_decode_attention_kernel(
    # Query: 現在のトークンのQ (1, head_dim) — FP16
    Q_ptr,
    # 圧縮Key: indices (seq_len, head_dim) — int8
    K_indices_ptr,
    # 圧縮Key: norms (seq_len,) — FP16
    K_norms_ptr,
    # コードブック: centroids (n_levels,) — FP32
    codebook_ptr,
    # 回転行列: Π (head_dim, head_dim) — FP32
    rotation_ptr,
    # Value: (seq_len, head_dim) — FP16（非圧縮）
    V_ptr,
    # 出力: (1, head_dim) — FP16
    Out_ptr,
    # メタデータ
    seq_len, head_dim, n_levels,
    # ブロックサイズ
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """圧縮keyのまま1トークン分のattention出力を計算する。

    各スレッドブロックがseq_lenのチャンクを担当し:
    1. 圧縮keyを復元（codebook lookup + 逆回転）
    2. Q @ K^T の部分内積を計算
    3. online softmax で累積
    4. softmax(scores) @ V で出力を累積
    """
    ...
```

**カーネル内の処理フロー（1ブロック分）:**
```
for each block of BLOCK_SEQ tokens:
    1. indices[block] → codebook lookup → rotated_k (BLOCK_SEQ, head_dim)
    2. norms[block] → scale
    3. rotated_k → Π^T @ rotated_k → restored_k (BLOCK_SEQ, head_dim)
    4. scores = Q @ restored_k^T → (1, BLOCK_SEQ)
    5. scores *= scale（ノルム適用）
    6. online softmax 更新（running max, running sum）
    7. V[block] → output += softmax_weight @ V_block
```

**重要: 回転行列の逆変換（Π^T）がカーネル内にある。** head_dim×head_dimの行列乗算がブロックごとに走る。head_dim=128なので128×128。Tritonのtl.dotで処理可能だが、shared memoryに回転行列をキャッシュすること。

### TQ-6b. transformers attentionの差し替え

Qwen3-8Bのattention層のforward()をモンキーパッチし、decode phase（seq_len=1のquery）のときだけTritonカーネルを使う。prefill phaseは通常のattention（flash attention等）をそのまま使用。

```python
def patch_attention_for_turboquant(model, compressor):
    """Qwen3-8Bの各attention層にTurboQuant対応をパッチする。"""

    for layer in model.model.layers:
        original_forward = layer.self_attn.forward

        def turboquant_forward(*args, **kwargs):
            # prefill（seq_len > 1）: 通常のattention
            # その後keyを圧縮して保存
            if is_prefill:
                output = original_forward(*args, **kwargs)
                compress_and_store_keys(...)
                return output

            # decode（seq_len = 1）: Tritonカーネル
            return turboquant_decode_attention(...)

        layer.self_attn.forward = turboquant_forward
```

### TQ-6c. VRAM管理

prefill後にFP16のkey cacheを解放し、圧縮keyのみ保持する。value cacheはFP16のまま。

```
prefill完了時:
  key cache (FP16): 解放 → 圧縮key (int8 indices + FP16 norms) に置き換え
  value cache (FP16): そのまま保持

VRAM見積もり:
  model: 6.1 GB
  compressed keys: 元の ~1/5（3-bit） ≈ KVの半分のさらに1/5
  values (FP16): 元の半分
  合計: 大幅にVRAM内に収まるはず
```

### TQ-6d. テスト

| テスト | 内容 |
|--------|------|
| test_triton_kernel_correctness | Tritonカーネルの出力とPyTorch参照実装の出力がほぼ一致（cosine sim > 0.999） |
| test_triton_kernel_numerical | 異なるseq_len（32, 128, 512）でattention scoreが安定 |
| test_decode_with_compressed_keys | 圧縮keyでのdecode生成がFP16と同等の応答品質 |
| test_prefill_key_release | prefill後にFP16 key cacheが解放されていること（VRAM計測） |
| test_end_to_end_generate | generate_with_field()でTritonカーネル経由の応答生成が完走 |

---

## 更新された作業順序

```
TQ-2-pre: DynamicCacheスモークテスト（実施済み: PASS）
  ↓
TQ-1: コアアルゴリズム実装（実施済み: PASS）
  ↓
TQ-4: コアアルゴリズムテスト（実施済み: PASS）
  ↓
TQ-2: transformers KVキャッシュ統合（実施済み: VRAM削減確認、速度改善なし）
  ↓
TQ-6a: Triton fused decode attention カーネル実装 ← NEW
  ↓
TQ-6d-partial: カーネル単体テスト（PyTorch参照実装との一致確認）
  ↓
TQ-6b: transformers attention差し替え（モンキーパッチ）
  ↓
TQ-6c: VRAM管理（prefill後のkey解放）
  ↓
TQ-6d: 統合テスト
  ↓
TQ-5: 実機検証（VRAM・速度・品質）
```

### Tritonの環境確認

Tritonはtorchに同梱されている（torch >= 2.0）。追加インストール不要のはず。確認:
```python
import triton
print(triton.__version__)
```

RTX 3060 Ti（compute capability 8.6）はTritonがサポートしている。

---

## リスクと撤退基準

| リスク | 影響 | 対処 |
|--------|------|------|
| DynamicCache継承がgenerate()で受け入れられない | TQ-2全体が実装困難 | TQ-2-preで最初に判明。フォールバック: generate()を手動ステップループで回し、各ステップでKVキャッシュを圧縮/復元する |
| Tritonカーネル内の回転行列逆変換が遅い | decode速度が改善しない | 回転行列をshared memoryにキャッシュ。それでも遅ければWalsh-Hadamard Transform（O(d log d)）に切り替え |
| Tritonカーネルの数値精度問題 | attention scoreが不正確 | PyTorch参照実装との比較テスト（cosine sim > 0.999）で早期検出。FP32累積で精度確保 |
| attention モンキーパッチがQwen3-8Bで機能しない | TQ-6b実装困難 | Qwen3のattentionクラスのソースを確認し、forward()のシグネチャに合わせる |
| Python実装の圧縮/復元が遅すぎる | tok/s改善が限定的 | 全演算をtorch GPU上で実行（torch.matmul, torch.searchsorted等）。CPU-GPU転送を排除。それでもダメなら撤退 |
| QJLのbool配列がメモリを食う | 圧縮率が理論値を下回る | uint8にbit pack（128次元 → 16bytes）。pack/unpack関数を実装 |
| 3-bitで応答品質が劣化する | 実用に耐えない | 4-bitに切り替え（圧縮率は下がるがまだ有効） |
| VRAMが依然として溢れる | 根本解決にならない | bits を下げるか、別アプローチ（モデル変更等）に切り替え |

**撤退判断**: TQ-5でVRAMがピーク8GB以下に収まらない、またはtok/sが5以下の場合、この実装は保留とし別アプローチを検討する。
