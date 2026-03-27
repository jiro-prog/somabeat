# 指示書: 視覚モジュール統合 — Phase C実行

**日付:** 2026-03-26
**対象:** somabeat（integrated-system）
**親文書:** docs/vision_integration_taskflow.md
**前提:** Phase B完了（α=0.5, D=384, PASS）

---

## メモリ制約（全ステップ共通）

| リソース | 容量 | 常駐 |
|---------|------|------|
| VRAM | 8,192 MiB (RTX 3060 Ti) | Qwen3-8B (Ollama) ~5GB |
| RAM | 7.7 GB | OS + Python ~2GB |

**OOM防止の原則:**
- SigLIP ViT-B（~350MB VRAM）とQwen3-8B（~5GB VRAM）を同時にGPUにロードしない
- SigLIPは推論時にCPUで動かすか、Ollama一時停止時のみGPU使用
- 画像のバッチ処理は1枚ずつまたは小バッチ（最大8枚）
- 721件のre-encodeはバッチ処理。全件を一度にメモリに載せない

---

## 作業一覧

| # | 作業 | 影響範囲 |
|---|------|---------|
| C-0 | 事前確認（テスト数、Ollama状態） | — |
| C-1 | MultimodalFieldEncoderの実装 | shared_state/ |
| C-2 | ChromaDBコレクションの再構築 | shared_state/, data/ |
| C-3 | SensoryVisionモジュールの実装 | 新規ディレクトリ |
| C-4 | 既存テスト + 新規テストの実行 | tests/ |
| C-5 | 統合動作確認 | 全体 |

---

## C-0. 事前確認

### C-0a. テスト数の確認

現在のテスト数を確認し、Phase C合格条件のパスラインを確定する。

```bash
cd /home/jiro/integrated-system && .venv/bin/python -m pytest --co -q | tail -1
```

Phase B時点で292件（integrated-system 189 + SleepyJean 103）。変更があれば最新値を使用。

### C-0b. Ollamaの状態

```bash
nvidia-smi
ollama list
```

Qwen3-8Bがロードされている場合のVRAM使用量を記録。Phase C中のSigLIPとの共存計画に使う。

### C-0c. Phase B成果物の確認

```bash
ls -la data/vision_phase_b/alpha_0.5/
# projection_text.pt, projection_img.pt が存在すること
```

---

## C-1. MultimodalFieldEncoderの実装

### C-1a. FieldEncoder Protocolの拡張

**ファイル:** `shared_state/interface.py`

既存のFieldEncoder Protocolに`encode_image`メソッドを追加。

```python
class FieldEncoder(Protocol):
    def encode(self, text: str) -> NDArray[np.float32]: ...
    def encode_for_emit(self, text: str) -> NDArray[np.float32]: ...
    def encode_for_sense(self, text: str) -> NDArray[np.float32]: ...
    def encode_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]: ...  # NEW
    def dimensionality(self) -> int: ...
```

**既存のE5SmallEncoderへの影響:** `encode_image`が未実装でも、テキスト専用として動作し続ける必要がある。E5SmallEncoderに`encode_image`のスタブ（NotImplementedError）を追加するか、Protocolをオプショナルメソッドにする。

**推奨:** E5SmallEncoderにスタブを追加。テスト時にE5SmallEncoderを使っている箇所がencode_imageを呼ばないことを保証。

```python
class E5SmallEncoder:
    # ... 既存メソッド ...

    def encode_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        raise NotImplementedError("E5SmallEncoder does not support image encoding. Use MultimodalFieldEncoder.")
```

### C-1b. MultimodalFieldEncoderの実装

**新規ファイル:** `shared_state/multimodal_encoder.py`

```python
class MultimodalFieldEncoder:
    def __init__(
        self,
        e5_model_name: str = "intfloat/multilingual-e5-small",
        siglip_model_name: str = "google/siglip2-base-patch16-256",
        projection_text_path: str = "data/vision_phase_b/alpha_0.5/projection_text.pt",
        projection_img_path: str = "data/vision_phase_b/alpha_0.5/projection_img.pt",
        device: str = "cpu",  # デフォルトCPU（OOM防止）
    ):
        # e5-small（テキストエンコーダ）
        self.text_encoder = SentenceTransformer(e5_model_name)

        # SigLIP ViT-B（画像エンコーダ）— CPUにロード
        self.image_encoder = AutoModel.from_pretrained(siglip_model_name).to(device)
        self.image_processor = AutoProcessor.from_pretrained(siglip_model_name)

        # 学習済みprojection heads
        self.projection_text = self._load_projection(projection_text_path, 384, 512, 384)
        self.projection_img = self._load_projection(projection_img_path, 768, 512, 384)

        # 推論モード固定
        self.projection_text.eval()
        self.projection_img.eval()
        self.image_encoder.eval()
```

**メモリ管理の要点:**

| コンポーネント | ロード先 | 推定メモリ | 常駐 |
|-------------|---------|----------|------|
| e5-small | CPU (RAM) | ~120MB | 常駐（既存と同じ） |
| SigLIP ViT-B | CPU (RAM) | ~350MB | 常駐 |
| projection_text | CPU (RAM) | ~1.6MB | 常駐 |
| projection_img | CPU (RAM) | ~2.4MB | 常駐 |
| **合計追加** | | **~354MB** | |

RAM 7.7GBのうち空き~5.7GB。354MB追加は問題なし。SigLIPをCPUで動かすので、VRAMには影響なし。

**encode_imageの実装:**

```python
def encode_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """画像1枚をD次元共通空間にエンコードする。"""
    with torch.no_grad():
        # SigLIPで画像embedding取得
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        img_features = self.image_encoder.get_image_features(**inputs)  # (1, 768)

        # projection_imgで共通空間に射影
        projected = self.projection_img(img_features)  # (1, 384)

    return projected.squeeze(0).numpy()  # (384,) — 非正規化
```

**注意: 非正規化で返す。** ノルムが濃度を担う設計原則（shared_field_design.md セクション6.2）に従い、正規化しない。画像信号の「確信度」はノルムに反映される。

**encode（テキスト）の実装:**

```python
def encode_for_emit(self, text: str) -> NDArray[np.float32]:
    """テキストをD次元共通空間にエンコードする（emit用）。"""
    with torch.no_grad():
        raw = self.text_encoder.encode(f"passage: {text}", convert_to_numpy=True)  # (384,)
        raw_tensor = torch.tensor(raw).unsqueeze(0)  # (1, 384)
        projected = self.projection_text(raw_tensor)  # (1, 384)
    return projected.squeeze(0).numpy()  # (384,) — 非正規化

def encode_for_sense(self, text: str) -> NDArray[np.float32]:
    """テキストをD次元共通空間にエンコードする（sense用）。"""
    with torch.no_grad():
        raw = self.text_encoder.encode(f"query: {text}", convert_to_numpy=True)  # (384,)
        raw_tensor = torch.tensor(raw).unsqueeze(0)  # (1, 384)
        projected = self.projection_text(raw_tensor)  # (1, 384)
    return projected.squeeze(0).numpy()  # (384,)
```

### C-1c. 設定による切り替え

**ファイル:** `config/system.yaml.example`

```yaml
encoder:
  type: "multimodal"  # "e5_small" or "multimodal"
  model_name: "intfloat/multilingual-e5-small"
  siglip_model_name: "google/siglip2-base-patch16-256"
  projection_text_path: "data/vision_phase_b/alpha_0.5/projection_text.pt"
  projection_img_path: "data/vision_phase_b/alpha_0.5/projection_img.pt"
  device: "cpu"  # SigLIPのデバイス。"cpu"推奨（VRAM節約）
```

FieldEncoderの生成ロジック（おそらくorchestrator.pyかmain.py）で、`encoder.type`に応じてE5SmallEncoderかMultimodalFieldEncoderを生成する。

**フォールバック:** multimodalの初期化に失敗した場合（SigLIPのダウンロード失敗等）、E5SmallEncoderにフォールバックしてログにwarningを出す。視覚機能は無効になるが、テキスト機能は維持される。

---

## C-2. ChromaDBコレクションの再構築

### C-2a. 既存コレクションのバックアップ

```python
# snapshotで場の全信号を取得
snap = await field.snapshot()
# JSON or pickleでバックアップ
save_snapshot(snap, "data/chromadb_backup_pre_vision.json")
```

**バックアップに含めるもの:**
- 全Signalのtrace, origin, emitted_at, extra
- embeddingは含めない（re-encodeするので不要）

### C-2b. 新コレクションの作成

次元数は384のまま（D=384、変更なし）。

**重要: 次元数が変わらないので、コレクションの再作成は不要。** 既存の信号のembeddingをprojection_textで変換して上書きする方式の方がシンプル。

手順:
1. 既存信号を全件取得（snapshot）
2. 各信号のtraceをMultimodalFieldEncoderでre-encode
3. 既存信号のembeddingを新しいembeddingで置き換え

**ただし、ChromaDBのAPIで既存エントリのembedding更新が可能かを確認すること。** 不可能な場合は、コレクション削除→再作成→全件再投入。

### C-2c. re-encodeの実行

```python
encoder = MultimodalFieldEncoder(...)
snap = await field.snapshot()

for signal in snap.signals:
    new_embedding = encoder.encode_for_emit(signal.trace)
    # ChromaDB updateまたは delete + add
    collection.update(
        ids=[signal.signal_id],
        embeddings=[new_embedding.tolist()]
    )
```

**OOM防止:** 721件を1件ずつ処理。バッチ化してもRAM的に問題ないが（721×384×4≈1MB）、安全のため100件ずつのバッチで処理し、進捗ログを出す。

**推定時間:** e5エンコード + projection推論で1件あたり数ms。721件で数秒。

### C-2d. re-encode後の検証

- コレクションの信号数が721件のままであること
- re-encode前後でtrace、origin、emitted_at、extraが不変であること
- re-encode後のembeddingがprojection_textの出力と一致すること（ランダム10件で検証）

---

## C-3. SensoryVisionモジュールの実装

### C-3a. スケルトン実装

**新規ファイル:** `sensory/vision.py`（ディレクトリ`sensory/`を新設）

```python
class SensoryVision:
    def __init__(
        self,
        encoder: FieldEncoder,
        field: SharedField,
        thalamus_threshold: float = 0.1,  # コサイン距離の閾値
    ):
        self.encoder = encoder
        self.field = field
        self.thalamus_threshold = thalamus_threshold
        self._last_embedding: NDArray | None = None

    async def process_frame(self, image: NDArray[np.uint8]) -> bool:
        """画像フレームを処理し、必要に応じて場にemitする。
        Returns: emitしたかどうか。
        """
        embedding = self.encoder.encode_image(image)

        # 視床フィルタ: 前フレームとの距離が閾値以下なら抑制
        if self._last_embedding is not None:
            cosine_dist = 1.0 - np.dot(embedding, self._last_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(self._last_embedding)
            )
            if cosine_dist < self.thalamus_threshold:
                return False  # 抑制（変化が小さい）

        # 場にemit
        signal_id = await self.field.emit(
            embedding=embedding,
            origin=SignalOrigin(system="sensory:vision", context="frame"),
            trace="[視覚] 画面キャプチャの視覚信号",
        )

        self._last_embedding = embedding
        return True

    def clear(self):
        """入眠時にリセット。"""
        self._last_embedding = None
```

### C-3b. 視床フィルタの設計

| パラメータ | 初期値 | 根拠 |
|-----------|--------|------|
| thalamus_threshold | 0.1 | Phase B検証データから類似画像のコサイン距離を計測して調整。0.1は暫定値 |

閾値はconfig/system.yamlに外出し:

```yaml
sensory:
  vision:
    enabled: false  # デフォルト無効。手動で有効化
    thalamus_threshold: 0.1
    capture_interval_sec: 5.0  # キャプチャ間隔
```

**`enabled: false`がデフォルト。** Phase Cのテスト通過を確認してから有効化する。既存の運用に影響しない。

### C-3c. 画面キャプチャ

画面キャプチャの具体的な実装はスコープ外（プラットフォーム依存）。SensoryVisionはimage: NDArrayを受け取るインターフェースとし、キャプチャ方法は呼び出し側に委ねる。

テスト用にはダミー画像（numpy配列）を渡す。

### C-3d. traceフィールドの扱い

現時点ではtrace（人間可読な記述）は固定文字列。将来的にVLMでキャプション生成してtraceに設定することも可能だが、Phase Cのスコープ外。

```python
trace="[視覚] 画面キャプチャの視覚信号"
```

### C-3e. 睡眠中の停止

orchestrator.pyのenter_sleep()でSensoryVision.clear()を呼ぶ。睡眠中は視覚モジュールがemitしない。設計原則4（睡眠は全身状態）との整合。

---

## C-4. テスト

### C-4a. 既存テストの通過確認

```bash
cd /home/jiro/integrated-system && .venv/bin/python -m pytest -x -q
```

FieldEncoder Protocolの変更（encode_image追加）で既存テストが壊れないことを確認。E5SmallEncoderにスタブを追加しているので、encode_imageを呼ばないテストは影響なし。

### C-4b. 新規テスト

**新規ファイル:** `tests/test_multimodal_encoder.py`

| テスト | 内容 |
|--------|------|
| test_encode_image_shape | encode_imageが(384,)のベクトルを返すこと |
| test_encode_text_shape | encode_for_emit/senseが(384,)のベクトルを返すこと |
| test_same_space | encode_for_emitとencode_imageの出力のコサイン類似度が[0,1]範囲であること |
| test_projection_text_not_identity | projection_text通過後のベクトルが元のe5ベクトルと完全一致しないこと（α=0.5なので変形される） |
| test_e5_fallback | E5SmallEncoder.encode_imageがNotImplementedErrorを出すこと |
| test_config_switching | encoder.type設定でE5SmallEncoder/MultimodalFieldEncoderが切り替わること |
| test_initialization_failure_fallback | SigLIPロード失敗時にE5SmallEncoderにフォールバックすること |

**新規ファイル:** `tests/test_sensory_vision.py`

| テスト | 内容 |
|--------|------|
| test_first_frame_always_emits | 初回フレームは常にemitされること |
| test_similar_frame_suppressed | 同一画像の連続でemitが抑制されること |
| test_different_frame_emits | 異なる画像ではemitされること |
| test_clear_resets_state | clear()後に同じ画像を渡すとemitされること |
| test_emit_signal_format | emitされたSignalのorigin.systemが"sensory:vision"であること |

**OOM防止のテスト設計:**
- SigLIPのロードが重いので、test_multimodal_encoder.pyではfixture scopeを`module`にして1回だけロード
- テスト用ダミー画像は小さいサイズ（64×64 RGB）を使用

```python
@pytest.fixture(scope="module")
def multimodal_encoder():
    """モジュール全体で1回だけMultimodalFieldEncoderをロード。"""
    try:
        return MultimodalFieldEncoder(device="cpu")
    except Exception:
        pytest.skip("MultimodalFieldEncoder initialization failed")
```

### C-4c. SleepyJean側テストへの影響

SleepyJean側のテストはFieldEncoderに依存しない（独自のOllamaクライアントを使用）。影響なし。確認のため一度実行する。

---

## C-5. 統合動作確認

### C-5a. 手動テスト: re-encode後のsenseが正常動作するか

```python
# MultimodalFieldEncoderでsense
encoder = MultimodalFieldEncoder(...)
query = encoder.encode_for_sense("最近の自分の調子")
reading = await field.sense(query, SenseParams(max_signals=10))

# 結果にテキスト信号が含まれること
for ws in reading.signals:
    print(f"{ws.signal.origin.context}: {ws.signal.trace[:50]}... (weight={ws.effective_weight:.3f})")
```

re-encode前後でsense結果の順位が大きく変わっていないことを確認。projection_textはα=0.5で学習されているので、元のe5空間からの乖離は小さい（Spearman=0.925）が、実運用での確認は必要。

### C-5b. 手動テスト: 視覚信号のemitとsense

```python
# ダミー画像でemit
dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
vision = SensoryVision(encoder, field, thalamus_threshold=0.1)
emitted = await vision.process_frame(dummy_image)
assert emitted

# テキストクエリで視覚信号がsenseできるか
query = encoder.encode_for_sense("画面の様子")
reading = await field.sense(query, SenseParams(max_signals=10))
# 視覚信号が結果に含まれること
vision_signals = [ws for ws in reading.signals if ws.signal.origin.system == "sensory:vision"]
print(f"Vision signals in reading: {len(vision_signals)}")
```

### C-5c. レイテンシ計測

| 操作 | 目標 | 計測 |
|------|------|------|
| encode_image（CPU） | <500ms | `time.time()`で計測 |
| encode_for_emit（projection_text付き） | <50ms | 同上 |
| encode_for_sense（projection_text付き） | <50ms | 同上 |
| sense（re-encode後のChromaDB） | <500ms | 同上 |

encode_image（SigLIP on CPU）が最もレイテンシが大きい。500ms以内を目安。超過する場合はcapture_interval_secを長くして頻度を下げる対応。

### C-5d. 睡眠サイクルへの影響確認

視覚モジュールはデフォルト無効（`sensory.vision.enabled: false`）なので、睡眠サイクルに影響しないことを確認。

有効化した場合に、enter_sleep()でSensoryVision.clear()が呼ばれることを確認。

---

## C-6. Phase C合格条件

- [ ] MultimodalFieldEncoderがFieldEncoder Protocolを満たしている
- [ ] E5SmallEncoderにencode_imageスタブが追加され、既存テストが壊れない
- [ ] config設定でE5SmallEncoder/MultimodalFieldEncoderが切り替え可能
- [ ] 初期化失敗時にE5SmallEncoderにフォールバックする
- [ ] 721件の既存信号がre-encode済み
- [ ] re-encode後のtrace/origin/emitted_at/extraが不変
- [ ] SensoryVisionの視床フィルタが動作する
- [ ] テキストクエリで視覚信号がsenseでヒットする
- [ ] 視覚モジュールのデフォルト設定がdisabled
- [ ] 既存テスト全件（C-0aで確認した件数）がpass
- [ ] 新規テスト（encoder 7件 + vision 5件 = 12件）がpass
- [ ] encode_imageのレイテンシが500ms以内（CPU）
- [ ] 睡眠サイクルに影響がない

---

## 作業順序

```
C-0: 事前確認（5分）
  ↓
C-1: MultimodalFieldEncoder実装（1-2時間）
  ↓
C-4a: 既存テスト通過確認（10分）
  ↓
C-2: ChromaDB re-encode（30分）
  ↓
C-3: SensoryVision実装（1時間）
  ↓
C-4b: 新規テスト追加・実行（30分）
  ↓
C-5: 統合動作確認（30分）
  ↓
C-6: 合格判定
```

**C-1の後、C-4aを先に実行する理由:** FieldEncoder Protocolの変更で既存テストが壊れていないことを、ChromaDB再構築の前に確認する。re-encode後に「実はProtocol変更でテスト壊れてました」だと手戻りが大きい。

**推定合計: 4-5時間**
