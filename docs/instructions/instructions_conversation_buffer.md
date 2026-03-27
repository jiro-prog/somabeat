# 指示書: llamarcute-live 会話バッファの実装

**日付:** 2026-03-22
**対象:** integrated-system — llamarcute-live
**目的:** マルチターン対話の文脈維持（ワーキングメモリの導入）

---

## 背景

llamarcute-liveのプロンプトに直近の会話履歴が含まれていない。場からのsense結果は意味検索ベースの長期記憶であり、「さっき話していたこと」のような直近の会話文脈を保持する仕組みがない。ユーザーが指示代名詞（「あれ」「それ」）や前のターンの話題を前提とした発言をしても、LLMには文脈がわからない。

## 設計判断

- メモリ上のリングバッファを採用（案B）。DB読み取りやDiscord API依存を避け、dialogue.py内で完結する
- 再起動で消えるが、会話の自然な区切りとして許容する
- 場のsense（長期記憶の想起）との役割分担: 会話バッファは「ワーキングメモリ」、senseは「長期記憶からの検索」。生体の前頭前野と海馬の関係に相当。設計原則2（間接協調）には抵触しない

---

## T1: DialogueManagerに会話バッファを追加

### 変更箇所: `llamarcute_live/dialogue.py`

DialogueManagerクラスに会話履歴のリングバッファを追加する。

```python
class DialogueManager:
    def __init__(self, ...):
        # ... 既存の初期化 ...
        self._history: list[dict] = []
        self._max_history: int = 10  # user+assistantで5往復分
```

`_max_history`はconfig/system.yamlから読み込み可能にする:

```yaml
llamarcute_live:
  max_conversation_history: 10
```

### process_input()の変更

応答生成後にuser/assistantのペアを追記し、上限を超えたら古い方から捨てる。

```python
async def process_input(self, user_input: str) -> str:
    # ... 既存のsense処理 + プロンプト構築 + LLM推論 ...
    response = await self.generate(prompt)

    # NEW: 会話バッファに追記
    self._history.append({"role": "user", "content": user_input})
    self._history.append({"role": "assistant", "content": response})
    if len(self._history) > self._max_history:
        self._history = self._history[-self._max_history:]

    # ... 既存のemit処理 ...
    return response
```

### テスト

- 会話バッファが対話ごとに蓄積されること
- `_max_history`を超えたら古いターンから削除されること
- 空の状態（初回対話）で正常動作すること

---

## T2: build_prompt()にバッファを注入

### 変更箇所: `llamarcute_live/dialogue.py`

プロンプト構造を以下に変更する:

```
[System]
## 不変制約（自己改善の対象外）
- 応答言語: 日本語で応答すること
- 応答長: Discordの2000文字制限を意識し、自然な区切りで収めること
- 本セクションは自己改善による変更の対象外である

## 行動規範
{behavioral_rules_yaml}

## 自己認識（共有状態の場から取得）
{自己認識クエリの結果}

## 最近の記憶（共有状態の場から取得）
{対話コンテキストクエリの結果}

## 直近の会話                          ← NEW
{会話バッファの内容}

[User]
{ユーザの入力}
```

### 会話バッファのフォーマット

```python
def _format_history(self) -> str:
    if not self._history:
        return ""
    lines = []
    for h in self._history:
        role = "ユーザー" if h["role"] == "user" else "あなた"
        lines.append(f"{role}: {h['content']}")
    return "\n".join(lines)
```

### 配置順序の根拠

「直近の会話」を「最近の記憶」の後、ユーザー入力の直前に置く。LLMにとって、直近の会話がユーザー入力に最も近い位置にあることで、文脈の接続が自然になる。

### テスト

- 会話バッファが空のとき「直近の会話」セクションがプロンプトに含まれないこと
- 会話バッファがあるとき、正しいフォーマットでプロンプトに含まれること
- 不変制約→行動規範→自己認識→最近の記憶→直近の会話→ユーザー入力の順序が正しいこと

---

## T3: /sleepコマンドと再起動時のバッファクリア

### 変更箇所: `llamarcute_live/dialogue.py`

バッファをクリアするメソッドを追加:

```python
def clear_history(self):
    self._history.clear()
```

### 呼び出し箇所

| タイミング | 箇所 | 理由 |
|-----------|------|------|
| 入眠時 | `orchestrator.py` — `enter_sleep()`の`stop_dialogue()`後 | 睡眠前後で文脈が切れるのは自然 |

Bot再起動時は`_history`が空リストで初期化されるので、明示的なクリアは不要。

### テスト

- `clear_history()`後に`_history`が空であること
- 入眠→覚醒後の最初の対話でバッファが空の状態から始まること

---

## T4: トークン数の安全確認

### 目的

会話バッファ追加によるプロンプト全体のトークン数増加が、Qwen3-8Bのコンテキストウィンドウ（32768トークン）を圧迫しないことを確認する。

### 確認方法

以下の最大ケースでのトークン数を計測する:

| セクション | 最大文字数目安 | トークン数目安 |
|-----------|-------------|-------------|
| 不変制約 | ~150字 | ~80 |
| 行動規範 | ~650字（WARNING閾値） | ~350 |
| 自己認識（sense結果10件） | ~2000字 | ~1000 |
| 最近の記憶（sense結果10件） | ~2000字 | ~1000 |
| 直近の会話（10ターン） | ~2000字 | ~1000 |
| ユーザー入力 | ~500字 | ~250 |
| **合計** | **~7300字** | **~3680** |

32kに対して約11%。十分な余裕がある。

### テスト

- 最大バッファ + 最大sense結果でプロンプトを構築し、トークン数が上限以内であることを確認
- 上限を超える場合の対処（バッファの古いターンをさらに削除）は現時点では不要だが、将来の拡張点として認識しておく

---

## 確認事項

全作業完了後、以下を確認:

1. 既存テスト（170件）が全パスすること
2. 新規テスト（T1: 3件、T2: 3件、T3: 2件、T4: 1件、計9件程度）が全パスすること
3. Discord上でマルチターン対話が成立すること（「ゲームの話をして」→「それの続き教えて」で文脈が維持されるか）
4. /sleep後の最初の対話でバッファがクリーンであること
5. 長時間対話（10ターン超）でバッファが正しくローテーションし、古いターンが消えること
