# 指示書: GitHub公開用リポジトリ整理

**日付:** 2026-03-24
**目的:** integrated-system + SleepyJean のコードベースをGitHub公開可能な状態に整理する

---

## 方針

### リポジトリ構成: 2リポジトリ

**メインリポジトリ: `somabeat`**
- 統合システム本体（llamarcute-live, 共有状態の場, オーケストレーター, 免疫系, ブリッジ層, Discord Bot）
- 全設計文書
- テスト

**サブリポジトリ: `sleepyjean`（既存リポジトリをクリーンアップ）**
- SleepyJean夜間学習パイプライン
- SleepyJean Discord Bot
- メインリポジトリからREADMEでリンク参照

元のllamarcuteパイプラインは公開しない（統合システムのself_improve.pyに機能が再構成されているため）。設計文書で言及するのみ。

---

## 作業1: メインリポジトリのディレクトリ構造

現在のintegrated-system/を以下の構造に整理する。

```
somabeat/
├── README.md                          ← 新規作成（作業3）
├── LICENSE                            ← 新規作成（作業4）
├── .gitignore                         ← 新規作成（作業5）
├── requirements.txt                   ← 既存から整理
│
├── docs/
│   ├── bio_ai_architecture.md         ← 上位設計
│   ├── shared_field_design.md         ← 場のインターフェース設計
│   ├── llamarcute_live_design.md      ← llamarcute-live + SleepyJean変更 + 統合サイクル
│   ├── individuality_design.md        ← 個体性の設計
│   ├── individuality_taskflow.md      ← 個体性検証タスクフロー
│   ├── vision_integration_taskflow.md ← 視覚モジュールタスクフロー（未着手）
│   └── findings.md                    ← 論文のmarkdown版（既存PDFから変換）
│
├── shared_state/
│   ├── __init__.py
│   ├── interface.py                   ← Signal, SharedField Protocol, FieldEncoder等
│   └── backends/
│       ├── __init__.py
│       └── chromadb_backend.py
│
├── llamarcute_live/
│   ├── __init__.py
│   ├── dialogue.py                    ← 対話管理（会話バッファ、プロンプト構築、sense）
│   ├── self_improve.py                ← 自己改善（変異生成、二軸評価、選出）
│   ├── fitness.py                     ← fitness評価
│   ├── cuteness.py                    ← cuteness評価
│   ├── selection.py                   ← 人格選出
│   ├── personality.py                 ← 行動規範管理
│   ├── metrics.py                     ← 計測インフラ
│   └── data/
│       └── personality_v0.yaml        ← シード人格（公開OK）
│
├── orchestrator/
│   ├── __init__.py
│   └── orchestrator.py                ← 睡眠・覚醒サイクル管理
│
├── immune/                            ← 免疫系（現在の配置に応じてパスを調整）
│   ├── __init__.py
│   └── ...
│
├── bridge/
│   ├── __init__.py
│   ├── sleep_ingest.py                ← 入眠ブリッジ
│   └── wake_export.py                 ← 覚醒ブリッジ
│
├── discord_bot/
│   ├── __init__.py
│   └── bot.py
│
├── config/
│   └── system.yaml.example            ← テンプレート（実値なし）
│
├── scripts/
│   ├── run_experiment1_gate.py        ← 個体性実験1
│   └── run_experiment2_metacognition.py ← 個体性実験2
│
└── tests/
    ├── conftest.py
    ├── test_shared_state.py
    ├── test_chromadb_backend.py
    ├── test_llamarcute_live.py
    ├── test_bridge.py
    ├── test_orchestrator.py
    ├── test_self_improve.py
    ├── test_immune.py
    ├── test_cleaner.py
    ├── test_discord_bot.py
    ├── test_metrics.py
    ├── test_conversation_buffer.py
    ├── test_integration.py
    └── ...
```

### 注意

- 現在のディレクトリ構造と上記が一致しない場合は、ファイルを移動して上記に合わせる。ただし、import文の書き換えが大量に発生する場合は、現在の構造を維持しつつREADMEで説明する方式にする（無理に移動しない）
- `__init__.py`が存在しないディレクトリには追加する

---

## 作業2: SleepyJeanリポジトリの整理

SleepyJeanの既存リポジトリをクリーンアップする。

### 最低限必要なこと

- README.mdに「統合システム（somabeat）の記憶・学習系統として動作する」旨を追記
- 統合システムからの参照方法（ブリッジ層の説明）を記載
- APIキーやトークンがコードにハードコードされていないことを確認

### SleepyJean側の.gitignoreに追加すべきもの

```
# モデル・学習データ
models/
*.gguf
*.safetensors

# データベース
*.db

# 学習データ
data/training/

# 環境
.env
__pycache__/
```

---

## 作業3: README.md作成

メインリポジトリのREADME.md。以下の構成で作成する。

```markdown
# Somabeat — Bio-inspired Integrated AI Architecture

複数のLLMエージェントを「認知機能の分割」で統合し、一つの個体として振る舞うシステム。

## 何が面白いのか

- 毎晩寝て、記憶を整理して、翌朝少し賢くなって起きてくる
- メタ認知ルール1つから、3サイクルで独自の自己評価フレームワークが創発された
- 壊れたら免疫系が自動で治す

## アーキテクチャ概要

[bio_ai_architecture.md セクション7のASCIIアートを挿入]

### 三系統

| 系統 | 認知機能 | 担当 |
|------|---------|------|
| 神経系（llamarcute-live） | 人格・推論・自己改善 | リアルタイム対話、行動規範の漸進的更新 |
| 内分泌系（SleepyJean） | 記憶・学習・忘却 | 夜間バッチでの記憶定着と知識蒸留 |
| 免疫系 | 障害検知・修復 | ヘルスチェック、ロールバック |

### 設計原則

1. 認知機能の三系統分離
2. 共有状態を介した間接協調（APIで直接呼び合わない）
3. 自己同一性（人格の変化は漸進的にのみ）
4. 睡眠は全身状態

## 主な成果

- 共有状態の場を介した間接協調による認知的循環の成立
- 睡眠中の自己改善メカニズムによる人格の漸進的進化（v0→v16、20サイクル）
- 免疫系による異常検知・自己修復の実装と運用検証
- メタ認知ルールの自律的進化（手動1ルール→3サイクルで自己評価フレームワーク創発）
- 個体性の操作的定義と検証（方向B: 場の読み取りパターンからの創発）

## 設計文書

| ドキュメント | 内容 |
|------------|------|
| [bio_ai_architecture.md](docs/bio_ai_architecture.md) | 上位設計。ビジョン・設計原則・全体像 |
| [shared_field_design.md](docs/shared_field_design.md) | 共有状態の場のインターフェース設計 |
| [llamarcute_live_design.md](docs/llamarcute_live_design.md) | 統合サイクルの詳細設計 |
| [individuality_design.md](docs/individuality_design.md) | 個体性の操作的定義と検証計画 |
| [findings.md](docs/findings.md) | 得られた知見のまとめ |

## 関連プロジェクト

- [SleepyJean](リンク) — 記憶・学習系統。夜間学習パイプライン

## セットアップ

### 必要なもの

- Python 3.11+
- Ollama（qwen3:8bモデル）
- ChromaDB
- Discord Bot Token

### インストール

[手順を記載]

### 設定

`config/system.yaml.example` をコピーして `config/system.yaml` を作成し、各種設定値を入力する。

### 実行

[起動コマンドを記載]

## ライセンス

[LICENSE参照]

## 引用

このプロジェクトを参照する場合:
[引用フォーマット]
```

README.mdの内容はオーナーが最終調整する前提で、骨格を作成する。

---

## 作業4: LICENSE

MITライセンスを推奨。学術・研究用途での参照を妨げないため。オーナーに確認してから作成。

---

## 作業5: .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# データベース
*.db
*.sqlite3

# ChromaDB
chroma_data/

# モデル・学習データ
models/
*.gguf
*.safetensors
*.pt
*.pth

# 設定（実値）
config/system.yaml
.env

# 運用データ
data/cycle_history.json
data/personality_backups/
data/personality_current.yaml
data/experiment*_results.json
data/experiment*_evaluation.json

# LoRA（停止中だが念のため）
lora_output/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
```

---

## 作業6: config/system.yaml.example

現在のsystem.yamlから実値を除去し、テンプレートを作成する。

```yaml
# Somabeat — Bio-inspired Integrated AI Architecture — Configuration Template
# Copy this file to system.yaml and fill in your values.

discord:
  token: "YOUR_DISCORD_BOT_TOKEN"
  channel_id: 0  # Your Discord channel ID

llamarcute_live:
  model: "qwen3:8b"
  max_conversation_history: 10
  self_awareness_query: "最近の自分の調子"

sleepyjean:
  local_model: "qwen3:8b"
  target_qa_count:
    homework: 5
    curiosity: 3
  max_homework_per_night: 2
  min_curiosity_per_night: 2

orchestrator:
  night_cycle_timeout_sec: 1800
  self_improvement_timeout_sec: 1800
  sleep_schedule: "03:00"

ollama:
  base_url: "http://localhost:11434"
  parallel: 2

claude_api:
  # Used only for Q&A reference generation (5 calls/cycle)
  api_key: "YOUR_CLAUDE_API_KEY"

metrics:
  enabled: true

immune:
  fitness_warning: 0.80
  fitness_critical: 0.70
  token_warning: 650
  token_critical: 800
  rule_warning: 13
  rule_critical: 15
```

実際の設定項目は現在のsystem.yamlの内容に合わせて調整すること。上記は構造の例。

---

## 作業7: 公開してはいけないファイルの確認

以下のファイルがリポジトリに含まれていないことを確認する。

### 絶対に含めてはいけない

| 対象 | 理由 |
|------|------|
| Discordトークン | セキュリティ |
| Claude APIキー | セキュリティ |
| Ollama APIキー（もしあれば） | セキュリティ |
| .envファイル | 環境変数にキーが含まれる可能性 |

### 含めない方がいい

| 対象 | 理由 |
|------|------|
| personality_current.yaml | 現在の人格の生データ |
| cycle_history.json | 運用履歴の実データ |
| *.db（SQLiteファイル） | 対話ログ、メトリクス等の実データ |
| ChromaDBのデータディレクトリ | 場の信号の実データ |
| experiment*_results.json | 実験の生応答データ |
| personality_backups/ | バックアップの実データ |
| models/ | モデルファイル（サイズが大きい） |

### 確認手順

1. `git status`と`git diff`で追跡対象を確認
2. 上記のファイルが.gitignoreに含まれていることを確認
3. 過去のコミット履歴にAPIキー等が含まれていないか確認（含まれている場合は`git filter-branch`や`BFG Repo-Cleaner`で削除）
4. **初回公開前に`git log -p | grep -i "token\|api_key\|secret"`で検索**

---

## 作業8: findings.mdの作成

bio_ai_architecture_findings.pdf（論文）のmarkdown版を作成し、docs/findings.mdに配置する。

- PDFの内容をmarkdownに変換
- T12の成果（メタ認知ルールの創発、SleepyJeanリファクタリング、個体性の暫定評価）を追加セクションとして含める
- 参考文献のリンクを有効にする

---

## 作業順序

```
作業7（セキュリティ確認）       ← 最初に必ず実施
  ↓
作業5（.gitignore作成）
  ↓
作業1（ディレクトリ整理）       ← 最も作業量が大きい
  ↓
作業6（config template）
  ↓
作業3（README.md）
  ↓
作業4（LICENSE）               ← オーナー確認後
  ↓
作業8（findings.md）
  ↓
作業2（SleepyJeanリポジトリ）
```

作業7を最初に実施する理由: セキュリティ問題を残したまま他の整理を進めると、うっかり公開するリスクがある。

---

## 確認事項

全作業完了後、公開前に以下を最終確認:

1. `grep -r "sk-" . --include="*.py" --include="*.yaml"` でAPIキーの残留がないこと
2. `grep -r "token" config/ --include="*.yaml"` で実トークンが含まれていないこと
3. .gitignoreが全ての除外対象をカバーしていること
4. README.mdのリンクが全て有効であること
5. `pip install -r requirements.txt` が通ること
6. 全テストがパスすること
7. system.yaml.exampleの設定項目が現在のコードと一致していること
