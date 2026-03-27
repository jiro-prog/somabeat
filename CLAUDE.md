# CLAUDE.md

設計判断を含む作業を行う場合は [architect.md](architect.md) を読むこと。

## プロジェクト概要
integrated-system: Somabeatの統合システム。Discord Bot (Sleepy Jean)
を通じた対話、自己改善、夜間学習サイクルを統合。

## アーキテクチャ
- `llamarcute_live/` — 対話エンジン (DialogueManager, FieldAwareLLM, Personality)
- `discord_bot/` — Discord Bot エントリポイント
- `orchestrator/` — システム統合・ライフサイクル管理
- `shared_state/` — SharedField, FieldReceptor, TurboQuant, MultimodalEncoder
- `sleepyjean/` — 夜間学習サイクル (外部リポジトリ)
- `config/system.yaml` — 全体設定

## LLM推論
- モデル: Qwen/Qwen3-8B (4-bit NF4, bitsandbytes)
- KVキャッシュ: TurboQuant 4-bit圧縮
- Field embedding注入: system tokens と user tokens の間に挿入
- /no_think: system content末尾に付与、skip_special_tokens=False で処理

## 重要な設計判断
- 行動規範は英語で記述（LLMの指示追従性が英語で最も高いため）
- 不変制約はbuild_prompt()にハードコード（自己改善スクリプトからアクセス不能にする）
- NF4由来の軽微なトークン崩壊（低頻度外来語）は受容
- FR出力ノルムはL2正規化不可、一律スケーリングで保持

## テスト
- `python -m pytest tests/` (225件)
- テスト失敗はスキップせず修正する

## debug設定 (config/system.yaml)
- `debug.empty_perceive`: perceive結果を空にする（field embedding切り分け用）
- `debug.disable_turboquant`: TurboQuant無効化（FP16 KV、VRAM注意）
