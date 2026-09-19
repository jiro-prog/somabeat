"""Triple extraction from dialogue logs using generate_bare.

Runs during sleep to extract factual triples from conversations.

Reference: instructions_kg_ollama_removal.md K-2
"""

from __future__ import annotations

import logging
from datetime import datetime

from sleepyjean.knowledge_graph import Triple

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
あなたは対話ログから事実を抽出するアシスタントです。

【最重要ルール】
対話中に話者が明示的に述べた事実のみを抽出してください。
あなた自身の知識で補完・推測してはいけません。
対話に書かれていない情報は、たとえ事実であっても出力禁止です。

【除外対象】
- 挨拶、感想、主観的評価
- 対話に書かれていない背景知識や関連情報
- 話者が述べていない属性や特徴

【出力形式】1行1トリプル:
主語 | 関係 | 目的語

事実がない場合は「なし」とだけ出力してください。

【正しい抽出の例】
対話: 「フォレトスはむし・はがねタイプだよ」
正しい出力: フォレトス | タイプ | むし・はがね

【誤った抽出の例】
対話: 「フォレトスはむし・はがねタイプだよ」
誤った出力: フォレトス | 生息地 | 森（←対話に書かれていない。これを出力してはいけない）
誤った出力: フォレトス | 進化前 | クヌギダマ（←事実だが対話に書かれていない。これも出力禁止）"""


class TripleExtractor:
    """Extract factual triples from dialogue logs via LLM (generate_bare)."""

    def __init__(self, llm, config: dict) -> None:
        self._llm = llm
        self._chunk_size = config.get("chunk_size", 3)
        self._max_new_tokens = config.get("max_new_tokens", 512)
        self._temperature = config.get("temperature", 0.3)

    async def extract(self, dialogue_logs: list[dict]) -> list[Triple]:
        """Extract triples from dialogue logs.

        1. Split logs into chunks (chunk_size turns each)
        2. For each chunk, call generate_bare with extraction prompt
        3. Parse LLM output into Triple objects
        4. Skip parse failures (log warning)
        """
        if not dialogue_logs:
            return []

        chunks = self._chunk_logs(dialogue_logs)
        all_triples: list[Triple] = []

        for i, chunk in enumerate(chunks):
            episode_id = f"chunk_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            user_input = self._format_chunk(chunk)

            try:
                response, duration = await self._llm.generate_bare(
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_input=user_input,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                )

                if response is None:
                    logger.warning("Triple extraction chunk %d: LLM returned None", i)
                    continue

                triples = self._parse_triples(response, episode_id)
                all_triples.extend(triples)
                logger.info(
                    "Triple extraction chunk %d/%d: %d triples (%.1fs)",
                    i + 1, len(chunks), len(triples), duration,
                )

            except Exception as e:
                logger.error("Triple extraction chunk %d failed: %s", i, e)
                continue

        logger.info("Total triples extracted: %d from %d chunks", len(all_triples), len(chunks))
        return all_triples

    def _chunk_logs(self, logs: list[dict]) -> list[list[dict]]:
        """Split dialogue logs into chunks of chunk_size turns."""
        # A "turn" is a user+assistant pair. chunk_size is in pairs.
        pairs_per_chunk = self._chunk_size
        entries_per_chunk = pairs_per_chunk * 2  # user + assistant

        chunks = []
        for start in range(0, len(logs), entries_per_chunk):
            chunk = logs[start:start + entries_per_chunk]
            if chunk:
                chunks.append(chunk)
        return chunks

    def _format_chunk(self, chunk: list[dict]) -> str:
        """Format a dialogue chunk for the extraction prompt."""
        lines = []
        for entry in chunk:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")[:300]
            if role == "user":
                lines.append(f"ユーザー: {content}")
            else:
                lines.append(f"AI: {content}")
        return "\n".join(lines)

    def _parse_triples(self, llm_output: str, source_episode_id: str) -> list[Triple]:
        """Parse LLM output into Triple objects.

        Expected format: '主語 | 関係 | 目的語' per line.
        Lines that fail to parse are skipped.
        """
        if not llm_output or "なし" in llm_output.strip():
            return []

        triples = []
        now = datetime.now()

        for line in llm_output.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                # Strip common markdown list markers
                line = line.lstrip("-#* ").strip()
                if not line:
                    continue

            parts = line.split("|")
            if len(parts) != 3:
                logger.debug("Triple parse skip (not 3 parts): %s", line[:80])
                continue

            subj = parts[0].strip()
            rel = parts[1].strip()
            obj = parts[2].strip()

            if not subj or not rel or not obj:
                logger.debug("Triple parse skip (empty field): %s", line[:80])
                continue

            triples.append(Triple(
                subject=subj,
                relation=rel,
                object=obj,
                source_episode_id=source_episode_id,
                created_at=now,
            ))

        return triples
