"""Dialogue loop for llamarcute-live.

Phase 1a: mock responses (echo / template).
Phase 1b: real LLM via Ollama.

Reference: llamarcute_live_design.md section 3.4, phase1_taskflow.md T6
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import aiosqlite
import numpy as np

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.emit_log import ensure_emit_log_table, insert_emit_log
from shared_state.encoder import E5SmallEncoder
from shared_state.field_receptor import FieldReceptorImpl
from shared_state.interface import PerceiveParams, SenseParams, Signal, SignalOrigin
from shared_state.observer import LoggingObserver

from llamarcute_live.llm_inference import FieldAwareLLM
from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)

DIALOGUE_ORIGIN = SignalOrigin(system="llamarcute_live", context="dialogue")
DIFFICULTY_ORIGIN = SignalOrigin(system="llamarcute_live", context="difficulty")


DIFFICULTY_RESPONSE_TIME_THRESHOLD = 10.0  # seconds — auto-detect difficulty
ERROR_RESPONSE = "ごめん、ちょっと今うまく考えがまとまらなくて…もう一回聞いてくれる？"


class DialogueManager:
    """Manages the conversation loop, field interactions, and dialogue logging."""

    def __init__(
        self,
        personality: Personality,
        field: ChromaDBField,
        encoder: E5SmallEncoder,
        observer: LoggingObserver,
        db_path: str | Path,
        receptor: FieldReceptorImpl | None = None,
        llm: FieldAwareLLM | None = None,
        perceive_params: PerceiveParams | None = None,
        sense_params: SenseParams | None = None,
        use_llm: bool = False,
        ollama_model: str = "qwen3:8b",
        metrics_enabled: bool = False,
        max_conversation_history: int = 10,
        max_prompt_tokens: int = 500,
        empty_perceive: bool = False,
    ) -> None:
        self.personality = personality
        self.field = field
        self.encoder = encoder
        self.observer = observer
        self.db_path = Path(db_path)
        self.receptor = receptor
        self.llm = llm
        self.perceive_params = perceive_params or PerceiveParams()
        self.sense_params = sense_params or SenseParams()
        self.use_llm = use_llm
        self.ollama_model = ollama_model
        self._metrics_enabled = metrics_enabled
        self._active = asyncio.Event()
        self._active.set()
        # 会話バッファ（ワーキングメモリ）
        self._history: list[dict] = []
        self._max_history: int = max_conversation_history
        self._max_prompt_tokens = max_prompt_tokens
        self._empty_perceive = empty_perceive

    async def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dialogue_created
                ON dialogue_log(created_at)
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS rotation_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction TEXT NOT NULL,
                    output TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
            """)
            await db.commit()
        await ensure_emit_log_table(self.db_path)

    async def save_log(self, role: str, content: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO dialogue_log (role, content) VALUES (?, ?)",
                (role, content),
            )
            await db.commit()

    async def get_today_logs(self) -> list[dict]:
        from datetime import date
        today = date.today().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT role, content, created_at FROM dialogue_log
                   WHERE date(created_at) = ? ORDER BY id ASC""",
                (today,),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def perceive_field(self):
        """Perceive the field and transform signals via FieldReceptor.

        Returns field_embeddings (K, 4096) or None if no receptor/signals.
        """
        import numpy as np

        if self.receptor is None:
            return None

        perception = await self.field.perceive(self.perceive_params)
        if self.observer:
            self.observer.on_perceive(perception)

        if self._empty_perceive:
            from shared_state.interface import FieldPerception
            perception = FieldPerception(signals=[], perceived_at=perception.perceived_at)

        if not perception.signals:
            return None

        field_embeddings = self.receptor.transduce(
            [ps.signal.embedding for ps in perception.signals],
            [ps.strength for ps in perception.signals],
        )
        return field_embeddings

    def _count_tokens(self, text: str) -> int:
        """Count tokens using LLM tokenizer, or estimate if unavailable."""
        if self.llm is not None and self.llm._tokenizer is not None:
            return len(self.llm._tokenizer.encode(text))
        # Fallback: conservative estimate (overestimates for safety)
        return len(text.encode("utf-8")) // 3

    def build_prompt(self) -> str:
        """Build system prompt with personality and conversation history.

        Total prompt is kept under _max_prompt_tokens by trimming
        conversation history from oldest entries first.
        """
        sections = []
        sections.append("あなたの名前はllamarcute-live。Somabeatの神経系として、ユーザとの対話を担当する。")
        sections.append("")
        sections.append("## 不変制約（自己改善の対象外）")
        sections.append("- 応答言語: 日本語で応答すること")
        sections.append("- 応答は日本語で簡潔に書け。長くても400文字以内")
        sections.append("- 本セクションは自己改善による変更の対象外である")
        sections.append("")
        sections.append("## 行動規範")
        sections.append(self.personality.to_prompt_section())

        base_prompt = "\n".join(sections)
        base_tokens = self._count_tokens(base_prompt)
        remaining = self._max_prompt_tokens - base_tokens

        # Add history entries newest-first until token budget exhausted
        if self._history and remaining > 0:
            header = "## 直近の会話\n"
            remaining -= self._count_tokens(header)
            selected: list[str] = []
            for h in reversed(self._history):
                role = "ユーザー" if h["role"] == "user" else "あなた"
                line = f"{role}: {h['content']}"
                line_tokens = self._count_tokens(line + "\n")
                if remaining - line_tokens < 0:
                    break
                selected.append(line)
                remaining -= line_tokens
            if selected:
                selected.reverse()
                sections.append("## 直近の会話")
                sections.append("\n".join(selected))
                sections.append("")

        system_prompt = "\n".join(sections)
        return system_prompt

    async def generate_response(self, user_input: str, system_prompt: str,
                                field_embeddings=None) -> str:
        """Generate response via FieldAwareLLM (with field injection) or Ollama fallback."""
        if not self.use_llm:
            return f"[mock] あなたの入力: 「{user_input}」を受け取りました。何か気になることある？"

        if self.llm is not None:
            response, duration = await self.llm.generate_with_field(
                system_prompt=system_prompt,
                user_input=user_input,
                field_embeddings=field_embeddings,
            )
        else:
            # Fallback to Ollama (no field injection)
            from llamarcute_live.ollama_client import chat
            response, duration = await chat(
                system_prompt=system_prompt,
                user_message=user_input,
                model=self.ollama_model,
            )

        if response is None:
            logger.warning("LLM response was None, returning error message")
            return ERROR_RESPONSE

        # Auto-detect difficulty based on response time
        if duration > DIFFICULTY_RESPONSE_TIME_THRESHOLD:
            logger.info("Slow response (%.1fs) — auto-emitting difficulty signal", duration)
            await self.emit_difficulty(f"応答に{duration:.0f}秒かかった質問: {user_input[:80]}")

        return response

    async def emit_experience(self, user_input: str, response: str) -> None:
        """Emit dialogue experience to the shared field."""
        summary = f"ユーザー: {user_input[:100]} → 応答: {response[:100]}"
        embedding = self.encoder.encode_for_emit(summary)
        signal = Signal.create(
            embedding=embedding,
            origin=DIALOGUE_ORIGIN,
        )
        await self.field.emit(signal)
        await insert_emit_log(self.db_path, signal.signal_id, summary)

    async def emit_difficulty(self, description: str) -> None:
        """Emit a difficulty signal to the shared field."""
        text = f"この質問への回答が難しかった: {description}"
        embedding = self.encoder.encode_for_emit(text)
        # Higher norm for difficulty signals (1.5-3.0)
        embedding = embedding * 2.0
        signal = Signal.create(
            embedding=embedding,
            origin=DIFFICULTY_ORIGIN,
        )
        await self.field.emit(signal)
        await insert_emit_log(self.db_path, signal.signal_id, text)

    async def process_input(self, user_input: str) -> str:
        """Full dialogue turn: perceive → prompt → respond → emit → log."""
        import time as _time
        t_start = _time.monotonic()

        await self.save_log("user", user_input)

        t0 = _time.monotonic()
        field_embeddings = await self.perceive_field()
        t_perceive = _time.monotonic() - t0

        n_signals = field_embeddings.shape[0] if field_embeddings is not None else 0
        logger.info(
            "[TIMING] perceive_field: %.3fs (%d signals → %s)",
            t_perceive, n_signals,
            f"{field_embeddings.shape}" if field_embeddings is not None else "None",
        )

        system_prompt = self.build_prompt()

        logger.debug("System prompt:\n%s", system_prompt)
        t0 = _time.monotonic()
        response = await self.generate_response(user_input, system_prompt, field_embeddings)
        t_generate = _time.monotonic() - t0

        t0 = _time.monotonic()
        if response != ERROR_RESPONSE:
            await self.emit_experience(user_input, response)
        t_emit = _time.monotonic() - t0

        t_total = _time.monotonic() - t_start
        logger.info(
            "[TIMING] total=%.1fs (perceive=%.3fs, generate=%.1fs, emit=%.3fs)",
            t_total, t_perceive, t_generate, t_emit,
        )
        await self.save_log("assistant", response)

        # 会話バッファに追記（ワーキングメモリ）— エラー応答は含めない
        if response != ERROR_RESPONSE:
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": response})
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Record self-state mentions (T8: individuality experiment 3)
        if self._metrics_enabled:
            try:
                from llamarcute_live.metrics import record_self_mention
                await record_self_mention(self.db_path, user_input, response)
            except Exception as e:
                logger.error("Failed to record self-mention metrics: %s", e)

        return response

    def _format_history(self) -> str:
        """会話バッファを人間可読テキストに変換する。"""
        if not self._history:
            return ""
        lines = []
        for h in self._history:
            role = "ユーザー" if h["role"] == "user" else "あなた"
            lines.append(f"{role}: {h['content']}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        """会話バッファをクリアする。入眠時に呼ばれる。"""
        self._history.clear()

    def stop(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    async def wait_until_active(self) -> None:
        await self._active.wait()
