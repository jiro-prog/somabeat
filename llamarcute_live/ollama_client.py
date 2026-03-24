"""Ollama API client for llamarcute-live.

Based on SleepyJean's ollama_client.py, adapted for the integrated system.

Reference: llamarcute_live_design.md section 3.4, phase1_taskflow.md T10
"""

from __future__ import annotations

import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = 180


async def check_health() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def chat(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    timeout_sec: int = DEFAULT_TIMEOUT,
) -> tuple[str | None, float]:
    """Send a chat request to Ollama.

    Returns (response_text, duration_seconds).
    response_text is None on error.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 4096,
        },
    }

    start = time.monotonic()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Ollama API error %d: %s", resp.status, error_text)
                    return None, time.monotonic() - start

                data = await resp.json()
                content = data.get("message", {}).get("content", "").strip()
                thinking = data.get("message", {}).get("thinking", "")

                if thinking:
                    logger.debug("Think: %d chars", len(thinking))

                # Fallback: if content is empty but thinking exists
                if not content and thinking:
                    lines = [l.strip() for l in thinking.split("\n") if l.strip()]
                    if lines:
                        content = lines[-1]

                duration = time.monotonic() - start
                logger.info(
                    "Ollama: %d chars, %.1fs (think: %dc)",
                    len(content), duration, len(thinking),
                )
                return content, duration

    except aiohttp.ClientError as e:
        logger.error("Ollama connection error: %s", e)
        return None, time.monotonic() - start
    except TimeoutError:
        logger.error("Ollama timeout after %ds", timeout_sec)
        return None, time.monotonic() - start


async def chat_messages(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    timeout_sec: int = DEFAULT_TIMEOUT,
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> tuple[str | None, float]:
    """Send a multi-turn chat request to Ollama.

    Used for cuteness conversations and other multi-turn interactions.
    Returns (response_text, duration_seconds).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": max_tokens,
        },
    }

    start = time.monotonic()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Ollama API error %d: %s", resp.status, error_text)
                    return None, time.monotonic() - start

                data = await resp.json()
                content = data.get("message", {}).get("content", "").strip()
                thinking = data.get("message", {}).get("thinking", "")

                if not content and thinking:
                    lines = [l.strip() for l in thinking.split("\n") if l.strip()]
                    if lines:
                        content = lines[-1]

                duration = time.monotonic() - start
                return content, duration

    except aiohttp.ClientError as e:
        logger.error("Ollama connection error: %s", e)
        return None, time.monotonic() - start
    except TimeoutError:
        logger.error("Ollama timeout after %ds", timeout_sec)
        return None, time.monotonic() - start
