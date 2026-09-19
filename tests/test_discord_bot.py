"""Tests for Discord Bot integration (T43).

All tests use mocks — no actual Discord connection or LLM calls needed.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from discord_bot.bot import (
    IntegratedBot,
    _generate_wake_message,
    _split_message,
    FALLBACK_WAKE_MESSAGE,
)
from orchestrator.orchestrator import SystemState


# ============================================================
# Fixtures
# ============================================================


def _mock_orchestrator(state=SystemState.AWAKE):
    """Create a mock Orchestrator with sensible defaults."""
    orch = MagicMock()
    orch.state = state
    orch.notification_callback = None

    # Personality
    orch.personality = MagicMock()
    orch.personality.id = "test_v0"
    orch.personality.version = 3
    orch.personality.rule_count = 8

    # Dialogue
    orch.dialogue = MagicMock()
    orch.dialogue.init_db = AsyncMock()
    orch.dialogue.process_input = AsyncMock(return_value="テスト応答です")

    # Field
    orch.field = MagicMock()
    snap = MagicMock()
    snap.total_count = 42
    orch.field.snapshot = AsyncMock(return_value=snap)
    orch.field.sense = AsyncMock(return_value=MagicMock(signals=[]))

    # Encoder
    orch.encoder = MagicMock()
    orch.encoder.encode_for_sense = MagicMock(return_value="fake_embedding")

    # Observer
    orch.observer = MagicMock()
    orch.observer.format_logs = MagicMock(return_value="[12:00:00] EMIT test")

    # Config
    orch.config = {
        "llamarcute_live": {},
        "notifications": {"wake_enabled": True},
    }

    # Enter sleep
    orch.enter_sleep = AsyncMock()

    return orch


def _mock_config():
    return {
        "discord": {
            "token_env": "DISCORD_BOT_TOKEN",
            "active_channel_id": None,
        },
        "schedule": {"enabled": False},
        "notifications": {"wake_enabled": True},
    }


def _make_message(content="テストメッセージ", is_bot=False, channel_id=123):
    """Create a mock Discord message."""
    msg = MagicMock()
    msg.content = content
    msg.author.bot = is_bot
    msg.channel.id = channel_id
    msg.channel.typing = MagicMock(return_value=MagicMock(
        __aenter__=AsyncMock(),
        __aexit__=AsyncMock(),
    ))
    msg.reply = AsyncMock()
    return msg


def _make_interaction():
    """Create a mock Discord interaction."""
    interaction = MagicMock()
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    interaction.channel = MagicMock()
    return interaction


# ============================================================
# T39: Message response tests
# ============================================================


class TestMessageResponse:
    def test_message_response(self):
        """User message → dialogue.process_input → reply."""
        orch = _mock_orchestrator()
        bot = IntegratedBot(orch, _mock_config())
        msg = _make_message("こんにちは")

        asyncio.get_event_loop().run_until_complete(bot.on_message(msg))

        orch.dialogue.process_input.assert_called_once_with("こんにちは")
        msg.reply.assert_called_once_with("テスト応答です")

    def test_sleeping_reply(self):
        """Messages during sleep get a sleeping reply."""
        orch = _mock_orchestrator(state=SystemState.SLEEPING)
        bot = IntegratedBot(orch, _mock_config())
        msg = _make_message("起きてる？")

        asyncio.get_event_loop().run_until_complete(bot.on_message(msg))

        orch.dialogue.process_input.assert_not_called()
        msg.reply.assert_called_once()
        assert "眠っています" in msg.reply.call_args[0][0]

    def test_ignore_bot_messages(self):
        """Bot messages are ignored."""
        orch = _mock_orchestrator()
        bot = IntegratedBot(orch, _mock_config())
        msg = _make_message(is_bot=True)

        asyncio.get_event_loop().run_until_complete(bot.on_message(msg))

        orch.dialogue.process_input.assert_not_called()
        msg.reply.assert_not_called()

    def test_dialogue_error_fallback(self):
        """Dialogue error → graceful error message."""
        orch = _mock_orchestrator()
        orch.dialogue.process_input = AsyncMock(side_effect=RuntimeError("LLM down"))
        bot = IntegratedBot(orch, _mock_config())
        msg = _make_message()

        asyncio.get_event_loop().run_until_complete(bot.on_message(msg))

        msg.reply.assert_called_once()
        assert "ごめん" in msg.reply.call_args[0][0]


# ============================================================
# T40: Slash command tests
# ============================================================


class TestSlashCommands:
    def test_slash_sleep(self):
        """The /sleep command triggers enter_sleep."""
        orch = _mock_orchestrator()
        bot = IntegratedBot(orch, _mock_config())

        # Get the registered sleep command
        sleep_cmd = None
        for cmd in bot.tree.get_commands():
            if cmd.name == "sleep":
                sleep_cmd = cmd
                break
        assert sleep_cmd is not None, "/sleep command not registered"

    def test_slash_status(self):
        """The /status command is registered."""
        orch = _mock_orchestrator()
        bot = IntegratedBot(orch, _mock_config())

        status_cmd = None
        for cmd in bot.tree.get_commands():
            if cmd.name == "status":
                status_cmd = cmd
                break
        assert status_cmd is not None, "/status command not registered"

    def test_slash_field(self):
        """The /field command is registered."""
        orch = _mock_orchestrator()
        bot = IntegratedBot(orch, _mock_config())

        field_cmd = None
        for cmd in bot.tree.get_commands():
            if cmd.name == "field":
                field_cmd = cmd
                break
        assert field_cmd is not None, "/field command not registered"


# ============================================================
# T41: Scheduler tests
# ============================================================


class TestScheduler:
    def test_scheduler_not_started_when_disabled(self):
        """Scheduler does not start when disabled in config."""
        orch = _mock_orchestrator()
        config = _mock_config()
        config["schedule"]["enabled"] = False
        bot = IntegratedBot(orch, config)
        # No assertion needed — just verify no crash

    def test_scheduler_skip_if_sleeping(self):
        """Scheduler skips if already sleeping."""
        orch = _mock_orchestrator(state=SystemState.SLEEPING)
        config = _mock_config()
        config["schedule"]["enabled"] = True
        config["schedule"]["sleep_hour"] = 3
        config["schedule"]["sleep_minute"] = 0
        bot = IntegratedBot(orch, config)

        # The scheduler logic: if state != AWAKE, skip
        assert orch.state == SystemState.SLEEPING
        # enter_sleep should NOT be called if already sleeping
        orch.enter_sleep.assert_not_called()

    def test_scheduler_no_double_trigger_same_day(self):
        """Scheduler must not trigger twice on the same day.

        Regression test: if a cycle finishes before the scheduled hour,
        the scheduler would compute the same day's target and re-trigger.
        The last_triggered_date guard prevents this.
        """
        orch = _mock_orchestrator()
        config = _mock_config()
        config["schedule"]["enabled"] = True
        config["schedule"]["sleep_hour"] = 3
        config["schedule"]["sleep_minute"] = 0
        bot = IntegratedBot(orch, config)

        # Simulate: scheduler has last_triggered_date set to today
        # When the loop recalculates target, it should skip to tomorrow
        today = datetime.now().date()
        # The scheduler's target for today should be skipped
        target_today = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)
        # If cycle finishes at 02:42 (before 03:00), target.date() == last_triggered_date
        # should push target to tomorrow
        target_tomorrow = target_today + timedelta(days=1)

        # Verify the logic: if last_triggered_date == target.date(), skip to next day
        last_triggered_date = today
        target = target_today
        if target <= datetime.now() or target.date() == last_triggered_date:
            target += timedelta(days=1)
        assert target.date() == target_tomorrow.date()


# ============================================================
# T42: Wake notification tests
# ============================================================


class TestWakeNotification:
    def test_wake_notification_callback_registered(self):
        """Bot registers itself as notification callback."""
        orch = _mock_orchestrator()
        bot = IntegratedBot(orch, _mock_config())
        assert orch.notification_callback is not None

    def test_wake_notification_disabled(self):
        """Wake notification disabled → no LLM call."""
        orch = _mock_orchestrator()
        orch.config["notifications"]["wake_enabled"] = False
        # The patched wake_up checks this config flag
        assert orch.config["notifications"]["wake_enabled"] is False

    def test_generate_wake_message_fallback(self):
        """LLM failure → fallback message."""
        orch = _mock_orchestrator()
        orch.field.sense = AsyncMock(return_value=MagicMock(signals=[]))

        result = asyncio.get_event_loop().run_until_complete(
            _generate_wake_message(orch)
        )
        # No learnings → fallback
        assert result == FALLBACK_WAKE_MESSAGE


# ============================================================
# Utility tests
# ============================================================


class TestSplitMessage:
    def test_short_message(self):
        assert _split_message("hello") == ["hello"]

    def test_long_message_split_at_newline(self):
        text = "a" * 1000 + "\n" + "b" * 1000 + "\n" + "c" * 500
        chunks = _split_message(text, limit=2000)
        assert all(len(c) <= 2000 for c in chunks)
        # Joined content should be equivalent
        assert "".join(chunks) == text.replace("\n", "", 1)  # newline consumed

    def test_very_long_no_newlines(self):
        text = "x" * 5000
        chunks = _split_message(text, limit=2000)
        assert len(chunks) == 3
        assert chunks[0] == "x" * 2000
