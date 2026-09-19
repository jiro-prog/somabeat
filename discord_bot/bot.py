"""Discord Bot entry point for the integrated system.

Replaces the CLI as the primary interface. llamarcute-live's dialogue
logic (dialogue.py) is unchanged — only the I/O channel switches
from stdin/stdout to Discord messages.

Design principle 1.3 (self-identity): A single personality faces the
outside world through a single bot.
Design principle 1.2 (indirect coordination): SleepyJean never writes
to Discord directly. All external communication goes through
llamarcute-live.

Reference: operational_deployment_taskflow.md T39-T42
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.orchestrator import Orchestrator, SystemState, load_config

logger = logging.getLogger(__name__)

FALLBACK_WAKE_MESSAGE = "おはようございます。今日もよろしくお願いします。"


class IntegratedBot(commands.Bot):
    """Discord Bot wrapping the integrated system orchestrator.

    Provides:
    - Message-based dialogue (on_message → dialogue.process_input)
    - Slash commands (/sleep, /status, /field)
    - Sleep scheduler (auto-sleep at configured time)
    - Wake notification (morning message after sleep cycle)
    """

    def __init__(self, orchestrator: Orchestrator, config: dict) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

        self.orchestrator = orchestrator
        self.config = config
        self.discord_cfg = config.get("discord", {})
        self.schedule_cfg = config.get("schedule", {})
        self.notification_cfg = config.get("notifications", {})

        self._active_channel_id: int | None = self.discord_cfg.get("active_channel_id")
        self._sleep_lock = asyncio.Lock()
        self._last_sleep_date = None  # Track last sleep cycle date (shared by scheduler + /sleep)

        # Register the orchestrator's wake notification callback
        self.orchestrator.notification_callback = self._send_wake_notification

        # Register slash commands
        self._register_commands()

    def _register_commands(self) -> None:
        """Register Discord slash commands."""

        @self.tree.command(name="sleep", description="睡眠サイクルを開始")
        async def sleep_command(interaction: discord.Interaction):
            if self.orchestrator.state == SystemState.SLEEPING:
                await interaction.response.send_message("既に眠っています 💤")
                return
            await interaction.response.send_message(
                "おやすみなさい… 💤 睡眠サイクルを開始します"
            )
            asyncio.create_task(self._run_sleep_cycle(interaction.channel))

        @self.tree.command(name="status", description="システムの状態を表示")
        async def status_command(interaction: discord.Interaction):
            state = self.orchestrator.state.value
            p = self.orchestrator.personality
            snap = await self.orchestrator.field.snapshot()

            # VRAM diagnostic
            import torch
            vram_lines = []
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                vram_lines.append(f"**VRAM:** alloc={alloc:.0f}MiB, reserved={reserved:.0f}MiB")

            # Check device of key models
            devices = []
            llm = self.orchestrator.dialogue.llm
            if llm and llm.is_loaded():
                dev = next(llm._model.parameters()).device
                devices.append(f"LLM: {dev}")
            receptor = self.orchestrator.dialogue.receptor
            if receptor:
                dev = next(receptor.parameters()).device
                devices.append(f"FieldReceptor: {dev}")
            enc = getattr(self.orchestrator, '_multimodal_encoder', None)
            if enc is None:
                enc = getattr(self.orchestrator.dialogue.encoder, '_projection_text', None)
            if devices:
                vram_lines.append(f"**Devices:** {', '.join(devices)}")

            await interaction.response.send_message(
                f"**状態:** {state}\n"
                f"**人格:** {p.id} v{p.version} ({p.rule_count} rules)\n"
                f"**場の信号数:** {snap.total_count}\n"
                + "\n".join(vram_lines)
            )

        @self.tree.command(name="field", description="場の直近ログを表示")
        async def field_command(interaction: discord.Interaction):
            logs = self.orchestrator.observer.format_logs(10)
            # Discord message limit is 2000 chars
            if len(logs) > 1900:
                logs = logs[:1900] + "\n..."
            await interaction.response.send_message(f"```\n{logs}\n```")

    async def setup_hook(self) -> None:
        """Called when the bot is ready. Sync slash commands."""
        await self.tree.sync()
        logger.info("Slash commands synced")

    async def on_ready(self) -> None:
        logger.info("Bot ready: %s (id=%s)", self.user, self.user.id)
        print(f"[DISCORD] Bot ready: {self.user}")

        # Initialize dialogue DB
        await self.orchestrator.dialogue.init_db()

        # Initialize metrics tables if enabled (T8)
        metrics_cfg = self.orchestrator.config.get("metrics", {})
        if metrics_cfg.get("enabled", False):
            from llamarcute_live.metrics import init_metrics_db
            db_path = self.orchestrator.config["llamarcute_live"].get(
                "dialogue_db_path", "data/llamarcute_live.db"
            )
            await init_metrics_db(db_path)

        # Start sleep scheduler if enabled
        if self.schedule_cfg.get("enabled", False):
            asyncio.create_task(self._sleep_scheduler())
            logger.info("[SCHEDULER] Started")

    async def on_message(self, message: discord.Message) -> None:
        # Ignore bot messages
        if message.author.bot:
            return

        # Check channel restriction
        if (
            self._active_channel_id
            and message.channel.id != self._active_channel_id
        ):
            return

        # Check if sleeping
        if self.orchestrator.state != SystemState.AWAKE:
            await message.reply("💤 今は眠っています…")
            return

        # Process through llamarcute-live dialogue
        async with message.channel.typing():
            try:
                response = await self.orchestrator.dialogue.process_input(
                    message.content
                )
            except Exception as e:
                logger.error("Dialogue error: %s", e)
                response = "ごめん、ちょっと今うまく考えがまとまらなくて…もう一回聞いてくれる？"

        # Split long responses for Discord's 2000 char limit
        for chunk in _split_message(response):
            await message.reply(chunk)

    # -----------------------------------------------------------------
    # Sleep cycle with Discord feedback
    # -----------------------------------------------------------------

    async def _run_sleep_cycle(self, channel: discord.abc.Messageable) -> None:
        """Run sleep cycle with progress messages to Discord.

        Uses a lock to prevent concurrent sleep cycles from /sleep command
        and scheduler racing each other.
        """
        if self._sleep_lock.locked():
            logger.info("Sleep cycle already running, skipping")
            return
        async with self._sleep_lock:
            self._last_sleep_date = datetime.now().date()
            try:
                await self.orchestrator.enter_sleep()
            except Exception as e:
                logger.error("Sleep cycle error: %s", e)
                try:
                    await channel.send(f"⚠️ 睡眠サイクルでエラーが発生しました: {e}")
                except Exception:
                    pass

    # -----------------------------------------------------------------
    # T41: Sleep scheduler
    # -----------------------------------------------------------------

    async def _sleep_scheduler(self) -> None:
        """Auto-trigger sleep at the configured time."""
        sleep_hour = self.schedule_cfg.get("sleep_hour", 3)
        sleep_minute = self.schedule_cfg.get("sleep_minute", 0)

        while True:
            now = datetime.now()
            target = now.replace(
                hour=sleep_hour, minute=sleep_minute, second=0, microsecond=0
            )
            # Skip to tomorrow if target has passed OR already ran today (including manual /sleep)
            if target <= now or target.date() == self._last_sleep_date:
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            logger.info(
                "[SCHEDULER] Next sleep at %s (%.0fs from now)",
                target.strftime("%Y-%m-%d %H:%M"),
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)

            if self.orchestrator.state == SystemState.AWAKE:
                logger.info("[SCHEDULER] Triggering scheduled sleep cycle")
                channel = self._get_notification_channel()
                if channel:
                    await channel.send("💤 定時の睡眠サイクルを開始します…")
                    await self._run_sleep_cycle(channel)
                else:
                    # No channel available, run directly with lock
                    async def _null_send(*a, **kw):
                        pass
                    _NullChannel = type("_NullChannel", (), {"send": _null_send})
                    await self._run_sleep_cycle(_NullChannel())
            else:
                logger.info("[SCHEDULER] Already sleeping, skipping")

    # -----------------------------------------------------------------
    # T42: Wake notification
    # -----------------------------------------------------------------

    async def _send_wake_notification(self, message: str) -> None:
        """Callback invoked by orchestrator after wake-up."""
        channel = self._get_notification_channel()
        if channel:
            await channel.send(f"☀️ {message}")
            logger.info("[WAKE] Notification sent to channel %s", channel.id)

    def _get_notification_channel(self) -> discord.TextChannel | None:
        """Get the channel for notifications."""
        channel_id = (
            self.notification_cfg.get("wake_channel_id")
            or self._active_channel_id
        )
        if channel_id:
            return self.get_channel(channel_id)
        return None


# -----------------------------------------------------------------
# Orchestrator extension: wake message generation + notification
# -----------------------------------------------------------------


async def _generate_wake_message(
    orchestrator: Orchestrator,
    recon_result: dict | None = None,
    dialogue_logs: list[dict] | None = None,
) -> str:
    """Generate a wake-up message using generate_bare, informed by Reconsolidation results.

    Uses recon_result (cluster stats) and dialogue_logs (recent topics) directly.
    """
    try:
        # Build context from Reconsolidation results and dialogue topics
        context_parts = []

        if recon_result:
            ep = recon_result.get("episode_count", 0)
            cl = recon_result.get("new_cluster_count", 0)
            mg = recon_result.get("merge_count", 0)
            fg = recon_result.get("forget_count", 0)
            context_parts.append(
                f"記憶整理: {ep}件の記憶を処理し、{cl}個の新しいまとまり、"
                f"{mg}個の統合、{fg}件の忘却"
            )

        if dialogue_logs:
            # Extract recent user topics (last 5 user messages, truncated)
            user_msgs = [
                log["content"][:80]
                for log in dialogue_logs
                if log.get("role") == "user" and log.get("content")
            ][-5:]
            if user_msgs:
                context_parts.append(
                    "昨日の話題:\n" + "\n".join(f"- {m}" for m in user_msgs)
                )

        if not context_parts:
            return FALLBACK_WAKE_MESSAGE

        prompt = (
            "あなたは今目覚めたところです。"
            "昨夜の睡眠中の記憶整理と、昨日の対話を踏まえて、"
            "短いおはようメッセージを書いてください。"
            "2〜3文で、自然な日本語で。\n\n"
            + "\n".join(context_parts)
        )

        if orchestrator.llm is None:
            return FALLBACK_WAKE_MESSAGE

        response, _ = await orchestrator.llm.generate_bare(
            system_prompt="",
            user_input=prompt,
            max_new_tokens=200,
        )

        if response and response.strip():
            return response.strip()

    except Exception as e:
        logger.error("[WAKE] Failed to generate wake message: %s", e)

    return FALLBACK_WAKE_MESSAGE


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a message into chunks that fit Discord's character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        idx = text.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


# -----------------------------------------------------------------
# Patch orchestrator's wake_up to generate + send notification
# -----------------------------------------------------------------


_original_wake_up = Orchestrator.wake_up


async def _patched_wake_up(self: Orchestrator) -> None:
    """Extended wake_up: send pre-generated wake message via notification callback.

    The wake message is pre-generated by generate_bare during the sleep
    sequence and stashed in _pending_wake_message.
    """
    self.state = SystemState.AWAKE
    self.dialogue.resume()
    print("[AWAKE] Good morning. System is awake.")
    logger.info("=== SYSTEM AWAKE ===")

    # Send wake notification (message was pre-generated before LLM reload)
    message = getattr(self, "_pending_wake_message", None)
    if message and hasattr(self, "notification_callback") and self.notification_callback:
        try:
            await self.notification_callback(message)
        except Exception as e:
            logger.error("[WAKE] Notification failed: %s", e)
    self._pending_wake_message = None


Orchestrator.wake_up = _patched_wake_up
Orchestrator.notification_callback = None


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------


def main() -> None:
    log_fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/bot.log", encoding="utf-8"),
        ],
    )

    config = load_config()

    # Ensure data directories exist
    Path(config["shared_state"]["chromadb"]["persist_directory"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["llamarcute_live"]["dialogue_db_path"]).parent.mkdir(
        parents=True, exist_ok=True
    )

    orchestrator = Orchestrator(config)
    bot = IntegratedBot(orchestrator, config)

    # Get token from environment
    token_env = config.get("discord", {}).get("token_env", "DISCORD_BOT_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        # Try .env file
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        if key.strip() == token_env:
                            token = val.strip().strip('"').strip("'")
                            break

    if not token:
        print(f"ERROR: Discord bot token not found. Set {token_env} in .env or environment.")
        sys.exit(1)

    bot.run(token)


if __name__ == "__main__":
    main()
