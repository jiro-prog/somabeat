"""CLI entry point for llamarcute-live.

Reference: llamarcute_live_design.md section 11.1, phase1_taskflow.md T6
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.encoder import E5SmallEncoder
from shared_state.interface import SenseParams
from shared_state.observer import LoggingObserver

from llamarcute_live.dialogue import DialogueManager
from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/system.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class LlamarcuteLiveCLI:
    """CLI interface wrapping DialogueManager + commands."""

    def __init__(
        self,
        dialogue: DialogueManager,
        observer: LoggingObserver,
        sleep_callback=None,
    ) -> None:
        self.dialogue = dialogue
        self.observer = observer
        self.sleep_callback = sleep_callback
        self._running = True

    async def run(self) -> None:
        await self.dialogue.init_db()
        print("[AWAKE] System started. Type your message or a command.")

        while self._running:
            await self.dialogue.wait_until_active()

            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("[AWAKE] > ")
                )
            except (EOFError, KeyboardInterrupt):
                break

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                await self.handle_command(line)
            else:
                response = await self.dialogue.process_input(line)
                print(f"[AWAKE] llamarcute-live: {response}")

    async def handle_command(self, cmd: str) -> None:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/quit":
            print("Shutting down...")
            self._running = False

        elif command == "/sleep":
            if self.sleep_callback:
                await self.sleep_callback()
            else:
                print("[INFO] Sleep callback not configured.")

        elif command == "/status":
            snap = await self.dialogue.field.snapshot()
            active = "AWAKE" if self.dialogue.is_active else "SLEEPING"
            print(f"[STATUS] State: {active}")
            print(f"[STATUS] Personality: {self.dialogue.personality.id} v{self.dialogue.personality.version}")
            print(f"[STATUS] Field signals: {snap.total_count}")

        elif command == "/field":
            print(self.observer.format_logs())

        elif command == "/difficulty":
            if arg:
                await self.dialogue.emit_difficulty(arg)
                print(f"[INFO] Difficulty signal emitted: {arg}")
            else:
                print("[INFO] Usage: /difficulty <description>")

        else:
            print(f"[INFO] Unknown command: {command}")
            print("[INFO] Available: /sleep /status /field /difficulty /quit")


def create_components(config: dict) -> tuple:
    """Create and wire all components from config."""
    observer = LoggingObserver()

    field = ChromaDBField(
        persist_directory=config["shared_state"]["chromadb"]["persist_directory"],
        collection_name=config["shared_state"]["chromadb"]["collection_name"],
        observer=observer,
    )

    encoder = E5SmallEncoder(config["encoder"]["model_name"])

    personality = Personality.load(config["llamarcute_live"]["personality_path"])

    sense_cfg = config["llamarcute_live"]["sense"]
    from datetime import timedelta
    sense_params = SenseParams(
        max_signals=sense_cfg["max_signals"],
        min_relevance=sense_cfg["min_relevance"],
        time_horizon=timedelta(hours=sense_cfg["time_horizon_hours"]),
    )

    dialogue = DialogueManager(
        personality=personality,
        field=field,
        encoder=encoder,
        observer=observer,
        db_path=config["llamarcute_live"]["dialogue_db_path"],
        sense_params=sense_params,
    )

    return dialogue, observer, field, encoder


def main() -> None:
    setup_logging()
    config = load_config()
    dialogue, observer, field, encoder = create_components(config)
    cli = LlamarcuteLiveCLI(dialogue, observer)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
