"""Orchestrator — manages system-wide sleep/wake state transitions.

Analogous to the ventrolateral preoptic area (VLPO) in the hypothalamus.
Not a cognitive system — it is infrastructure for state management.

Reference: llamarcute_live_design.md section 4, phase1_taskflow.md T8
"""

from __future__ import annotations

import asyncio
import enum
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
import yaml

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.encoder import E5SmallEncoder
from shared_state.field_receptor import FieldReceptorImpl
from shared_state.interface import FieldEncoder, PerceiveParams, PurgeCriteria, SenseParams
from shared_state.observer import FieldSnapshotLogger, LoggingObserver

from llamarcute_live.llm_inference import FieldAwareLLM

from llamarcute_live.dialogue import DialogueManager
from llamarcute_live.main import LlamarcuteLiveCLI
from llamarcute_live.personality import Personality

from bridge.sleep_ingest import fetch_dialogue_logs, run_reconsolidation, save_ingest_watermark

from llamarcute_live.immune import (
    AnomalyThresholds,
    AnomalySeverity,
    check_health,
    execute_repair,
    save_personality_backup,
)
from llamarcute_live.cycle_history import append_cycle, load_history

from llamarcute_live.self_improve import determine_max_changes, generate_mutation_candidates
from llamarcute_live.fitness import evaluate_fitness
from llamarcute_live.cuteness import evaluate_cuteness
from llamarcute_live.selection import select_and_update

logger = logging.getLogger(__name__)


def _create_encoder(config: dict) -> FieldEncoder:
    """Create FieldEncoder from config. Falls back to E5SmallEncoder on failure."""
    encoder_cfg = config.get("encoder", {})
    encoder_type = encoder_cfg.get("type", "e5_small")

    if encoder_type == "multimodal":
        try:
            from shared_state.multimodal_encoder import MultimodalFieldEncoder
            return MultimodalFieldEncoder(
                e5_model_name=encoder_cfg.get("model_name", "intfloat/multilingual-e5-small"),
                siglip_model_name=encoder_cfg.get("siglip_model_name", "google/siglip2-base-patch16-256"),
                projection_text_path=encoder_cfg.get("projection_text_path", "data/vision_phase_b/alpha_0.5/projection_text.pt"),
                projection_img_path=encoder_cfg.get("projection_img_path", "data/vision_phase_b/alpha_0.5/projection_img.pt"),
                device=encoder_cfg.get("device", "cpu"),
            )
        except Exception:
            logger.warning("Failed to initialize MultimodalFieldEncoder, falling back to E5SmallEncoder", exc_info=True)

    return E5SmallEncoder(encoder_cfg.get("model_name", "intfloat/multilingual-e5-small"))


class SystemState(enum.Enum):
    AWAKE = "awake"
    SLEEPING = "sleeping"


class Orchestrator:
    """Manages the integrated system lifecycle: awake ↔ sleeping."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.state = SystemState.AWAKE

        # Shared components
        db_path = config["llamarcute_live"]["dialogue_db_path"]
        self.observer = FieldSnapshotLogger(db_path)
        self.field = ChromaDBField(
            persist_directory=config["shared_state"]["chromadb"]["persist_directory"],
            collection_name=config["shared_state"]["chromadb"]["collection_name"],
            observer=self.observer,
        )
        self.encoder: FieldEncoder = _create_encoder(config)

        # Personality
        self.personality = Personality.load(config["llamarcute_live"]["personality_path"])

        # Sense params
        sense_cfg = config["llamarcute_live"]["sense"]
        self.sense_params = SenseParams(
            max_signals=sense_cfg["max_signals"],
            min_relevance=sense_cfg["min_relevance"],
            time_horizon=timedelta(hours=sense_cfg["time_horizon_hours"]),
        )

        # Dialogue + LLM config
        ll_cfg = config["llamarcute_live"]
        metrics_cfg = config.get("metrics", {})

        # Perceive params
        perceive_cfg = ll_cfg.get("perceive", {})
        self.perceive_params = PerceiveParams(
            min_strength=perceive_cfg.get("min_strength", 1.0),
            max_signals=perceive_cfg.get("max_signals", 30),
            time_horizon=timedelta(hours=perceive_cfg["time_horizon_hours"])
            if "time_horizon_hours" in perceive_cfg else None,
            strength_exponent=perceive_cfg.get("strength_exponent", 0.5),
        )

        # FieldReceptor (load trained weights if available)
        receptor_path = config.get("field_receptor", {}).get(
            "weights_path", "data/fieldreceptor/field_receptor.pt",
        )
        self.receptor: FieldReceptorImpl | None = None
        if Path(receptor_path).exists():
            self.receptor = FieldReceptorImpl.load(receptor_path)
            logger.info("FieldReceptor loaded from %s", receptor_path)

        # FieldAwareLLM (lazy load — will be loaded on first use)
        self.llm: FieldAwareLLM | None = None
        if ll_cfg.get("use_llm", False):
            llm_cfg = ll_cfg.get("llm", {})
            quantization = llm_cfg.get("quantization", "gptq_marlin")
            debug_cfg = config.get("debug", {})
            disable_tq = debug_cfg.get("disable_turboquant", False)
            kv_bits_default = llm_cfg.get("kv_cache_bits", 0)
            kv_bits = 0 if disable_tq else kv_bits_default
            self.llm = FieldAwareLLM(
                model_name=llm_cfg.get("transformers_model", "Qwen/Qwen3-8B"),
                kv_cache_bits=kv_bits,
                quantization=quantization,
                gptq_model=llm_cfg.get("gptq_model", "AlphaGaO/Qwen3-8B-GPTQ"),
            )

        # SleepyJean (hippocampal module — LLM used only during sleep for triple extraction)
        self.sleepyjean = None
        try:
            from sleepyjean.sleepyjean import SleepyJean
            self.sleepyjean = SleepyJean(config, self.field, self.encoder, llm=self.llm)
            logger.info("SleepyJean initialized (KG-based, LLM=%s)", "yes" if self.llm else "no")
        except Exception as e:
            logger.warning("SleepyJean init failed (non-fatal): %s", e)

        # Dialogue manager
        self.dialogue = DialogueManager(
            personality=self.personality,
            field=self.field,
            encoder=self.encoder,
            observer=self.observer,
            db_path=ll_cfg["dialogue_db_path"],
            receptor=self.receptor,
            llm=self.llm,
            perceive_params=self.perceive_params,
            sense_params=self.sense_params,
            use_llm=ll_cfg.get("use_llm", False),
            ollama_model=ll_cfg.get("ollama_model", ""),
            metrics_enabled=metrics_cfg.get("enabled", False),
            max_conversation_history=ll_cfg.get("max_conversation_history", 10),
            empty_perceive=config.get("debug", {}).get("empty_perceive", False),
            sleepyjean=self.sleepyjean,
        )

        # Sensory: Vision (optional, default disabled)
        self._sensory_vision = None
        sensory_cfg = config.get("sensory", {}).get("vision", {})
        if sensory_cfg.get("enabled", False):
            from sensory.vision import SensoryVision
            self._sensory_vision = SensoryVision(
                encoder=self.encoder,
                field=self.field,
                thalamus_threshold=sensory_cfg.get("thalamus_threshold", 0.1),
            )
            logger.info("SensoryVision enabled (threshold=%.2f)", sensory_cfg.get("thalamus_threshold", 0.1))

        # CLI
        self.cli = LlamarcuteLiveCLI(
            dialogue=self.dialogue,
            observer=self.observer,
            sleep_callback=self.enter_sleep,
        )

    async def run(self) -> None:
        """Start the system in AWAKE state."""
        self.state = SystemState.AWAKE
        # Initialize metrics tables if enabled
        metrics_cfg = self.config.get("metrics", {})
        if metrics_cfg.get("enabled", False):
            from llamarcute_live.metrics import init_metrics_db
            db_path = self.config["llamarcute_live"].get("dialogue_db_path", "data/llamarcute_live.db")
            await init_metrics_db(db_path)
        await self.cli.run()

    async def enter_sleep(self) -> None:
        """Execute the full sleep sequence."""
        logger.info("=== ENTERING SLEEP SEQUENCE ===")
        print("[SLEEPING] Entering sleep sequence...")

        # 1. Stop dialogue, clear conversation buffer, reset sensory modules
        self.dialogue.stop()
        self.dialogue.clear_history()
        if self._sensory_vision is not None:
            self._sensory_vision.clear()
        self.state = SystemState.SLEEPING

        # 2. Glymphatic cleanup
        try:
            from orchestrator.glymphatic import glymphatic_cleanup
            gly_result = await glymphatic_cleanup(
                self.config["llamarcute_live"]["dialogue_db_path"],
            )
            if any(v > 0 for v in gly_result.values()):
                print(f"[SLEEPING] Glymphatic: {gly_result}")
        except Exception as e:
            logger.warning("Glymphatic cleanup failed (non-fatal): %s", e)

        # 3. Fetch dialogue logs for Reconsolidation (watermark-based dedup)
        dialogue_logs = []
        ingest_max_id = 0
        sj_db_path = self.config.get("sleepyjean", {}).get(
            "memory", {},
        ).get("sqlite_path", "data/sleepyjean_memory.db")
        try:
            dialogue_logs, ingest_max_id = await fetch_dialogue_logs(
                self.config["llamarcute_live"]["dialogue_db_path"],
                sleepyjean_db_path=sj_db_path,
            )
            print(f"[SLEEPING] Fetched {len(dialogue_logs)} new dialogue entries")
        except Exception as e:
            logger.error("fetch_dialogue_logs failed: %s", e)
            print(f"[SLEEPING] fetch_dialogue_logs: ERROR - {e}")

        # 4. SleepyJean Reconsolidation (non-LLM, no VRAM needed)
        recon_result = None
        if self.sleepyjean is not None:
            try:
                recon_result = await run_reconsolidation(
                    self.sleepyjean, dialogue_logs,
                )
                # Save watermark only after successful consolidation
                if ingest_max_id > 0:
                    await save_ingest_watermark(sj_db_path, ingest_max_id)
                print(
                    f"[SLEEPING] Reconsolidation: "
                    f"{recon_result['episode_count']} episodes, "
                    f"{recon_result['new_cluster_count']} new clusters, "
                    f"{recon_result['merge_count']} merged, "
                    f"{recon_result['forget_count']} forgotten"
                )
            except Exception as e:
                logger.error("Reconsolidation failed: %s", e)
                print(f"[SLEEPING] Reconsolidation: ERROR - {e}")
        else:
            logger.warning("SleepyJean not initialized, skipping Reconsolidation")

        # 5. Self-improvement — model stays loaded (generate_bare, no VRAM swap needed)
        si_timeout = self.config.get("self_improvement", {}).get("timeout_sec", 1800)
        try:
            await asyncio.wait_for(
                self._run_self_improvement(),
                timeout=si_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Self-improvement timed out after %ds", si_timeout)
            print(f"[SLEEPING] self_improvement: TIMEOUT after {si_timeout}s")
        except Exception as e:
            logger.error("Self-improvement failed: %s", e)
            print(f"[SLEEPING] self_improvement: ERROR - {e}")

        # 5b. Generate wake message (generate_bare, model already loaded)
        if self.config.get("notifications", {}).get("wake_enabled", False):
            try:
                from discord_bot.bot import _generate_wake_message
                self._pending_wake_message = await _generate_wake_message(
                    self, recon_result=recon_result, dialogue_logs=dialogue_logs,
                )
            except Exception as e:
                logger.warning("Wake message generation failed: %s", e)
                self._pending_wake_message = None

        # 6. Purge old signals
        try:
            await self._run_purge()
        except Exception as e:
            logger.error("Purge failed: %s", e)
            print(f"[SLEEPING] purge: ERROR - {e}")

        # 7. Wake up
        await self.wake_up()

    async def _run_self_improvement(self) -> None:
        """Run the full self-improvement cycle during sleep.

        Steps:
        0. Save personality backup (immune system)
        1. Determine max_changes from difficulty signal accumulation
        2. Generate mutation candidates (3 mutations + current = 4 candidates)
        3. Evaluate fitness on core + rotation tasks
        4. Evaluate cuteness via Japanese conversations
        5. Integrate scores and select winner
        6. Update personality if a mutation wins
        7. Health check + repair if needed (immune system)
        """
        import time as _time

        si_cfg = self.config.get("self_improvement", {})
        if not si_cfg.get("enabled", False):
            logger.info("Self-improvement is disabled in config, skipping")
            return

        immune_cfg = self.config.get("immune", {})

        print("[SLEEPING] Running self-improvement...")
        cycle_start = _time.monotonic()

        # 0. Save personality backup before the cycle
        if immune_cfg.get("enabled", False):
            backup_dir = Path(immune_cfg.get("backup_dir", "data/personality_backups"))
            max_backups = immune_cfg.get("max_backups", 5)
            cycle_id = datetime.now().strftime("%Y%m%dT%H%M%S")
            save_personality_backup(
                self.personality, backup_dir, cycle_id, max_backups,
            )

        # 1. Determine mutation intensity
        difficulty_hours = si_cfg.get("difficulty_time_horizon_hours", 24)
        mc_result = await determine_max_changes(
            self.field, self.encoder,
            difficulty_time_horizon_hours=difficulty_hours,
        )
        max_changes = mc_result.max_changes
        logger.info("Self-improvement: max_changes=%d (conservative_mode=%s, difficulty_based=%d)",
                     max_changes, mc_result.conservative_mode_active, mc_result.difficulty_based_value)

        # Record fallback metrics (T8: individuality experiment 3)
        metrics_cfg = self.config.get("metrics", {})
        if metrics_cfg.get("enabled", False):
            from llamarcute_live.metrics import record_fallback
            db_path = self.config["llamarcute_live"]["dialogue_db_path"]
            await record_fallback(
                db_path,
                conservative_mode_active=mc_result.conservative_mode_active,
                max_changes_before=mc_result.difficulty_based_value,
                max_changes_after=max_changes,
            )

        # 2. Generate mutation candidates (FieldAwareLLM loaded temporarily)
        mutations = await generate_mutation_candidates(
            current_personality=self.personality,
            max_changes=max_changes,
            field=self.field,
            encoder=self.encoder,
            llm=self.llm,
            receptor=self.receptor,
            perceive_params=self.perceive_params,
        )

        min_candidates = si_cfg.get("min_candidates_for_eval", 2)
        # all_candidates = current + mutations
        all_candidates = [self.personality] + mutations
        if len(all_candidates) < min_candidates:
            logger.warning(
                "Only %d candidates (need %d). Skipping evaluation.",
                len(all_candidates), min_candidates,
            )
            print(f"[SLEEPING] self_improvement: Only {len(all_candidates)} candidates, skipping")
            return

        print(f"[SLEEPING] self_improvement: {len(all_candidates)} candidates generated")

        # 3. Load tasks
        import json
        core_tasks_path = si_cfg.get("core_tasks_path", "llamarcute_live/tasks/core_tasks.json")
        cuteness_topics_path = si_cfg.get("cuteness_topics_path", "llamarcute_live/tasks/cuteness_topics_ja.json")

        with open(core_tasks_path) as f:
            core_tasks = json.load(f)

        with open(cuteness_topics_path) as f:
            cuteness_topics = json.load(f)

        # Retrieve rotation tasks from SQLite (direct transfer, not via field)
        rotation_tasks = []
        try:
            async with aiosqlite.connect(self.dialogue.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT instruction, output FROM rotation_tasks ORDER BY id DESC LIMIT 8"
                )
                rows = await cursor.fetchall()
                rotation_tasks = [
                    {"instruction": row["instruction"], "output": row["output"]}
                    for row in rows
                ]
        except Exception as e:
            logger.warning("Failed to load rotation tasks from SQLite: %s", e)

        logger.info(
            "Self-improvement: %d core tasks, %d rotation tasks",
            len(core_tasks), len(rotation_tasks),
        )

        # 4. Evaluate fitness (generate_bare — model stays loaded)
        eval_parallel = si_cfg.get("eval_parallel", 2)
        print(f"[SLEEPING] self_improvement: Evaluating fitness (parallel={eval_parallel})...")
        fitness_results = await evaluate_fitness(
            candidates=all_candidates,
            core_tasks=core_tasks,
            rotation_tasks=rotation_tasks,
            encoder=self.encoder,
            llm=self.llm,
            parallel=eval_parallel,
        )

        # 5. Evaluate cuteness (generate_bare — model stays loaded)
        print("[SLEEPING] self_improvement: Evaluating cuteness...")
        cuteness_result = await evaluate_cuteness(
            candidates=all_candidates,
            topics=cuteness_topics,
            llm=self.llm,
        )
        cuteness_scores = cuteness_result["scores"]
        cuteness_conversation_logs = cuteness_result.get("conversation_logs", {})

        # 6. Integrate and select
        personality_path = self.config["llamarcute_live"]["personality_path"]
        result = await select_and_update(
            current_personality=self.personality,
            candidates=all_candidates,
            fitness_scores=fitness_results,
            cuteness_scores=cuteness_scores,
            personality_path=personality_path,
            field=self.field,
            encoder=self.encoder,
            fitness_weight=si_cfg.get("fitness_weight", 0.75),
            cuteness_weight=si_cfg.get("cuteness_weight", 0.25),
            db_path=self.dialogue.db_path,
        )

        if result["personality_updated"]:
            # Reload personality for next awake session
            self.personality = Personality.load(personality_path)
            self.dialogue.personality = self.personality
            print(f"[SLEEPING] self_improvement: Personality updated to {result['winner_id']}")
        else:
            print("[SLEEPING] self_improvement: Current personality retained")

        # 6b. Save winner's cuteness dialogue logs
        try:
            await _save_cuteness_dialogue_logs(
                self.dialogue.db_path,
                result["winner_id"],
                cuteness_scores,
                cuteness_conversation_logs,
            )
        except Exception as e:
            logger.warning("Failed to save cuteness dialogue logs: %s", e)

        # 7. Immune system: health check + repair
        if immune_cfg.get("enabled", False):
            elapsed = _time.monotonic() - cycle_start
            cycle_result = {
                "winner_id": result["winner_id"],
                "is_current_winner": not result["personality_updated"],
                "fitness": fitness_results,
                "cuteness": cuteness_scores,
                "post_personality": {
                    "rule_count": self.personality.rule_count,
                    "approx_tokens": self.personality.approx_tokens,
                },
                "timings": {"total": elapsed},
            }

            # Load thresholds from config
            threshold_cfg = immune_cfg.get("thresholds", {})
            thresholds = AnomalyThresholds.from_config(threshold_cfg)

            # Load history for trend detection
            history_path = Path("data/cycle_history.json")
            history = load_history(history_path)

            report = check_health(cycle_result, thresholds=thresholds, history=history)

            if report.severity != AnomalySeverity.OK:
                print(
                    f"[IMMUNE] {report.severity.value.upper()}: "
                    f"{len(report.anomalies)} anomalies detected"
                )
                repair_result = await execute_repair(
                    report,
                    Path(personality_path),
                    backup_dir,
                    self.field,
                    self.encoder,
                    db_path=self.dialogue.db_path,
                )
                if repair_result.get("personality_rolled_back"):
                    self.personality = Personality.load(personality_path)
                    self.dialogue.personality = self.personality
                    print("[IMMUNE] Personality rolled back to last known good")

                # Record immune event metrics (T8)
                if metrics_cfg.get("enabled", False):
                    from llamarcute_live.metrics import record_immune_event
                    db_path = self.config["llamarcute_live"]["dialogue_db_path"]
                    for anomaly in report.anomalies:
                        await record_immune_event(
                            db_path,
                            event_type=f"health_check_{report.severity.value}",
                            severity=anomaly["severity"],
                            details=anomaly["message"],
                        )
            else:
                print("[IMMUNE] Health check: OK")

            # Record cycle in history
            append_cycle(history_path, cycle_result)

    async def _run_purge(self) -> None:
        """Purge old signals from the shared field + SleepyJean data cleanup."""
        # 1. Shared field purge (existing)
        purge_cfg = self.config.get("purge", {})
        if purge_cfg.get("enabled", False):
            max_age_days = purge_cfg.get("max_age_days", 7)
            criteria = PurgeCriteria(older_than=timedelta(days=max_age_days))
            result = await self.field.purge(criteria)
            logger.info(
                "Purge complete: removed=%d, remaining=%d",
                result.purged_count, result.remaining_count,
            )
            print(
                f"[SLEEPING] purge: Removed {result.purged_count} old signals, "
                f"{result.remaining_count} remaining"
            )
        else:
            logger.info("Purge is disabled in config, skipping")

        # 2. SleepyJean data cleanup (Phase 4)
        cleaner_cfg = self.config.get("cleaner", {})
        if not cleaner_cfg.get("enabled", False):
            return

    async def wake_up(self) -> None:
        """Execute the wake-up sequence."""
        self.state = SystemState.AWAKE
        self.dialogue.resume()
        print("[AWAKE] Good morning. System is awake.")
        logger.info("=== SYSTEM AWAKE ===")

        # Design invariant check (Phase 1 only — static structural verification).
        # Runs every N cycles as a lightweight infrastructure health check.
        # Phase 2 (semantic review) requires human/Claude Code invocation of
        # /design-conformance and is NOT automated here (architect decision 2026-03-30).
        self._run_design_invariant_check()

    def _run_design_invariant_check(self) -> None:
        """Run structural design invariant checks periodically after wake-up."""
        interval = self.config.get("design_check", {}).get("interval_cycles", 4)
        history_path = Path("data/cycle_history.json")
        history = load_history(history_path)
        cycle_count = len(history)

        if cycle_count == 0 or cycle_count % interval != 0:
            return

        import subprocess
        script = Path(__file__).resolve().parent.parent / "scripts" / "check_design_invariants.py"
        if not script.exists():
            return

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.warning("DESIGN INVARIANT CHECK FAILED:\n%s", result.stdout)
                print(f"[DESIGN CHECK] VIOLATION DETECTED — run /design-conformance")
            else:
                logger.info("Design invariant check: all pass (cycle %d)", cycle_count)
        except Exception as e:
            logger.warning("Design invariant check error: %s", e)


async def _save_cuteness_dialogue_logs(
    db_path: str,
    winner_id: str,
    cuteness_scores: dict[str, float],
    conversation_logs: dict[str, dict],
) -> None:
    """Save winner's cuteness conversation logs to SQLite."""
    import aiosqlite
    import json

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cuteness_dialogue_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT,
                winner_id TEXT,
                partner_id TEXT,
                topic TEXT,
                dialogue TEXT,
                cuteness_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cycle_id = datetime.now().date().isoformat()
        winner_score = cuteness_scores.get(winner_id, 0.0)

        for key, conv_data in conversation_logs.items():
            # Only save conversations involving the winner
            if winner_id not in key:
                continue

            parts = key.split("_vs_")
            partner_id = parts[1] if parts[0] == winner_id else parts[0]

            dialogue_json = json.dumps(conv_data.get("log", []), ensure_ascii=False)
            topic = conv_data.get("topic", "")

            await db.execute(
                """INSERT INTO cuteness_dialogue_log
                   (cycle_id, winner_id, partner_id, topic, dialogue, cuteness_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cycle_id, winner_id, partner_id, topic, dialogue_json, winner_score),
            )

        await db.commit()
        logger.info("Saved cuteness dialogue logs for winner %s", winner_id)


def load_config(path: str = "config/system.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()

    # Ensure data directories exist
    Path(config["shared_state"]["chromadb"]["persist_directory"]).mkdir(parents=True, exist_ok=True)
    Path(config["llamarcute_live"]["dialogue_db_path"]).parent.mkdir(parents=True, exist_ok=True)

    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
