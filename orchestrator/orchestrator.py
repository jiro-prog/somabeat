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
from shared_state.observer import LoggingObserver

from llamarcute_live.llm_inference import FieldAwareLLM

from llamarcute_live.dialogue import DialogueManager
from llamarcute_live.main import LlamarcuteLiveCLI
from llamarcute_live.personality import Personality

from bridge.sleep_ingest import sleep_ingest
from bridge.wake_export import wake_export
from bridge.cleaner import clean_chromadb_rag, clean_lora_adapters

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
        self.observer = LoggingObserver()
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
            debug_cfg = config.get("debug", {})
            disable_tq = debug_cfg.get("disable_turboquant", False)
            kv_bits = 0 if disable_tq else 4
            self.llm = FieldAwareLLM(
                model_name=ll_cfg.get("transformers_model", "Qwen/Qwen3-8B"),
                kv_cache_bits=kv_bits,
                max_new_tokens=128 if disable_tq else 512,
            )

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
            ollama_model=ll_cfg.get("ollama_model", "qwen3:8b"),
            metrics_enabled=metrics_cfg.get("enabled", False),
            max_conversation_history=ll_cfg.get("max_conversation_history", 10),
            empty_perceive=config.get("debug", {}).get("empty_perceive", False),
        )

        # SleepyJean paths
        sj = config["sleepyjean"]
        self.sj_root = Path(sj["root_path"])
        self.sj_night_cycle = sj["night_cycle_script"]
        self.sj_db_path = sj["db_path"]
        self.sj_ki_path = sj["knowledge_index_path"]
        self.sj_ki_backup = str(Path(self.sj_ki_path).parent / "knowledge_index_before.json")
        self.sj_training_dir = str(self.sj_root / "data" / "training")

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

        # 1b. Unload FieldAwareLLM to free VRAM for SleepyJean (Ollama)
        if self.llm is not None and self.llm.is_loaded():
            self.llm.unload()
            logger.info("FieldAwareLLM unloaded for sleep cycle (VRAM freed for Ollama)")

        # 1c. Wait for Ollama to be reachable (needed by sleep_ingest + night cycle)
        if not await self._wait_for_ollama(timeout=30):
            logger.error("Ollama not reachable — sleep cycle may fail")
            print("[SLEEPING] WARNING: Ollama not reachable")

        # 2. Run sleep_ingest (timeout: 5 min)
        try:
            ingest_result = await asyncio.wait_for(
                sleep_ingest(
                    field=self.field,
                    encoder=self.encoder,
                    llamarcute_db_path=self.config["llamarcute_live"]["dialogue_db_path"],
                    sleepyjean_db_path=self.sj_db_path,
                    knowledge_index_path=self.sj_ki_path,
                    knowledge_index_backup_path=self.sj_ki_backup,
                ),
                timeout=300,
            )
            print(f"[SLEEPING] sleep_ingest: Transferred {ingest_result['dialogue_transferred']} dialogue entries")
            if ingest_result["homework_added"]:
                print(f"[SLEEPING] sleep_ingest: Found {ingest_result['homework_added']} difficulty signal(s) → added homework")
        except asyncio.TimeoutError:
            logger.error("sleep_ingest timed out after 300s")
            print("[SLEEPING] sleep_ingest: TIMEOUT")
        except Exception as e:
            logger.error("sleep_ingest failed: %s", e)
            print(f"[SLEEPING] sleep_ingest: ERROR - {e}")

        # 3. Run SleepyJean night cycle (subprocess)
        try:
            print("[SLEEPING] Running SleepyJean night cycle...")
            exit_code = await self._run_night_cycle()

            if exit_code != 0:
                logger.error("Night cycle failed with exit code %d", exit_code)
                print(f"[SLEEPING] Night cycle failed (exit code {exit_code}). Proceeding with wake_export using available data.")

            # 4. Run wake_export (even if night cycle failed — prior training data may exist)
            #    Timeout: 5 min
            try:
                export_result = await asyncio.wait_for(
                    wake_export(
                        field=self.field,
                        encoder=self.encoder,
                        knowledge_index_path=self.sj_ki_path,
                        knowledge_index_backup_path=self.sj_ki_backup,
                        night_result_dir=self.sj_training_dir,
                        db_path=self.sj_db_path,
                        llamarcute_db_path=self.dialogue.db_path,
                    ),
                    timeout=300,
                )
                total = (
                    export_result["new_topics_emitted"]
                    + export_result["confidence_changes_emitted"]
                    + export_result["deleted_topics_emitted"]
                    + export_result["qa_pairs_emitted"]
                    + export_result.get("dream_signals_emitted", 0)
                )
                print(f"[SLEEPING] wake_export: Emitted {total} signals")
            except asyncio.TimeoutError:
                logger.error("wake_export timed out after 300s")
                print("[SLEEPING] wake_export: TIMEOUT")
            except Exception as e:
                logger.error("wake_export failed: %s", e)
                print(f"[SLEEPING] wake_export: ERROR - {e}")
        except Exception as e:
            logger.error("Night cycle execution failed: %s", e)
            print(f"[SLEEPING] Night cycle ERROR: {e}")

        # 5. Free Ollama VRAM, then reload FieldAwareLLM for self-improvement
        await self._unload_ollama()

        if self.llm is not None:
            try:
                self.llm.load()
                logger.info("FieldAwareLLM reloaded for self-improvement cycle")
            except Exception as e:
                logger.error("FieldAwareLLM reload failed: %s — skipping self-improvement", e)
                print(f"[SLEEPING] FieldAwareLLM reload failed: {e}")

        # 5b. Self-improvement (Phase 2) — with overall timeout
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

        # 6. Purge old signals
        try:
            await self._run_purge()
        except Exception as e:
            logger.error("Purge failed: %s", e)
            print(f"[SLEEPING] purge: ERROR - {e}")

        # 7. Wake up
        await self.wake_up()

    async def _unload_ollama(self) -> None:
        """Ask Ollama to unload all models, freeing VRAM."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                # Generate with keep_alive=0 triggers model unload
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "qwen3:8b", "keep_alive": 0},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Ollama model unloaded (VRAM freed)")
                    else:
                        logger.warning("Ollama unload returned status %d", resp.status)
        except Exception as e:
            logger.warning("Ollama unload failed: %s (may already be free)", e)

        # Wait briefly for VRAM to actually free
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            await asyncio.sleep(2)

    async def _wait_for_ollama(self, timeout: int = 30) -> bool:
        """Poll Ollama health endpoint until reachable or timeout."""
        import aiohttp

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "http://localhost:11434/api/tags",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status == 200:
                            logger.info("Ollama is reachable")
                            return True
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(2)

        return False

    async def _run_night_cycle(self) -> int:
        """Run SleepyJean's night_cycle.py as a subprocess."""
        night_script = self.sj_root / self.sj_night_cycle
        venv_python = self.sj_root / ".venv" / "bin" / "python"

        if not night_script.exists():
            logger.warning("Night cycle script not found: %s", night_script)
            print("[SLEEPING] (Night cycle script not found — simulating)")
            return 0

        python_exec = str(venv_python) if venv_python.exists() else sys.executable

        # Load SleepyJean .env into subprocess environment
        import os
        env = os.environ.copy()
        dotenv_path = self.sj_root / ".env"
        if dotenv_path.exists():
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        env[key.strip()] = val.strip()

        # Suppress SleepyJean's wake message — orchestrator sends its own after
        # all processing (self_improvement etc.) completes
        env["SLEEPYJEAN_SUPPRESS_WAKE_MSG"] = "1"

        proc = await asyncio.create_subprocess_exec(
            python_exec,
            str(night_script),
            cwd=str(self.sj_root / "scripts" / "night"),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        night_cycle_timeout = self.config.get("orchestrator", {}).get(
            "night_cycle_timeout_sec", 1800,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=night_cycle_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Night cycle subprocess timed out after %ds, killing PID %d",
                night_cycle_timeout, proc.pid,
            )
            proc.kill()
            await proc.wait()
            return -1

        if stdout:
            logger.info("Night cycle stdout:\n%s", stdout.decode()[-2000:])
        if stderr:
            logger.warning("Night cycle stderr:\n%s", stderr.decode()[-2000:])

        return proc.returncode

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
        ollama_model = self.config["llamarcute_live"].get("ollama_model", "qwen3:8b")

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

        # 2. Generate mutation candidates
        mutations = await generate_mutation_candidates(
            current_personality=self.personality,
            max_changes=max_changes,
            field=self.field,
            encoder=self.encoder,
            llm=self.llm,
            receptor=self.receptor,
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

        # 4. Evaluate fitness
        ollama_parallel = si_cfg.get("ollama_parallel", 2)
        print(f"[SLEEPING] self_improvement: Evaluating fitness (parallel={ollama_parallel})...")
        fitness_results = await evaluate_fitness(
            candidates=all_candidates,
            core_tasks=core_tasks,
            rotation_tasks=rotation_tasks,
            encoder=self.encoder,
            ollama_model=ollama_model,
            ollama_parallel=ollama_parallel,
        )

        # 5. Evaluate cuteness (with field perception)
        print("[SLEEPING] self_improvement: Evaluating cuteness...")
        cuteness_field_embs = None
        if self.receptor is not None:
            perception = await self.field.perceive(PerceiveParams())
            if perception.signals:
                cuteness_field_embs = self.receptor.transduce(
                    [ps.signal.embedding for ps in perception.signals],
                    [ps.strength for ps in perception.signals],
                )
        cuteness_result = await evaluate_cuteness(
            candidates=all_candidates,
            topics=cuteness_topics,
            ollama_model=ollama_model,
            llm=self.llm,
            field_embeddings=cuteness_field_embs,
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

        # 2a. ChromaDB RAG cleanup
        chroma_cfg = cleaner_cfg.get("chromadb", {})
        if chroma_cfg.get("path"):
            try:
                chroma_result = clean_chromadb_rag(
                    chromadb_path=chroma_cfg["path"],
                    collection_name=chroma_cfg.get(
                        "collection_name", "sleepyjean_knowledge"
                    ),
                    max_generations_per_topic=chroma_cfg.get(
                        "max_generations_per_topic", 1
                    ),
                    max_age_days=chroma_cfg.get("max_age_days", 90),
                )
                print(
                    f"[CLEANER] ChromaDB: dup={chroma_result['duplicate_removed']}, "
                    f"aged={chroma_result['aged_removed']}, "
                    f"retained={chroma_result['retained']}"
                )
            except Exception as e:
                logger.error("ChromaDB cleanup failed: %s", e)
                print(f"[CLEANER] ChromaDB: ERROR - {e}")

        # 2b. LoRA adapter cleanup
        lora_cfg = cleaner_cfg.get("lora", {})
        if lora_cfg.get("adapter_dir"):
            try:
                lora_result = clean_lora_adapters(
                    adapter_dir=lora_cfg["adapter_dir"],
                    max_keep=lora_cfg.get("max_keep", 3),
                )
                print(
                    f"[CLEANER] LoRA: removed={lora_result['removed']}, "
                    f"retained={lora_result['retained']}"
                )
            except Exception as e:
                logger.error("LoRA cleanup failed: %s", e)
                print(f"[CLEANER] LoRA: ERROR - {e}")

    async def wake_up(self) -> None:
        """Execute the wake-up sequence."""
        self.state = SystemState.AWAKE
        self.dialogue.resume()
        print("[AWAKE] Good morning. System is awake.")
        logger.info("=== SYSTEM AWAKE ===")


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
