"""Immune system — anomaly detection, health monitoring, and self-repair.

Monitors self-improvement cycle health after each run. When anomalies
are detected, triggers conservative mode or personality rollback.

All coordination uses the shared field (ChromaDB) for indirect cooperation,
maintaining the stigmergic design principle.

Thresholds derived from T28 baseline data (5 cycles, fitness ~0.87, rules 8-10).

Reference: Phase 3 immune system design, T30-T31
"""

from __future__ import annotations

import enum
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.emit_log import insert_emit_log
from shared_state.encoder import E5SmallEncoder
from shared_state.interface import (
    Signal,
    SignalOrigin,
)

from llamarcute_live.cycle_history import get_consecutive_retains
from llamarcute_live.personality import Personality

logger = logging.getLogger(__name__)

IMMUNE_ROLLBACK_ORIGIN = SignalOrigin(system="llamarcute_live", context="immune:rollback")
IMMUNE_CONSERVATIVE_ORIGIN = SignalOrigin(system="llamarcute_live", context="immune:conservative_mode")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class AnomalySeverity(enum.Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AnomalyThresholds:
    """Thresholds for anomaly detection, derived from T28 baseline."""

    fitness_warning: float = 0.80
    fitness_critical: float = 0.70
    rule_count_warning: int = 13
    rule_count_critical: int = 15
    token_count_warning: int = 650
    token_count_critical: int = 800
    cycle_time_warning_s: float = 2500.0
    cycle_time_critical_s: float = 3600.0
    cuteness_warning: float = 2.0
    consecutive_retain_warning: int = 3

    @classmethod
    def from_config(cls, cfg: dict) -> AnomalyThresholds:
        """Create thresholds from config dict, using defaults for missing keys."""
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class AnomalyReport:
    severity: AnomalySeverity
    anomalies: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add(
        self,
        metric: str,
        value: float,
        threshold: float,
        severity: AnomalySeverity,
        message: str,
    ) -> None:
        self.anomalies.append({
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "severity": severity.value,
            "message": message,
        })
        # Escalate overall severity
        if severity == AnomalySeverity.CRITICAL:
            self.severity = AnomalySeverity.CRITICAL
        elif severity == AnomalySeverity.WARNING and self.severity == AnomalySeverity.OK:
            self.severity = AnomalySeverity.WARNING


# ---------------------------------------------------------------------------
# Health check — pure function, no side effects
# ---------------------------------------------------------------------------


def check_health(
    cycle_result: dict,
    thresholds: AnomalyThresholds | None = None,
    history: list[dict] | None = None,
) -> AnomalyReport:
    """Check cycle result for anomalies. Returns an AnomalyReport.

    This is a pure function — no async, no field access, no LLM calls.
    Takes the same cycle_result structure as data/t28/cycle_*_summary.json.

    Args:
        cycle_result: Dict with keys: winner_id, is_current_winner,
            fitness (dict of {candidate_id: {combined_fitness: float}}),
            cuteness (dict of {candidate_id: float}),
            post_personality ({rule_count: int, approx_tokens: int}),
            timings ({total: float}).
        thresholds: Optional custom thresholds. Uses defaults if None.
        history: Optional list of past cycle results for trend detection.
    """
    if thresholds is None:
        thresholds = AnomalyThresholds()

    report = AnomalyReport(severity=AnomalySeverity.OK)

    winner_id = cycle_result.get("winner_id", "")

    # 1. Winner fitness
    fitness_data = cycle_result.get("fitness", {})
    winner_fitness_entry = fitness_data.get(winner_id, {})
    winner_fitness = winner_fitness_entry.get("combined_fitness")
    if winner_fitness is not None:
        if winner_fitness < thresholds.fitness_critical:
            report.add(
                "winner_fitness", winner_fitness, thresholds.fitness_critical,
                AnomalySeverity.CRITICAL,
                f"Winner fitness {winner_fitness:.3f} below critical threshold "
                f"{thresholds.fitness_critical}",
            )
        elif winner_fitness < thresholds.fitness_warning:
            report.add(
                "winner_fitness", winner_fitness, thresholds.fitness_warning,
                AnomalySeverity.WARNING,
                f"Winner fitness {winner_fitness:.3f} below warning threshold "
                f"{thresholds.fitness_warning}",
            )

    # 2. Rule count
    post = cycle_result.get("post_personality", {})
    rule_count = post.get("rule_count")
    if rule_count is not None:
        if rule_count >= thresholds.rule_count_critical:
            report.add(
                "rule_count", rule_count, thresholds.rule_count_critical,
                AnomalySeverity.CRITICAL,
                f"Rule count {rule_count} reached critical limit "
                f"{thresholds.rule_count_critical}",
            )
        elif rule_count >= thresholds.rule_count_warning:
            report.add(
                "rule_count", rule_count, thresholds.rule_count_warning,
                AnomalySeverity.WARNING,
                f"Rule count {rule_count} above warning threshold "
                f"{thresholds.rule_count_warning}",
            )

    # 3. Token count
    approx_tokens = post.get("approx_tokens")
    if approx_tokens is not None:
        if approx_tokens >= thresholds.token_count_critical:
            report.add(
                "token_count", approx_tokens, thresholds.token_count_critical,
                AnomalySeverity.CRITICAL,
                f"Token count {approx_tokens} reached critical limit "
                f"{thresholds.token_count_critical}",
            )
        elif approx_tokens >= thresholds.token_count_warning:
            report.add(
                "token_count", approx_tokens, thresholds.token_count_warning,
                AnomalySeverity.WARNING,
                f"Token count {approx_tokens} above warning threshold "
                f"{thresholds.token_count_warning}",
            )

    # 4. Cycle time
    timings = cycle_result.get("timings", {})
    cycle_time = timings.get("total")
    if cycle_time is not None:
        if cycle_time > thresholds.cycle_time_critical_s:
            report.add(
                "cycle_time", cycle_time, thresholds.cycle_time_critical_s,
                AnomalySeverity.CRITICAL,
                f"Cycle time {cycle_time:.0f}s exceeds critical threshold "
                f"{thresholds.cycle_time_critical_s:.0f}s",
            )
        elif cycle_time > thresholds.cycle_time_warning_s:
            report.add(
                "cycle_time", cycle_time, thresholds.cycle_time_warning_s,
                AnomalySeverity.WARNING,
                f"Cycle time {cycle_time:.0f}s exceeds warning threshold "
                f"{thresholds.cycle_time_warning_s:.0f}s",
            )

    # 5. Cuteness
    cuteness_data = cycle_result.get("cuteness", {})
    winner_cuteness = cuteness_data.get(winner_id)
    if winner_cuteness is not None:
        if winner_cuteness <= 0:
            report.add(
                "cuteness", winner_cuteness, 0,
                AnomalySeverity.CRITICAL,
                f"Winner cuteness score is {winner_cuteness} (zero or negative)",
            )
        elif winner_cuteness < thresholds.cuteness_warning:
            report.add(
                "cuteness", winner_cuteness, thresholds.cuteness_warning,
                AnomalySeverity.WARNING,
                f"Winner cuteness {winner_cuteness:.1f} below warning threshold "
                f"{thresholds.cuteness_warning}",
            )

    # 6. Consecutive retains (trend detection from history)
    if history is not None:
        # Include current result in the consecutive count
        extended = history + [cycle_result]
        consec = get_consecutive_retains(extended)
        if consec >= thresholds.consecutive_retain_warning:
            # WARNING only — no action (see design notes)
            report.add(
                "consecutive_retain", consec,
                thresholds.consecutive_retain_warning,
                AnomalySeverity.WARNING,
                f"Personality retained {consec} consecutive cycles. "
                f"Consider whether exploration is sufficient.",
            )

    return report


# ---------------------------------------------------------------------------
# Backup management
# ---------------------------------------------------------------------------


def save_personality_backup(
    personality: Personality,
    backup_dir: Path,
    cycle_id: str,
    max_backups: int = 5,
) -> Path:
    """Save a backup of the current personality before self-improvement.

    Keeps at most max_backups files in backup_dir (oldest deleted first).
    Returns the path of the saved backup.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / f"personality_backup_{cycle_id}.yaml"
    personality.save(backup_path)
    logger.info("Personality backup saved: %s", backup_path)

    # Prune old backups
    backups = sorted(backup_dir.glob("personality_backup_*.yaml"))
    while len(backups) > max_backups:
        oldest = backups.pop(0)
        oldest.unlink()
        logger.info("Pruned old backup: %s", oldest)

    return backup_path


def find_last_known_good(backup_dir: Path) -> Path | None:
    """Find the most recent valid personality backup.

    Returns None if no valid backup exists.
    """
    if not backup_dir.exists():
        return None

    backups = sorted(backup_dir.glob("personality_backup_*.yaml"), reverse=True)
    for backup_path in backups:
        try:
            Personality.load(backup_path)
            return backup_path
        except Exception as e:
            logger.warning("Invalid backup %s: %s", backup_path, e)
            continue
    return None


# ---------------------------------------------------------------------------
# Self-repair actions
# ---------------------------------------------------------------------------


async def execute_repair(
    report: AnomalyReport,
    personality_path: Path,
    backup_dir: Path,
    field: ChromaDBField,
    encoder: E5SmallEncoder,
    db_path: str | Path | None = None,
) -> dict:
    """Execute repair actions based on anomaly report severity.

    WARNING: Emit conservative_mode signal to the field.
        (Except for consecutive_retain, which logs only.)
    CRITICAL: Rollback personality + emit conservative_mode +
        purge stale self_improvement signals + emit rollback record.

    Returns a dict with repair actions taken.
    """
    result: dict = {
        "severity": report.severity.value,
        "anomaly_count": len(report.anomalies),
        "actions": [],
        "personality_rolled_back": False,
    }

    # Check if only consecutive_retain anomalies (no action needed)
    non_retain_anomalies = [
        a for a in report.anomalies
        if a["metric"] != "consecutive_retain"
    ]

    if report.severity == AnomalySeverity.CRITICAL:
        # 1. Rollback personality
        good_backup = find_last_known_good(backup_dir)
        if good_backup:
            # Load the backup to get version info for the log
            backup_personality = Personality.load(good_backup)
            current_personality = Personality.load(personality_path)

            shutil.copy2(good_backup, personality_path)
            result["personality_rolled_back"] = True
            result["rolled_back_from"] = current_personality.version
            result["rolled_back_to_file"] = str(good_backup)
            result["actions"].append(
                f"Rolled back personality from v{current_personality.version} "
                f"using backup {good_backup.name}"
            )
            logger.critical(
                "IMMUNE CRITICAL: Personality rolled back from v%d using %s",
                current_personality.version, good_backup,
            )

            # 2. Emit rollback record to field
            emit_text = (
                f"personality_rollback from v{current_personality.version} "
                f"to backup {good_backup.name}"
            )
            embedding = encoder.encode_for_emit(emit_text)
            embedding = embedding * 2.5  # high norm for visibility
            signal = Signal.create(
                embedding=embedding, origin=IMMUNE_ROLLBACK_ORIGIN,
            )
            await field.emit(signal)
            if db_path:
                await insert_emit_log(db_path, signal.signal_id, emit_text)
            result["actions"].append("Emitted rollback record to field")

            # Note: stale self_improvement signals are left to natural time-decay
            # rather than origin-based purge (non-directionality principle).
            # The rollback signal emitted above has high norm (2.5) and will
            # dominate perception until self_improvement signals decay away.
        else:
            logger.error("IMMUNE CRITICAL: No valid backup found for rollback!")
            result["actions"].append("No valid backup for rollback")

        # 4. Emit conservative_mode signal
        emit_text = "conservative_mode: CRITICAL anomaly detected, forcing max_changes=1"
        embedding = encoder.encode_for_emit(emit_text)
        embedding = embedding * 2.0
        signal = Signal.create(
            embedding=embedding, origin=IMMUNE_CONSERVATIVE_ORIGIN,
        )
        await field.emit(signal)
        if db_path:
            await insert_emit_log(db_path, signal.signal_id, emit_text)
        result["actions"].append("Emitted conservative_mode signal")

    elif report.severity == AnomalySeverity.WARNING and non_retain_anomalies:
        # Only emit conservative_mode for non-retain warnings
        emit_text = "conservative_mode: WARNING anomaly detected, forcing max_changes=1"
        embedding = encoder.encode_for_emit(emit_text)
        embedding = embedding * 1.5
        signal = Signal.create(
            embedding=embedding, origin=IMMUNE_CONSERVATIVE_ORIGIN,
        )
        await field.emit(signal)
        if db_path:
            await insert_emit_log(db_path, signal.signal_id, emit_text)
        result["actions"].append("Emitted conservative_mode signal")
        logger.warning(
            "IMMUNE WARNING: %d anomalies detected, conservative mode activated",
            len(non_retain_anomalies),
        )

    # Log all anomalies
    for anomaly in report.anomalies:
        logger.warning(
            "IMMUNE [%s] %s: %s",
            anomaly["severity"], anomaly["metric"], anomaly["message"],
        )

    return result
