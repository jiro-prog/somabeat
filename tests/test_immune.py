"""Tests for Phase 3 immune system (T30-T34)."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llamarcute_live.immune import (
    AnomalyReport,
    AnomalySeverity,
    AnomalyThresholds,
    IMMUNE_CONSERVATIVE_ORIGIN,
    IMMUNE_ROLLBACK_ORIGIN,
    check_health,
    execute_repair,
    find_last_known_good,
    save_personality_backup,
)
from llamarcute_live.cycle_history import (
    append_cycle,
    get_consecutive_retains,
    load_history,
)


# ============================================================
# Test fixtures
# ============================================================


def _normal_cycle_result():
    """A normal cycle result based on T28 baseline data."""
    return {
        "winner_id": "personality_v0",
        "is_current_winner": True,
        "fitness": {
            "personality_v0": {
                "core_accuracy": 0.87,
                "rotation_score": 0.89,
                "combined_fitness": 0.87,
            },
            "mutation_1_A": {
                "core_accuracy": 0.80,
                "rotation_score": 0.89,
                "combined_fitness": 0.83,
            },
        },
        "cuteness": {
            "personality_v0": 5.0,
            "mutation_1_A": 7.0,
        },
        "post_personality": {
            "rule_count": 9,
            "approx_tokens": 278,
        },
        "timings": {
            "total": 1550.0,
        },
    }


def _make_personality_data():
    """Return a valid personality data dict with 6 rules (above RULE_COUNT_MIN=5)."""
    return {
        "id": "test_v0",
        "version": 0,
        "updated_at": None,
        "previous_version": None,
        "behavioral_rules": {
            "reasoning": [
                {"rule": "Break complex problems into smaller parts", "added_ver": 0, "modified_ver": None},
                {"rule": "Consider multiple approaches", "added_ver": 0, "modified_ver": None},
            ],
            "response": [
                {"rule": "Keep responses concise", "added_ver": 0, "modified_ver": None},
            ],
            "meta": [
                {"rule": "Acknowledge when uncertain", "added_ver": 0, "modified_ver": None},
            ],
            "identity": [
                {"rule": "Speak with warmth and curiosity", "added_ver": 0, "modified_ver": None},
                {"rule": "Be honest about what you know", "added_ver": 0, "modified_ver": None},
            ],
        },
    }


# ============================================================
# check_health tests
# ============================================================


class TestCheckHealth:
    def test_all_ok(self):
        """Normal T28 data should report OK."""
        report = check_health(_normal_cycle_result())
        assert report.severity == AnomalySeverity.OK
        assert len(report.anomalies) == 0

    def test_fitness_warning(self):
        result = _normal_cycle_result()
        result["fitness"]["personality_v0"]["combined_fitness"] = 0.75
        report = check_health(result)
        assert report.severity == AnomalySeverity.WARNING
        assert any(a["metric"] == "winner_fitness" for a in report.anomalies)

    def test_fitness_critical(self):
        result = _normal_cycle_result()
        result["fitness"]["personality_v0"]["combined_fitness"] = 0.65
        report = check_health(result)
        assert report.severity == AnomalySeverity.CRITICAL
        assert any(
            a["metric"] == "winner_fitness" and a["severity"] == "critical"
            for a in report.anomalies
        )

    def test_rule_count_warning(self):
        result = _normal_cycle_result()
        result["post_personality"]["rule_count"] = 14
        report = check_health(result)
        assert report.severity == AnomalySeverity.WARNING
        assert any(a["metric"] == "rule_count" for a in report.anomalies)

    def test_rule_count_critical(self):
        result = _normal_cycle_result()
        result["post_personality"]["rule_count"] = 15
        report = check_health(result)
        assert report.severity == AnomalySeverity.CRITICAL

    def test_token_count_warning(self):
        result = _normal_cycle_result()
        result["post_personality"]["approx_tokens"] = 700
        report = check_health(result)
        assert report.severity == AnomalySeverity.WARNING
        assert any(a["metric"] == "token_count" for a in report.anomalies)

    def test_token_count_critical(self):
        result = _normal_cycle_result()
        result["post_personality"]["approx_tokens"] = 800
        report = check_health(result)
        assert report.severity == AnomalySeverity.CRITICAL

    def test_cycle_time_warning(self):
        result = _normal_cycle_result()
        result["timings"]["total"] = 2800.0
        report = check_health(result)
        assert report.severity == AnomalySeverity.WARNING
        assert any(a["metric"] == "cycle_time" for a in report.anomalies)

    def test_cycle_time_critical(self):
        result = _normal_cycle_result()
        result["timings"]["total"] = 4000.0
        report = check_health(result)
        assert report.severity == AnomalySeverity.CRITICAL

    def test_cuteness_warning(self):
        result = _normal_cycle_result()
        result["cuteness"]["personality_v0"] = 1.5
        report = check_health(result)
        assert report.severity == AnomalySeverity.WARNING
        assert any(a["metric"] == "cuteness" for a in report.anomalies)

    def test_cuteness_critical_zero(self):
        result = _normal_cycle_result()
        result["cuteness"]["personality_v0"] = 0
        report = check_health(result)
        assert report.severity == AnomalySeverity.CRITICAL

    def test_multiple_anomalies_severity_max(self):
        """Multiple anomalies: overall severity is the maximum."""
        result = _normal_cycle_result()
        result["fitness"]["personality_v0"]["combined_fitness"] = 0.75  # WARNING
        result["post_personality"]["rule_count"] = 15  # CRITICAL
        report = check_health(result)
        assert report.severity == AnomalySeverity.CRITICAL
        assert len(report.anomalies) == 2

    def test_consecutive_retains(self):
        result = _normal_cycle_result()
        history = [
            {"is_current_winner": True},
            {"is_current_winner": True},
        ]
        # history has 2 retains + current result = 3 consecutive
        report = check_health(result, history=history)
        assert report.severity == AnomalySeverity.WARNING
        assert any(a["metric"] == "consecutive_retain" for a in report.anomalies)

    def test_consecutive_retains_not_triggered(self):
        result = _normal_cycle_result()
        history = [
            {"is_current_winner": False},  # update breaks the streak
            {"is_current_winner": True},
        ]
        report = check_health(result, history=history)
        # Only 2 consecutive retains (the last history entry + current), below threshold 3
        assert report.severity == AnomalySeverity.OK

    def test_custom_thresholds(self):
        result = _normal_cycle_result()
        result["fitness"]["personality_v0"]["combined_fitness"] = 0.85
        # Default threshold: 0.80 → OK. Custom threshold: 0.90 → WARNING
        custom = AnomalyThresholds(fitness_warning=0.90)
        report = check_health(result, thresholds=custom)
        assert report.severity == AnomalySeverity.WARNING

    def test_thresholds_from_config(self):
        cfg = {"fitness_warning": 0.85, "rule_count_warning": 10}
        t = AnomalyThresholds.from_config(cfg)
        assert t.fitness_warning == 0.85
        assert t.rule_count_warning == 10
        # Unset values use defaults
        assert t.fitness_critical == 0.70


# ============================================================
# Backup management tests
# ============================================================


class TestBackupManagement:
    def test_save_personality_backup(self):
        from llamarcute_live.personality import Personality

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Personality(_make_personality_data())
            backup_dir = Path(tmpdir) / "backups"

            path = save_personality_backup(p, backup_dir, "cycle_1")
            assert path.exists()

            loaded = Personality.load(path)
            assert loaded.id == "test_v0"

    def test_backup_pruning(self):
        from llamarcute_live.personality import Personality

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Personality(_make_personality_data())
            backup_dir = Path(tmpdir) / "backups"

            # Create 7 backups with max_backups=5
            for i in range(7):
                save_personality_backup(p, backup_dir, f"cycle_{i}", max_backups=5)

            backups = list(backup_dir.glob("personality_backup_*.yaml"))
            assert len(backups) == 5

    def test_find_last_known_good(self):
        from llamarcute_live.personality import Personality

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Personality(_make_personality_data())
            backup_dir = Path(tmpdir) / "backups"

            save_personality_backup(p, backup_dir, "cycle_1")
            save_personality_backup(p, backup_dir, "cycle_2")

            found = find_last_known_good(backup_dir)
            assert found is not None
            assert "cycle_2" in found.name

    def test_find_last_known_good_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            found = find_last_known_good(Path(tmpdir))
            assert found is None

    def test_find_last_known_good_nonexistent(self):
        found = find_last_known_good(Path("/nonexistent/dir"))
        assert found is None


# ============================================================
# Cycle history tests
# ============================================================


class TestCycleHistory:
    def test_load_history_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            assert load_history(path) == []

    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            append_cycle(path, {"cycle": 1, "winner": "A"})
            append_cycle(path, {"cycle": 2, "winner": "B"})

            history = load_history(path)
            assert len(history) == 2
            assert history[0]["cycle"] == 1
            assert history[1]["cycle"] == 2

    def test_append_max_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            for i in range(25):
                append_cycle(path, {"cycle": i}, max_entries=20)

            history = load_history(path)
            assert len(history) == 20
            # Should keep the most recent 20
            assert history[0]["cycle"] == 5
            assert history[-1]["cycle"] == 24

    def test_get_consecutive_retains(self):
        history = [
            {"is_current_winner": False},
            {"is_current_winner": True},
            {"is_current_winner": True},
            {"is_current_winner": True},
        ]
        assert get_consecutive_retains(history) == 3

    def test_get_consecutive_retains_broken(self):
        history = [
            {"is_current_winner": True},
            {"is_current_winner": False},
            {"is_current_winner": True},
        ]
        assert get_consecutive_retains(history) == 1

    def test_get_consecutive_retains_all_updates(self):
        history = [
            {"is_current_winner": False},
            {"is_current_winner": False},
        ]
        assert get_consecutive_retains(history) == 0

    def test_get_consecutive_retains_empty(self):
        assert get_consecutive_retains([]) == 0


# ============================================================
# execute_repair tests
# ============================================================


class TestExecuteRepair:
    @pytest.fixture
    def mock_field_encoder(self):
        field = AsyncMock()
        field.emit = AsyncMock()
        field.purge = AsyncMock(return_value=MagicMock(purged_count=3))

        encoder = MagicMock()
        rng = np.random.RandomState(42)
        encoder.encode_for_emit = MagicMock(
            return_value=rng.randn(384).astype(np.float32)
        )
        return field, encoder

    def test_warning_emits_conservative_mode(self, mock_field_encoder):
        field, encoder = mock_field_encoder

        report = AnomalyReport(severity=AnomalySeverity.WARNING)
        report.add("winner_fitness", 0.75, 0.80, AnomalySeverity.WARNING, "Low fitness")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = asyncio.get_event_loop().run_until_complete(
                execute_repair(
                    report, Path(tmpdir) / "p.yaml", Path(tmpdir) / "backups",
                    field, encoder,
                )
            )
        assert "Emitted conservative_mode signal" in result["actions"]
        assert not result["personality_rolled_back"]
        field.emit.assert_called_once()

    def test_warning_consecutive_retain_no_action(self, mock_field_encoder):
        """consecutive_retain warnings should NOT emit conservative_mode."""
        field, encoder = mock_field_encoder

        report = AnomalyReport(severity=AnomalySeverity.WARNING)
        report.add(
            "consecutive_retain", 3, 3, AnomalySeverity.WARNING,
            "Personality retained 3 consecutive cycles",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = asyncio.get_event_loop().run_until_complete(
                execute_repair(
                    report, Path(tmpdir) / "p.yaml", Path(tmpdir) / "backups",
                    field, encoder,
                )
            )
        # No actions should be taken for retain-only warnings
        assert len(result["actions"]) == 0
        field.emit.assert_not_called()

    def test_critical_rollback(self, mock_field_encoder):
        from llamarcute_live.personality import Personality

        field, encoder = mock_field_encoder

        report = AnomalyReport(severity=AnomalySeverity.CRITICAL)
        report.add(
            "winner_fitness", 0.60, 0.70, AnomalySeverity.CRITICAL, "Very low fitness",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a personality file and backup
            p = Personality(_make_personality_data())
            personality_path = Path(tmpdir) / "personality.yaml"
            p.save(personality_path)

            backup_dir = Path(tmpdir) / "backups"
            save_personality_backup(p, backup_dir, "cycle_1")

            result = asyncio.get_event_loop().run_until_complete(
                execute_repair(report, personality_path, backup_dir, field, encoder)
            )

        assert result["personality_rolled_back"]
        assert any("Rolled back" in a for a in result["actions"])
        assert any("conservative_mode" in a for a in result["actions"])
        # emit called for rollback record + conservative_mode = 2 calls
        assert field.emit.call_count == 2

    def test_critical_no_backup(self, mock_field_encoder):
        field, encoder = mock_field_encoder

        report = AnomalyReport(severity=AnomalySeverity.CRITICAL)
        report.add(
            "winner_fitness", 0.60, 0.70, AnomalySeverity.CRITICAL, "Very low",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = asyncio.get_event_loop().run_until_complete(
                execute_repair(
                    report, Path(tmpdir) / "p.yaml",
                    Path(tmpdir) / "empty_backups",
                    field, encoder,
                )
            )
        assert not result["personality_rolled_back"]
        assert any("No valid backup" in a for a in result["actions"])
        # Should still emit conservative_mode
        assert any("conservative_mode" in a for a in result["actions"])
