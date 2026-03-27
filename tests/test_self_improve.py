"""Tests for Phase 2 self-improvement modules (T18-T21)."""

import asyncio
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from llamarcute_live.parser import (
    extract_answer, extract_code, grade_code, grade_logic, grade_math,
)
from llamarcute_live.selection import ScoredCandidate, integrate_scores, z_score_normalize


# ============================================================
# Parser tests
# ============================================================

class TestParser:
    def test_extract_answer_basic(self):
        assert extract_answer("Some text\nANSWER: 42") == "42"

    def test_extract_answer_with_think(self):
        text = "<think>reasoning</think>\nSome text\nANSWER: 42"
        assert extract_answer(text) == "42"

    def test_extract_answer_none(self):
        assert extract_answer("No answer marker here") is None

    def test_extract_code_markdown(self):
        text = "Here is code:\n```python\ndef foo():\n    return 1\n```"
        assert "def foo():" in extract_code(text)

    def test_extract_code_none(self):
        assert extract_code("No code here") is None

    def test_grade_math_correct(self):
        assert grade_math("Let me solve this.\nANSWER: 273", 273.0)

    def test_grade_math_incorrect(self):
        assert not grade_math("ANSWER: 100", 273.0)

    def test_grade_math_with_commas(self):
        assert grade_math("ANSWER: 1,234", 1234.0)

    def test_grade_logic_correct(self):
        assert grade_logic("The answer is\nANSWER: A", "A")

    def test_grade_logic_incorrect(self):
        assert not grade_logic("ANSWER: B", "A")

    def test_grade_logic_lowercase(self):
        assert grade_logic("ANSWER: a", "A")

    def test_grade_code_passes(self):
        response = '```python\ndef min_of_three(a, b, c):\n    return min(a, b, c)\n```'
        result = grade_code(response, [
            {"assert": "assert min_of_three(10,20,0)==0"},
            {"assert": "assert min_of_three(19,15,18)==15"},
        ])
        assert result["score"] == 1.0

    def test_grade_code_fails(self):
        response = '```python\ndef min_of_three(a, b, c):\n    return a\n```'
        result = grade_code(response, [
            {"assert": "assert min_of_three(10,20,0)==0"},
            {"assert": "assert min_of_three(19,15,18)==15"},
        ])
        assert result["score"] < 1.0

    def test_grade_code_no_code(self):
        result = grade_code("I don't know how to write code", [{"assert": "assert True"}])
        assert result["score"] == 0.0


# ============================================================
# Personality mutation tests
# ============================================================

class TestPersonalityMutation:
    def _make_personality_data(self):
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

    def test_apply_add(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        changes = [
            {"action": "add", "category": "reasoning", "new": "Use analogies to explain"},
        ]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1
        assert "ADD" in applied[0]
        assert new_p.rule_count == p.rule_count + 1

    def test_apply_modify(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        changes = [
            {
                "action": "modify",
                "category": "response",
                "original": "Keep responses concise",
                "new": "Keep responses brief and focused",
            },
        ]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1
        assert "MODIFY" in applied[0]

    def test_apply_delete(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        # Add an extra rule first to stay above minimum
        p.rules["reasoning"].append(
            {"rule": "Extra rule for testing", "added_ver": 0, "modified_ver": None}
        )
        changes = [
            {"action": "delete", "category": "reasoning", "original": "Extra rule for testing"},
        ]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1
        assert "DELETE" in applied[0]

    def test_cooling_period(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        # Rule was added in ver 1
        p.rules["reasoning"].append(
            {"rule": "Recently added rule", "added_ver": 1, "modified_ver": None}
        )
        # Try to modify it in ver 1 (should be blocked)
        changes = [
            {
                "action": "modify",
                "category": "reasoning",
                "original": "Recently added rule",
                "new": "Modified recently added rule",
            },
        ]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 0  # Change should be blocked

    def test_max_changes_respected(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        changes = [
            {"action": "add", "category": "reasoning", "new": "Rule A"},
            {"action": "add", "category": "reasoning", "new": "Rule B"},
            {"action": "add", "category": "reasoning", "new": "Rule C"},
        ]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1

    def test_save_and_load(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "personality.yaml"
            p.save(str(path))
            loaded = Personality.load(str(path))
            assert loaded.id == p.id
            assert loaded.version == p.version

    def test_numbered_rules_format(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        text = p.to_numbered_rules()
        lines = text.strip().split("\n")
        assert len(lines) == 6  # 6 rules total
        assert lines[0].startswith("[R1] (reasoning)")
        assert lines[5].startswith("[R6] (identity)")

    def test_build_rule_index(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        index = p.build_rule_index()
        assert len(index) == 6
        assert "R1" in index
        assert "R6" in index
        cat, idx = index["R1"]
        assert cat == "reasoning"
        assert idx == 0

    def test_apply_modify_by_rule_id(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        # R3 = first rule in "response" = "Keep responses concise"
        index = p.build_rule_index()
        r3_cat, r3_idx = index["R3"]
        assert p.rules[r3_cat][r3_idx]["rule"] == "Keep responses concise"

        changes = [{"action": "modify", "rule_id": "R3", "new": "Keep responses brief and focused"}]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1
        assert "MODIFY" in applied[0]
        assert "R3" in applied[0]
        assert new_p.rules[r3_cat][r3_idx]["rule"] == "Keep responses brief and focused"

    def test_apply_delete_by_rule_id(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        # Add extra rule to stay above minimum
        p.rules["reasoning"].append(
            {"rule": "Extra rule for testing", "added_ver": 0, "modified_ver": None}
        )
        # R7 = the newly added extra rule
        index = p.build_rule_index()
        assert "R7" in index

        changes = [{"action": "delete", "rule_id": "R7"}]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1
        assert "DELETE" in applied[0]
        assert "R7" in applied[0]
        assert new_p.rule_count == p.rule_count - 1

    def test_apply_invalid_rule_id(self):
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        changes = [{"action": "modify", "rule_id": "R99", "new": "Something"}]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 0  # Invalid ID, no change

    def test_delete_then_modify_same_category_no_index_shift(self):
        """Delete + modify in same category must not shift indices."""
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        # reasoning has R1="Break complex...", R2="Consider multiple..."
        # Add extra to stay above minimum after delete
        p.rules["reasoning"].append(
            {"rule": "Extra reasoning rule", "added_ver": 0, "modified_ver": None}
        )
        # Delete R1, modify R2 — should not accidentally modify R3 (the extra)
        changes = [
            {"action": "delete", "rule_id": "R1"},
            {"action": "modify", "rule_id": "R2", "new": "Modified R2 text"},
        ]
        new_p, applied = p.apply_changes(changes, max_changes=2, current_ver=1)
        assert len(applied) == 2
        # R2 was "Consider multiple approaches" — verify it was modified, not the extra
        modified_rules = [r["rule"] for r in new_p.rules["reasoning"]]
        assert "Modified R2 text" in modified_rules
        assert "Extra reasoning rule" in modified_rules
        assert "Break complex problems into smaller parts" not in modified_rules

    def test_legacy_exact_match_still_works(self):
        """Legacy format (original + category) should still work as fallback."""
        from llamarcute_live.personality import Personality

        p = Personality(self._make_personality_data())
        changes = [
            {
                "action": "modify",
                "category": "response",
                "original": "Keep responses concise",
                "new": "Keep responses brief and focused",
            },
        ]
        new_p, applied = p.apply_changes(changes, max_changes=1, current_ver=1)
        assert len(applied) == 1
        assert "MODIFY" in applied[0]


# ============================================================
# Parse changes tests
# ============================================================

class TestParseChanges:
    def test_parse_single_change(self):
        from llamarcute_live.self_improve import parse_changes

        text = """ANALYSIS:
The current rules need adjustment.

CHANGES:
- action: add
  category: reasoning
  new: "Use step-by-step thinking for complex problems"
  reason: "Improves systematic problem solving"
"""
        changes = parse_changes(text)
        assert changes is not None
        assert len(changes) == 1
        assert changes[0]["action"] == "add"
        assert changes[0]["category"] == "reasoning"

    def test_parse_multiple_changes(self):
        from llamarcute_live.self_improve import parse_changes

        text = """ANALYSIS:
Multiple areas need improvement.

CHANGES:
- action: modify
  category: response
  original: "Keep responses concise"
  new: "Provide detailed explanations"
  reason: "Better understanding"
- action: add
  category: meta
  new: "Reflect on approach before answering"
  reason: "More thoughtful responses"
"""
        changes = parse_changes(text)
        assert changes is not None
        assert len(changes) == 2

    def test_parse_no_changes_section(self):
        from llamarcute_live.self_improve import parse_changes

        text = "ANALYSIS:\nEverything is fine, no changes needed."
        assert parse_changes(text) is None

    def test_parse_rule_id_modify(self):
        from llamarcute_live.self_improve import parse_changes

        text = """ANALYSIS:
R3 needs updating.

CHANGES:
- action: modify
  rule_id: R3
  new: "Keep responses brief and focused"
  reason: "Clarity improvement"
"""
        changes = parse_changes(text)
        assert changes is not None
        assert len(changes) == 1
        assert changes[0]["action"] == "modify"
        assert changes[0]["rule_id"] == "R3"
        assert "new" in changes[0]

    def test_parse_rule_id_delete(self):
        from llamarcute_live.self_improve import parse_changes

        text = """ANALYSIS:
R5 is redundant.

CHANGES:
- action: delete
  rule_id: R5
  reason: "Overlaps with R2"
"""
        changes = parse_changes(text)
        assert changes is not None
        assert len(changes) == 1
        assert changes[0]["action"] == "delete"
        assert changes[0]["rule_id"] == "R5"

    def test_parse_mixed_rule_id_and_add(self):
        from llamarcute_live.self_improve import parse_changes

        text = """ANALYSIS:
Rebalance rules.

CHANGES:
- action: modify
  rule_id: R1
  new: "Decompose problems step by step"
  reason: "Clearer instruction"
- action: delete
  rule_id: R4
  reason: "Redundant"
- action: add
  category: identity
  new: "Use humor when appropriate"
  reason: "More engaging"
"""
        changes = parse_changes(text)
        assert changes is not None
        assert len(changes) == 3


# ============================================================
# Selection / scoring tests
# ============================================================

class TestSelection:
    def test_z_score_normalize(self):
        result = z_score_normalize([1.0, 2.0, 3.0])
        assert len(result) == 3
        assert abs(sum(result)) < 1e-6  # z-scores sum to ~0
        assert result[2] > result[0]  # highest value gets highest z

    def test_z_score_all_same(self):
        result = z_score_normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_z_score_empty(self):
        assert z_score_normalize([]) == []

    def test_integrate_scores(self):
        fitness = {"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
        cuteness = {"A": 3.0, "B": 6.0, "C": 9.0, "D": 12.0}

        scored = integrate_scores(fitness, cuteness, fitness_weight=0.75, cuteness_weight=0.25)
        assert len(scored) == 4
        # First one should have highest integrated score
        assert scored[0].integrated >= scored[-1].integrated
        # Check it's a ScoredCandidate
        assert isinstance(scored[0], ScoredCandidate)

    def test_integrate_scores_fitness_dominant(self):
        """With high fitness weight, fitness should dominate."""
        fitness = {"A": 1.0, "B": 0.0}
        cuteness = {"A": 0.0, "B": 1.0}

        scored = integrate_scores(fitness, cuteness, fitness_weight=0.9, cuteness_weight=0.1)
        assert scored[0].id == "A"  # A wins due to high fitness


# ============================================================
# Cuteness parsing tests
# ============================================================

class TestCutenessParser:
    def test_parse_rankings_success(self):
        from llamarcute_live.cuteness import parse_rankings

        text = """
RANK_1: AI-B
REASON: Very creative and engaging

RANK_2: AI-C
REASON: Good listener

RANK_3: AI-A
REASON: A bit too formal
"""
        labels = ["AI-A", "AI-B", "AI-C"]
        result = parse_rankings(text, labels)
        assert result is not None
        assert len(result) == 3
        assert result[0]["label"] == "AI-B"
        assert result[0]["points"] == 3
        assert result[1]["points"] == 2
        assert result[2]["points"] == 1

    def test_parse_rankings_failure(self):
        from llamarcute_live.cuteness import parse_rankings

        text = "I enjoyed all conversations equally."
        labels = ["AI-A", "AI-B", "AI-C"]
        assert parse_rankings(text, labels) is None

    def test_parse_rankings_duplicate(self):
        from llamarcute_live.cuteness import parse_rankings

        text = """
RANK_1: AI-A
REASON: Best

RANK_2: AI-A
REASON: Also best

RANK_3: AI-B
REASON: Ok
"""
        labels = ["AI-A", "AI-B", "AI-C"]
        assert parse_rankings(text, labels) is None

    def test_parse_rankings_japanese_pattern(self):
        from llamarcute_live.cuteness import parse_rankings

        text = "1位: AI-B, 2位: AI-A, 3位: AI-C"
        labels = ["AI-A", "AI-B", "AI-C"]
        result = parse_rankings(text, labels)
        assert result is not None
        assert len(result) == 3
        assert result[0]["label"] == "AI-B"
        assert result[0]["points"] == 3
        assert result[1]["label"] == "AI-A"
        assert result[1]["points"] == 2
        assert result[2]["label"] == "AI-C"
        assert result[2]["points"] == 1

    def test_parse_rankings_with_preamble(self):
        from llamarcute_live.cuteness import parse_rankings

        text = """会話を振り返って、楽しかった順に並べます。

RANK_1: AI-C
REASON: とても面白い会話でした

RANK_2: AI-A
REASON: 良い議論ができました

RANK_3: AI-B
REASON: 少し堅い印象でした
"""
        labels = ["AI-A", "AI-B", "AI-C"]
        result = parse_rankings(text, labels)
        assert result is not None
        assert len(result) == 3
        assert result[0]["label"] == "AI-C"
        assert result[0]["points"] == 3
        assert result[1]["label"] == "AI-A"
        assert result[2]["label"] == "AI-B"

    def test_parse_rankings_appearance_order_fallback(self):
        from llamarcute_live.cuteness import parse_rankings

        text = "一番楽しかったのはAI-Cで、次にAI-Aが良く、AI-Bは普通でした。"
        labels = ["AI-A", "AI-B", "AI-C"]
        result = parse_rankings(text, labels)
        assert result is not None
        assert len(result) == 3
        # Appearance order: AI-C first, AI-A second, AI-B third
        assert result[0]["label"] == "AI-C"
        assert result[0]["points"] == 3
        assert result[1]["label"] == "AI-A"
        assert result[1]["points"] == 2
        assert result[2]["label"] == "AI-B"
        assert result[2]["points"] == 1
