"""Behavioral rules (personality) management.

Reference: llamarcute_live_design.md section 3.2, 5.4
"""

from __future__ import annotations

import copy
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

RULE_COUNT_MIN = 5
RULE_COUNT_MAX = 15
TOKEN_LIMIT = 800  # approximate token count for all rules combined
VALID_CATEGORIES = {"reasoning", "response", "meta", "tone", "knowledge_attitude"}


class Personality:
    """Loads, validates, and manages the behavioral rules."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> Personality:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    def _validate(self) -> None:
        rules = self._data.get("behavioral_rules", {})
        total = sum(len(v) for v in rules.values())
        if total < RULE_COUNT_MIN:
            raise ValueError(
                f"Rule count {total} is below minimum {RULE_COUNT_MIN}"
            )
        if total > RULE_COUNT_MAX:
            raise ValueError(
                f"Rule count {total} exceeds maximum {RULE_COUNT_MAX}"
            )
        text = yaml.dump(rules, allow_unicode=True)
        approx_tokens = len(text) // 4
        if approx_tokens > TOKEN_LIMIT:
            raise ValueError(
                f"Approximate token count {approx_tokens} exceeds limit {TOKEN_LIMIT}"
            )

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def version(self) -> int:
        return self._data["version"]

    @property
    def rules(self) -> dict:
        return self._data["behavioral_rules"]

    @property
    def rule_count(self) -> int:
        return sum(len(v) for v in self.rules.values())

    @property
    def approx_tokens(self) -> int:
        """Approximate token count for all rules (same heuristic as _validate)."""
        text = yaml.dump(self.rules, allow_unicode=True)
        return len(text) // 4

    def to_yaml(self) -> str:
        return yaml.dump(
            self._data["behavioral_rules"], allow_unicode=True, default_flow_style=False,
        )

    def to_numbered_rules(self) -> str:
        """Format rules as numbered items for LLM mutation prompts.

        Output format:
            [R1] (reasoning) Break complex problems into smaller sub-problems
            [R2] (reasoning) When uncertain, consider two approaches
            ...

        Excludes added_ver/modified_ver metadata — LLMs don't need it.
        """
        lines = []
        idx = 1
        for category, rule_list in self.rules.items():
            for entry in rule_list:
                lines.append(f"[R{idx}] ({category}) {entry['rule']}")
                idx += 1
        return "\n".join(lines)

    def build_rule_index(self) -> dict[str, tuple[str, int]]:
        """Build a mapping from rule ID (R1, R2, ...) to (category, list_index).

        Used by apply_changes() to resolve rule references by number.
        """
        index: dict[str, tuple[str, int]] = {}
        idx = 1
        for category, rule_list in self.rules.items():
            for i in range(len(rule_list)):
                index[f"R{idx}"] = (category, i)
                idx += 1
        return index

    def to_full_yaml(self) -> str:
        """Serialize the full personality data (including metadata)."""
        return yaml.dump(self._data, allow_unicode=True, default_flow_style=False)

    def to_prompt_section(self) -> str:
        """Format rules for injection into the system prompt."""
        lines = []
        for category, rule_list in self.rules.items():
            lines.append(f"### {category}")
            for entry in rule_list:
                lines.append(f"- {entry['rule']}")
            lines.append("")
        return "\n".join(lines)

    def clone(self, new_id: str | None = None) -> Personality:
        cloned = Personality(copy.deepcopy(self._data))
        if new_id:
            cloned._data["id"] = new_id
        return cloned

    def save(self, path: str | Path) -> None:
        """Save personality to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self._data, f, allow_unicode=True, default_flow_style=False)
        logger.info("Personality saved to %s (v%d)", path, self.version)

    def apply_changes(
        self,
        changes: list[dict],
        max_changes: int,
        current_ver: int,
        protect_versions: int = 1,
    ) -> tuple[Personality, list[str]]:
        """Apply parsed changes to a clone of this personality.

        Supports two modes:
        - Rule ID reference (preferred): rule_id="R3" resolves via build_rule_index()
        - Legacy exact match: original="full rule text" + category (fallback)

        Cooling period: rules added or modified in recent versions cannot be changed.

        Returns (new_personality, list_of_applied_change_descriptions).
        """
        new_p = self.clone()
        applied: list[str] = []
        change_count = 0

        # Build rule index for R-number resolution
        rule_index = self.build_rule_index()

        # Process add/modify first, then delete.
        # Delete shifts list indices, so deferring it prevents
        # subsequent R-number lookups from targeting wrong rules.
        deferred_deletes: list[dict] = []

        for change in changes:
            if change_count >= max_changes:
                break

            action = change.get("action", "").lower()
            rule_id = change.get("rule_id", "").upper().strip()
            category = change.get("category", "")

            if action == "add":
                if category not in VALID_CATEGORIES:
                    logger.warning("Skipping add: unknown category '%s'", category)
                    continue
                new_text = change.get("new", "")
                if not new_text:
                    continue
                if category not in new_p.rules:
                    new_p.rules[category] = []
                new_p.rules[category].append({
                    "rule": new_text,
                    "added_ver": current_ver,
                    "modified_ver": None,
                })
                applied.append(f"ADD [{category}]: {new_text}")
                change_count += 1

            elif action == "modify":
                new_text = change.get("new", "")
                if not new_text:
                    continue
                entry, resolved_cat = self._resolve_rule(
                    rule_id, category, change.get("original", ""),
                    new_p, rule_index,
                )
                if entry is None:
                    continue
                # Cooling period check
                last_changed = max(
                    entry.get("added_ver", 0) or 0,
                    entry.get("modified_ver", 0) or 0,
                )
                if current_ver - last_changed < protect_versions:
                    logger.info(
                        "Skipping modification of recently changed rule %s: %s",
                        rule_id or "(legacy)", entry["rule"][:50],
                    )
                    continue
                old_text = entry["rule"][:40]
                entry["rule"] = new_text
                entry["modified_ver"] = current_ver
                applied.append(f"MODIFY [{resolved_cat}] {rule_id}: {old_text} -> {new_text[:40]}")
                change_count += 1

            elif action == "delete":
                # Defer to avoid index shift affecting subsequent lookups
                deferred_deletes.append(change)

        # Apply deferred deletes
        for change in deferred_deletes:
            if change_count >= max_changes:
                break

            rule_id = change.get("rule_id", "").upper().strip()
            category = change.get("category", "")
            entry, resolved_cat = self._resolve_rule(
                rule_id, category, change.get("original", ""),
                new_p, rule_index,
            )
            if entry is None:
                continue
            # Cooling period check
            added_v = entry.get("added_ver", 0) or 0
            if current_ver - added_v < protect_versions:
                logger.info(
                    "Skipping deletion of recently added rule %s: %s",
                    rule_id or "(legacy)", entry["rule"][:50],
                )
                continue
            new_p.rules[resolved_cat].remove(entry)
            applied.append(f"DELETE [{resolved_cat}] {rule_id}: {entry['rule'][:60]}")
            change_count += 1

        # Validate after changes
        try:
            new_p._validate()
        except ValueError as e:
            logger.warning("Changes resulted in invalid personality: %s", e)
            return self.clone(), []

        # Update version metadata
        new_p._data["version"] = current_ver
        new_p._data["previous_version"] = self.version
        new_p._data["updated_at"] = datetime.now().isoformat()

        return new_p, applied

    @staticmethod
    def _resolve_rule(
        rule_id: str,
        category: str,
        original: str,
        personality: Personality,
        rule_index: dict[str, tuple[str, int]],
    ) -> tuple[dict | None, str]:
        """Resolve a rule by ID or legacy exact match.

        Returns (rule_entry, category) or (None, "") if not found.
        """
        # Preferred: resolve by rule ID (R1, R2, ...)
        if rule_id and rule_id in rule_index:
            cat, idx = rule_index[rule_id]
            if cat in personality.rules and idx < len(personality.rules[cat]):
                return personality.rules[cat][idx], cat
            logger.warning(
                "Rule %s resolved to %s[%d] but index is out of range "
                "(category has %d rules). Rule may have been deleted by a prior change.",
                rule_id, cat, idx, len(personality.rules.get(cat, [])),
            )
            return None, ""

        # Legacy fallback: exact match by original text
        if original and category:
            for entry in personality.rules.get(category, []):
                if entry["rule"] == original:
                    return entry, category
            logger.warning(
                "Legacy match failed for '%s' in %s. "
                "No exact match found among %d rules.",
                original[:60], category,
                len(personality.rules.get(category, [])),
            )

        # Both methods failed
        if rule_id:
            logger.warning("Unknown rule_id '%s' (valid: %s)", rule_id, ", ".join(sorted(rule_index.keys())))
        return None, ""
