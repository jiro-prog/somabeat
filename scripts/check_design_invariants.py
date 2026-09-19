#!/usr/bin/env python3
"""Design invariant checker — enforces structural constraints from shared_field_design.md.

Runs as a Claude Code hook after edits to shared_state/ or bridge/ files,
and can also be invoked manually.

Exit code 0 = all checks pass, 1 = violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VIOLATIONS: list[str] = []


def fail(file: Path, line_num: int, rule: str, detail: str) -> None:
    rel = file.relative_to(ROOT)
    VIOLATIONS.append(f"  {rel}:{line_num}  [{rule}] {detail}")


# ---------------------------------------------------------------------------
# Rule 1: Signal must be frozen=True (shared_field_design.md §3.1)
# ---------------------------------------------------------------------------
def check_signal_frozen():
    f = ROOT / "shared_state" / "interface.py"
    if not f.exists():
        return
    lines = f.read_text().splitlines()
    in_signal = False
    for i, line in enumerate(lines, 1):
        if re.match(r"^@dataclass", line):
            in_signal = True
            decorator_line = i
            decorator_text = line
        elif in_signal and re.match(r"^class Signal\b", line):
            if "frozen=True" not in decorator_text:
                fail(f, decorator_line, "SIGNAL_FROZEN",
                     "Signal must be @dataclass(frozen=True)")
            in_signal = False
        elif in_signal and re.match(r"^class ", line):
            in_signal = False


# ---------------------------------------------------------------------------
# Rule 2: Signal must not have extra/trace fields (design principle 5.3)
# ---------------------------------------------------------------------------
def check_signal_no_text_fields():
    f = ROOT / "shared_state" / "interface.py"
    if not f.exists():
        return
    lines = f.read_text().splitlines()
    in_signal_class = False
    for i, line in enumerate(lines, 1):
        if re.match(r"^class Signal\b", line):
            in_signal_class = True
        elif in_signal_class and re.match(r"^class ", line):
            in_signal_class = False
        elif in_signal_class:
            stripped = line.strip()
            if re.match(r"extra\s*:", stripped):
                fail(f, i, "NO_EXTRA",
                     "Signal must not have 'extra' field — "
                     "no text payload in field (design principle 5.3)")
            if re.match(r"trace\s*:", stripped):
                fail(f, i, "NO_TRACE",
                     "Signal must not have 'trace' field — "
                     "no text in field (design principle 5.3)")


# ---------------------------------------------------------------------------
# Rule 3: PurgeCriteria must not have origin filters (non-directionality §6.1)
# ---------------------------------------------------------------------------
def check_purge_no_origin():
    f = ROOT / "shared_state" / "interface.py"
    if not f.exists():
        return
    lines = f.read_text().splitlines()
    in_purge = False
    for i, line in enumerate(lines, 1):
        if re.match(r"^class PurgeCriteria\b", line):
            in_purge = True
        elif in_purge and re.match(r"^class ", line):
            in_purge = False
        elif in_purge:
            stripped = line.strip()
            if re.match(r"origin_", stripped):
                fail(f, i, "PURGE_NO_ORIGIN",
                     f"PurgeCriteria must not have origin filter: {stripped.split(':')[0]} "
                     "— non-directionality principle (§6.1)")


# ---------------------------------------------------------------------------
# Rule 4: FieldObserver must not have on_sense (sense is deprecated)
# ---------------------------------------------------------------------------
def check_observer_no_sense():
    f = ROOT / "shared_state" / "interface.py"
    if not f.exists():
        return
    lines = f.read_text().splitlines()
    in_observer = False
    for i, line in enumerate(lines, 1):
        if re.match(r"^class FieldObserver\b", line):
            in_observer = True
        elif in_observer and re.match(r"^class ", line):
            in_observer = False
        elif in_observer and "on_sense" in line:
            fail(f, i, "NO_ON_SENSE",
                 "FieldObserver must not have on_sense — sense is deprecated")


# ---------------------------------------------------------------------------
# Rule 5: No extra_json in backend serialization
# ---------------------------------------------------------------------------
def check_backend_no_extra_json():
    backend_dir = ROOT / "shared_state" / "backends"
    if not backend_dir.exists():
        return
    for f in backend_dir.glob("*.py"):
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if "extra_json" in line and not line.strip().startswith("#"):
                fail(f, i, "NO_EXTRA_JSON",
                     "Backend must not serialize/deserialize extra_json")


# ---------------------------------------------------------------------------
# Rule 6: No query_text in observer (no text in logs, §7.2)
# ---------------------------------------------------------------------------
def check_observer_no_text():
    f = ROOT / "shared_state" / "observer.py"
    if not f.exists():
        return
    lines = f.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if "query_text" in line and not line.strip().startswith("#"):
            fail(f, i, "OBSERVER_NO_TEXT",
                 "Observer must not accept or log query_text — "
                 "no text in logs (§7.2)")


# ---------------------------------------------------------------------------
# Rule 7: No Signal.create(..., extra=) calls anywhere
# ---------------------------------------------------------------------------
def check_no_signal_extra_usage():
    for f in ROOT.rglob("*.py"):
        if ".venv" in f.parts or "__pycache__" in f.parts:
            continue
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            if re.search(r"extra\s*=\s*\{", line) and "Signal" in "".join(
                lines[max(0, i - 5): i + 1]
            ):
                fail(f, i, "NO_EXTRA_USAGE",
                     "Do not pass extra= to Signal — field carries embeddings only")


# ---------------------------------------------------------------------------
# Rule 8: Q&A pairs must not go through the field (§6.4, §7.4)
# ---------------------------------------------------------------------------
def check_qa_not_via_field():
    for f in (ROOT / "bridge").rglob("*.py"):
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            if "emit" in line.lower() and "qa" in line.lower():
                if "field" in line.lower() or "emit(" in line:
                    fail(f, i, "QA_NOT_VIA_FIELD",
                         "Q&A pairs must be transferred via SQLite, not field.emit "
                         "(§6.4, §7.4)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    check_signal_frozen()
    check_signal_no_text_fields()
    check_purge_no_origin()
    check_observer_no_sense()
    check_backend_no_extra_json()
    check_observer_no_text()
    check_no_signal_extra_usage()
    check_qa_not_via_field()

    if VIOLATIONS:
        print(f"DESIGN INVARIANT VIOLATIONS ({len(VIOLATIONS)}):")
        for v in VIOLATIONS:
            print(v)
        return 1
    else:
        print("All design invariants pass.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
