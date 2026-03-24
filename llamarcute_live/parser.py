"""LLM output parsing and answer extraction.

Ported from llamarcute/src/parser.py with minimal changes.
"""

import os
import re
import subprocess
import tempfile


def extract_answer(response: str) -> str | None:
    """Extract the text after 'ANSWER:' marker.

    Strips <think> blocks as a safety measure (Ollama normally returns
    thinking content in a separate field).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    match = re.search(r"ANSWER:\s*(.+?)$", cleaned, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None


def extract_code(response: str) -> str | None:
    """Extract Python code from response.

    Looks for markdown code blocks first, then falls back to ANSWER: marker.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    match = re.search(r"```python\s*\n(.+?)```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"ANSWER:\s*\n(.+?)$", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def grade_math(response: str, expected: float) -> bool:
    """Grade a math response by comparing extracted number to expected."""
    answer_str = extract_answer(response)
    if answer_str is None:
        return False
    try:
        cleaned = answer_str.replace(",", "").replace("$", "").strip()
        answer_val = float(cleaned)
        return abs(answer_val - expected) < 0.01
    except ValueError:
        return False


def grade_logic(response: str, expected: str) -> bool:
    """Grade a logic response by comparing extracted choice letter."""
    answer_str = extract_answer(response)
    if answer_str is None:
        return False
    match = re.search(r"[A-Da-d]", answer_str)
    if match:
        return match.group(0).upper() == expected.upper()
    return False


def grade_code(response: str, test_cases: list[dict], timeout: int = 10) -> dict:
    """Grade a code response by running test cases in a sandbox.

    Returns dict with:
        - score: float (0.0 to 1.0, fraction of tests passed)
        - errors: list[str] (stderr from failed tests, truncated)
    """
    code = extract_code(response)
    if code is None:
        return {"score": 0.0, "errors": ["No code block found in response"]}

    passed = 0
    errors = []
    for tc in test_cases:
        if "assert" in tc:
            test_code = f"{code}\n\n{tc['assert']}"
        else:
            test_code = (
                f"{code}\n\n"
                f"result = {tc['function_call']}\n"
                f'assert str(result) == "{tc["expected"]}", f"Got {{result}}"'
            )
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
            ) as f:
                f.write(test_code)
                tmp_path = f.name

            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                passed += 1
            else:
                stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    errors.append(stderr_text[-200:])
        except subprocess.TimeoutExpired:
            errors.append(f"Timeout: execution exceeded {timeout} seconds")
        except Exception as e:
            errors.append(f"Execution error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return {
        "score": passed / len(test_cases) if test_cases else 0.0,
        "errors": errors,
    }
