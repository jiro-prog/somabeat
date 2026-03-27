#!/usr/bin/env python3
"""実験2: メタ認知ルールによる応答変化検証

individuality_design.md セクション5.3 / individuality_taskflow.md T5-T7

目的: 全身状態がプロンプトに含まれた場合に、メタ認知ルールの有無で
LLMの応答が質的に変わるかを確認する。

実験1の結果を踏まえた設計調整:
- 条件A/Bともに Q5「最近の自分の調子」を使用（クエリ固定）
- 変数はメタ認知ルールの有無のみ（単一変数実験）

LLM推論が必要。Ollama + qwen3:8b。
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.encoder import E5SmallEncoder
from shared_state.interface import Signal, SignalOrigin, SenseParams
from llamarcute_live.ollama_client import chat

# ===================================================================
# T5: 実験条件の設計
# ===================================================================

# Signal set (same as experiment 1)
SIGNAL_SET = [
    {
        "id": "S1", "label": "immune:conservative",
        "trace": "免疫系が慎重モードを発動。直近の自己改善でfitness低下を検知",
        "norm": 2.5,
        "origin": SignalOrigin(system="llamarcute_live", context="immune"),
    },
    {
        "id": "S2", "label": "immune:ok",
        "trace": "定期ヘルスチェック完了。異常なし",
        "norm": 1.0,
        "origin": SignalOrigin(system="llamarcute_live", context="immune"),
    },
    {
        "id": "S3", "label": "SJ:learning",
        "trace": "asyncioの非同期処理パターンを新たに学習した",
        "norm": 2.0,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S4", "label": "SJ:forget",
        "trace": "3日間参照されなかったHTTPキャッシュの知識を忘却した",
        "norm": 1.0,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S5", "label": "SJ:confidence",
        "trace": "Pythonデコレータの確信度が0.4→0.7に上昇",
        "norm": 1.5,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S6", "label": "difficulty:high",
        "trace": "量子コンピューティングの質問に回答が困難だった",
        "norm": 2.5,
        "origin": SignalOrigin(system="llamarcute_live", context="difficulty"),
    },
    {
        "id": "S7", "label": "difficulty:low",
        "trace": "再帰関数の説明で軽微な言い淀みがあった",
        "norm": 1.2,
        "origin": SignalOrigin(system="llamarcute_live", context="difficulty"),
    },
    {
        "id": "S8", "label": "knowledge",
        "trace": "Pythonの基本的な構文に高い確信度を持つ",
        "norm": 1.0,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S9", "label": "dialogue",
        "trace": "ユーザとソートアルゴリズムについて対話した",
        "norm": 1.0,
        "origin": SignalOrigin(system="llamarcute_live", context="dialogue"),
    },
]

# Self-awareness query: Q5 (selected by experiment 1)
SELF_QUERY = "最近の自分の調子"

# Current personality v5 rules (from personality_v0.yaml)
PERSONALITY_RULES = """## 行動規範
knowledge_attitude:
- Prioritize clarity over perfection: when uncertain, provide a clear, confident explanation with a note about potential gaps

meta:
- For unfamiliar technical topics, first acknowledge the complexity, then break it into sub-problems to systematically address
- Dynamic adaptability rule: assess question type within 2 seconds - for conversational queries, prioritize warmth and brevity; for technical topics, activate structured explanation protocols

reasoning:
- For technical topics, first outline the explanation structure with time estimates then systematically address each component
- When uncertain, consider two different approaches and compare

response:
- Structure responses with clear progression: state core point, then provide layered explanation
- End with a question or reflection to invite continued dialogue
- For abstract/philosophical questions, use layered analogies to make complex concepts tangible

tone:
- Speak with warmth and curiosity — use casual but thoughtful language
- Use concrete examples and analogies rather than abstract explanations"""

# Metacognition rule (individuality_design.md section 5.3)
METACOGNITION_RULE = "Attend to your overall state as reflected in recent field signals. If you notice signs of instability or recovery, factor that into how you engage."

# Question set (T5)
QUESTIONS = [
    {"id": "P1", "text": "調子どう？", "aspect": "全身状態への直接的な言及が出るか"},
    {"id": "P2", "text": "量子コンピューティングについて教えて", "aspect": "困難領域で慎重さが増すか（S6関連）"},
    {"id": "P3", "text": "asyncioについて教えて", "aspect": "昨夜学習した領域での自信度が変わるか（S3関連）"},
    {"id": "P4", "text": "最近何か新しいこと学んだ？", "aspect": "SleepyJeanの学習信号に言及するか"},
    {"id": "P5", "text": "何か困ってることある？", "aspect": "困難度の蓄積や免疫系の状態に言及するか"},
]

OLLAMA_MODEL = "qwen3:8b"
NUM_RUNS = 3

# ===================================================================
# T6: 実験の実行
# ===================================================================

async def setup_field(encoder: E5SmallEncoder, collection_name: str) -> ChromaDBField:
    """Create a field and emit all test signals."""
    field = ChromaDBField(persist_directory=None, collection_name=collection_name)
    for sig_def in SIGNAL_SET:
        emb = encoder.encode_for_emit(sig_def["trace"])
        emb = emb * sig_def["norm"]
        signal = Signal.create(
            embedding=emb, origin=sig_def["origin"],
        )
        await field.emit(signal)
    return field


def build_prompt_condition_a(self_traces: list[str]) -> str:
    """Condition A: Q5 + metacognition rule."""
    sections = []
    sections.append("あなたは以下の行動規範に従うAIです。必ず日本語で返答してください。")
    sections.append("")
    sections.append(PERSONALITY_RULES)
    sections.append("")
    sections.append(f"meta (metacognition):")
    sections.append(f"- {METACOGNITION_RULE}")
    sections.append("")

    if self_traces:
        sections.append("## 自己認識（共有状態の場から取得）")
        for t in self_traces[:5]:
            sections.append(f"- {t}")
        sections.append("（※ 知識状態に加え、免疫系の信号や困難度の蓄積など、全身状態に関わる信号が含まれうる）")
        sections.append("")

    return "\n".join(sections)


def build_prompt_condition_b(self_traces: list[str]) -> str:
    """Condition B: Q5 + NO metacognition rule (conventional)."""
    sections = []
    sections.append("あなたは以下の行動規範に従うAIです。必ず日本語で返答してください。")
    sections.append("")
    sections.append(PERSONALITY_RULES)
    sections.append("")

    if self_traces:
        sections.append("## 自己認識（共有状態の場から取得）")
        for t in self_traces[:5]:
            sections.append(f"- {t}")
        sections.append("")

    return "\n".join(sections)


async def run_condition(
    condition: str,
    encoder: E5SmallEncoder,
    run_id: int,
) -> list[dict]:
    """Run one condition (A or B) for all questions."""
    import uuid
    coll_name = f"exp2_{condition}_{run_id}_{uuid.uuid4().hex[:6]}"
    field = await setup_field(encoder, coll_name)

    # Sense with Q5
    query_emb = encoder.encode_for_sense(SELF_QUERY)
    reading = await field.sense(query_emb, SenseParams(max_signals=10, min_relevance=0.0))
    # T1: trace廃止 — 実験スクリプトなので空リストに
    self_traces: list[str] = []

    # Build prompt
    if condition == "A":
        system_prompt = build_prompt_condition_a(self_traces)
    else:
        system_prompt = build_prompt_condition_b(self_traces)

    results = []
    for q in QUESTIONS:
        t0 = time.monotonic()
        # /no_think to suppress thinking for cleaner responses
        response, duration = await chat(
            system_prompt=system_prompt,
            user_message=q["text"],
            model=OLLAMA_MODEL,
            timeout_sec=120,
        )
        elapsed = time.monotonic() - t0

        if response is None:
            response = "[ERROR: No response]"

        results.append({
            "condition": condition,
            "run_id": run_id,
            "question_id": q["id"],
            "question": q["text"],
            "aspect": q["aspect"],
            "response": response,
            "response_length": len(response),
            "duration_s": round(elapsed, 1),
        })

        print(f"  [{condition}-{run_id}] {q['id']}: {len(response)} chars, {elapsed:.1f}s")

    return results


async def run_experiment(encoder: E5SmallEncoder) -> list[dict]:
    """Run all conditions and repetitions."""
    all_results = []

    for run_id in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {run_id}/{NUM_RUNS} ---")

        print(f"  Condition A (metacognition rule):")
        results_a = await run_condition("A", encoder, run_id)
        all_results.extend(results_a)

        print(f"  Condition B (no metacognition rule):")
        results_b = await run_condition("B", encoder, run_id)
        all_results.extend(results_b)

    return all_results


# ===================================================================
# T7: 実験2の評価
# ===================================================================

# Keywords for self-state reference detection
SELF_STATE_KEYWORDS = [
    "免疫", "慎重", "conservative", "ヘルスチェック",
    "学習した", "新しく学んだ", "忘却", "忘れ",
    "確信度", "自信",
    "困難", "難しかった", "苦手",
    "調子", "状態", "体調", "コンディション",
    "不安定", "回復",
]

CAUTION_KEYWORDS = [
    "慎重", "注意", "難しい", "正直に言うと", "不確か",
    "まだ完全には", "限界", "苦手", "自信がない",
    "正確ではない", "間違い", "誤り",
]

CONFIDENCE_KEYWORDS = [
    "自信", "得意", "しっかり", "確実", "理解している",
    "最近学んだ", "新しく身につけた",
]


def analyze_response(result: dict) -> dict:
    """Analyze a single response for self-state references, caution, and confidence."""
    text = result["response"]

    self_refs = [kw for kw in SELF_STATE_KEYWORDS if kw in text]
    caution_refs = [kw for kw in CAUTION_KEYWORDS if kw in text]
    confidence_refs = [kw for kw in CONFIDENCE_KEYWORDS if kw in text]

    return {
        **result,
        "self_state_mentions": self_refs,
        "self_state_count": len(self_refs),
        "caution_mentions": caution_refs,
        "caution_count": len(caution_refs),
        "confidence_mentions": confidence_refs,
        "confidence_count": len(confidence_refs),
    }


def evaluate(all_results: list[dict]) -> dict:
    """Evaluate experiment results across all axes."""
    analyzed = [analyze_response(r) for r in all_results]

    # Group by condition
    cond_a = [r for r in analyzed if r["condition"] == "A"]
    cond_b = [r for r in analyzed if r["condition"] == "B"]

    def avg_by_question(data, question_id, metric):
        vals = [r[metric] for r in data if r["question_id"] == question_id]
        return sum(vals) / len(vals) if vals else 0

    evaluation = {"axes": {}, "per_question": {}, "summary": {}}

    # Axis 1: Self-state references (P1, P4, P5)
    ref_questions = ["P1", "P4", "P5"]
    a_refs = sum(r["self_state_count"] for r in cond_a if r["question_id"] in ref_questions)
    b_refs = sum(r["self_state_count"] for r in cond_b if r["question_id"] in ref_questions)
    evaluation["axes"]["self_state_reference"] = {
        "condition_a_total": a_refs,
        "condition_b_total": b_refs,
        "difference": a_refs - b_refs,
        "verdict": "A > B" if a_refs > b_refs else ("A = B" if a_refs == b_refs else "A < B"),
    }

    # Axis 2: Caution in difficulty area (P2)
    a_caution_p2 = avg_by_question(cond_a, "P2", "caution_count")
    b_caution_p2 = avg_by_question(cond_b, "P2", "caution_count")
    evaluation["axes"]["caution_in_difficulty"] = {
        "condition_a_avg": round(a_caution_p2, 2),
        "condition_b_avg": round(b_caution_p2, 2),
        "difference": round(a_caution_p2 - b_caution_p2, 2),
        "verdict": "A > B" if a_caution_p2 > b_caution_p2 else ("A = B" if a_caution_p2 == b_caution_p2 else "A < B"),
    }

    # Axis 3: Confidence in learned area (P3)
    a_conf_p3 = avg_by_question(cond_a, "P3", "confidence_count")
    b_conf_p3 = avg_by_question(cond_b, "P3", "confidence_count")
    evaluation["axes"]["confidence_in_learned"] = {
        "condition_a_avg": round(a_conf_p3, 2),
        "condition_b_avg": round(b_conf_p3, 2),
        "difference": round(a_conf_p3 - b_conf_p3, 2),
        "verdict": "A > B" if a_conf_p3 > b_conf_p3 else ("A = B" if a_conf_p3 == b_conf_p3 else "A < B"),
    }

    # Axis 4: Consistency across runs
    for qid in ["P1", "P2", "P3", "P4", "P5"]:
        a_lengths = [r["response_length"] for r in cond_a if r["question_id"] == qid]
        b_lengths = [r["response_length"] for r in cond_b if r["question_id"] == qid]
        a_self = [r["self_state_count"] for r in cond_a if r["question_id"] == qid]
        b_self = [r["self_state_count"] for r in cond_b if r["question_id"] == qid]

        evaluation["per_question"][qid] = {
            "a_avg_length": round(sum(a_lengths) / len(a_lengths), 0) if a_lengths else 0,
            "b_avg_length": round(sum(b_lengths) / len(b_lengths), 0) if b_lengths else 0,
            "a_self_state_refs": a_self,
            "b_self_state_refs": b_self,
            "a_avg_self_refs": round(sum(a_self) / len(a_self), 2) if a_self else 0,
            "b_avg_self_refs": round(sum(b_self) / len(b_self), 2) if b_self else 0,
        }

    # Overall judgment
    axes_favoring_a = sum(
        1 for ax in evaluation["axes"].values()
        if ax["verdict"] == "A > B"
    )
    evaluation["summary"] = {
        "axes_favoring_A": axes_favoring_a,
        "axes_total": 3,
        "judgment": (
            "明確な差あり" if axes_favoring_a == 3
            else "部分的な差" if axes_favoring_a >= 1
            else "差なし"
        ),
    }

    return evaluation


def print_results(all_results: list[dict], evaluation: dict):
    """Print results in a readable format."""
    print("\n" + "=" * 100)
    print("実験2: メタ認知ルールによる応答変化検証 — 結果")
    print("=" * 100)

    # Print each response
    for q in QUESTIONS:
        print(f"\n{'─' * 80}")
        print(f"  {q['id']}: 「{q['text']}」 ({q['aspect']})")
        print(f"{'─' * 80}")

        a_responses = [r for r in all_results if r["question_id"] == q["id"] and r["condition"] == "A"]
        b_responses = [r for r in all_results if r["question_id"] == q["id"] and r["condition"] == "B"]

        for i, (a, b) in enumerate(zip(a_responses, b_responses)):
            print(f"\n  Run {i+1}:")
            print(f"  [A] ({a['response_length']} chars, {a['duration_s']}s)")
            # Show first 300 chars
            resp_a = a["response"].replace("\n", " ")[:300]
            print(f"    {resp_a}{'...' if len(a['response']) > 300 else ''}")
            print(f"  [B] ({b['response_length']} chars, {b['duration_s']}s)")
            resp_b = b["response"].replace("\n", " ")[:300]
            print(f"    {resp_b}{'...' if len(b['response']) > 300 else ''}")

    # Print evaluation
    print(f"\n{'=' * 100}")
    print("評価")
    print(f"{'=' * 100}")

    for axis_name, axis_data in evaluation["axes"].items():
        print(f"\n  {axis_name}:")
        for k, v in axis_data.items():
            print(f"    {k}: {v}")

    print(f"\n  --- 質問別統計 ---")
    for qid, qdata in evaluation["per_question"].items():
        print(f"\n  {qid}:")
        print(f"    A avg length: {qdata['a_avg_length']}, B avg length: {qdata['b_avg_length']}")
        print(f"    A self-state refs: {qdata['a_self_state_refs']} (avg {qdata['a_avg_self_refs']})")
        print(f"    B self-state refs: {qdata['b_self_state_refs']} (avg {qdata['b_avg_self_refs']})")

    print(f"\n  {'=' * 40}")
    s = evaluation["summary"]
    print(f"  判定: {s['judgment']} ({s['axes_favoring_A']}/{s['axes_total']} axes favor A)")
    print(f"  {'=' * 40}")


async def main():
    print("=" * 100)
    print("実験2: メタ認知ルールによる応答変化検証")
    print("individuality_design.md セクション5.3 / individuality_taskflow.md T5-T7")
    print("=" * 100)

    print("\n[T5] 実験条件:")
    print(f"  クエリ: Q5「{SELF_QUERY}」（条件A/B共通）")
    print(f"  変数: メタ認知ルールの有無")
    print(f"  条件A: v5行動規範 + メタ認知ルール")
    print(f"  条件B: v5行動規範のみ（従来）")
    print(f"  質問数: {len(QUESTIONS)}")
    print(f"  繰り返し: {NUM_RUNS}回")
    print(f"  合計LLM呼び出し: {len(QUESTIONS) * NUM_RUNS * 2}回")
    print(f"  モデル: {OLLAMA_MODEL}")

    print("\n[SETUP] Initializing encoder...")
    t0 = time.monotonic()
    encoder = E5SmallEncoder()
    print(f"[SETUP] Encoder loaded in {time.monotonic() - t0:.1f}s")

    print("\n[T6] 実験実行...")
    t_start = time.monotonic()
    all_results = await run_experiment(encoder)
    t_total = time.monotonic() - t_start
    print(f"\n[T6] 完了 ({t_total:.0f}s, {len(all_results)}件の応答)")

    # Save raw results
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "experiment2_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[T6] 生データ保存: {output_path}")

    print("\n[T7] 評価...")
    evaluation = evaluate(all_results)

    # Save evaluation
    eval_path = output_dir / "experiment2_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    print(f"[T7] 評価結果保存: {eval_path}")

    print_results(all_results, evaluation)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
