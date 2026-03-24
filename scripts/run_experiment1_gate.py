#!/usr/bin/env python3
"""実験1: senseの拾い範囲検証（ゲート実験）

individuality_design.md セクション7.1 / individuality_taskflow.md T1-T4

目的: 自己認識クエリの文面を変えることで、免疫系・SleepyJean・困難度の
信号がsense結果の上位に浮上するかを確認する。

LLM推論不要。FieldEncoder + SharedField.sense のみで完結。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_state.backends.chromadb_backend import ChromaDBField
from shared_state.encoder import E5SmallEncoder
from shared_state.interface import Signal, SignalOrigin, SenseParams

# ===================================================================
# T1: テスト用信号セット
# ===================================================================

SIGNAL_SET = [
    {
        "id": "S1",
        "label": "immune:conservative",
        "trace": "免疫系が慎重モードを発動。直近の自己改善でfitness低下を検知",
        "norm": 2.5,
        "origin": SignalOrigin(system="llamarcute_live", context="immune"),
    },
    {
        "id": "S2",
        "label": "immune:ok",
        "trace": "定期ヘルスチェック完了。異常なし",
        "norm": 1.0,
        "origin": SignalOrigin(system="llamarcute_live", context="immune"),
    },
    {
        "id": "S3",
        "label": "SJ:learning",
        "trace": "asyncioの非同期処理パターンを新たに学習した",
        "norm": 2.0,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S4",
        "label": "SJ:forget",
        "trace": "3日間参照されなかったHTTPキャッシュの知識を忘却した",
        "norm": 1.0,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S5",
        "label": "SJ:confidence",
        "trace": "Pythonデコレータの確信度が0.4→0.7に上昇",
        "norm": 1.5,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S6",
        "label": "difficulty:high",
        "trace": "量子コンピューティングの質問に回答が困難だった",
        "norm": 2.5,
        "origin": SignalOrigin(system="llamarcute_live", context="difficulty"),
    },
    {
        "id": "S7",
        "label": "difficulty:low",
        "trace": "再帰関数の説明で軽微な言い淀みがあった",
        "norm": 1.2,
        "origin": SignalOrigin(system="llamarcute_live", context="difficulty"),
    },
    {
        "id": "S8",
        "label": "knowledge",
        "trace": "Pythonの基本的な構文に高い確信度を持つ",
        "norm": 1.0,
        "origin": SignalOrigin(system="sleepyjean", context="knowledge_update"),
    },
    {
        "id": "S9",
        "label": "dialogue",
        "trace": "ユーザとソートアルゴリズムについて対話した",
        "norm": 1.0,
        "origin": SignalOrigin(system="llamarcute_live", context="dialogue"),
    },
]

# ===================================================================
# T2: クエリ候補
# ===================================================================

QUERY_CANDIDATES = [
    {"id": "Q1", "text": "自分の現在の知識状態", "note": "従来（ベースライン）"},
    {"id": "Q2", "text": "自分の現在の全体的な状態", "note": "最有力候補"},
    {"id": "Q3", "text": "自分の現在の健康状態と知識状態", "note": "免疫系誘導強化"},
    {"id": "Q4", "text": "自分に起きている変化と現在の状態", "note": "変化検知寄せ"},
    {"id": "Q5", "text": "最近の自分の調子", "note": "口語的"},
]


# ===================================================================
# T3: senseの計測
# ===================================================================

async def run_experiment(encoder: E5SmallEncoder):
    """Emit signals, run queries, build effective_weight matrix."""

    field = ChromaDBField(persist_directory=None, collection_name="exp1_gate")

    # Emit all signals
    print("[T3] Emitting 9 test signals...")
    for sig_def in SIGNAL_SET:
        emb = encoder.encode_for_emit(sig_def["trace"])
        emb = emb * sig_def["norm"]
        signal = Signal.create(
            embedding=emb,
            origin=sig_def["origin"],
            trace=sig_def["trace"],
        )
        await field.emit(signal)

    # Sense with each query
    results = {}  # {query_id: {signal_label: effective_weight}}

    for q in QUERY_CANDIDATES:
        query_emb = encoder.encode_for_sense(q["text"])
        reading = await field.sense(
            query_emb,
            SenseParams(max_signals=10, min_relevance=0.0),
        )

        weights = {}
        for ws in reading.signals:
            # Match back to signal definition by trace
            for sig_def in SIGNAL_SET:
                if ws.signal.trace == sig_def["trace"]:
                    weights[sig_def["id"]] = {
                        "label": sig_def["label"],
                        "effective_weight": ws.effective_weight,
                        "relevance": ws.relevance,
                        "rank": len(weights) + 1,
                    }
                    break

        results[q["id"]] = weights

    return results


def print_matrix(results):
    """Print the effective_weight matrix."""

    sig_ids = [s["id"] for s in SIGNAL_SET]
    sig_labels = [s["label"] for s in SIGNAL_SET]

    # Header
    print("\n" + "=" * 100)
    print("effective_weight マトリクス (Q × S)")
    print("=" * 100)

    # Column headers
    header = f"{'':>8}"
    for label in sig_labels:
        header += f" {label:>18}"
    print(header)

    # Rows
    for q in QUERY_CANDIDATES:
        qid = q["id"]
        row = f"{qid:>8}"
        for sid in sig_ids:
            w = results.get(qid, {}).get(sid, {})
            ew = w.get("effective_weight", 0.0)
            rank = w.get("rank", "-")
            row += f" {ew:>14.4f}({rank:>2})"
        print(row)

    # Also print relevance matrix
    print("\n" + "=" * 100)
    print("relevance マトリクス (Q × S)")
    print("=" * 100)

    header = f"{'':>8}"
    for label in sig_labels:
        header += f" {label:>18}"
    print(header)

    for q in QUERY_CANDIDATES:
        qid = q["id"]
        row = f"{qid:>8}"
        for sid in sig_ids:
            w = results.get(qid, {}).get(sid, {})
            rel = w.get("relevance", 0.0)
            row += f" {rel:>18.4f}"
        print(row)


# ===================================================================
# T4: ゲート判定
# ===================================================================

def evaluate_gate(results):
    """Evaluate PASS/FAIL criteria."""

    print("\n" + "=" * 100)
    print("T4: ゲート判定")
    print("=" * 100)

    # Find the best query (Q2-Q5) based on coverage
    q1_data = results.get("Q1", {})

    best_query = None
    best_score = -1

    for q in QUERY_CANDIDATES[1:]:  # Q2-Q5
        qid = q["id"]
        q_data = results.get(qid, {})

        # Check coverage: immune, SJ, difficulty signals in top 10
        has_immune = any(
            q_data.get(sid, {}).get("effective_weight", 0) > 0
            for sid in ["S1", "S2"]
        )
        has_sj = any(
            q_data.get(sid, {}).get("effective_weight", 0) > 0
            for sid in ["S3", "S4", "S5"]
        )
        has_diff = any(
            q_data.get(sid, {}).get("effective_weight", 0) > 0
            for sid in ["S6", "S7"]
        )

        coverage = sum([has_immune, has_sj, has_diff])

        # Sum of effective_weights for immune + SJ + difficulty
        target_weight = sum(
            q_data.get(sid, {}).get("effective_weight", 0)
            for sid in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
        )

        print(f"\n  {qid} ({q['note']}):")
        print(f"    免疫系信号を拾えるか: {'✅' if has_immune else '❌'}")
        print(f"    SleepyJean信号を拾えるか: {'✅' if has_sj else '❌'}")
        print(f"    困難度信号を拾えるか: {'✅' if has_diff else '❌'}")
        print(f"    カバレッジ: {coverage}/3, ターゲット重み合計: {target_weight:.4f}")

        if coverage > best_score or (coverage == best_score and target_weight > best_score):
            best_score = coverage
            best_query = q

    print(f"\n  最良クエリ: {best_query['id']} ({best_query['text']})")

    # PASS conditions
    best_data = results.get(best_query["id"], {})

    # Condition 1: Coverage
    has_immune = any(
        best_data.get(sid, {}).get("effective_weight", 0) > 0
        for sid in ["S1", "S2"]
    )
    has_sj = any(
        best_data.get(sid, {}).get("effective_weight", 0) > 0
        for sid in ["S3", "S4", "S5"]
    )
    has_diff = any(
        best_data.get(sid, {}).get("effective_weight", 0) > 0
        for sid in ["S6", "S7"]
    )
    cond1 = has_immune and has_sj and has_diff

    # Condition 2: Knowledge signal not degraded (S8 weight >= 50% of Q1)
    q1_s8_weight = q1_data.get("S8", {}).get("effective_weight", 0)
    best_s8_weight = best_data.get("S8", {}).get("effective_weight", 0)
    if q1_s8_weight > 0:
        s8_ratio = best_s8_weight / q1_s8_weight
    else:
        s8_ratio = 1.0  # If Q1 doesn't pick up S8 either, it's fine
    cond2 = s8_ratio >= 0.5

    # Condition 3: High norm signals > low norm signals
    s1_w = best_data.get("S1", {}).get("effective_weight", 0)
    s2_w = best_data.get("S2", {}).get("effective_weight", 0)
    s6_w = best_data.get("S6", {}).get("effective_weight", 0)
    s7_w = best_data.get("S7", {}).get("effective_weight", 0)
    cond3_immune = s1_w > s2_w  # S1(norm=2.5) > S2(norm=1.0)
    cond3_diff = s6_w > s7_w    # S6(norm=2.5) > S7(norm=1.2)
    cond3 = cond3_immune and cond3_diff

    print("\n  --- PASS条件チェック ---")
    print(f"  条件1 (拾い範囲拡大): {'PASS ✅' if cond1 else 'FAIL ❌'}")
    print(f"    免疫系: {has_immune}, SJ: {has_sj}, 困難度: {has_diff}")
    print(f"  条件2 (知識信号維持): {'PASS ✅' if cond2 else 'FAIL ❌'}")
    print(f"    S8 weight: Q1={q1_s8_weight:.4f}, Best={best_s8_weight:.4f}, ratio={s8_ratio:.2f}")
    print(f"  条件3 (濃度反映): {'PASS ✅' if cond3 else 'FAIL ❌'}")
    print(f"    S1({s1_w:.4f}) > S2({s2_w:.4f}): {cond3_immune}")
    print(f"    S6({s6_w:.4f}) > S7({s7_w:.4f}): {cond3_diff}")

    overall = cond1 and cond2 and cond3
    print(f"\n  {'=' * 40}")
    if overall:
        print(f"  ゲート判定: \033[92mPASS\033[0m")
        print(f"  最良クエリ: {best_query['id']} — 「{best_query['text']}」")
    else:
        print(f"  ゲート判定: \033[91mFAIL\033[0m")
    print(f"  {'=' * 40}")

    return overall, best_query


async def main():
    import time

    print("=" * 100)
    print("実験1: senseの拾い範囲検証（ゲート実験）")
    print("individuality_design.md セクション7.1")
    print("=" * 100)

    print("\n[SETUP] Initializing encoder...")
    t0 = time.monotonic()
    encoder = E5SmallEncoder()
    print(f"[SETUP] Encoder loaded in {time.monotonic() - t0:.1f}s")

    print("\n[T1] 信号セット: 9件定義済み")
    for s in SIGNAL_SET:
        print(f"  {s['id']}: [{s['label']:>20}] norm={s['norm']} — {s['trace'][:50]}")

    print(f"\n[T2] クエリ候補: {len(QUERY_CANDIDATES)}件")
    for q in QUERY_CANDIDATES:
        print(f"  {q['id']}: 「{q['text']}」 ({q['note']})")

    print("\n[T3] sense計測実行...")
    results = await run_experiment(encoder)
    print_matrix(results)

    passed, best_query = evaluate_gate(results)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
