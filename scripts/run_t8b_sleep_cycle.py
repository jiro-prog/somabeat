#!/usr/bin/env python3
"""T8-b 睡眠サイクル再実行 — 3件修正後のフル検証

Usage:
    cd /home/jiro/integrated-system
    source .venv/bin/activate
    python scripts/run_t8b_sleep_cycle.py

検証項目:
1. FieldAwareLLM unload
2. Ollamaヘルスチェック
3. sleep_ingest（対話ログ転送、homework追加、knowledge snapshot）
4. SleepyJean night cycle（Unsloth + LoRA学習）
5. wake_export（knowledge diff emit、Q&A転送、dream emit）
   - 修正: ensure_emit_log_table で emit_log テーブル確保
6. Ollama アンロード（keep_alive: 0）← 新規追加ステップ
7. FieldAwareLLM reload
   - 修正: try/except で graceful degradation
8. 自己改善（候補生成 + cuteness評価）
9. purge + wake_up
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/t8b_verification.log", mode="w"),
    ],
)
logger = logging.getLogger("t8b")


async def main():
    from orchestrator.orchestrator import Orchestrator, load_config

    print("=" * 60)
    print("T8-b: Sleep Cycle Verification (2nd run, post-fix)")
    print("=" * 60)

    config = load_config()

    # Ensure data directories
    Path(config["shared_state"]["chromadb"]["persist_directory"]).mkdir(parents=True, exist_ok=True)
    Path(config["llamarcute_live"]["dialogue_db_path"]).parent.mkdir(parents=True, exist_ok=True)

    print("\n[INIT] Loading Orchestrator...")
    orch = Orchestrator(config)

    # Pre-flight checks
    print("\n[PRE-FLIGHT] Checking system state...")

    # Check FieldAwareLLM
    if orch.llm is not None:
        print(f"  FieldAwareLLM: configured (model={config['llamarcute_live'].get('transformers_model', 'Qwen/Qwen3-8B')})")
    else:
        print("  FieldAwareLLM: NOT configured (use_llm=false)")

    # Check FieldReceptor
    if orch.receptor is not None:
        print(f"  FieldReceptor: loaded (field_dim={orch.receptor.field_dimensionality()}, agent_dim={orch.receptor.agent_dimensionality()})")
    else:
        print("  FieldReceptor: NOT loaded")

    # Check field state
    snap = await orch.field.snapshot()
    print(f"  Field signals: {snap.total_count}")

    # Check dialogue data
    import aiosqlite
    db_path = config["llamarcute_live"]["dialogue_db_path"]
    if Path(db_path).exists():
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM dialogue_log")
            count = (await cursor.fetchone())[0]
            print(f"  Dialogue log entries: {count}")
    else:
        print("  Dialogue DB: not found")

    # Check SleepyJean
    sj_root = Path(config["sleepyjean"]["root_path"])
    night_script = sj_root / config["sleepyjean"]["night_cycle_script"]
    print(f"  SleepyJean night script: {'exists' if night_script.exists() else 'NOT FOUND'}")
    print(f"  SleepyJean DB: {'exists' if Path(config['sleepyjean']['db_path']).exists() else 'NOT FOUND'}")

    # Check Ollama
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:11434/api/tags",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    print(f"  Ollama: reachable, models={models}")
                else:
                    print(f"  Ollama: status {resp.status}")
    except Exception as e:
        print(f"  Ollama: NOT reachable ({e})")
        print("  WARNING: Night cycle requires Ollama. Proceeding anyway...")

    # GPU check
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        print(f"  GPU: {gpu} ({total:.1f}GB total, {free:.1f}GB free)")
    else:
        print("  GPU: CUDA not available")

    print("\n" + "=" * 60)
    input("Press Enter to start sleep cycle, or Ctrl+C to abort...")
    print("=" * 60)

    # Simulate being in AWAKE state with dialogue having happened
    # (The orchestrator was just created, so dialogue.stop() etc. are safe)

    start = time.monotonic()
    print("\n[T8-b] Starting full sleep cycle via Orchestrator.enter_sleep()...\n")

    await orch.enter_sleep()

    elapsed = time.monotonic() - start
    print(f"\n[T8-b] Sleep cycle completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Post-verification
    print("\n" + "=" * 60)
    print("POST-VERIFICATION")
    print("=" * 60)

    # Check field state after
    snap_after = await orch.field.snapshot()
    print(f"  Field signals after cycle: {snap_after.total_count} (was {snap.total_count})")

    # Check emit_log in SleepyJean DB (問題8の修正確認)
    sj_db = config["sleepyjean"]["db_path"]
    if Path(sj_db).exists():
        try:
            async with aiosqlite.connect(sj_db) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM emit_log")
                count = (await cursor.fetchone())[0]
                print(f"  SleepyJean emit_log entries: {count} ✓ (table exists)")
        except Exception as e:
            print(f"  SleepyJean emit_log: ERROR - {e}")

    # Check FieldAwareLLM state (問題9の修正確認)
    if orch.llm is not None:
        loaded = orch.llm.is_loaded()
        print(f"  FieldAwareLLM loaded after cycle: {loaded}")
    else:
        print("  FieldAwareLLM: not configured")

    # Check system state
    print(f"  Orchestrator state: {orch.state.value}")

    # GPU VRAM after cycle
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"  GPU VRAM allocated: {alloc:.2f}GB")

    print("\n[T8-b] Done. Check data/t8b_verification.log for full details.")


if __name__ == "__main__":
    asyncio.run(main())
