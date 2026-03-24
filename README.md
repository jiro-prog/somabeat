# Somabeat — Bio-inspired Integrated AI Architecture

複数のLLMエージェントを「認知機能の分割」で統合し、一つの個体として振る舞うシステム。

## What's Interesting

- Sleeps every night, consolidates memories, wakes up a little smarter the next morning
- A single metacognition rule spontaneously evolved into a self-assessment framework within 3 cycles
- When something breaks, the immune system automatically repairs it

## Architecture

```
 ┌─────────────────────────────┐
 │     外界（ユーザ、Web等）     │
 └──────┬──────────────┬────────┘
        │              │
 ┌──────▼──────┐ ┌─────▼──────┐
 │   感覚系    │ │   運動系    │
 └──────┬──────┘ └─────▲──────┘
        │              │
 ┌──────▼──────────────┴──────────────────────────────┐
 │                                                     │
 │       共有状態の場（エンベディング空間）               │
 │  ┌──────────────────────────────┐                   │
 │  │   減衰関数付きベクトル群      │                   │
 │  └──────────────────────────────┘                   │
 │    ↑ 書込  ↓ 読取                                   │
 │                                                     │
 ├──────────────┬──────────────┬────────────────────────┤
 │              │              │                        │
 ┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐         │
 │ llamarcute  │ │ SleepyJean │ │  自己修復   │         │
 │   -live     │ │ 記憶・学習  │ │ 障害検知   │         │
 │ (人格/推論) │ │  ・忘却    │ │  ・修復    │         │
 │ +自己改善   │ │ (内分泌系)  │ │  (免疫系)  │         │
 └─────────────┘ └────────────┘ └─────────────┘         │
 │                                                     │
 │              永続層（SQLite）                         │
 └─────────────────────────────────────────────────────┘
```

### Three Systems

| System | Cognitive Function | Role |
|--------|-------------------|------|
| Nervous (llamarcute-live) | Personality, reasoning, self-improvement | Real-time dialogue, gradual behavioral rule evolution |
| Endocrine (SleepyJean) | Memory, learning, forgetting | Nightly batch memory consolidation and knowledge distillation |
| Immune | Anomaly detection, repair | Health checks, personality rollback |

### Design Principles

1. **Cognitive function separation** — not task division, but division by cognitive role
2. **Indirect coordination via shared field** — systems never call each other's APIs; they read/write to a shared embedding space
3. **Self-identity** — personality changes are gradual and versioned
4. **Sleep as a whole-body state** — all systems participate in the sleep/wake cycle

## Key Results

- Cognitive loop established through shared field: dialogue → learning → memory → behavior change
- Gradual personality evolution through sleep-time self-improvement (v0 → v16, 20 cycles)
- Immune system: anomaly detection across 6 axes with automatic rollback
- Autonomous evolution of metacognition rules (1 manual rule → self-assessment framework in 3 cycles → self-pruning of redundancy)
- Operational definition and verification of "individuality" (Direction B: emergence from field reading patterns)

## Documentation

| Document | Contents |
|----------|----------|
| [bio_ai_architecture.md](docs/bio_ai_architecture.md) | Top-level design: vision, principles, architecture |
| [shared_field_design.md](docs/shared_field_design.md) | Shared field interface: Signal, sense/emit protocols |
| [llamarcute_live_design.md](docs/llamarcute_live_design.md) | Integrated cycle: dialogue, self-improvement, bridges |
| [individuality_design.md](docs/individuality_design.md) | Operational definition and verification of individuality |
| [findings.md](docs/findings.md) | Summary of findings |

## Related Projects

- [SleepyJean](https://github.com/jiro-prog/sleepyjean) — Memory and learning system. Nightly learning pipeline.

## Setup

### Requirements

- Python 3.11+
- [Ollama](https://ollama.ai/) with `qwen3:8b` model
- ChromaDB (installed via pip)
- Discord Bot Token
- (Optional) Anthropic API key for SleepyJean reference generation

### Installation

```bash
git clone https://github.com/jiro-prog/somabeat.git
cd somabeat
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuration

```bash
cp config/system.yaml.example config/system.yaml
# Edit system.yaml with your Discord channel ID, SleepyJean paths, etc.

# Create .env with your Discord bot token
echo 'DISCORD_TOKEN=your_token_here' > .env
```

### Running

```bash
# Start the Discord bot (normal operation)
python -m discord_bot.bot

# Or run as a systemd service
sudo cp integrated-system.service /etc/systemd/system/somabeat.service
sudo systemctl enable somabeat
sudo systemctl start somabeat
```

### Running Experiments

```bash
# Individuality Experiment 1: Gate test
python scripts/run_experiment1_gate.py

# Individuality Experiment 2: Metacognition A/B test
python scripts/run_experiment2_metacognition.py
```

## License

[LICENSE](LICENSE)

## Citation

If you reference this project:

```
@software{somabeat2026,
  title={Somabeat: Bio-inspired Integrated AI Architecture},
  author={jiro-prog},
  year={2026},
  url={https://github.com/jiro-prog/somabeat}
}
```
