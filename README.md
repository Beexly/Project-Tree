# Project-Tree

Standalone home for the **CRC knowledge-integrity wedge** (`sheaf_integrity/`).

Not part of Beexly/Sports until an explicit founder YES.

## Quick start

```bash
git clone https://github.com/Beexly/Project-Tree.git
cd Project-Tree/sheaf_integrity
pip install numpy fastapi uvicorn
python demo.py
python demo.py --serve   # :8080
```

## What this is

Classical multi-source consistency engine.

Repair classes: `consistent` | `statistical` | `structural` | `self_consistent_wrong_vs_ledger`

See `sheaf_integrity/AGENT_HANDOFF.md` and `sheaf_integrity/WHAT_THIS_IS.md`.

## Layout

```
sheaf_integrity/
  core/          CRC engine
  server/        FastAPI
  demo.py
  tests/
  benchmarks/
  docs/
  research/
```

## Status

Path A technically complete in sandbox. Repo is the durable home. Some core modules may still be landing on `main` — check `sheaf_integrity/core/`.
