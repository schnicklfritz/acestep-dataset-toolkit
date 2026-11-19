# HANDOFF - state as of 2026-11-18

Portable memory for any agent picking this up. Read this first, then
`.clinerules` (mode-scoped invariants), then the file for the task at hand.
Do not re-derive anything written here; if it is wrong, fix it here.

## What this repo is
A PySide6 desktop toolkit that builds ACE-Step 1.5 LoRA/LoKR training datasets:
import audio -> audit -> DSP normalise -> stem-split -> structure-split ->
caption -> export manifest. 34,174 live lines, 36 test files.

## Just landed (git log)
- `2d61db2` docs: hard limits generated from `modules/caption_spec.py`; SPECIFICATION
  drift fixed (it recommended the retired narrative caption); guides guarded
  against silent truncation.
- `5e6f305` rules: `.clinerules` rewritten to 6.9 KB, mode-scoped, deletions
  git-gated, `deletions/` retired.

## Open work, priority order
1. **Two failing tests.** Names unknown. Get them with
   `.venv/bin/python -m pytest --lf -q`. The full suite takes ~15 min and two
   runs were killed before finishing.
2. **UI overhaul.** Owner's words: "the UI needs an overhaul, it feels wrong."
   Measured: styling is NOT the problem (5 `setStyleSheet` calls, 0 hardcoded
   font sizes, tokens live in `ui/themes.py`). The problem is layout: one screen
   carries a health strip, 6 toolbar buttons, a view toggle, 2 quick actions, a
   9-column table, a permanent 40% inspector with 8 lock checkboxes, a lyrics
   editor and a diagnostics report. It is a cockpit for a six-stage pipeline.
   Target: stage rail, inspector as a drawer, table only at audit/edit.
3. **Bloat.** `deletions/` = 5.5 MB untracked (8 snapshots of
   `dataset_manager.py`, 27 dead test files) - retire it. `dataset_manager.py` =
   7,292 lines = 21% of live code - split along the seams that already exist.
   `pytest.ini` declares a `slow` marker used 0 times - mark the slow tests so
   the inner loop is `pytest -m "not slow" -q`.

## How to verify anything
```
cd ~/Desktop/acestep-dataset-toolkit
.venv/bin/python -m pytest -q                      # full, ~15 min
.venv/bin/python -m pytest -m "not slow" -q        # inner loop
.venv/bin/python scripts/doc_manifest.py --check   # guides not truncated
.venv/bin/python scripts/gen_doc_numbers.py --check # guide numbers == caption_spec
```

## Invariants that will bite you
- Caption = front-loaded keyword tags + 8-16 comma-separated acoustic facts.
  A prose narrative is BANNED (retired 2026-10-23, token-measured). The rule
  text lives in `modules/caption_spec.CAPTION_DENSITY_RULE`; never restate it,
  quote it.
- Budget: 35-70 words, max 900 chars, max 3 descriptors per bracket, 6-10
  syllables per lyric line. All generated into the guide - do not hand-edit
  that block.
- Docs are append-only. `tests/test_docs_integrity.py` enforces it.
- The Comprehensive LoRA/LoKR guide is the only one of its kind; there is no
  upstream replacement to re-download.

## Shared memory (both agents, one store)
- agentmemory MCP front: `http://127.0.0.1:9115` (48 tools).
- From this phone: `py_am`, bridged at `192.168.1.6:9113`.
- Coordination: `memory_action_create(title required)` -> `memory_lease`
  (`operation: acquire|release`, `ttlMs` max 1h) -> `release` with `result`.
  `memory_frontier {}` shows what is unblocked and who holds it.
- Obsidian export to a folder makes the whole store readable as Markdown:
  `memory_obsidian_export {"vaultDir": "~/agentmemory-vault"}`.

## Environment facts
- CachyOS, projects under `/home/fritz/Desktop`, git repo is the toolkit.
- `.venv/bin/python` is 3.14; Qt runs headless via `QT_QPA_PLATFORM=offscreen`
  (set in `tests/conftest.py`). Never add a network call to a test.
- `scripts/gen_changed_markers.sh` is currently dirty - that predates this
  handoff and is not part of the work above.
