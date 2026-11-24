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
1. ~~**Two failing tests.** Names unknown.~~ **Resolved 2026-11-18.** They were
   `tests/test_assistant_actions.py::test_gap_audit_says_ok_for_a_complete_track`
   and `tests/test_assistant_tool_routing.py::test_write_caption_validates_what_it_just_wrote`,
   both tripping `MIN_FACTS = 8` against a prose fixture. The fixtures were right:
   the CHECKER was wrong (see the first invariant below) and has been deleted.
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
- **Caption TEXT is judged by a human/LLM, never by a script** (2026-11-18).
  The "front-load 5-12 tags then 8-16 comma-separated facts" rule was a
  SONG-GENERATION prompt convention, not a LoRA requirement: upstream's
  `docs/en/LoRA_Training_Tutorial.md` caption example is one prose line and its
  LoRA captioner (`scripts/lora_data_prepare/gemini_caption.py`) constrains the
  shape not at all. `check_caption`, `count_facts`, `modules/caption_audit.py`,
  the export-time caption gate and `write_caption`'s self-review are DELETED.
  `trim_to_caption` survives as a runaway length guard only.
  `BUILTIN_SYSTEM_PROMPT` still asks for the dense shape — that is the captioner's
  guidance, and retiring it is an open decision, not a check to re-add.
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

## How the owner works (read before your first reply)

- **Peer register, not assistant register.** Direct. No "great question", no
  "I'd recommend", no restating what was just said. If something is wrong, say
  it is wrong and say what it costs.
- **Numbers are the currency.** A count, a timing, a diff, a measured command
  output. Rhetoric and reassurance read as noise. Leave the command that
  produced the number.
- **Do not re-propose a rejected idea.** Already decided against, with reasons:
  throttling as the explanation for his results, revenue-model pitches for the
  free app, hotspot bypass as a product, generic advice rules, deleting docs to
  "clean up", rewriting a doc wholesale.
- **He corrects confidently and is usually right.** Concede in one line, say what
  it changes, move on. Never soften it to "you may have a point".
- **No moralising.** He has thought the ethics through; a lecture is a tax.
- **One clarifying question at most**, and only when the answer changes what you
  build. Otherwise state the assumption and proceed.
- **Multiple agents share this repo.** Lease, do, release. Never edit a file
  another agent may hold.
- **He is short of time and attention, not advice.** One batched finished action
  beats five small confirmations.

## Where shared state lives
- `HANDOFF.md`: cold-start facts.  `.clinerules`: invariants, mode-scoped.
- agentmemory MCP `http://127.0.0.1:9115` (48 tools): actions, leases, signals.
- `~/phone-memory/memserve.py`: append-only journal. Corrections go there.
