# Agent Guardrails — DRAFT FOR REVIEW (2026-10-16)

> **Status: proposal, not in force.** Nothing here is referenced by `.clinerules`, any script,
> or any hook. Edit or cut freely — the review checklist at the end is the intended way to
> accept or reject each item, and the last section explains how (and whether) this migrates
> into `.clinerules`.
>
> **Citation convention.** Every premise below cites a **symbol** (`module.function`,
> `ClassName`, `CONSTANT`), never a line number. Line numbers were the mechanism that hid the
> original bug (see G1), and `git grep -n "<symbol>"` is a staleness check that cannot itself
> go stale. Where a document section rather than code is the premise, it is cited by heading.

## Why this file exists

On 2026-10-16 a task was reported complete — JSON sidecar written beside the audio, Save
dialog fixed, tests green — while the dataset it produced still could not be trained on with
its lyrics, because the step that produces training-ready lyrics wrote a filename the consumer
never opens. The bug was small. The *route to not finding it* is what this file addresses: a
justification that outlived its reason, a verification ritual that consumed an hour of CPU to
answer a question a one-line diff had already answered, a hypothesis chased four rounds past
the measurement that refuted it, and a live source tree overwritten three times to produce
evidence that a temp directory could have produced without risk.

The triggers below are deliberately **observable** ("second repetition", "no new number",
"path cited as `file:line`") rather than exhortative ("be careful", "don't loop"). A rule that
can only be obeyed by remembering it is not a guardrail.

## The set

### G1 — A boundary name has one owner, and premises cite symbols

**Rule.** Any name that crosses a boundary the app does not control — an exported filename,
a sidecar suffix, a JSON key, a column name — is defined **once**, in the module that owns
that boundary, and imported everywhere else. No second literal. Any note, docstring, or test
that asserts *what another file does* cites a **symbol**, not a line number. A test pins the
boundary against the consumer's documented layout, and quotes the document heading it relies
on, so a change to either side trips the test.

**Trigger.** Writing or editing a docstring/note/test that claims something about a file you
are not editing; or typing a filename suffix that already exists elsewhere.

**Incident.** `ui.tidy_panel._write_lyrics_sidecar` writes `<stem>_lyrics.txt`, justified in
its own docstring by "`modules.exporters.export_sidecar_captions` owns `<stem>.txt` for the
caption". By the time it mattered, the caption had moved to `<stem>.caption.txt` — the
justification was dead and the code was pointing at a slot that no longer existed.
`tests/test_tidy_panel.py`'s module docstring pinned that same dead reason as an assertion, so
the suite defended the bug. Correct names, per the guide's "Required Files" / "File lookup
priority" sections: `<stem>.lyrics.txt` (recommended) or bare `<stem>.txt` (legacy) for
**lyrics**; `<stem>.caption.txt` for captions. `<stem>_lyrics.txt` matches neither.

**Enforcement.** Import structure (mechanical, by construction) + one test + review.

**Cost.** Near zero. It is strictly less work than a second literal.

### G1b — Tests pin behaviour; reasons live in prose

**Rule.** A test asserts what the code *does*. It must not carry, as load-bearing text, the
*reason* the behaviour was chosen — a reason can expire while the assertion stays green.

**Trigger.** Writing `assert ...  # because X owns Y` or a module docstring that justifies a
test premise.

**Incident.** Same as G1: the "NOT `<stem>.txt` — that name belongs to the caption exporter"
premise was true when written and false later, and no mechanical thing could notice.

**Enforcement.** Review; G1's single-owner rule removes most of the motivation to do it.

### G2 — Bounded verification

**Rule.** The full suite is an **owner-requested** action, never a routine closing step.
Every run is targeted at the failure point just touched, carries a `timeout`, and its result
is reported from the **summary line** (`N passed`/`N failed`), never from a `tail`. Nothing is
ever launched with a bare `&`. A run that is still going after the point where it stopped
telling you anything new is stopped, and that is reported.

**Trigger.** Any test invocation; any temptation to "just run everything to be safe".

**Incident.** Three full-suite runs at ~38 min each on this 4-core box; a detached instance
accumulated 26 min of CPU and wedged the tool terminal, which was then typed into ~8 more
times; a "all green" claim was made from `tail -5` instead of the summary line. Meanwhile the
one measurement that settled the *actual* question (is the app busy at idle?) was a 6-second
`/proc` sample reading 0.00 % of one core.

**Enforcement.** Prose + the ergonomic path (`timeout <n> python -m pytest -q <file>`).

### G3 — Measure before hypothesizing

**Rule.** For any question about performance, hangs, or "is X doing Y", obtain **one number
first**, and write down which hypothesis that number kills. Before opening a *second* file to
support a theory, state what the first file would have had to show to kill it. If nothing
could kill it, it is not a hypothesis and it may not drive further reading.

**Trigger.** Any diagnostic session; any sentence beginning "maybe it's".

**Incident.** A slow-but-progressing run was theorised as a *hang* and chased through
`/proc/<pid>/syscall`, `wchan` and context-switch counters for four rounds; the run was
advancing the whole time, which one growing line count would have shown. Separately, an idle
CPU report was met with a search of `QTimer`s, `wheel_guard`, `main.py` and `model_manager`
*after* the first sample had already measured 0.00 %.

**Enforcement.** Prose (no cheap mechanical form exists).

### G4 — Perception reports: resolve the referent before reading code

**Rule.** When the report is about the owner's experience — "slow", "hot", "wrong",
"confusing" — the first action is to ask *which process / window / track / button*, and what
was on screen. Only then read code. **One question, then act.**

**Trigger.** A bug report phrased from the user's point of view rather than as a failing input.

**Incident.** Eight tool calls (process table, `/proc` sampling, `QTimer` search, `main.py`,
`model_manager`, `wheel_guard`) preceded the question "what were you looking at?", whose
answer ("the manager") would have collapsed the search to one window title.

**Enforcement.** Prose.

### G5 — Never mutate the live tree to produce evidence

**Rule.** Evidence is produced from `git show HEAD:<path>` into a temp directory, from the
artifacts on disk, or from a copy under `/tmp`. A tracked file is never overwritten, even
temporarily, to demonstrate that an older version behaves differently. Restoring afterwards
does not make it safe: the window between overwrite and restore is exactly the unrecoverable
state the deletion policy exists to prevent. `git stash` is not a substitute.

**Trigger.** Any plan of the form "put the old file back, run, then put the new one back".

**Incident.** The same swap-and-restore red/green proof was performed three times. Each run
had a real chance of leaving the working tree reverted if the session had been interrupted.

**Enforcement.** Prose + an amendment to the existing DELETION POLICY, which today covers
*deletions* but does not name *temporary overwrites* as the same hazard.

### G6 — Classify a finding by goal impact, not by authorship

**Rule.** "Pre-existing" is not a triage label. Every finding is classified by its effect on
the outcome the owner asked for. A pre-existing defect that defeats that outcome is either
**in scope** or is reported as a **blocker before work continues** — never filed as "not mine"
and stepped over.

**Trigger.** Any time the phrase "pre-existing" is forming.

**Incident.** The lyrics file naming was found, written down as a pre-existing finding, and
left, in the same session whose stated goal was to get the dataset into ACE-Step. The dataset
reached training with tags and a caption and no lyrics. The finding *was* the task.

**Enforcement.** Prose + the task-close report requiring a goal-impact line per finding.

### G7 — Loop breaker: two strikes, then change the kind of move

**Rule.** On the **second** repetition of an action class — same file re-read, same suite
re-run, same command re-issued, same ritual performed — stop and name the loop in one
sentence before any third attempt. The third attempt, if it happens at all, must be a
**different kind** of action: measure → read → ask the owner → check prior art → test the
real artifact.

**Trigger.** The second occurrence. Also the phrase "just one more".

**Incident.** This file. Every item above is a third attempt that should have been a first
question or a one-line measurement.

**Enforcement.** Prose. The observable trigger is repetition, which is countable.

### G8 — Report environment damage; do not type around it

**Rule.** Long or background work is launched as
`setsid <cmd> </dev/null >>/tmp/<name>.log 2>&1 &`, and immediately followed by a check that
the terminal still reports completion. If it does not, say so and stop using that terminal,
routing further reads through files. A wedged shell is damage to report, not a puzzle to solve
by persistence.

**Trigger.** Any command that will outlive the call; any command that returns
"completion could not be observed".

**Incident.** After the first wedged run, roughly eight more commands were issued into the
same terminal instead of reporting it. The workaround (write to `/tmp`, read the file) was
correct and arrived several rounds later than the report would have.

**Enforcement.** Prose.

### G9 — Verify a contract on the real artifact

**Rule.** Where the thing being verified is a contract with an outside consumer, the
acceptance check uses a **real input** — a track copied from the owner's dataset into `/tmp`
— not only a synthetic fixture. Fixtures prove the code runs; the real file proves the
consumer can eat it.

**Trigger.** Any claim of the form "this is the format the trainer reads".

**Incident.** The sidecar's entire value is "ACE-Step reads it", and its acceptance evidence
was a `tmp_path` sample. One real track would have exercised dotted basenames, non-ASCII
captions and the audio's actual directory in seconds.

**Enforcement.** Prose; the tooling already has read access to the dataset root for exactly
this purpose.

### G10 — Look for an existing solution before building one

**Rule.** Before writing code, a script, or a rule, check in this order and record which
answered:

1. **This repo** — `git grep` for the concept; it may already exist here.
2. **`docs/`** — the contract may already be specified, and the citations may already be
   verified against upstream.
3. **The upstream whose contract you are honoring** — readers *and* writers; confirm what it
   ships instead of inferring it.
4. **The ecosystem** — the library, the platform, the standard tool.

**Corollary (important):** if `.clinerules` already covers the case, then *applying that rule*
is the solution. Do not propose a new mechanism on top of one that already exists.

**Trigger.** Before the first line of new tooling; whenever a solution is being designed rather
than looked up.

**Incident.** A bespoke `ANCHOR IMPACT` scanner was designed for `scripts/gen_changed_markers.sh`
and a `scripts/run_targeted.sh` was drafted, when (a) `.clinerules`' BOUNDARIES section already
prescribed "pin the mapping in ONE module and cover it with a test", which is the actual fix,
and (b) `git grep` on a symbol already is the anchor check, provided premises cite symbols —
which G1 now requires. The ecosystem equivalents (`timeout`, `pytest -q --lf`) already cover
the test wrapper. Two new moving parts were deleted before being written.

**Enforcement.** Prose + the "which of the four answered" line in the task-open report.

### G11 — Know when to abandon

**Rule.** Each hypothesis gets **two attempts**. If the second produces no new number, no new
file, no new error and no new quote from a source, abandon it and change axis. Never escalate
by repeating the same kind of move with more effort.

**Trigger.** Second attempt with no new information; a plan whose next step is the previous
step again.

**Incident.** The ACE-Step upstream check: one URL guess returned 404. The next move was *not*
a second and third URL guess, but the axis change to the verified contract already recorded in
`docs/` — which also revealed that upstream ships readers, not writers, making the toolkit the
party responsible for the naming contract. One failed attempt, then a different kind of move.

**Enforcement.** Prose.

## Task-open / task-close protocol

Four lines at the start, five at the end. It is the cheapest part of this file and the part
that would have caught the lyrics bug directly.

**Open**

1. **Scope** — the paths about to be read and why.
2. **Target** — the one measurement or failing input that defines "solved".
3. **Referent** — if the report is perceptual, the question asked before any code is read (G4).
4. **Prior art** — which of G10's four answered.

**Close**

1. **Result** — the summary line, quoted; not a `tail` (G2).
2. **Change + evidence** — what changed, and how it is known to work (G9).
3. **Findings** — every finding classified by goal impact (G6), pre-existing included.
4. **Environment** — terminal/process state, and anything left running (G8).
5. **Deletions** — every path archived to `deletions/`, per the deletion policy.

## Deliberately not included, and why

- **An `ANCHOR IMPACT` scanner in `scripts/gen_changed_markers.sh`** — superseded by G1. With
  symbols in notes, `git grep` is the check; a scanner would age with the note format.
- **`scripts/run_targeted.sh`** — superseded by `timeout <n> python -m pytest -q <file>`
  (G10). Add it only if the one-liner proves insufficient in practice.
- **A "don't get stuck in loops" rule** — unenforceable as written. Replaced by G7 and G11,
  which have countable triggers.
- **Any rule that requires remembering it under pressure.** That is the failure mode, not the
  fix. If an item here cannot be checked by a command, a grep, or a count, it belongs in
  review, not in a rules file.

## Measured facts, scoped to this machine (2026-10-16)

Recorded so they are not re-derived by ritual, and scoped so they are not mistaken for
universal truths (G1 — a premise carries its scope):

- `python main.py` at idle: **0.00 % of one core** (6-second per-thread sample). The app does
  not have an idle-CPU defect.
- Full test suite: **~38-40 minutes** on this 4-core box, with heavy btrfs I/O; load average
  reached **12.21**. It is not a five-minute check.
- A stray full-suite run reached **26 minutes of accumulated CPU** and a `pytest` process
  observed at **62.8 % of one core**. It has since exited on its own; exit status was not
  captured.

## How this becomes `.clinerules`

If, after review, any of this is accepted:

1. Append **one** dated section to `.clinerules` (`# === AGENT GUARDRAILS (accepted YYYY-MM-DD)`).
2. Keep each rule to its imperative plus its one-line reason. The incident histories stay in
   this file and are cross-referenced, not duplicated.
3. Promote only what is accepted. Silent promotion of the whole draft is the failure mode this
   file was written to prevent.
4. Add the G5 amendment inside the existing DELETION POLICY section rather than as a new
   section, since it modifies that policy rather than adding to it.
5. If an accepted rule is later shown to be wrong, strike it with a dated line — do not delete
   it. The reason it was wrong is the most useful part.

## Review checklist

Mark each line **keep / cut / rewrite**, then decide promotion:

| Item | Question to answer |
|---|---|
| G1 | Is single-owner naming + symbol citations acceptable as a working rule? |
| G1b | Is banning reasons-from-tests too strict (some tests legitimately document intent)? |
| G2 | Is "full suite is owner-requested only" the right default? |
| G3 | Keep, or is it too abstract to act on? |
| G4 | Keep. |
| G5 | Accept the DELETION POLICY amendment (temporary overwrite = deletion)? |
| G6 | Is "pre-existing, and it defeats the goal → blocker" the right escalation? |
| G7 | Is "second repetition" the right trigger, or should it be first? |
| G8 | Keep. |
| G9 | Does using a real track from the dataset for contract checks raise data concerns? |
| G10 | Keep the four-source order? |
| G11 | Is "two attempts" the right budget? |
