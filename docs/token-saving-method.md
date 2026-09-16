# The Git-as-Cache Method: cut your AI coding costs by treating every commit as a cache hit

> A single workflow that slashes token spend — and the unusual part is that
> it benefits *both* the developer *and* the customer, which almost never happens.

---

## The one-sentence version

**Don't ask your coding agent to re-scan your repository. Make git the cache,
and invent only one tiny missing piece: a transient "marker" file that lists
*only* the regions that changed. Everything already committed is a cache hit;
only marked regions get read.**

The marker is deleted the moment its change is committed — at which point git
(the cache) has absorbed it and it never needs to be read again.

---

## The problem

LLM coding agents re-send large amounts of context on every turn. The more files
the agent "knows about," the more every single message costs. Let a session run
long against a big repo and the costs compound — people routinely burn $5 in
half an hour without writing anything wrong. It's not the code that's expensive;
it's **re-reading what you already knew.**

The intuitive "fix" people reach for is expensive too:

- "Let me scan the whole repo once and build a dependency graph" — that scan
  *itself* is the most expensive thing you can do.
- "Let me run an in-memory graph database" — RAM-hungry, and it doesn't cut
  tokens.
- "Let me be more careful" — unbounded, unenforceable, and evaporates when the
  session ends.

None of those survive a budget, and none of them survive a weak laptop.

---

## The method

### 1. Git is already the cache. Use it.

Your committed history is a content-addressed store of "exactly what changed and
when." You don't need to build a cache. It already exists. The only thing you
need is a way to point the agent at the *delta* instead of the *whole*.

```bash
git diff --stat       # one line per changed file — effectively free
git diff -- <file>    # only the hunks that changed — not the 5,000-line file
```

This is the "cache lookup." It tells you precisely where something changed,
for near-zero tokens.

### 2. A transient marker file is the only new piece.

A single tiny file at the repo root (`.changed_markers`) records *only* what is
not yet committed:

```text
[UNREVIEWED]
dataset_manager.py   L~2130-2200   scope-string bug fixed
modules/mvsep_gui.py  (new file, full review)

[FLAGGED - REVIEW]
dataset_manager.py   GATEKEEPER credential check removed — intentional?
```

That's it. No graph. No database. No dependencies. A 20-line text file.

### 3. Commit = cache absorb = marker deleted.

This is the cycle that makes it self-cleaning:

```
change -> marker -> commit -> marker removed
```

When you commit, git absorbs the change, and you **delete** the marker line.
The region is now a cache hit forever. The marker exists *only* between
"changed" and "absorbed" — it's a one-shot invalidation notice, not a growing
ledger.

### 4. One rule line makes it permanent.

Add a single directive to your agent's rules file (`.clinerules`), so every
future session inherits the discipline without re-deriving it:

```text
- Before reading any module, read `.changed_markers`.
- Treat committed state (git HEAD) as a full CACHE HIT: do not re-sweep the repo.
- Only open files / line-ranges listed under `[UNREVIEWED]`.
- Prefer `git diff` over whole-file reads.
```

---

## What this buys you (measured, not estimated)

Real numbers from a real session, at **double the normal per-token rate**
(premium/peak pricing):

| Task | Result |
|------|--------|
| Audit a **5,099-line** file for bugs and dead code | cost ~pennies |
| Fix **3 live bugs** (a crash, 2 silent duplicate-method overrides) | included above |
| Remove dead methods + a 496-line new module review | included above |
| Full **4-commit history rewrite** (reword-only, force-pushed) | included above |
| **Total for the entire session** | **~$0.11 USD** |

The point isn't the absolute dollar figure — it's that the savings are
**structured into the process**, not lucky. The same workflow at the same rate
would cost the same pennies a month from now.

---

## Why it benefits *both* sides (the part people miss)

Cost-saving advice is usually zero-sum: the *company* saves, or the *customer*
gets faster results — rarely both. This is the exception:

- **Developer / company:** fewer tokens = lower spend, less context bloat,
  better answers (the model spends its budget *thinking*, not *re-reading*).
- **Customer / end result:** lower cost removes the pressure to run a cheaper,
  dumber model to save money. You keep the strong model, or you run it longer
  on the problems that matter, because the *waste* — not the *quality* — is
  what you cut.

You aren't trading quality for cost. You're deleting the redundant re-reading
that never added value in the first place.

---

## How to adopt it in ~2 minutes

1. Create `.changed_markers` at the repo root (the template above).
2. Add the 4-line directive to your `.clinerules` (or equivalent).
3. When you start work: run `git diff --stat`, write the changed regions into
   `[UNREVIEWED]`, and let the agent read *only those*.
4. When you commit: delete the absorbed marker lines.

That's the entire system. Git does the heavy lifting; the marker file is just
the note that says "this part is dirty."

---

## Honest caveats

- **The agent's in-session context still wipes when you start a new chat.** This
  method works *precisely because* it makes the cache live on disk (git + the
  marker file), not in the model's ephemeral memory.
- **`git diff` only shows what changed since the last commit.** Genuinely
  *historical* dead code (committed long ago, never touched) still needs a real
  scan — but you do it *once*, flag it, and never re-sweep it again.
- **The deeper truth is anticlimactic:** the individual pieces are old (incremental
  caching, dirty bits, `git diff`). The *novelty* is combining them into a
  deliberate, named workflow that treats the repo's own commit history as the
  agent's cache — and being explicit that only the marker is new.

---

*If this pattern has a prior name, it deserves to be better known. If it doesn't,
it should. The workflow above costs cents to run and pays every session after.*
