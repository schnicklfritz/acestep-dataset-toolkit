First, make sure .changed_markers file is empty or create one, then run: 

# `git add -N .` on the REAL index records intent-to-add for every untracked
# file, which changes what a later `git commit -a` picks up and lets a binary
# asset inject "Binary files ... differ" noise. So staging happens in a private
# shadow index that is deleted afterwards; the real .git/index is byte-identical
# before and after.
{
  tmp_idx="$(mktemp)"; cp .git/index "$tmp_idx" 2>/dev/null || : > "$tmp_idx"
  export GIT_INDEX_FILE="$tmp_idx"
  git add -N . 2>/dev/null || true
  {
    echo "# AUTO-GENERATED — do not edit. Regenerate before every agent session."
    echo "# HEAD: $(git rev-parse HEAD)"
    echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo
    echo "# --- NAME-STATUS (renames included) ---"
    git diff --name-status HEAD
    echo
    echo "# --- FUNCTION-CONTEXT DIFF ---"
    git diff -W -U3 -p HEAD
  } > .changed_markers
  unset GIT_INDEX_FILE
  rm -f "$tmp_idx"
}

# Repos with binary assets use scripts/gen_changed_markers.sh instead: same
# output, plus a TEXT pathspec allowlist (a bare `add -N .` emits a
# "Binary files ... differ" line for every binary in the tree), a temp file +
# atomic rename, and a loud failure instead of a silently truncated index.

Then, Make sure .gitattributes in active repo contains:

*.py diff=python

Add the following to a system prompt, and/or treat this file as part of the system prompt:

# RULES
1. Regenerate .changed_markers at the start of every session. Treat it as a cache with a TTL of zero. Never reuse a file from a previous turn.
2. Treat HEAD as a cache hit. Do not re-read files that have not changed.
3. Read .changed_markers first, before anything else. Use the --name-status section to orient: M modified, A added, R renamed, D deleted.
4. Read only the hunks and functions in that file. Do not open whole files unless a rule below requires it.
5. Anchor on function and class names, never line numbers. Line numbers go stale within one edit.
6. When a function signature changes, run git grep -n "name" to find callers. Read each caller before editing. A diff-only review misses ripple effects, and "no callers" is not the same as "safe to change" until the grep confirms it.
7. Check callees as well as callers. A changed function may look correct in isolation while an unchanged module it calls now receives different arguments or returns a different type.
8. An anchor that matches nothing means UNKNOWN. UNKNOWN requires investigation, never action. Do not infer a plausible function body from a dangling hunk header.
9. Never edit .changed_markers. It is generated output, not source.
10. After every edit, run the verification gate before declaring the change complete. At minimum: ruff check on the changed files, and vulture or deadtrace on the changed files. Do not ask the model to review its own work; run a deterministic tool and feed the output back.
11. Treat a D or R in --name-status as a mandatory investigation, not an automatic delete. A rename may have broken imports elsewhere; a deletion may have removed a framework-registered entry point that has no ordinary caller.
12. When the diff is empty but you expected changes, stop. An empty .changed_markers means either nothing changed or the generator failed silently. Investigate before generating anything.
13. After a successful commit, regenerate the index (not just empty it), so the next session starts from a file that matches HEAD.
14. If `.changed_markers` is empty or stale relative to `git status`, the agent must refuse the task and regenerate the file           never  invent a scope. 
15. If rename detection and old/new greps disagree, stop and ask rather than “fix” callers from memory. 

# The rules above are only as good as the tools behind rule 10. Run these on the
# changed files, not the whole repo:
git diff --name-only --diff-filter=ACMR HEAD -- '*.py' | xargs -r ruff check
git diff --name-only --diff-filter=ACMR HEAD -- '*.py' | xargs -r vulture --min-confidence 80
git diff --name-only --diff-filter=ACMR HEAD -- '*.py' | xargs -r deadtrace

# WHAT THIS STILL CANNOT CATCH (this is the part that keeps the protocol honest)
# 1. UNCHANGED-FILE INTERACTIONS. The diff shows the changed function looking
#    correct; the bug is that an invariant in a module that did not change no
#    longer holds. Rules 6/7 narrow this, but they depend on knowing which
#    symbol to grep.
# 2. SILENT DEFAULTS. A misspelled config key, a missing metadata field, a
#    fallback value that looks like real data. Nothing changed, so nothing
#    appears in the diff.
# 3. FRAMEWORK MAGIC. Decorators, dependency injection and lifecycle hooks make
#    a function live with no textual caller. This is why vulture alone is
#    insufficient, and why the whitelist has to be maintained deliberately.
# The protocol is robust against generator failures and rule failures. It is not
# robust against the diff's fundamental shape: that one requires a human, or an
# agent that knows when to stop reading the diff and start reading the unchanged
# file.


==========================================================
# CHANGE-MARKER SYSTEM (cost optimization) — READ .changed_markers FIRST
# ============================================================
# `.changed_markers` is the SINGLE exception to the cache-hit rule:
# it is always read first, before any code, because it is the index.
- Before reading ANY module, read `.changed_markers`.
- Treat committed state (git HEAD) as an IMMUTABLE CACHE HIT. Never run
  whole-repo file sweeps; never scan clean files to "confirm" behavior. Trust git.
- Only inspect the specific hunks / functions identified in `.changed_markers`.
- STALE LINE PROTECTION: anchor every edit to function/class names + surrounding
  hunk context, never raw line numbers (marker line ranges can go stale).
- SIDE-EFFECT AUDIT: if an edit changes a function signature, parameter list, or
  return type, run `git grep -n "<fn>"` to inspect ONLY the call sites for
  breakage. Do not re-read entire caller files.
- Prefer `git diff` (base vs working tree) over whole-file reads to learn what
  changed. Never `cat` a 5000-line file to find a delta.
- Once changes are committed to git, REGENERATE `.changed_markers`
  (`scripts/gen_changed_markers.sh`) so the next session starts from a file that
  matches HEAD. It is GITIGNORED derived data: never commit it (a commit cannot
  carry its own SHA, so a committed marker is stale by construction) and never
  hand-edit it — the hand-curated
  ledger lives in `.agent_notes.md`, which no script ever writes.

# ------------------------------------------------------------
# INDEX VALIDITY — a stale index is worse than no index
# ------------------------------------------------------------
# The marker file is only useful if it agrees with reality. If it disagrees with
# `git status`, an agent scopes its work off stale information and never finds
# out — the same silent-wrong-answer failure the UNKNOWN rule guards against,
# but applied to the index itself rather than to an anchor.
- BEFORE scoping any task, validate the index in this order:
    1. `grep '^# HEAD:' .changed_markers` must equal `git rev-parse HEAD`. The
       file also carries `# Generated:` (UTC) and `# dirty: <n> path(s)`.
    2. every dirty/untracked file in `git status` must be listed in the index,
       and every file the index lists must actually be dirty.
- SANCTIONED EXEMPTION: `.changed_markers` and its `.changed_markers.*`
  scratch file are gitignored and excluded from their own diff. Their absence from the index is
  NOT a staleness signal. Everything else must be listed.
- On ANY disagreement: STOP. Do not scope, do not edit, do not "work around it".
  Regenerate the index first (`scripts/gen_changed_markers.sh`). Then re-read it
  and continue.
- NEVER invent a scope. If the index is empty while `git status` is not, that is
  not "nothing changed" — it is a broken index.
- After a successful commit, regenerate the index (not just empty it) so the
  next session starts from a file that matches HEAD.
- `.agent_notes.md` is the other half of the picture and does NOT go stale with
  the diff: unresolved causes, unverified claims, machine-local damage, the
  baseline sweep. Read it together with the index. It is HAND-OWNED — never
  regenerate, reflow or "tidy" it, and never move its items into the generated
  file, where the next run would delete them.

# INDEX GENERATION RECIPE (verified)
# ---------------------------------------------------------------------------
#   scripts/gen_changed_markers.sh
#
# That script is the ONLY supported way to build this file. Two properties are
# load-bearing and were each measured — do not "simplify" them away:
#
#   * IT STAGES INTO A SHADOW INDEX (GIT_INDEX_FILE), never the real one. A
#     bare `git add -A` silently stages the entire working tree, which changes
#     what a later `git commit` picks up and is how someone commits files they
#     never intended to. Measured: the real `.git/index` is byte-identical
#     before and after the script.
#   * IT USES A TEXT-PATHSPEC ALLOWLIST, NOT A DENYLIST. `-diff` in
#     .gitattributes marks a file as binary but git STILL emits a one-line
#     "Binary files ... differ" header for it (re-verified with
#     `git check-attr`), and a denylist can never enumerate every binary
#     extension this toolkit produces. A single added .flac therefore injects
#     noise for every binary in the working tree, and that is exactly how this
#     index was corrupted once before (69KB of binary lines). Restricting the
#     diff to text pathspecs removes the header entirely and shrinks the index.
#     Measured on one identical tree: allowlist 95 bytes / 1 file / 0 headers
#     vs denylist 317 bytes / 3 files / 2 headers.
#   * THE TRAP IN AN ALLOWLIST: it is BLIND to files with NO extension. That
#     included `.clinerules`, `.gitattributes` and `.gitignore` themselves, so
#     editing the rules file produced an index with no sign of it. It was found
#     by validating the index against `git status` — which is exactly why that
#     rule below exists. `':(top,glob).*'` in TEXTS closes it. The `glob`
#     magic matters: without it `*` also matches `/`, so the spec silently
#     pulled in everything under .agents/ and .claude/ too. An extensionless text file elsewhere must be named
#     in TEXTS explicitly.
#
#   * THE SHADOW ADD IS `git add -N` (intent-to-add) ON UNTRACKED ALLOWLIST
#     PATHS ONLY. A shadow `git add -A` hashes every untracked file into
#     .git/objects, binaries included: one 20 MB .wav grew it by 19.5 MB per
#     run (measured). `-N` records the path without its content.
#
# The shadow intent-to-add is also what makes not-yet-tracked files visible to
# `git diff HEAD`, and what makes git emit real `rename from/to` entries for a
# rename that is still unstaged — plain `git diff --summary -M` shows only the
# deletion (measured). When adding a new TEXT format, add its extension to
# TEXTS in the script, not to a denylist here.


# ------------------------------------------------------------
# RENAME SAFETY — git diff is blind to cross-file fallout
# ------------------------------------------------------------
# A rename commit looks clean. The breakage is in the CALLERS, which did not
# change and therefore do not appear in any diff.
- After any rename, before committing — START WITH THE INDEX YOU ALREADY HAVE,
  because the generator's shadow index is the only thing that makes a
  still-unstaged rename visible at all:
      grep -n -E '^(rename |rename from|rename to|similarity)' .changed_markers
      git grep -n "old_name"                 # who still points to the old name?
      git grep -n "new_name"                 # did the new name actually land?
- MEASURED, so this mistake is not repeated: on an unstaged rename,
  `git diff --summary -M` prints ONLY `delete mode 100644 old_name` — the new
  name is invisible — and `git status --porcelain -M` reports it as
  `?? new_name` with no rename pairing. Detection requires the shadow index
  (`cp .git/index /tmp/x && GIT_INDEX_FILE=/tmp/x git add -N <new paths>`), which the
  generator already does for you; the marker file carries the `rename from` /
  `rename to` lines. Do NOT substitute `git log --diff-filter=R --summary -1`
  here either: it inspects the LAST COMMIT and cannot see uncommitted work.
- If rename detection and those greps DISAGREE, stop and ask. Do not reconstruct
  callers from memory — that is how live code gets deleted.
- An anchor with no `git grep` hits means UNKNOWN, not "gone". Investigate
  before editing or removing anything.

# ------------------------------------------------------------
# BOUNDARIES — the contracts a local hunk cannot show you
# ------------------------------------------------------------
# Some breakage is invisible in a per-file diff: shared types, field/format
# names at an export or API boundary, schemas, migrations. Changing these can
# break a consumer in another repository with no error on this side.
- Keep a short ALWAYS-READ allowlist for such files (e.g. schema/contract
  modules). Reading those few files is cheaper than unbounded exploration and
  prevents "the diff is green so the interface must be fine".
- Field/format names are a CONTRACT the moment data leaves the app (export,
  training loader, API). Pin the mapping in ONE module and cover it with a test;
  internal names are free to change, boundary names are not.

# ------------------------------------------------------------
# SCOPE DECLARATION — make context growth visible
# ------------------------------------------------------------
- Before opening anything, STATE the paths you intend to read and why.
  Declared scope makes accidental context growth obvious, to the user and to
  yourself, instead of it happening silently over many tool calls.
- Stay inside that declared scope. If the task genuinely needs more, say so and
  get agreement before expanding it.
- Keep the generated index OUT of source directories and out of the diff it
  describes (binary/data extensions should be `-diff` in `.gitattributes` so a
  stray asset cannot flood the index).

# ============================================================
# TWO-MODE MAINTENANCE (fix for historical dead code / tech debt)
# ============================================================
# `git diff` only sees DIRTY files. A dead function committed long ago and
# never edited looks 100% clean to git (a cache hit) and would never be found.
# So maintenance is split into two explicit modes:
#
# 1. DAILY ITERATIVE MODE (runs ~99% of the time) — git-as-cache above.
#    Only touch dirty hunks + direct callers. Cost ~pennies.
#
# 2. ONE-TIME BASELINE SWEEP (explicit, when auditing an old repository):
#    - Do NOT pay the LLM to read tens of thousands of lines searching for
#      dead code. Run a FREE local static analyzer first, e.g.
#        Python:     vulture . --exclude .venv,kernels   (dead code)
#                    flake8 --select=F401 .              (unused imports)
#        TypeScript: knip  /  ts-prune                    (unused/dead)
#    - Record ONLY the flagged lines in `.agent_notes.md` under its
#      `[BASELINE SWEEP]` section — NEVER in `.changed_markers`, which the
#      generator overwrites on every run and is gitignored. The LLM then inspects just those
#      candidates (judgment: is this really dead?) instead of the whole repo.
#    - The free, deterministic tool does the scan; the model only reviews flags.
