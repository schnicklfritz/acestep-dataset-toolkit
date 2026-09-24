#!/usr/bin/env bash
# Regenerate .changed_markers: the diff of everything dirty since HEAD.
#
# Read .clinerules -> "INDEX GENERATION RECIPE" before changing anything here.
#
# Design notes, all four measured (see the measurements at the bottom):
#
#   1. THE REAL STAGING AREA IS NEVER TOUCHED. Everything is staged into a
#      private shadow index (GIT_INDEX_FILE), so whatever you had staged
#      before the run is exactly what you have staged after it. The old inline
#      recipe ran a bare `git add -A`, which silently staged the whole working
#      tree and changed what `git commit` would have picked up.
#      Verified: the real .git/index is byte-identical before and after.
#
#   2. TEXT-PATHSPEC ALLOWLIST, NOT A DENYLIST. `-diff`/`binary` in
#      .gitattributes does NOT suppress the "Binary files ... differ" header,
#      and a denylist can never enumerate every binary extension a toolkit
#      like this one produces. An allowlist fails safe: an unknown extension
#      is simply not dumped.
#      THE TRAP IN AN ALLOWLIST: it is blind to files with NO extension, and
#      that included .clinerules, .gitattributes and .gitignore themselves —
#      an edit to the rules file produced an index with no sign of it. The
#      ':(top).*' spec in TEXTS closes that. When adding a new TEXT format, add
#      its extension to TEXTS below; an extensionless file must be named there.
#
#   3. THE SHADOW `git add -A` IS LOAD-BEARING, not just a staging trick. It is
#      what makes not-yet-tracked files visible to `git diff HEAD`, and it is
#      what makes git emit real `rename from/to` entries for a rename that is
#      still unstaged. Measured: plain `git diff --summary -M` on an unstaged
#      rename prints only the deletion; the new name is invisible to it.
#
#   4. WRITTEN VIA A TEMP FILE IN THE REPO ROOT, then renamed into place. A
#      failure mid-diff leaves the previous index intact instead of truncating
#      it. The temp file is created in the repo root (not $TMPDIR) so the final
#      `mv` is an atomic rename rather than a cross-device copy, and it is
#      covered by `.changed_markers.*` in .gitignore.
#
# THIS SCRIPT DOES NOT TOUCH .agent_notes.md. The generated index and the
# hand-curated ledger are deliberately separate files: this one is overwritten
# on every run, the other is never written by a machine.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git_dir="$(git rev-parse --absolute-git-dir 2>/dev/null || git rev-parse --git-dir)"
markers="$repo_root/.changed_markers"
tmp_markers="$(mktemp "$repo_root/.changed_markers.XXXXXX")"
# mktemp creates the file 0600. Preserve the mode of the file we are about to
# replace, so regenerating a tracked file never quietly makes it owner-only.
markers_mode="$(stat -c '%a' "$markers" 2>/dev/null || echo 644)"
chmod "$markers_mode" "$tmp_markers"

# Optional locks off: no git command below rewrites the real index's stat
# cache. `git add` still writes its own (shadow) index, which is required.
export GIT_OPTIONAL_LOCKS=0

old_git_index_file="${GIT_INDEX_FILE-}"
tmp_index=""
cleanup() {
  if [ -n "$tmp_index" ]; then rm -f "$tmp_index"; fi
  rm -f "$tmp_markers"
  if [ -n "$old_git_index_file" ]; then
    export GIT_INDEX_FILE="$old_git_index_file"
  else
    unset GIT_INDEX_FILE
  fi
  # Explicit: a trap whose last command fails can REPLACE the script's exit
  # status, turning a clean run into a reported failure.
  return 0
}
trap cleanup EXIT

# Snapshot the REAL index's status BEFORE the shadow index is exported. Once
# GIT_INDEX_FILE points at the shadow index (whole worktree staged), `git
# status` reports a different, coarser path set, so the validation lines in the
# generated header would disagree with the `git status` a human runs.
# `.changed_markers` is this script's own output and `.changed_markers.XXXXXX`
# is its scratch file; neither may be reported as if it were your work. Match on
# the marker name rather than trusting .gitignore, so the script behaves the
# same in a tree where that ignore rule is missing (e.g. a fresh clone).
marker_re='\.changed_markers($|\.)'

real_status="$(git --no-pager status --porcelain)"
if [ -n "$real_status" ]; then
  real_dirty_count="$(printf '%s\n' "$real_status" | grep -v -E "$marker_re" | wc -l || true)"
else
  real_dirty_count=0
fi

# --- Fail loudly instead of writing a plausible-but-wrong index -------------
head_sha="$(git rev-parse --verify -q HEAD)" || {
  echo "gen_changed_markers: no HEAD commit to diff against" >&2
  exit 1
}
[ -f "$git_dir/index" ] || {
  echo "gen_changed_markers: no index at $git_dir/index (did git crash?)" >&2
  exit 1
}

# Unique per run: two concurrent invocations must not share a staging area.
# The file is then REMOVED before use -- a leftover empty file makes git die
# with "index file smaller than expected", while a missing one is fine because
# we cp the real index onto it.
tmp_index="$(mktemp)"
rm -f "$tmp_index"
cp "$git_dir/index" "$tmp_index"
export GIT_INDEX_FILE="$tmp_index"

# Stage everything -- into the shadow index only.
git add -A

TEXTS=( '*.py' '*.md' '*.json' '*.txt' '*.toml' '*.ini' '*.cfg' '*.yaml' '*.yml'
        '*.sh' '*.html' '*.css'
        # An extension allowlist is BLIND to files with no extension, and that
        # included .clinerules, .gitattributes and .gitignore themselves: an
        # edit to the rules file produced an index with no sign of it. This spec
        # matched exactly 6 tracked paths and none of them were binary. Any
        # extensionless text file elsewhere in the tree must be named here too.
        ':(top).*' )
EXCLUDES=( ':(exclude).changed_markers' ':(exclude).changed_markers.*' )

{
  echo "# AUTO-GENERATED -- do not edit. Regenerate: scripts/gen_changed_markers.sh"
  echo "# HEAD: $head_sha"
  echo "# generated: $(date -Iseconds)"
  echo "# dirty: $real_dirty_count path(s), this file excluded"
  echo "#"
  echo "# VALIDATE ME BEFORE SCOPING ANY WORK:"
  echo "#   - '# HEAD:' must equal \`git rev-parse HEAD\`"
  echo "#   - the DIRTY block below must match \`git status --porcelain\`"
  echo "#     (ignoring .changed_markers itself -- tracked and rewritten by this"
  echo "#      script -- and .changed_markers.*, which is gitignored scratch)."
  echo "#   - git status collapses an untracked DIRECTORY to one line while the"
  echo "#     diff below expands it to one entry per file. Expected, not drift."
  echo "#   - cross-check against .agent_notes.md, which is hand-owned."
  echo "# On ANY disagreement: STOP, regenerate, re-read. See .clinerules."
  echo "#"
  echo "# ---- DIRTY (real index; one line per git status --porcelain entry) ----"
  if [ -n "$real_status" ]; then
    printf '%s\n' "$real_status" | grep -v -E "$marker_re" | sed 's/^/#   /' || true
    if printf '%s\n' "$real_status" | grep -q -E "$marker_re"; then
      echo "#   (plus .changed_markers and its scratch file: excluded by design)"
    fi
  else
    echo "#   (nothing dirty: HEAD == worktree, excluding this file)"
  fi
  echo "#"
  echo "# ------ SCOPE SUMMARY (read this before the hunks) ------"
  git --no-pager diff --no-color --no-ext-diff --stat HEAD -- \
    "${TEXTS[@]}" "${EXCLUDES[@]}"
  echo
  echo "# ------ FULL HUNKS (3 lines of context) ------"
  echo "# Anchor edits to function/class names + hunk context, never line numbers."
  git --no-pager diff --no-color --no-ext-diff -U3 -p HEAD -- \
    "${TEXTS[@]}" "${EXCLUDES[@]}"
  echo
  echo "# ------ RENAME GUARD ------"
  echo "# If the hunks above contain 'rename from', run before committing:"
  echo "#   git grep -n \"old_name\"   # who still points at the old name?"
  echo "#   git grep -n \"new_name\"   # did the new name actually land?"
  echo "# Disagreement means UNKNOWN -> stop and ask. See .clinerules."
  if git --no-pager diff --no-color --no-ext-diff --summary -M HEAD -- \
       "${TEXTS[@]}" "${EXCLUDES[@]}" | grep -qE '^ (rename|copy) '; then
    echo "# RENAMES DETECTED (rename/copy lines only):"
    git --no-pager diff --no-color --no-ext-diff --summary -M HEAD -- \
      "${TEXTS[@]}" "${EXCLUDES[@]}" | grep -E '^ (rename|copy) ' | sed 's/^/## /'
  else
    echo "# (no renames detected in this diff)"
  fi
} > "$tmp_markers"

mv "$tmp_markers" "$markers"
echo "gen_changed_markers: wrote $markers ($(wc -l < "$markers") lines)"
