#!/usr/bin/env bash
# py_dead_scan.sh - repo-agnostic Python dead-code and hygiene runner.
#
#   ./py_dead_scan.sh --changed [target]   lint/type/dead-check what this session touched
#   ./py_dead_scan.sh --sweep   [target]   whole-repo dead code + unused deps
#
# Every missing tool prints SKIP and the run continues: check what exists with
# command -v, never assume an install and never install one from here.
set -euo pipefail

MODE="${1:---changed}"
TARGET="${2:-.}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

has() { command -v "$1" >/dev/null 2>&1; }

LIST=""
cleanup() { rm -f ${LIST:+"$LIST"}; }
trap cleanup EXIT

collect_py() {
  local -a out=()
  while IFS= read -r -d '' f; do out+=("$f"); done < <(
    { git diff --name-only -z --diff-filter=ACMR HEAD -- '*.py'
      git ls-files  -z --others --exclude-standard    -- '*.py'
    } 2>/dev/null
  )
  [ "${#out[@]}" -gt 0 ] && printf '%s\0' "${out[@]}"
  return 0
}

if [ "$MODE" = "--changed" ]; then
  PY_FILES=()
  while IFS= read -r -d '' f; do PY_FILES+=("$f"); done < <(collect_py)

  if [ "${#PY_FILES[@]}" -eq 0 ]; then
    echo "No changed .py files."
  else
    printf 'scanning %d changed file(s):\n' "${#PY_FILES[@]}"
    printf '  %s\n' "${PY_FILES[@]}"

    echo "=== [1] Unused imports/vars, undefined names, syntax ==="
    if has ruff; then
      ruff check --select F401,F841,F821,E9 -- "${PY_FILES[@]}" || true
    elif has flake8; then
      flake8 --select=F,E9 -- "${PY_FILES[@]}" || true
    else
      echo "SKIP: neither ruff nor flake8 found"
    fi

    echo "=== [2] Unreachable branches ==="
    if has mypy; then
      # --follow-imports=silent: a per-file run has no package context, so it
      # follows imports without reporting on them. Errors in the target file
      # still surface; phantom "cannot find implementation" noise does not.
      mypy --warn-unreachable --enable-error-code truthy-bool \
           --follow-imports=silent -- "${PY_FILES[@]}" || true
    else
      echo "SKIP: mypy not found"
    fi

    echo "=== [3] Dead symbols, filtered exactly to changed files ==="
    if has vulture; then
      WL=()
      [ -f vulture_whitelist.py ] && WL=(vulture_whitelist.py)
      LIST="$(mktemp)"
      printf '%s\n' "${PY_FILES[@]}" > "$LIST"
      vulture "$TARGET" "${WL[@]+"${WL[@]}"}" --min-confidence 80 \
        | awk -F: 'NR==FNR{a[$0]=1; next} {f=$1; sub(/^\.\//, "", f); if (f in a) print}' "$LIST" - \
        || true
    else
      echo "SKIP: vulture not found"
    fi
  fi

  # --no-renames exposes both deleted modules AND old paths of renamed modules.
  GONE="$(git diff --no-renames --name-only --diff-filter=D HEAD -- '*.py' || true)"
  if [ -n "$GONE" ]; then
    echo "=== [4] Deleted/renamed-from modules - verify no importer remains ==="
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      base="$(basename "$f" .py)"
      if [ "$base" = "__init__" ]; then
        mod="$(basename "$(dirname "$f")")"
        [ "$mod" = "." ] && continue
      else
        mod="$base"
      fi
      hits="$(git grep -n -E -- "(^|[[:space:]])(from|import)[[:space:]].*\b${mod}\b" -- '*.py' 2>/dev/null | head -5 || true)"
      if [ -n "$hits" ]; then printf '  %s still referenced:\n%s\n' "$f" "$hits"
      else printf '  %s: no remaining importer\n' "$f"; fi
    done <<< "$GONE"
  fi
  exit 0
fi

if [ "$MODE" = "--sweep" ]; then
  echo "=== [BASELINE SWEEP] Cross-module dead code ==="
  if has deadcode; then deadcode "$TARGET" || true
  elif has skylos; then skylos "$TARGET" || true
  elif has vulture; then
    WL=(); [ -f vulture_whitelist.py ] && WL=(vulture_whitelist.py)
    vulture "$TARGET" "${WL[@]+"${WL[@]}"}" --min-confidence 80 --sort-by-size || true
  else
    echo "SKIP: install vulture, deadcode, or skylos for whole-repo dead code"
  fi

  echo "=== [BASELINE SWEEP] Unused dependencies ==="
  if has deptry; then deptry "$TARGET" || true
  else echo "SKIP: deptry not found"; fi

  echo "=== [BASELINE SWEEP] Unreachable branches ==="
  if has mypy; then mypy --warn-unreachable --enable-error-code truthy-bool "$TARGET" || true
  else echo "SKIP: mypy not found"; fi
  exit 0
fi

echo "Usage: $0 [--changed | --sweep] [target_dir]" >&2
exit 2
