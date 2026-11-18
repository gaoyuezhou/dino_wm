#!/bin/bash
# Sync this checkout with the remote, preserving local edits.
#
# Why not just `git stash; git pull; git stash pop`:
#   - `git stash` on a CLEAN tree stashes nothing but still exits 0, so the trailing
#     `git stash pop` would pop an UNRELATED older stash entry into the working tree.
#   - Without `set -e`, a failed pull (conflict / no network / detached HEAD) would
#     still run `git stash pop`, leaving a half-applied state.
# So: only stash when there is something to stash, and only pop what we pushed.
set -euo pipefail

cd "$(dirname "$0")"

# Refuse to run outside a git repo (guards against being invoked from another dir).
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "gitpull: not a git repository: $(pwd)" >&2
  exit 1
}

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "gitpull: repo=$(pwd) branch=${BRANCH}"

# Stash only if the tree is dirty (unstaged OR staged changes). Untracked files are
# left alone -- they don't conflict with a pull.
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  git stash push -m "gitpull-autostash"
  STASHED=1
  echo "gitpull: stashed local changes"
else
  echo "gitpull: working tree clean, nothing to stash"
fi

# --rebase keeps history linear (no merge commit on every sync). On failure we stop
# BEFORE popping, so we never pop onto a broken pull.
if ! git pull --rebase; then
  echo "gitpull: pull failed" >&2
  if [ "${STASHED}" -eq 1 ]; then
    echo "gitpull: your changes are safe in the stash -- restore with: git stash pop" >&2
  fi
  exit 1
fi

if [ "${STASHED}" -eq 1 ]; then
  # A conflicting pop leaves markers in the tree AND keeps the stash entry; say so
  # loudly rather than letting a half-merged config reach a training run.
  if git stash pop; then
    echo "gitpull: restored local changes"
  else
    echo "gitpull: CONFLICT restoring your changes -- resolve, then: git stash drop" >&2
    exit 1
  fi
fi

echo "gitpull: done -- $(git log -1 --oneline)"

