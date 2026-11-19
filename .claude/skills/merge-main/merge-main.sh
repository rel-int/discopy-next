#!/bin/bash
# Merge origin/main into the current branch: CHANGELOG.md by the skill's
# driver, then rename-check. Never pushes, never rebases (RULES.md 4).
# Usage: merge-main.sh [<what came in>], from anywhere in the checkout.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$(git rev-parse --show-toplevel)" || { echo "merge-main: not in a git checkout"; exit 2; }
rel="${here#"$(pwd)"/}"
branch="$(git symbolic-ref --short -q HEAD)" || { echo "merge-main: detached HEAD, check out the PR branch"; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "merge-main: uncommitted changes, commit or stash them first"; exit 2; }
git fetch -q origin main || { echo "merge-main: fetch origin main failed"; exit 2; }
if git fetch -q origin "$branch" 2>/dev/null && git rev-parse -q --verify "origin/$branch" >/dev/null && ! git merge-base --is-ancestor "origin/$branch" HEAD; then
  git merge -q --ff-only "origin/$branch" 2>/dev/null \
    || { echo "merge-main: $branch and origin/$branch diverged; git merge origin/$branch first (never rebase)"; exit 2; }
  echo "merge-main: fast-forwarded to origin/$branch ($(git rev-parse --short HEAD))"
fi
if git merge-base --is-ancestor origin/main HEAD; then
  echo "merge-main: $branch already contains origin/main ($(git rev-parse --short origin/main))"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$here/changelog-merge.py" "$tmp/"
global="$(git config --path core.attributesFile || true)"
{ [ -n "$global" ] && [ -f "$global" ] && cat "$global"; echo "CHANGELOG.md merge=changelog"; } > "$tmp/attributes"

count="$(git rev-list --count HEAD..origin/main)"
what="${1:-$(git log --first-parent --reverse --format=%s HEAD..origin/main | grep -o '^Merge pull request #[0-9]*' | grep -o '#[0-9]*' | paste -sd ' ' -)}"
what="${what:-$count commits}"
message="Merge origin/main into $branch: $what"
trailers="$(git config --get-all next.trailer)"
full="$message"
[ -n "$trailers" ] && full="$message"$'\n\n'"$trailers"

git -c core.attributesFile="$tmp/attributes" \
  -c merge.changelog.name="CHANGELOG.md: main's file plus the branch's own changes" \
  -c merge.changelog.driver="python3 -I $tmp/changelog-merge.py %O %A %B" \
  merge -q --no-commit --no-ff -m "$full" origin/main >/dev/null 2>"$tmp/err"
conflicted="$(git diff --name-only --diff-filter=U)"
if [ -n "$conflicted" ]; then
  echo "merge-main: conflicts left in:"
  echo "$conflicted" | sed 's/^/  /'
  echo "merge-main: resolve them, run python3 -I $rel/rename-check.py, then git commit --no-edit"
  exit 1
fi
if ! git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  cat "$tmp/err"
  echo "merge-main: git merge failed"
  exit 2
fi

hits="$(python3 -I "$here/rename-check.py")"
rc=$?
if [ "$rc" -ne 0 ] && [ -z "$hits" ]; then
  echo "merge-main: rename-check failed (exit $rc); merge left uncommitted"
  exit 2
fi
if [ -n "$hits" ]; then
  echo "$hits"
  echo "merge-main: $(echo "$hits" | wc -l) references to names main removed; fix them, then git commit --no-edit"
  exit 1
fi
git commit -q --no-edit || { echo "merge-main: git commit failed"; exit 2; }

base="$(git merge-base HEAD^1 HEAD^2)"
changelog=untouched
if ! git diff --quiet "$base" HEAD^1 -- CHANGELOG.md && ! git diff --quiet "$base" HEAD^2 -- CHANGELOG.md; then
  changelog=merged
fi
echo "merge-main: $(git rev-parse --short HEAD) $message ($count commits, CHANGELOG.md $changelog, rename-check clean); not pushed: run next-checks, then git push"
