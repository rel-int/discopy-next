#!/bin/bash
# Install the full development environment in a Claude Code on the web
# session, so that `uv run pflake8 discopy` and `uv run coverage run -m pytest`
# run as CONTRIBUTING.md says, extras included.
#
# `pyproject.toml` pins torch to the CPU index at download.pytorch.org on
# Linux, whose index page links the wheels and their metadata from
# download-r2.pytorch.org. A session whose network policy does not allow both
# hosts cannot sync the `quantum` and `grammar` extras, nor run `uv lock`, and
# every torch test used to be skipped: the fallback syncs everything but torch
# and installs the locked version from PyPI, which ships CUDA wheels that run
# on the CPU. The allowlist is the better fix.
#
# Every session, local ones included, registers the CHANGELOG.md merge driver
# of the merge-main skill, which `.gitattributes` names; without this config
# git merges the file as text. A checkout without the driver's script merges
# it as text too.
set -uo pipefail

log() { echo "session-start: $*" >&2; }
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 0

driver=.claude/skills/merge-main/changelog-merge.py
if git rev-parse --git-dir >/dev/null 2>&1; then
  git config merge.changelog.name "CHANGELOG.md: main's file plus the branch's own changes"
  git config merge.changelog.driver "if [ -f $driver ]; then python3 -I $driver %O %A %B; else git merge-file -L ours -L base -L theirs %A %O %B; fi"
fi

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

if uv sync --dev --group all >/dev/null 2>&1; then
  log "uv sync --dev --group all"
  exit 0
fi

log "uv sync --dev --group all failed, syncing without torch"
if ! uv sync --dev --group all --no-install-package torch >/dev/null 2>&1; then
  log "uv sync failed: run 'uv sync --dev' by hand"
  exit 0
fi

version="$(awk '/^name = "torch"$/ { found = 1 } found && /^version/ { print $3; exit }' uv.lock | tr -d '"')"
if uv pip install --index-url https://pypi.org/simple "torch==$version" >/dev/null 2>&1; then
  log "torch $version installed from PyPI instead of download.pytorch.org"
else
  log "torch unavailable: run 'uv run pytest --skip-extra'"
fi
exit 0
