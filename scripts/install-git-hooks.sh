#!/usr/bin/env bash
#
# Point this repo's git at the tracked hooks in scripts/git-hooks/.
# Run once after cloning: ./scripts/install-git-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
git -C "$REPO_ROOT" config core.hooksPath scripts/git-hooks
chmod +x "$REPO_ROOT"/scripts/git-hooks/* 2>/dev/null || true
echo "Installed git hooks (core.hooksPath = scripts/git-hooks)."
