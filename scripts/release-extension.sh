#!/usr/bin/env bash
# Release the Chrome EXTENSION.
#
# Bumps the version in extension/manifest.config.ts and extension/package.json,
# commits that to main, pushes it, then creates and pushes an `ext-v*` tag on the
# bumped commit. The tag triggers CI to build the production .zip and attach it to
# a draft GitHub Release. (The Web Store upload stays manual.)
#
# Usage:
#   scripts/release-extension.sh [major|minor|patch|X.Y.Z]   (default: patch)
#   ASSUME_YES=1 scripts/release-extension.sh minor          (skip the prompt)
#
# Examples:
#   scripts/release-extension.sh         # 0.1.0 -> ext-v0.1.1 (+ manifest 0.1.1)
#   scripts/release-extension.sh minor   # 0.1.0 -> ext-v0.2.0
#   scripts/release-extension.sh 1.0.0   # -> ext-v1.0.0
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/release-common.sh"

TAG_PREFIX="ext-v"
BUMP=${1:-patch}
EXT_DIR="$REPO_ROOT/extension"
MANIFEST="$EXT_DIR/manifest.config.ts"

command -v npm >/dev/null 2>&1 || die "npm is required (it bumps package.json)"
[ -f "$MANIFEST" ] || die "missing $MANIFEST"

require_clean_main

# Read the version currently declared in the manifest source.
manifest_ver=$(sed -n -E "s/^[[:space:]]*version:[[:space:]]*'([0-9]+\.[0-9]+\.[0-9]+)'.*/\1/p" "$MANIFEST" | head -1)
[ -n "$manifest_ver" ] || die "couldn't read a version from $MANIFEST"

# "Previous" = the higher of the latest ext-v tag and the in-tree manifest
# version, so bootstrapping (no tags yet) never regresses below the manifest.
tag_ver=$(latest_tag_version "$TAG_PREFIX")
prev=$(printf '%s\n%s\n' "${tag_ver:-0.0.0}" "$manifest_ver" | _semver_sort | tail -1)
new=$(bump_version "$prev" "$BUMP")
tag="${TAG_PREFIX}${new}"
ensure_tag_free "$tag"

step "Extension release: ${prev} -> ${new}   (tag: ${tag})"
info "This will:"
info "  1. set version ${new} in manifest.config.ts + package.json"
info "  2. commit to main and push"
info "  3. tag ${tag} and push it (triggers the build + draft release)"
confirm "Proceed?" || die "aborted"

# 1a) manifest.config.ts — replace only the `version: '...'` line (manifest_version
#     is a different key and is left untouched).
tmp=$(mktemp)
sed -E "s/(^[[:space:]]*version:[[:space:]]*)'[0-9]+\.[0-9]+\.[0-9]+'/\1'${new}'/" "$MANIFEST" >"$tmp"
mv "$tmp" "$MANIFEST"
grep -q "version: '${new}'" "$MANIFEST" || die "failed to update version in $MANIFEST"

# 1b) package.json (+ lockfile) — let npm own this; skip npm's own commit/tag.
( cd "$EXT_DIR" && npm version "$new" --no-git-tag-version --allow-same-version >/dev/null )

# 2) commit the bump to main and push it.
git -C "$REPO_ROOT" add "$MANIFEST" "$EXT_DIR/package.json" "$EXT_DIR/package-lock.json"
git -C "$REPO_ROOT" commit -m "release: extension v${new}"
git -C "$REPO_ROOT" push origin main

# 3) tag that commit and push the tag.
git -C "$REPO_ROOT" tag -a "$tag" -m "Extension release ${new}"
git -C "$REPO_ROOT" push origin "$tag"

step "Pushed the v${new} bump + ${tag}. The extension build / draft release is running."
