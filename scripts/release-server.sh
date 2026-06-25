#!/usr/bin/env bash
# Release the Forge SERVER.
#
# Computes the next `api-v*` version tag from the latest one and pushes it.
# Pushing the tag is what triggers CI to build and publish the Docker image
# (:latest and :vX.Y.Z). No files change — the image version is the tag.
#
# Usage:
#   scripts/release-server.sh [major|minor|patch|X.Y.Z]   (default: patch)
#   ASSUME_YES=1 scripts/release-server.sh minor          (skip the prompt)
#
# Examples:
#   scripts/release-server.sh            # 1.2.3 -> api-v1.2.4
#   scripts/release-server.sh minor      # 1.2.3 -> api-v1.3.0
#   scripts/release-server.sh 2.0.0      # -> api-v2.0.0
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/release-common.sh"

TAG_PREFIX="api-v"
BUMP=${1:-patch}

require_clean_main

prev=$(latest_tag_version "$TAG_PREFIX"); prev=${prev:-0.0.0}
new=$(bump_version "$prev" "$BUMP")
tag="${TAG_PREFIX}${new}"
ensure_tag_free "$tag"

step "Server release: ${prev} -> ${new}   (tag: ${tag})"
info "Pushing this tag triggers the Docker build + publish (:latest and :v${new})."
confirm "Create and push ${tag}?" || die "aborted"

git -C "$REPO_ROOT" tag -a "$tag" -m "Server release ${new}"
git -C "$REPO_ROOT" push origin "$tag"

step "Pushed ${tag}. Track the build on the repo's Actions tab."
