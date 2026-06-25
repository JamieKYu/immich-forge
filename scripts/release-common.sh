#!/usr/bin/env bash
# Shared helpers for the component release scripts (release-server.sh and
# release-extension.sh). Sourced, not executed directly.
#
# Portable to the stock macOS bash 3.2 + BSD userland: no `sort -V`, no `-i`
# in-place sed, no bash-4 features.

set -euo pipefail

# ---- output -----------------------------------------------------------------
_c_bold=$(printf '\033[1m'); _c_red=$(printf '\033[31m'); _c_rst=$(printf '\033[0m')
info() { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "$_c_bold" "$_c_rst" "$*"; }
die()  { printf '%serror:%s %s\n' "$_c_red" "$_c_rst" "$*" >&2; exit 1; }

# ---- semver -----------------------------------------------------------------
is_semver() { case "$1" in [0-9]*.[0-9]*.[0-9]*) return 0 ;; *) return 1 ;; esac; }

# Numeric major.minor.patch sort (BSD-sort friendly), highest last.
_semver_sort() { sort -t. -k1,1n -k2,2n -k3,3n; }

# Highest released version (X.Y.Z, prefix stripped) among tags "${1}*", or empty.
# $1 is the FULL tag prefix including the v, e.g. "ext-v".
latest_tag_version() {
  prefix=$1
  git tag --list "${prefix}[0-9]*.[0-9]*.[0-9]*" 2>/dev/null \
    | sed "s/^${prefix}//" \
    | _semver_sort \
    | tail -1
}

# bump_version CURRENT BUMP  ->  new X.Y.Z on stdout.
# BUMP is one of: major | minor | patch | an explicit X.Y.Z.
bump_version() {
  current=$1; bump=${2:-patch}
  if is_semver "$bump"; then echo "$bump"; return; fi
  is_semver "$current" || die "current version '$current' is not X.Y.Z"
  major=${current%%.*}; rest=${current#*.}; minor=${rest%%.*}; patch=${rest##*.}
  case "$bump" in
    major) major=$((major + 1)); minor=0; patch=0 ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    patch) patch=$((patch + 1)) ;;
    *) die "unknown bump '$bump' (use major|minor|patch or an explicit X.Y.Z)" ;;
  esac
  echo "$major.$minor.$patch"
}

# ---- git preflight ----------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a git repository"

# Releases publish straight from main, so refuse to run anywhere risky: wrong
# branch, uncommitted changes, or a local main that has diverged from origin.
require_clean_main() {
  branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
  [ "$branch" = "main" ] || die "must be on 'main' to release (currently on '$branch')"
  [ -z "$(git -C "$REPO_ROOT" status --porcelain)" ] \
    || die "working tree is dirty — commit or stash changes first"
  step "Syncing tags + main from origin…"
  git -C "$REPO_ROOT" fetch --quiet origin main --tags
  local_ref=$(git -C "$REPO_ROOT" rev-parse @)
  remote_ref=$(git -C "$REPO_ROOT" rev-parse '@{u}')
  [ "$local_ref" = "$remote_ref" ] \
    || die "local 'main' is out of sync with origin/main — pull/push to reconcile first"
}

# Abort if the tag already exists (locally or on the remote we just fetched).
ensure_tag_free() {
  tag=$1
  git -C "$REPO_ROOT" rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1 \
    && die "tag $tag already exists"
  return 0
}

# Prompt unless ASSUME_YES=1 in the environment.
confirm() {
  [ "${ASSUME_YES:-0}" = "1" ] && return 0
  printf '%s [y/N] ' "$1"
  read -r reply
  case "$reply" in y | Y | yes | YES) return 0 ;; *) return 1 ;; esac
}
