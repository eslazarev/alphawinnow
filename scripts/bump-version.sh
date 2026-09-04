#!/usr/bin/env bash
# Raise the workspace version and refresh the lockfile.
#
#   scripts/bump-version.sh [auto|patch|minor|major]
#
# Prints the new version on stdout; everything explanatory goes to stderr so the
# caller can capture the version directly. `auto` derives the level from the
# Conventional Commit subjects since the last `v*` tag.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT_DIR}/Cargo.toml"
LEVEL="${1:-auto}"

case "${LEVEL}" in
  auto | patch | minor | major) ;;
  *)
    echo "Error: level must be auto, patch, minor, or major; got '${LEVEL}'." >&2
    exit 1
    ;;
esac

current="$(awk -F'"' '/^version = "/ {print $2; exit}' "${MANIFEST}")"
if [[ ! "${current}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  echo "Error: cannot read a three-part version from ${MANIFEST}; got '${current}'." >&2
  exit 1
fi
major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"
patch="${BASH_REMATCH[3]}"

# Classify the change set when the caller did not decide for us.
if [[ "${LEVEL}" == "auto" ]]; then
  last_tag="$(git -C "${ROOT_DIR}" describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
  range="${last_tag:+${last_tag}..}HEAD"
  subjects="$(git -C "${ROOT_DIR}" log --format='%s%n%b' "${range}" 2>/dev/null || true)"

  if grep -qiE '^BREAKING[ -]CHANGE:|^[a-z]+(\([^)]*\))?!:' <<< "${subjects}"; then
    change="breaking"
  elif grep -qiE '^feat(\([^)]*\))?:' <<< "${subjects}"; then
    change="feature"
  else
    change="fix"
  fi
  echo "Change class since ${last_tag:-the first commit}: ${change}" >&2

  # Below 1.0 the minor position carries breakage, because Cargo treats
  # 0.1.x as one compatibility range and 0.2.0 as a different one. Promoting a
  # breaking change to `major` there would claim a stability this crate has
  # not declared yet.
  if [[ "${major}" -eq 0 ]]; then
    case "${change}" in
      breaking) LEVEL="minor" ;;
      *) LEVEL="patch" ;;
    esac
  else
    case "${change}" in
      breaking) LEVEL="major" ;;
      feature) LEVEL="minor" ;;
      *) LEVEL="patch" ;;
    esac
  fi
fi

case "${LEVEL}" in
  major) major=$((major + 1)); minor=0; patch=0 ;;
  minor) minor=$((minor + 1)); patch=0 ;;
  patch) patch=$((patch + 1)) ;;
esac
next="${major}.${minor}.${patch}"

# Rewrite only the first `version = "..."`, which is the one under
# [workspace.package]; dependency versions are `name = { version = ... }`.
awk -v next_version="${next}" '
  BEGIN { done = 0 }
  done == 0 && /^version = "/ { print "version = \"" next_version "\""; done = 1; next }
  { print }
  END { if (!done) exit 1 }
' "${MANIFEST}" > "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "${MANIFEST}"

# Cargo.lock records the workspace version too; a stale lockfile would fail
# every `--locked` build in the release run that follows.
cargo update --workspace --manifest-path "${MANIFEST}" --quiet

echo "Bumped ${current} -> ${next} (${LEVEL})" >&2
echo "${next}"
