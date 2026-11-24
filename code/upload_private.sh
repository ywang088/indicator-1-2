#!/usr/bin/env bash
# Run only after gh auth login. Never creates or updates a public repository.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
gh auth status >/dev/null
owner="$(gh api user --jq .login)"
repo="$owner/parking-indicator-1-2"
if git remote get-url origin >/dev/null 2>&1; then
  echo 'An origin already exists; inspect it before uploading.' >&2
  exit 1
fi
if gh repo view "$repo" --json name >/dev/null 2>&1; then
  echo 'Repository name already exists; refusing to overwrite it.' >&2
  exit 1
fi
test -f release_assets/SHA256SUMS
gh repo create "$repo" --private --source . --remote origin --push
visibility="$(gh repo view "$repo" --json visibility --jq .visibility)"
test "$visibility" = PRIVATE
gh release create sealed-18-v1 --repo "$repo" --title '固定18条轨迹 Docker交付' --notes '代码、固定数据和容器实跑结果。解包说明见README。' release_assets/parking-data.tar.gz.part* release_assets/parking-results.tar.gz release_assets/SHA256SUMS release_assets/release_manifest.json
printf 'Private repository: https://github.com/%s\n' "$repo"
