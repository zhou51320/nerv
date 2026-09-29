#!/usr/bin/env bash
set -u

repo="${1:-zhou51320/nerv}"
run_id="${2:?usage: $0 [owner/repo] <run-id> [interval-seconds]}"
interval="${3:-1800}"

while :; do
  status_json=$(gh run view "$run_id" --repo "$repo" --json status,conclusion,url)
  status=$(printf '%s' "$status_json" | jq -r '.status')
  conclusion=$(printf '%s' "$status_json" | jq -r '.conclusion')
  url=$(printf '%s' "$status_json" | jq -r '.url')
  printf '%s status=%s conclusion=%s %s\n' "$(date -u +%FT%TZ)" "$status" "$conclusion" "$url"
  if [[ "$status" == "completed" ]]; then
    if [[ "$conclusion" == "success" ]]; then exit 0; else exit 1; fi
  fi
  sleep "$interval"
done
