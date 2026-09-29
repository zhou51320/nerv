#!/usr/bin/env bash
set -euo pipefail

# Poll the single-version KVMem Win7 CUDA workflow. The default interval is
# deliberately 30 minutes because a clean CUDA build takes about 20 minutes.
# If a run fails, keep watching until a newer push starts a replacement run.
REPO="${REPO:-zhou51320/nerv}"
WORKFLOW="${WORKFLOW:-build-kvmem-win7-cuda.yml}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"

command -v gh >/dev/null || { echo 'gh is required' >&2; exit 2; }

last_run="${1:-}"
while true; do
    row="$(gh run list --repo "$REPO" --workflow "$WORKFLOW" --limit 1 \
        --json databaseId,status,conclusion,headSha,url \
        --jq '.[0] // empty')"
    if [[ -z "$row" ]]; then
        echo "[$(date -Is)] no workflow run yet; sleeping ${INTERVAL_SECONDS}s"
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    run_id="$(jq -r '.databaseId' <<<"$row")"
    status="$(jq -r '.status' <<<"$row")"
    conclusion="$(jq -r '.conclusion // empty' <<<"$row")"
    url="$(jq -r '.url' <<<"$row")"
    echo "[$(date -Is)] run=$run_id status=$status conclusion=$conclusion $url"

    if [[ "$status" == completed && "$conclusion" == success ]]; then
        echo "workflow succeeded: $url"
        exit 0
    fi

    if [[ "$status" == completed && "$conclusion" != success ]]; then
        if [[ "$run_id" != "$last_run" ]]; then
            echo 'latest run failed; waiting for a newer replacement run'
            gh run view "$run_id" --repo "$REPO" --log-failed 2>&1 | tail -80 || true
            last_run="$run_id"
        fi
    else
        last_run="$run_id"
    fi

    sleep "$INTERVAL_SECONDS"
done
