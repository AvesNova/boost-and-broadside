#!/bin/bash
# hub.sh COMMAND ...: the Hugging Face hub as the transfer path between the boxes and the
# laptop, beside ssh/rsync over the direct address. Uploads resume where they stopped and
# nothing rides the vast ssh relay (which drops transfers longer than about 2 minutes).
#
#   hub.sh push-cache ARCHIVE [NAME]     the compile-cache archive (cache.sh export) and its
#                                        SHA-256, to <prefix>/caches/NAME (default its file name)
#   hub.sh pull-cache NAME DIR           download it into DIR and check the SHA-256
#   hub.sh push-runs [--loop MIN] [RUN ...]   each run directory whole (checkpoints too, so
#                                        another box can continue it; half-written .tmp left
#                                        out), the ratings and the launch records, to
#                                        <prefix>/runs/; every MIN minutes with --loop
#   hub.sh pull-runs DIR [RUN ...] [--checkpoints]   download runs (default all) into DIR,
#                                        without the training checkpoints unless asked
#
# The repo: $HUB_REPO (default julianzia/boost-and-broadside-runs, public, the project's run
# store), under $HUB_PREFIX (default final). Uploads need a write token, HF_TOKEN, read
# from $WORKSPACE/.env on a box (`HF_TOKEN=hf_...`); downloads of the public repo need none.
# The `hf` command runs from uv's tool cache (`uvx`): nothing is added to the locked env.
set -euo pipefail
WORKSPACE=${WORKSPACE:-/workspace}
[ -f "$WORKSPACE/.env" ] && { set -a; . "$WORKSPACE/.env"; set +a; }
REPO=${HUB_REPO:-julianzia/boost-and-broadside-runs}
PREFIX=${HUB_PREFIX:-final}
hf() { uvx --quiet --from 'huggingface_hub[hf_xet]' hf "$@"; }
say() { echo "$(date -u +%FT%TZ) hub: $*"; }

push_runs() {
  local runs=$WORKSPACE/runs names=("$@")
  [ ${#names[@]} -gt 0 ] || mapfile -t names < <(ls "$runs")
  for name in "${names[@]}"; do
    [ -d "$runs/$name" ] || continue
    for attempt in 1 2 3; do
      if hf upload "$REPO" "$runs/$name" "$PREFIX/runs/$name" --exclude "*.tmp/*" \
          --exclude "*.tmp/**" --commit-message "sync $name" > /dev/null; then
        say "pushed $name"; break
      fi
      say "push $name failed (attempt $attempt)"; sleep 30
    done
  done
}

case "${1:-}" in
  push-cache)
    archive=${2:?push-cache ARCHIVE [NAME]}; name=${3:-$(basename "$archive")}
    sha256sum "$archive" | cut -d' ' -f1 > "$archive.sha256"
    hf upload "$REPO" "$archive" "$PREFIX/caches/$name" --commit-message "cache $name" > /dev/null
    hf upload "$REPO" "$archive.sha256" "$PREFIX/caches/$name.sha256" > /dev/null
    say "pushed $archive ($(du -h "$archive" | cut -f1), sha256 $(cat "$archive.sha256"))"
    ;;
  pull-cache)
    name=${2:?pull-cache NAME DIR}; dir=${3:?pull-cache NAME DIR}
    mkdir -p "$dir"
    hf download "$REPO" "$PREFIX/caches/$name" "$PREFIX/caches/$name.sha256" --local-dir "$dir" > /dev/null
    file=$dir/$PREFIX/caches/$name
    if [ "$(sha256sum "$file" | cut -d' ' -f1)" != "$(cat "$file.sha256")" ]; then
      say "SHA-256 MISMATCH for $file: download it again"; exit 1
    fi
    say "pulled $file (sha256 checked)"
    ;;
  push-runs)
    shift
    if [ "${1:-}" = --loop ]; then
      every=${2:?--loop MINUTES}; shift 2
      while true; do push_runs "$@"; sleep $((every * 60)); done
    fi
    push_runs "$@"
    ;;
  pull-runs)
    dir=${2:?pull-runs DIR [RUN ...] [--checkpoints]}; shift 2
    includes=(); excludes=(--exclude "*/checkpoints/*")
    for arg in "$@"; do
      if [ "$arg" = --checkpoints ]; then excludes=(); else includes+=(--include "$PREFIX/runs/$arg/*"); fi
    done
    [ ${#includes[@]} -gt 0 ] || includes=(--include "$PREFIX/runs/*")
    hf download "$REPO" "${includes[@]}" "${excludes[@]}" --local-dir "$dir" > /dev/null
    say "pulled into $dir/$PREFIX/runs"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
