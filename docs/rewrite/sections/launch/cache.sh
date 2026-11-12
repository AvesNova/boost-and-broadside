#!/bin/bash
# cache.sh export ARCHIVE | import ARCHIVE: the compile caches (inductor and Triton) as one
# zstd tar, with a manifest of what they are valid for: the GPU model, torch, Triton, CUDA
# and the code's commit. A cache entry is keyed by all of those and by the shapes, so it
# helps only a box of the same GPU model running the same tree (the import warns on any
# difference and imports anyway: entries that do not match are simply not used).
#
# Moving it between boxes: through the laptop, over each box's DIRECT address (the vast
# ssh relay, sshN.vast.ai, drops transfers longer than about 2 minutes):
#   scp -P PORT root@IP:/workspace/compile-cache.tar.zst .           (from the dry-run box)
#   scp -P PORT2 compile-cache.tar.zst root@IP2:/workspace/          (to the next box)
set -euo pipefail
action=${1:?usage: cache.sh export|import ARCHIVE}
archive=${2:?usage: cache.sh export|import ARCHIVE}
source "$(dirname "$0")/env.sh"
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
manifest() {
  "${PYTHON:-$repo/.venv/bin/python}" - "$repo" <<'PY'
import json, subprocess, sys, torch, triton
commit = subprocess.run(["git", "-C", sys.argv[1], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(json.dumps(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                      triton=triton.__version__, cuda=torch.version.cuda, commit=commit)))
PY
}
case "$action" in
  export)
    manifest > $WORKSPACE/cache/manifest.json
    tar -C $WORKSPACE/cache -I "zstd -T0 -3" -cf "$archive" manifest.json inductor triton
    echo "exported $(du -h "$archive" | cut -f1) to $archive: $(cat $WORKSPACE/cache/manifest.json)"
    ;;
  import)
    mkdir -p $WORKSPACE/cache/imported
    tar -C $WORKSPACE/cache/imported -I zstd -xf "$archive" manifest.json
    theirs=$(cat $WORKSPACE/cache/imported/manifest.json); ours=$(manifest)
    if [ "$theirs" != "$ours" ]; then
      echo "WARNING: the archive was made for $theirs; this box is $ours (entries that differ are not used)"
    fi
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    tar -C $WORKSPACE/cache -I zstd -xf "$archive" inductor triton
    echo "imported $archive into $WORKSPACE/cache"
    ;;
  *) echo "usage: cache.sh export|import ARCHIVE"; exit 2 ;;
esac
