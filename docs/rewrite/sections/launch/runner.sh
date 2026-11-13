#!/bin/bash
# runner.sh K: the queue's runner for slot K (GPU K), as a supervisor program. It recovers
# its slot's jobs that a restart interrupted (queued again as resumes) and runs the next
# ready job, one at a time, with only GPU K visible. No bench lock on a box.
set -u
cd "$(dirname "$0")/../../../.."
source docs/rewrite/sections/launch/env.sh
exec ${BNB:-.venv/bin/bnb} queue run --slot "$1" --lock none --poll 10 --queue-file "$WORKSPACE/queue/queue.json"
