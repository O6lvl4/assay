#!/usr/bin/env bash
# Run golemide against Terminal-Bench 2.0 and print the score.
#
#   bench/tb2.sh pilot                 the 10-task spread below
#   bench/tb2.sh all                   all 89 tasks
#   bench/tb2.sh fix-git regex-log     named tasks
#
# Environment: HB (path to the hb binary), GOLEMIDE_BINARY, MODEL (default cf:glm-5.3),
# CONCURRENT (default 4), ATTEMPTS (harbor attempts per task, default 1),
# AGENT_TIMEOUT_MULT (default 1.0 = 1800s per trial).
#
# Why a pilot before the full slate: 89 tasks at up to 30 minutes each is over five hours
# of wall clock even at concurrency 4, and a systematic fault in the adapter -- a verify
# derivation that misfires, an arch mismatch, a cost line that never appears -- costs the
# whole run to discover. The pilot spans the shapes cheaply: git surgery, regex and log
# work, HTML parsing, a certificate, a two-language build, a corrupted database, a formal
# proof, and a data-processing task. About an hour, and pennies.
#
# What `-n` is NOT: it is --n-concurrent, the parallelism. Attempts per task is -k. An
# earlier run of this harness passed `-n 1` believing it set trials; it set concurrency.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

HB="${HB:?set HB to the hb binary}"
GOLEMIDE_BINARY="${GOLEMIDE_BINARY:?set GOLEMIDE_BINARY to the golemide binary}"
MODEL="${MODEL:-cf:glm-5.3}"
CONCURRENT="${CONCURRENT:-4}"
ATTEMPTS="${ATTEMPTS:-1}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1.0}"
JOBS_DIR="${JOBS_DIR:-$HERE/jobs}"

PILOT=(fix-git sanitize-git-repo regex-log log-summary-date-ranges filter-js-from-html
       openssl-selfsigned-cert polyglot-c-py sqlite-db-truncate prove-plus-comm
       count-dataset-tokens)

case "${1:-pilot}" in
  pilot) TASKS=("${PILOT[@]}") ;;
  all)   TASKS=() ;;
  *)     TASKS=("$@") ;;
esac

args=(run -a adapters.golemide_agent:GolemideAgent -m "$MODEL"
      -d terminal-bench/terminal-bench-2
      -k "$ATTEMPTS" -n "$CONCURRENT"
      --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT"
      -o "$JOBS_DIR" -y)
for t in "${TASKS[@]+"${TASKS[@]}"}"; do args+=(-i "$t"); done

echo "model=$MODEL concurrent=$CONCURRENT attempts=$ATTEMPTS timeout_mult=$AGENT_TIMEOUT_MULT"
echo "tasks: ${TASKS[*]-<all 89>}"
echo

before="$(ls -1 "$JOBS_DIR" 2>/dev/null | sort | tail -1 || true)"
GOLEMIDE_BINARY="$GOLEMIDE_BINARY" PYTHONPATH=. "$HB" "${args[@]}"
after="$(ls -1 "$JOBS_DIR" | sort | tail -1)"

[ "$after" = "$before" ] && { echo "no new job directory appeared"; exit 1; }
echo
echo "job: $after"
bench/tb2-score.py "$JOBS_DIR/$after"

# Leftovers are their own finding: a killed trial can leave the host-side solve running,
# which then spends money with no record and starves the next trial of memory.
echo
echo "--- leftovers ---"
pgrep -fl 'golemide solve' || echo "no golemide processes"
docker ps --format '{{.Names}}' | grep -v '^harness-otel-' || echo "no task containers"
