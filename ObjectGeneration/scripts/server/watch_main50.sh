#!/usr/bin/env bash
# Watch the main50 DualCraft batch on the server; exit (waking the supervisor) when
# something needs attention:
#   DONE      the batch printed BATCH_DONE
#   FAILED    an object finished with a non-zero exit since the watcher started, other than
#             "Agent hit max turns limit" (a genuine outcome under the senior's settings,
#             recorded as is and not retried)
#   DIED      no batch process on the server and no BATCH_DONE
#   STALLED   the batch log has not changed for STALL_MIN minutes
# Usage: watch_main50.sh <log name, e.g. batch_main.log> [stall minutes]
LOG=${1:-batch_main.log}
STALL_MIN=${2:-100}
O=/mnt/data/usr/wzt/submodules/dualcraft/runs_main50_qiniu_astra
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -p 1021 root@139.224.9.204"

snapshot() {
  $SSH "cd $O && printf 'LINES=%s\n' \$(wc -l < $LOG) && printf 'AGE=%s\n' \$(( \$(date +%s) - \$(stat -c %Y $LOG) )) \
    && printf 'ALIVE=%s\n' \$(pgrep -fc 'run_dualcraft_main50.py' || true) \
    && printf 'DONE=%s\n' \$(grep -c BATCH_DONE $LOG || true) \
    && printf 'FAILS=%s\n' \$(grep -E ': exit (-?[1-9][0-9]*) in' $LOG | grep -vc 'max turns limit' || true)" 2>/dev/null
}

base_fails=""
misses=0
while true; do
  snap=$(snapshot)
  if [ -z "$snap" ]; then
    misses=$((misses + 1))
    [ $misses -ge 15 ] && { echo "UNREACHABLE: server did not answer 15 times in a row"; exit 0; }
    sleep 60; continue
  fi
  misses=0
  eval "$snap"
  [ -z "$base_fails" ] && base_fails=$FAILS
  if [ "${DONE:-0}" -ge 1 ]; then echo "DONE"; $SSH "tail -4 $O/$LOG"; exit 0; fi
  if [ "${FAILS:-0}" -gt "$base_fails" ]; then echo "FAILED"; $SSH "grep -E ': exit ' $O/$LOG | tail -3"; exit 0; fi
  if [ "${ALIVE:-0}" -eq 0 ]; then echo "DIED"; $SSH "tail -5 $O/$LOG"; exit 0; fi
  if [ "${AGE:-0}" -gt $((STALL_MIN * 60)) ]; then echo "STALLED: log unchanged ${AGE}s"; $SSH "tail -3 $O/$LOG"; exit 0; fi
  sleep 120
done
