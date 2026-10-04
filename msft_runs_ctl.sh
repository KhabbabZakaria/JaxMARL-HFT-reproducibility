#!/bin/bash
# Pause / resume the in-flight MSFT training runs.
#
# Uses SIGSTOP/SIGCONT, so the processes keep their full state in memory and
# carry on from the exact update they were on. Nothing is recomputed.
#
# Caveat: suspended processes do NOT survive a reboot or logout. They do survive
# the Mac sleeping. If the machine restarts, the runs are lost -- the trainer has
# no resume-from-checkpoint path wired up, so they would start from zero.
#
# Usage: ./msft_runs_ctl.sh {pause|resume|status}
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids() { pgrep -f 'ippo_rnn_JAXMARL|caffeinate -i python3' || true; }

case "${1:-status}" in
  pause)
    P=$(pids); [ -z "$P" ] && { echo "no runs found"; exit 1; }
    kill -STOP $P && echo "paused: $(echo $P | tr '\n' ' ')"
    ;;
  resume)
    P=$(pids); [ -z "$P" ] && { echo "no runs found -- were they killed or did the Mac reboot?"; exit 1; }
    kill -CONT $P && echo "resumed: $(echo $P | tr '\n' ' ')"
    ;;
  status)
    P=$(pids)
    if [ -z "$P" ]; then echo "no training processes alive"; else
      ps -o pid,stat,%cpu,etime -p $(echo $P | tr ' ' ',')
      echo "(STAT T = paused, R/S = running)"
    fi
    for f in run_MSFT_seed*.log; do
      [ -f "$f" ] && printf "  %-22s %4s / 975 updates\n" "$f" "$(grep -c 'completed' "$f")"
    done
    ;;
  *) echo "usage: $0 {pause|resume|status}"; exit 1 ;;
esac
