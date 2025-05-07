#!/usr/bin/env bash
# kill_high_mem_gunicorn.sh
#
# Kill any Gunicorn (launched via nohup) serving pwsgi:app if its memory > 70%

# --- Config ---
THRESHOLD=80.0
# match your exact invocation; this will catch the master process
PATTERN="gunicorn.*pwsgi:app"
WATCHDOG="/data/aadhaarmask/restart_watchdog.sh"

# sanity checks
for cmd in pgrep ps awk bc date kill; do
  command -v $cmd >/dev/null 2>&1 || { echo "ERROR: '$cmd' not found"; exit 1; }
done

[ -x "$WATCHDOG" ] || { echo "ERROR: watchdog script not executable at $WATCHDOG"; exit 1; }

# grab pids
PIDS=$(pgrep -f "$PATTERN") || true
[ -z "$PIDS" ] && exit 0

for PID in $PIDS; do
  MEM=$(ps -p "$PID" -o %mem=)   # e.g. "12.3"
  # compare floats
  if (( $(echo "$MEM > $THRESHOLD" | bc -l) )); then
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TS] PID $PID using $MEM% memory (> $THRESHOLD%) – sending TERM"
    kill -TERM "$PID"

    # wait up to 10s
    for i in {1..10}; do
      sleep 1
      kill -0 "$PID" 2>/dev/null || { 
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $PID exited cleanly"; 
        break
      }
    done

    # if still alive, force kill
    if kill -0 "$PID" 2>/dev/null; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $PID still running – sending KILL"
      kill -9 "$PID"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Invoking Restart_watchdog.sh"
    "$WATCHDOG"
  fi
done
