#!/bin/bash
# Stop the StoryForge app and the ComfyUI that start.sh launched.
#
#   ./stop.sh             stop both
#   ./stop.sh --app       stop only the app, leave ComfyUI resident
set -u
cd "$(dirname "$0")"
PORT="${PORT:-8100}"
COMFY_PID="data/logs/comfy.pid"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
ONLY_APP=0
[ "${1:-}" = "--app" ] && ONLY_APP=1

say() { printf '  %s\n' "$*"; }

pids=$(pgrep -f "uvicorn backend.main:app" || true)
if [ -n "$pids" ]; then
  kill $pids 2>/dev/null
  sleep 1
  pgrep -f "uvicorn backend.main:app" >/dev/null && kill -9 $(pgrep -f "uvicorn backend.main:app") 2>/dev/null
  say "app: stopped (:${PORT})"
else
  say "app: not running"
fi

[ "$ONLY_APP" = "1" ] && exit 0

# ComfyUI re-launches itself at startup, so a single recorded pid is not
# enough — match every process running a main.py out of the ComfyUI tree, and
# confirm against the port rather than against the pids.
# Matches ComfyUI however it was launched — absolute path, bare `main.py`, or
# by hand — by checking each candidate's working directory, not its cmdline.
comfy_pids() {
  { for pid in $(pgrep -f "main\.py" || true); do
      [ "$(readlink -f /proc/$pid/cwd 2>/dev/null)" = "$(readlink -f "$COMFY_DIR")" ] && echo "$pid"
    done
    cat "$COMFY_PID" 2>/dev/null || true
  } | sort -u | tr '\n' ' '
}
comfy_up() { curl -s --max-time 3 -o /dev/null "$COMFY_URL/system_stats"; }

cpids=$(comfy_pids)
if [ -z "${cpids// /}" ] && ! comfy_up; then
  say "comfyui: not running"
else
  kill $cpids 2>/dev/null
  for i in $(seq 1 20); do
    cpids=$(comfy_pids)
    [ -z "${cpids// /}" ] && ! comfy_up && break
    sleep 1
  done
  cpids=$(comfy_pids)
  [ -n "${cpids// /}" ] && kill -9 $cpids 2>/dev/null
  rm -f "$COMFY_PID"
  if comfy_up; then
    say "comfyui: still answering on ${COMFY_URL#http://} — started outside this script?"
  else
    say "comfyui: stopped, VRAM released"
  fi
fi
