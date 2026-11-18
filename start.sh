#!/bin/bash
# StoryForge — brings up everything the app needs, then serves it.
#
#   ./start.sh              start ComfyUI if it is down, then the app (foreground)
#   ./start.sh --no-comfy   assume ComfyUI is someone else's problem
#   ./stop.sh               stop the app and the ComfyUI this script started
#
# The narrator is cloud-hosted (Ollama Cloud), so no LLM runs locally unless
# config.json sets llm.provider to "local" — then Ollama must be up on the
# Windows side at :11434.
set -u
cd "$(dirname "$0")"
ROOT="$PWD"

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"
COMFY_WAIT_S="${COMFY_WAIT_S:-180}"

LOG_DIR="$ROOT/data/logs"
COMFY_LOG="$LOG_DIR/comfy.log"
COMFY_PID="$LOG_DIR/comfy.pid"
mkdir -p "$LOG_DIR"

START_COMFY=1
UVICORN_ARGS=()
for a in "$@"; do
  case "$a" in
    --no-comfy) START_COMFY=0 ;;
    *) UVICORN_ARGS+=("$a") ;;
  esac
done

say()  { printf '  %s\n' "$*"; }
warn() { printf '  warning: %s\n' "$*"; }
die()  { printf '  error: %s\n' "$*" >&2; exit 1; }

comfy_up() { curl -s --max-time 3 -o /dev/null "$COMFY_URL/system_stats"; }

# ---------------------------------------------------------------- preflight

command -v python3 >/dev/null || die "python3 not found"
python3 - <<'PY' || die "missing deps — pip install -r requirements.txt"
import importlib, sys
for m in ("fastapi", "uvicorn", "aiohttp", "PIL"):
    if importlib.util.find_spec(m) is None:
        print(f"  missing: {m}", file=sys.stderr); sys.exit(1)
PY

PROVIDER=$(python3 -c "import json;print((json.load(open('config.json'))['llm']).get('provider','cloud'))" 2>/dev/null) \
  || die "config.json unreadable"

if [ "$PROVIDER" = "cloud" ]; then
  KEY="${OLLAMA_API_KEY:-}"
  [ -z "$KEY" ] && [ -f .env ] && KEY=$(grep -m1 '^OLLAMA_API_KEY=' .env | cut -d= -f2-)
  if [ -z "$KEY" ]; then
    warn "no OLLAMA_API_KEY in env or .env — the cloud narrator will fail"
  else
    say "narrator: cloud ($(python3 -c "import json;print(json.load(open('config.json'))['llm']['model'])"))"
  fi
else
  curl -s --max-time 3 -o /dev/null "$OLLAMA_URL/api/tags" \
    || warn "Ollama not reachable on :11434 — start it on Windows (provider=local)"
  say "narrator: local ($(python3 -c "import json;print(json.load(open('config.json'))['llm']['model'])"))"
fi

# ------------------------------------------------------------------ ComfyUI

if comfy_up; then
  say "comfyui: already up on ${COMFY_URL#http://}"
elif [ "$START_COMFY" = "0" ]; then
  warn "comfyui down and --no-comfy given — images will fail"
else
  [ -x "$COMFY_DIR/venv/bin/python" ] || die "no ComfyUI venv at $COMFY_DIR/venv (set COMFY_DIR=)"
  say "comfyui: starting — log at $COMFY_LOG"
  # Absolute path to main.py on purpose: ComfyUI re-launches itself, so the
  # pid from $! is stale within a second. The cmdline is what stop.sh matches.
  ( cd "$COMFY_DIR" && nohup ./venv/bin/python "$COMFY_DIR/main.py" >"$COMFY_LOG" 2>&1 & )
  for i in $(seq 1 "$COMFY_WAIT_S"); do
    if comfy_up; then say "comfyui: up after ${i}s"; break; fi
    if ! pgrep -f "$COMFY_DIR/main.py" >/dev/null; then
      tail -n 15 "$COMFY_LOG" >&2
      die "comfyui exited while starting — see $COMFY_LOG"
    fi
    sleep 1
  done
  comfy_up || { tail -n 15 "$COMFY_LOG" >&2; die "comfyui did not answer within ${COMFY_WAIT_S}s"; }
  pgrep -f "$COMFY_DIR/main.py" >"$COMFY_PID"
fi

if command -v nvidia-smi >/dev/null; then
  say "gpu:     $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1) used"
fi

# ---------------------------------------------------------------- the app

if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:${PORT}/api/stories"; then
  die "something is already serving :${PORT} — ./stop.sh first"
fi

# hostname -I leads with eth0's 169.254 link-local under mirrored networking,
# which a phone cannot reach. Take the first routable LAN address instead.
IP=$(hostname -I 2>/dev/null | tr ' ' '\n' \
     | grep -vE '^(169\.254\.|127\.|10\.255\.|172\.(1[6-9]|2[0-9]|3[01])\.|$)' | head -1)
echo
say "local:  http://127.0.0.1:${PORT}"
[ -n "$IP" ] && say "phone:  http://${IP}:${PORT}   (same wifi)"
echo
say "ctrl-c stops the app; ComfyUI keeps running (./stop.sh ends both)"
echo

exec python3 -m uvicorn backend.main:app --host "$HOST" --port "$PORT" "${UVICORN_ARGS[@]}"
