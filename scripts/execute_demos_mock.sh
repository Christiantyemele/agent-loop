#!/usr/bin/env bash
# Offline headless execution check for runbook demos 1-2.
#
# Validates the demos end-to-end WITHOUT a live model or real API key:
#   * builds a local Jupyter venv (nbformat/nbclient/jupyterlab) if absent,
#   * regenerates the notebooks against THIS checkout so the `:dep agent_loop`
#     path resolves (the committed notebooks carry a machine-specific path),
#   * starts scripts/mock_server.py (an offline OpenAI-compatible endpoint),
#   * wires the evcxr kernel + user deps via scripts/setup.sh,
#   * executes notebooks 01 and 02 headlessly with --fail-on-error and fails
#     the run if any cell produced an error output.
#
# This is the command the (manual) "notebook-execute" CI job invokes, and it is
# also the way a contributor re-verifies demos 1-2 offline:
#     ./scripts/execute_demos_mock.sh
#
# Env overrides: MOCK_PORT (default 8000).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PORT="${MOCK_PORT:-8000}"
PY="$REPO_DIR/.venv/bin/python"

# 1) Jupyter venv (checkout-local; .gitignore already ignores /.venv).
if [ ! -x "$PY" ]; then
  echo "==> creating .venv and installing jupyter deps ..."
  python3 -m venv "$REPO_DIR/.venv"
  "$REPO_DIR/.venv/bin/pip" install --quiet jupyterlab notebook nbformat nbclient
fi
"$PY" -c "import nbformat, nbclient" || {
  echo "!! nbformat/nbclient missing in .venv; reinstalling" >&2
  "$REPO_DIR/.venv/bin/pip" install --quiet nbformat nbclient
}

# 2) Regenerate notebooks so `:dep agent_loop` points at THIS checkout.
echo "==> regenerating notebooks against $REPO_DIR ..."
"$PY" scripts/generate_notebooks.py

# 3) Start the mock server and wait for it to answer /v1/chat/completions.
echo "==> starting mock server on 127.0.0.1:$PORT ..."
"$PY" scripts/mock_server.py "$PORT" &
MOCK_PID=$!
cleanup() { kill "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 50); do
  if curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d '{"messages":[]}' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [ "$ready" != "1" ]; then
  echo "!! mock server did not become ready on port $PORT" >&2
  exit 1
fi

# 4) Point the demos at the mock server (no real API key needed).
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="sk-mock"
export AGENT_LOOP_MODEL="mock-model"

# 5) Exercise setup.sh: verifies Rust/Jupyter/evcxr and writes the evcxr
#    user-dependency config that lets notebooks `:dep agent_loop` by name.
echo "==> running scripts/setup.sh ..."
./scripts/setup.sh

# 6) Execute demos 1-2 headlessly; fail the run on any cell error.
echo "==> executing notebook 01 ..."
"$PY" scripts/run_nb.py --fail-on-error notebooks/01_request_response_tools.ipynb
echo "==> executing notebook 02 ..."
"$PY" scripts/run_nb.py --fail-on-error notebooks/02_parallel_tool_execution.ipynb

echo "==> ok: demos 1-2 executed successfully against the offline mock server."
