#!/usr/bin/env bash
# Re-execute all notebooks against the endpoint configured in the environment,
# writing the live outputs back into notebooks/*.ipynb.
#
# Uses whatever endpoint the process sees: exported OPENAI_* vars, or a `.env`
# file (the crate loads it via dotenvy). It does NOT use the mock server.
#
#   export OPENAI_BASE_URL=https://api.openai.com/v1   # or your real one
#   export OPENAI_API_KEY=sk-...
#   export AGENT_LOOP_MODEL=gpt-4o-mini
#   ./scripts/execute_notebooks.sh
#
# Or, to load the values from a `.env` file you have committed nearby:
#   set -a; [ -f .env ] && source .env; set +a
#   ./scripts/execute_notebooks.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

JUP="$REPO_DIR/.venv/bin/jupyter"
[ -x "$JUP" ] || JUP="$(command -v jupyter || true)"
[ -x "$JUP" ] || { echo "jupyter not found" >&2; exit 1; }

echo "==> endpoint: ${OPENAI_BASE_URL:-<from .env / default>}"
echo "==> model   : ${AGENT_LOOP_MODEL:-<default>}"

"$JUP" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    --ExecutePreprocessor.kernel_name=rust \
    notebooks/*.ipynb

echo "==> done. notebooks/*.ipynb now carry outputs from your configured model."
