# agent-loop · a Rust runbook on agent loops

An interactive, Jupyter-notebook presentation that takes an agent loop **from
conception to implementation** in Rust. Each notebook is a self-contained demo;
together they build a working tool-calling agent loop against any
OpenAI-compatible endpoint.

## The demos

| Notebook | Topic |
|----------|-------|
| `notebooks/00_welcome.ipynb` | Overview, agenda, mental model, configuration |
| `notebooks/01_request_response_tools.ipynb` | Model request/response, the system-message format, and how a request **exposes tools** (file ops, bash, web search) |
| `notebooks/02_parallel_tool_execution.ipynb` | Parallel vs sequential tool execution with timing |
| `notebooks/03_dynamic_tools.ipynb` | Registering / replacing / removing tools at runtime |
| `notebooks/04_hooks.ipynb` | What hooks are and how the agent incorporates them |
| `notebooks/05_hook_execution.ipynb` | Live examples of `pretooluse`, `posttooluse`, `userpromptsubmit`, `stop`, `subagentstart`, `subagentstop` |

The underlying implementation lives in `src/`:

- `src/chat.rs` — the OpenAI-compatible wire types and client
- `src/tools.rs` — file / bash / web tools plus the dynamic registry
- `src/executor.rs` — sequential and parallel tool execution
- `src/hooks.rs` — the six-lifecycle-event hook system
- `src/agent.rs` — the loop that ties them together

## Quick start

```bash
# 1. One-time setup (Rust, Jupyter venv, evcxr kernel, evcxr user-deps)
./scripts/setup.sh --install-kernel

# 2. Configure the endpoint — either export the vars, or use a .env file
cp .env.example .env          # then edit with your real endpoint/model

# 3. (Optional) generate/refresh executed outputs against your live model
./scripts/execute_notebooks.sh

# 4. Launch JupyterLab and open notebooks/00_welcome.ipynb
.venv/bin/jupyter lab --notebook-dir notebooks
```

> The notebooks call the **live model** configured in your environment: each
> cell builds the client from `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
> `AGENT_LOOP_MODEL` (via `ChatClient::from_env()` + dotenv). They do **not**
> mock responses — set the env vars above to your real endpoint and the demos
> run against it. The included `scripts/mock_server.py` is only an optional,
> offline stand-in for the *first* two demos when no model is reachable.

### Environment variables

The runbook reads its configuration from environment variables and automatically
loads a `.env` file (via [`dotenvy`](https://crates.io/crates/dotenvy)) if one is
present. `cp .env.example .env`, edit it, and you're set — no need to export.

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_BASE_URL` | Base URL of the OpenAI-compatible endpoint (`.../v1`) | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | API key for the endpoint | — |
| `AGENT_LOOP_MODEL` | Model name used by default | `gpt-4o-mini` |
| `TAVILY_API_KEY` | Optional key for the `web_search` tool | — |

Precedence: **real shell environment variables always win** over `.env`;
`.env` fills in anything not already set. The playback scripts
(`jupyter`, `cargo run`) load `.env` automatically because the library calls
`dotenv()` on first read.

> Point `OPENAI_BASE_URL` at any OpenAI-compatible server (Ollama, vLLM, LM
> Studio, a corporate gateway, ...). If no endpoint is reachable, the live cells
> print a friendly note instead of failing — demos **2**, **3** and most of
> **5** are fully local.

## Running headless (to validate or rehearse)

```bash
# Re-generate executed outputs against your configured (live) endpoint/model:
./scripts/execute_notebooks.sh

# Or run the equivalent directly with the rust kernel:
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=rust notebooks/*.ipynb
```

## CLI fallback

`cargo run` runs the same pieces from the terminal (no Jupyter required):

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=sk-mock \
  AGENT_LOOP_MODEL=mock-model cargo run -- "comment: run echo hello"
```

## Layout

```
src/            the library (also importable from notebooks as `agent_loop`)
notebooks/*.ipynb   the presentation
scripts/
  generate_notebooks.py   regenerate all notebooks (edits the :dep path)
  mock_server.py          tiny OpenAI-compatible server for offline demos
  setup.sh                one-time environment setup
  run_nb.py               execute one notebook and print all outputs
```
