#!/usr/bin/env python3
"""Generate the agent-loop runbook notebooks.

Writes one .ipynb per demo into notebooks/. Each notebook is self-contained:
its first code cell loads the local `agent_loop` crate via an absolute path dep,
so it does not depend on any global evcxr configuration.

Usage:  python3 scripts/generate_notebooks.py
Regenerate:  python3 scripts/generate_notebooks.py
"""

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "notebooks")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(title: str, cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Rust",
                "language": "rust",
                "name": "rust",
            },
            "language_info": {
                "codemirror_mode": "rust",
                "file_extension": ".rs",
                "mimetype": "text/x-rust",
                "name": "rust",
                "nbconvert_exporter": "rust",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(filename: str, title: str, cells: list) -> str:
    path = os.path.join(OUT, filename)
    with open(path, "w") as f:
        json.dump(notebook(title, cells), f, indent=1)
        f.write("\n")
    return path


# The :dep line every notebook needs. Using an absolute path so the notebook is
# self-contained on this machine. serde_json is added explicitly because cells
# reference `serde_json::json!` directly, and transitive deps of agent_loop are
# not guaranteed in the notebook's extern prelude.
DEP = (
    ':dep agent_loop = { path = "' + REPO + '" }\n'
    ':dep serde_json = "1"'
)


# ---------------------------------------------------------------------------
# 00 - welcome / overview
# ---------------------------------------------------------------------------
def welcome() -> list:
    cells = [
        md(
            """# Agent Loops: from conception to implementation
### A Rust runbook

This notebook series walks through how a tool-calling agent loop works — from a
single model request all the way to hooks that observe the whole lifecycle.

**What we build and demonstrate**

| # | Topic | Where |
|---|-------|-------|
| 1 | Model request/response, the message format, and how a request *exposes tools* (file ops, web, calculator) | `01_request_response_tools` |
| 2 | Single tool execution, then parallel tool execution | `02_parallel_tool_execution` |
| 3 | Adding a tool (the dynamic registry) | `03_dynamic_tools` |
| 4 | Hooks: what they are and how the agent incorporates them | `04_hooks` |
| 5 | How a hook is executed — real examples for `pretooluse`, `posttooluse`, `userpromptsubmit`, `stop`, `subagentstart`, `subagentstop` | `05_hook_execution` |

**The mental model.** An agent is *not* one magic call. It is a **loop** — and
everything downstream (tools, parallel execution, dynamic registration, hooks)
is a refinement of this single loop:

<div style="width:100%; text-align:center;">
  <img
    src="agent-loop-diagram-dark.svg"
    alt="Agent loop diagram"
    style="max-width:90%; height:auto; display:block; margin:0 auto;"
  />
</div>
"""
        ),
        md(
            """## Prerequisites & configuration

This demo talks to an **OpenAI-compatible endpoint**. Point the client at yours
with environment variables:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1     # e.g. Ollama, vLLM, LM Studio, a gateway
export OPENAI_API_KEY=sk-...                         # optional for local servers
export AGENT_LOOP_MODEL=llama3.1                     # or gpt-4o-mini, qwen2.5, ...
```

> Demo **2 (single then parallel)** and **3 (adding a tool)** and most of
> **5 (hooks)** are fully local — they need no network. Demo **1** and the live
> `agent.run()` cells need a reachable endpoint, and degrade gracefully if none
> is up.
"""
        ),
        code(
            DEP
            + """

// Let us see what the client is pointed at right now.
use agent_loop::chat::ChatClient;
use agent_loop::chat::default_model;

let client = ChatClient::from_env();
println!("base_url : {}", client.base_url);
println!("api_key  : {}", client.api_key.as_deref().unwrap_or("<none set>"));
println!("model    : {}", default_model());
println!("chat api : {}", client.chat_api_url());
"""
        ),
        md(
            """## Agenda (execution order)

1. **Request / response & tools exposed** — open `01_request_response_tools.ipynb`
2. **Single tool execution, then parallel** — open `02_parallel_tool_execution.ipynb`
3. **Adding a tool** — open `03_dynamic_tools.ipynb`
4. **Hooks** (conception) — open `04_hooks.ipynb`
5. **Hooks** (execution) — open `05_hook_execution.ipynb`

Run each notebook top-to-bottom. Cells that need a live endpoint print a clear
note if none is reachable. Happy looping.
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# 01 - request / response, system message format, tools exposed
# ---------------------------------------------------------------------------
def demo01() -> list:
    cells = [
        md(
            """# 1 · The request → response exchange, and how a request exposes tools

Every agent loop starts with a plain HTTP exchange: send one JSON object, get
one JSON object back. Nothing magical happens here — no state, no memory. The
entire secret of an agent is **what goes into that request** and **what you do
with the reply**.

## The request

A Chat Completions request is roughly:

```json
{
  "model": "gpt-4o-mini",
  "messages": [ { "role": "system", "content": "..." },
                { "role": "user",   "content": "..." } ],
  "tools": [ { "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } } ],
  "tool_choice": "auto"
}
```

Two things matter for an *agent*:

- **`messages`** — the conversation, including a `system` message that sets the
  agent's behaviour. This is where the "system message format" lives.
- **`tools`** — the surface of what the model *may* call. **This is how a
  request exposes tools**: each tool is described as a JSON Schema the model
  can read, so the model knows it exists and how to fill in its arguments.

Below we build the exact request this runbook's agent sends, and print it as the
bytes that go over the wire. You will literally *see* the file, web and
calculator tools advertised to the model.
"""
        ),
        code(
            DEP
            + """

// Load the crate pieces we need.
use agent_loop::chat::{ChatRequest, ChatMessage, ChatClient, default_model};
use agent_loop::tools::{ToolRegistry, file_tools, web_tools, calc_tools};

// Build the built-in tool surface explicitly so we can inspect it:
// three families — file ops, web, and a calculator.
let mut registry = ToolRegistry::new();
for t in [file_tools::read_tool(), file_tools::write_tool(), file_tools::list_tool(),
          web_tools::fetch_tool(), web_tools::search_tool(),
          calc_tools::calc_tool()] {
    registry.register(t);
}
println!("Registered {} tools.", registry.len());

// The messages: a system message (the agent behaviour) + the user prompt.
let messages = vec![
    ChatMessage::system("You are a coding agent. You can read and write files, search the web, and do arithmetic."),
    ChatMessage::user("Summarise the files in my project and tell me what hello.rs contains."),
];

// Assemble the request exactly as the loop does.
let request = ChatRequest {
    model: default_model(),
    messages,
    tools: Some(registry.as_chat_tools()),
    tool_choice: Some(serde_json::json!("auto")),
    temperature: Some(0.2),
};
request
"""
        ),
        md(
            """## The exact bytes the model receives

`to_json_pretty()` returns the precise JSON that gets POSTed to
`/chat/completions`. Look at the **`tools`** array: the model is handed the
`name`, a human-readable `description`, and a JSON `parameters` schema for each
of `file_read`, `file_write`, `file_list`, `web_fetch`, `web_search`, `calc`.
That schema is the *contract* the model uses to produce arguments.
"""
        ),
        code(
            """
// Dump the request body as it is serialized on the wire.
println!("{}", agent_loop::chat::pretty(&request.to_json_pretty()));
"""
        ),
        md(
            """### System message format

The `system` message is special: it is the *standing instructions* that frame
every turn. The format is simply one message with `"role": "system"`. Tools are
not "the system" — they are a separate `tools` array. The system message tells
the model *how to behave*, while the `tools` array tells it *what it can touch*.

Here is the same request, but only its messages, so the shape of the `system`
message is crystal clear:
"""
        ),
        code(
            """
// Just the messages, to isolate the system-message format.
for m in &request.messages {
    println!("{:?}: {}", m.role, m.content.as_deref().unwrap_or("<tool_calls>"));
}
"""
        ),
        md(
            """### A sample system message

A system message is just an object with `"role": "system"` and a `content`
string. It carries the agent's standing behaviour. Here is one built the same
way the loop builds it, then printed as the JSON the endpoint actually receives:
"""
        ),
        code(
            """
// Build one system message by hand.
let system_message = agent_loop::chat::ChatMessage::system(
    "You are a coding agent. You can read and write files, search the web, and do arithmetic. Be concise."
);
println!("{}", agent_loop::chat::pretty(&serde_json::to_value(&system_message).unwrap()));
"""
        ),
        md(
            """## The response

Now send it live. Two things can happen:

- The endpoint answers with **text** (`finish_reason = "stop"`): the model is
  done.
- The endpoint answers with **`tool_calls`** (`finish_reason = "tool_calls"`):
  the model wants us to run tools. Note the model **never runs them itself** —
  it only *declares* the calls. Running them is the loop's job — demo 2 executes
  a single call, then parallel calls; demo 3 shows how a tool is added.

The cell below attempts the live exchange. If no endpoint is reachable it prints
a friendly note instead of failing the notebook.
"""
        ),
        code(
            """
// A tiny request with NO tools, so we see the plain text-exchange shape first.
let simple = agent_loop::chat::ChatClient::from_env();
match simple.complete_raw(&agent_loop::chat::simple_request(
        &agent_loop::chat::default_model(),
        "Answer in one short sentence.",
        "Say hello.")) {
    Ok(v) => println!("{}", agent_loop::chat::pretty(&v)),
    Err(e) => println!("[no reachable endpoint] live response skipped.\n  {e}\n  -> set OPENAI_BASE_URL and re-run this cell."),
}
"""
        ),
        md(
            """## Putting the pieces together

You now have the two halves of the loop:

- **Request** = `messages` (system + user + history) **+** `tools` (what the
  model may call).
- **Response** = either a final answer (`stop`) or a set of **`tool_calls`**
  that the loop must execute and feed back.

That "feed back and repeat" is precisely what demo 2 makes concrete (executing
that single call, then many in parallel), demo 3 grows the toolset, and demos
`04`/`05` observe the whole loop with hooks.
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# 02 - single tool execution, then parallel
# ---------------------------------------------------------------------------
def demo02() -> list:
    cells = [
        md(
            """# 2 · Single tool execution, then parallel

Demo 1 only *built* a request — it serialized tools but never actually ran one.
Here we do the real thing, in two steps that mirror how the loop thinks:

1. **A single tool call** — the model asks for one thing, we run it once.
2. **Parallel tool execution** — the model asks for several *independent*
   things in one message; we fan them out onto threads and collect the results.

We use the built-in **file, web and calculator** tools throughout. Below, a
single `calc` call is executed exactly once via `execute_one`.
"""
        ),
        code(
            DEP
            + """

use agent_loop::tools::ToolRegistry;
use agent_loop::chat::ToolCall;
use agent_loop::executor::execute_one;

// The built-ins include file, web and calculator tools.
let registry = ToolRegistry::with_builtins();
println!("built-ins ({}): {}", registry.len(),
    registry.all().iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));

// The model emitted exactly ONE tool call. Running a loop is, at bottom, just
// doing this: name + arguments -> registry lookup -> execute -> tool result.
let call = ToolCall {
    id: "call_calc_1".into(),
    name: "calc".into(),
    arguments: r#"{"expression": "(2 + 3) * 4 ^ 2 + sqrt(9)"}"#.into(),
};

let started = std::time::Instant::now();
let result = execute_one(&registry, &call);
let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

println!("single call `{}` took {:.3} ms", call.name, elapsed_ms);
println!("result -> {}", result.content.as_deref().unwrap_or("<none>"));
"""
        ),
        md(
            """### What "a single tool call" means

That one call is the **smallest unit of the entire loop**. Everything else is a
refinement of it:

```
model says  call(calc, {"expression":"(2+3)*4^2+sqrt(9)"})
                    │
                    ▼
              execute_one(registry, &call)
                    │  1. parse arguments
                    │  2. find tool by name
                    │  3. run its executor
                    ▼
            tool_result message -> fed back to the model
```

A **file** tool works identically — same shape, different target. Try a single
`file_list` call on the notebook directory:

> Note: the kernel's working directory is the `notebooks/` folder (JupyterLab
> was launched with `--notebook-dir notebooks`), so a relative `"notebooks"`
> path would resolve to `notebooks/notebooks` and fail. We pass the absolute
> path instead so the demo works no matter where the kernel starts.
"""
        ),
        code(
            """
// A single FILE call: same execute_one, different tool. We use the absolute
// notebooks path because the kernel CWD is the notebooks dir itself.
let file_call = ToolCall {
    id: "call_list_1".into(),
    name: "file_list".into(),
    arguments: r#"{"path": "@NOTES_DIR@"}"#.into(),
};
let out = execute_one(&registry, &file_call);
println!("file_list ->\\n{}", out.content.as_deref().unwrap_or("<none>"));
""".replace("@NOTES_DIR@", OUT)
        ),
        md(
            """### Several independent calls arrive at once

A real model rarely returns exactly one `tool_calls` entry — it often returns
several *independent* ones in a single assistant message. They could be N file
reads, N web fetches, or N separate calculations. Because they don't depend on
each other, they are prime candidates for running **in parallel**.

The executor exposes two strategies:
- **sequential** — run one at a time, in order;
- **parallel** — fan out onto worker threads and collect results back in the
  original order.

This demo is **fully local** (no network): we register `N` independent compute
tools that each sleep `300 ms`, then run the same batch both ways and compare
the wall-clock time. The ordering of results is preserved in *both* modes —
parallelism buys speed, never reordering.
"""
        ),
        code(
            """

use agent_loop::tools::{Tool, ToolResult, s_required};
use agent_loop::executor::execute_many;
use serde_json::{json, Value};
use std::time::Duration;

// Register 4 independent tools. Each claims 300 ms of work.
let mut registry = ToolRegistry::new();
for i in 0..4usize {
    registry.register(Tool::new(
        format!("compute_{i}"),
        format!("Simulate compute task {i}"),
        s_required(json!({}), &[]),
        move |_| {
            std::thread::sleep(Duration::from_millis(300));
            Ok(ToolResult::ok(format!("task {i} done")))
        },
    ));
}

// The model "emitted" 4 tool calls in a single assistant message.
let calls: Vec<agent_loop::chat::ToolCall> = (0..4).map(|i| agent_loop::chat::ToolCall {
    id: format!("call_{i}"),
    name: format!("compute_{i}"),
    arguments: "{}".into(),
}).collect();
println!("model emitted {} tool calls in ONE message", calls.len());
"""
        ),
        md(
            """### Sequential run

Every call waits for the previous one: 4 × 300 ms ≈ **1200 ms**.
"""
        ),
        code(
            """
let (seq_results, seq_stats) = execute_many(&registry, &calls, false);
println!("sequential: {} calls took {:.1} ms", seq_stats.n_calls, seq_stats.elapsed_ms);
for r in &seq_results {
    println!("   ⇠ {}", r.content.as_deref().unwrap_or(""));
}
"""
        ),
        md(
            """### Parallel run

Workers run concurrently: ≈ **300 ms** for the whole batch, ~4× faster.
Results are returned in the *same order* as the calls.
"""
        ),
        code(
            """
let (par_results, par_stats) = execute_many(&registry, &calls, true);
println!("parallel  : {} calls took {:.1} ms", par_stats.n_calls, par_stats.elapsed_ms);
for r in &par_results {
    println!("   ⇠ {}", r.content.as_deref().unwrap_or(""));
}
"""
        ),
        md(
            """### Side-by-side

```
sequential : 1200 ms   ████████████████████████████
parallel   :  300 ms   ██████
```

The speedup scales with the number of *independent* calls. The loop chooses the
strategy via `AgentConfig.parallel_tools`. The same `execute_many` is what the
agent loop in demos 4–5 calls under the hood — it is just `execute_one` (from
the single-call step above) repeated, optionally on worker threads.
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# 03 - dynamic tools
# ---------------------------------------------------------------------------
def demo03() -> list:
    cells = [
        md(
            """# 3 · Adding a tool (and the dynamic registry)

Demo 1 showed how a request *exposes* tools, and demo 2 *executed* them. But a
fixed set of tools is limiting. An agent should be able to **gain a new
capability** — this is "dynamic tools": the tool surface is **not fixed at
compile time**. The `ToolRegistry` is just an in-memory map: you can register,
replace, and unregister tools **while the agent is alive**.

We keep using the **file, web and calculator** families. Here we bolt a brand-
new *file* tool (`file_head`) directly onto the running registry.

This demo is **fully local**.
"""
        ),
        code(
            DEP
            + """

use agent_loop::tools::{ToolRegistry, Tool, ToolResult, s_required, builtin_tools};
use serde_json::{json, Value};

let mut registry = ToolRegistry::with_builtins();
println!("initial tools ({}): {}", registry.len(),
    registry.all().iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));
"""
        ),
        md(
            """### How to add a tool

A tool is two things: a **description** the model can read (name + JSON-schema
parameters) and an **executor** (the closure that does the work). `Tool::new`
bundles them. Below we add `file_head`, which returns the first `n` lines of a
file — part of the file-ops family, added live.
"""
        ),
        code(
            """
// The tool did NOT exist a moment ago; we register it at runtime.
registry.register(Tool::new(
    "file_head",
    "Return the first n lines of a text file.",
    s_required(json!({
        "path": {"type": "string"},
        "n": {"type": "integer"}
    }), &["path"]),
    |args| {
        let path = args.get("path").and_then(Value::as_str).unwrap_or("");
        let n = args.get("n").and_then(Value::as_i64).unwrap_or(5) as usize;
        let text = std::fs::read_to_string(path)?;
        let head: Vec<&str> = text.lines().take(n).collect();
        Ok(ToolResult::ok(head.join("\\n")))
    },
));

println!("after adding file_head ({}): {}", registry.len(),
    registry.all().iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));
println!("call it now ->\n{}", registry.get("file_head").unwrap()
    .run(&json!({"path": "Cargo.toml", "n": 3})).unwrap().output);
"""
        ),
        md(
            """### Replace a handler at runtime

Same name, new behaviour: drop-in upgrades without changing the call site. The
model never knows — it just continues emitting `file_head(...)` calls.
"""
        ),
        code(
            """
let before = registry.get("file_head").unwrap()
    .run(&json!({"path":"Cargo.toml","n":2})).unwrap().output;

// Same tool name, new handler: now it returns a file *snippet* instead.
registry.register(Tool::new(
    "file_head",
    "Return the first n lines of a text file, labelled.",
    s_required(json!({
        "path": {"type": "string"},
        "n": {"type": "integer"}
    }), &["path"]),
    |args| {
        let path = args.get("path").and_then(Value::as_str).unwrap_or("");
        let n = args.get("n").and_then(Value::as_i64).unwrap_or(5) as usize;
        let text = std::fs::read_to_string(path)?;
        let head: Vec<String> = text.lines().take(n).map(|l| format!("> {l}")).collect();
        Ok(ToolResult::ok(head.join("\\n")))
    },
));

let after = registry.get("file_head").unwrap()
    .run(&json!({"path":"Cargo.toml","n":2})).unwrap().output;
println!("before replacement:\n{before}\\n");
println!("after  replacement:\n{after}");
"""
        ),
        md(
            """### Unregister a tool

Removing a capability is just as easy. Once unregistered, calls to it report the
tool as unknown so the model can adapt.
"""
        ),
        code(
            """
registry.unregister("file_head");
println!("after removing file_head ({}): {}", registry.len(),
    registry.all().iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));

let unknown = agent_loop::chat::ToolCall { id: "x".into(), name: "file_head".into(), arguments: "{}".into() };
let msg = agent_loop::executor::execute_one(&registry, &unknown);
println!("call to removed tool -> {}", msg.content.unwrap());
"""
        ),
        md(
            """### Why this matters

The request in demo 1 serializes whatever is in the registry at request time
(`registry.as_chat_tools()`). So adding a tool dynamically *changes the next
request* the model sees — the model learns about the new tool on the very next
turn. The **calculator** `calc` tool shipped in the crate is itself just
something added to the registry this way. Dynamic + single/parallel execution
(demos 1–2) + the loop from demo 5 = a live, growing agent.
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# 04 - hooks: what they are & how the agent incorporates them
# ---------------------------------------------------------------------------
def demo04() -> list:
    cells = [
        md(
            """# 4 · Hooks: conception

A **hook** is a callback the agent invokes at a defined lifecycle boundary.
Hooks *observe* — they log, they gate, they record telemetry — without being
part of the decision-making core. Think of them as the "spies" around the loop.

## The six hook events

| Hook | Fired when |
|------|-----------|
| `pretooluse`   | just *before* a tool runs |
| `posttooluse`  | just *after* a tool returns its result |
| `userpromptsubmit` | when the user's message enters the loop |
| `stop`         | when the turn finishes (`finish_reason = stop`) |
| `subagentstart`| when a sub-agent is spawned |
| `subagentstop` | when a sub-agent finishes |

## How the agent incorporates hooks

The loop calls `hooks.fire(event, payload)` at exactly six places. In this
project those call sites live in `src/agent.rs`:

- `UserPromptSubmit` — at the very top of `Agent::run`.
- `PreToolUse` — once per tool call, *before* execution.
- `PostToolUse` — once per tool call, *after* its result is produced.
- `Stop` — right before `run` returns the final answer.
- `SubAgentStart` / `SubAgentStop` — bracketing `run_subagent`.

The core loop does **not** change based on which hooks are registered; hooks are
an additive side-channel. Let's inspect that mechanism directly.
"""
        ),
        code(
            DEP
            + """

use agent_loop::hooks::{Hooks, HookEvent, HookPayload};
use std::sync::{Arc, Mutex};

// A Hooks registry with a shared log buffer enabled.
let mut hooks = Hooks::new().with_log();

// Attach one callback per event. Each prints a distinguishable line.
hooks.on_pre_tool_use(        |p| println!("  [handle {:<4}] {}", "PRE",  label(p)));
hooks.on_post_tool_use(       |p| println!("  [handle {:<4}] {}", "POST", label(p)));
hooks.on_user_prompt_submit(  |p| println!("  [handle {:<12}] {}", "USER", label(p)));
hooks.on_stop(                |p| println!("  [handle {:<8}] {}", "STOP", label(p)));
hooks.on_sub_agent_start(     |p| println!("  [handle {:<8}] {}", "SSTART", label(p)));
hooks.on_sub_agent_stop(      |p| println!("  [handle {:<8}] {}", "SSTOP", label(p)));

// Small helper to render any payload headline.
fn label(p: &HookPayload) -> String {
    match p {
        HookPayload::PreToolUse { tool, .. } => format!("about to call `{tool}`"),
        HookPayload::PostToolUse { tool, is_error, .. } => format!("`{tool}` finished, is_error={is_error}"),
        HookPayload::UserPromptSubmit { prompt, .. } => format!("prompt: {:?}", &prompt[..prompt.len().min(20)]),
        HookPayload::Stop { reason } => format!("turn over ({reason})"),
        HookPayload::SubAgentStart { name, .. } => format!("spawn {name}"),
        HookPayload::SubAgentStop { name, .. } => format!("finished {name}"),
    }
}

println!("registered callbacks: {}", hooks.count());
println!("supported events    : {:?}", HookEvent::ALL.iter().map(|e| e.as_str()).collect::<Vec<_>>());
"""
        ),
        md(
            """## Registry ↔ agent wiring

The loop only ever calls two methods:

- `hooks.fire(event, payload)` — run every callback for that event **and**
  append to the shared log (when enabled);
- `hook_name()` / the six `on_*` builders — how you subscribe.

Because callbacks are just closures, multiple hooks can subscribe to one event,
and you add/remove them dynamically — the same spirit as dynamic tools. Demo 5
fires each of the six for real and shows the ordered log.
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# 05 - hook execution with examples
# ---------------------------------------------------------------------------
def demo05() -> list:
    cells = [
        md(
            """# 5 · How a hook is executed — real examples

This demo *fires* each of the six hooks with a concrete payload and a real
callback, and prints the resulting ordered log. It is fully local: we drive the
hook machinery directly, no endpoint required.

The payload type tells you what data each hook can see:

| Event | Payload the callback receives |
|-------|-------------------------------|
| `pretooluse`      | `{ tool, args }` |
| `posttooluse`     | `{ tool, output, is_error }` |
| `userpromptsubmit`| `{ prompt }` |
| `stop`            | `{ reason }` |
| `subagentstart`   | `{ name, prompt }` |
| `subagentstop`    | `{ name, result }` |
"""
        ),
        code(
            DEP
            + """

use agent_loop::hooks::{Hooks, HookEvent, HookPayload};
use serde_json::json;

// Enable the shared log so we can replay the exact firing order at the end.
let mut hooks = Hooks::new().with_log();

// Subscribe: one callback per event, each just logs what it saw.
hooks.on_user_prompt_submit(|p| if let HookPayload::UserPromptSubmit { prompt } = p {
    println!("userpromptsubmit: user asked: {:?}", &prompt[..prompt.len().min(40)]);
});
hooks.on_pre_tool_use(|p| if let HookPayload::PreToolUse { tool, args } = p {
    println!("pretooluse: about to call `{tool}` args={args}");
});
hooks.on_post_tool_use(|p| if let HookPayload::PostToolUse { tool, output, is_error } = p {
    let frag: String = output.chars().take(40).collect();
    println!("posttooluse: `{tool}` is_error={is_error} -> {frag}…");
});
hooks.on_stop(|p| if let HookPayload::Stop { reason } = p {
    println!("stop: turn finished with {reason}");
});
hooks.on_sub_agent_start(|p| if let HookPayload::SubAgentStart { name, .. } = p {
    println!("subagentstart: spawning sub-agent `{name}`");
});
hooks.on_sub_agent_stop(|p| if let HookPayload::SubAgentStop { name, result } = p {
    let frag: String = result.chars().take(30).collect();
    println!("subagentstop: sub-agent `{name}` returned: {frag}…");
});
println!("attached callbacks: {}", hooks.count());
"""
        ),
        md(
            """### 5.1 · `userpromptsubmit`

Fired the instant a user message enters the loop — the natural place for
approval prompts, input scrubbing, or logging "who asked what".
"""
        ),
        code(
            """
hooks.fire(HookEvent::UserPromptSubmit, &HookPayload::UserPromptSubmit {
    prompt: "Summarize this repo and fix the build".into(),
});
"""
        ),
        md(
            """### 5.2 · `pretooluse`

Fired *before* a tool runs. It carries the tool name and the parsed arguments —
the right place for a guardrail that can veto an expensive or dangerous call.
"""
        ),
        code(
            """
hooks.fire(HookEvent::PreToolUse, &HookPayload::PreToolUse {
    tool: "bash_run".into(),
    args: json!({"command": "rm -rf /tmp/scratch", "timeout_secs": 30}),
});
"""
        ),
        md(
            """### 5.3 · `posttooluse`

Fired *after* a tool returns. It carries the output (and whether it errored) —
the natural place for telemetry, error alerting, or caching results.
"""
        ),
        code(
            """
hooks.fire(HookEvent::PostToolUse, &HookPayload::PostToolUse {
    tool: "bash_run".into(),
    output: "build finished in 4.2s, tests passed".into(),
    is_error: false,
});
"""
        ),
        md(
            """### 5.4 · `stop`

Fired when the turn ends (`finish_reason = "stop"`). Good for logging the final
answer, persisting the transcript, or emitting metrics about turn length.
"""
        ),
        code(
            """
hooks.fire(HookEvent::Stop, &HookPayload::Stop { reason: "stop".into() });
"""
        ),
        md(
            """### 5.5 & 5.6 · `subagentstart` / `subagentstop`

Fired around spawning and finishing a sub-agent — the bracket around any
delegated child task. Use them to track fan-out, tag spans, or sum sub-costs.
"""
        ),
        code(
            """
hooks.fire(HookEvent::SubAgentStart, &HookPayload::SubAgentStart {
    name: "code_reviewer".into(),
    prompt: "Review the latest diff for correctness.".into(),
});
hooks.fire(HookEvent::SubAgentStop, &HookPayload::SubAgentStop {
    name: "code_reviewer".into(),
    result: "2 issues found: unused import, unhandled panic.".into(),
});
"""
        ),
        md(
            """### The ordered log — proof the hooks fired

Because we enabled the shared log, here is the exact chronological sequence of
every hook that fired, in the order it happened. This is what an observability
pipeline would collect:
"""
        ),
        code(
            """
for (i, entry) in hooks.log_entries().iter().enumerate() {
    println!("{:2}. {}", i + 1, entry.event.as_str());
}
"""
        ),
        md(
            """### Bonus: see the hooks fire inside a real agent run

If you have a reachable endpoint, the cell below runs the actual `Agent::run`
loop with the registered hooks wired in, then replays the hook log. If no
endpoint is reachable it prints a note instead. (Try it once you've set
`OPENAI_BASE_URL`.)
"""
        ),
        code(
            """
// Wire the same hooks into a live agent and watch them fire in context.
use agent_loop::tools::ToolRegistry;
use agent_loop::chat::{ChatClient, default_model};
use agent_loop::{Agent, AgentConfig};

let registry = ToolRegistry::with_builtins();
let client = ChatClient::from_env();
let mut agent = Agent::new(AgentConfig::new(
    default_model(),
    "You are a terse assistant. Prefer bash for computations.",
    registry,
    hooks.clone(),
    client,
));
agent.config.parallel_tools = true;

match agent.run("Run `echo hooked` in the shell and tell me the output.") {
    Ok(outcome) => {
        println!("final answer: {}", outcome.final_text);
        println!("\\nhook order: {}", agent.config.hooks.log_entries()
            .iter().map(|e| e.event.as_str()).collect::<Vec<_>>().join(" → "));
    }
    Err(e) => println!("[no reachable endpoint] live run skipped: {e}"),
}
"""
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    specs = [
        ("00_welcome.ipynb", "Agent Loops: Welcome", welcome()),
        ("01_request_response_tools.ipynb", "1 · Request/Response & Tools", demo01()),
        ("02_parallel_tool_execution.ipynb", "2 · Single Tool, then Parallel", demo02()),
        ("03_dynamic_tools.ipynb", "3 · Adding a Tool", demo03()),
        ("04_hooks.ipynb", "4 · Hooks (Conception)", demo04()),
        ("05_hook_execution.ipynb", "5 · Hook Execution", demo05()),
    ]
    for filename, title, cells in specs:
        path = write_nb(filename, title, cells)
        print(f"wrote {path} ({len(cells)} cells)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
