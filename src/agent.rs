//! # The agent loop.
//!
//! Everything else in this crate is a part that plugs into one canonical loop:
//!
//! ```text
//!   user prompt
//!       │
//!       ▼
//!  [UserPromptSubmit hook] ──────────────► observe
//!       │
//!       ▼
//!  build request (messages + tools) ──────► RAW REQUEST (transparent)
//!       │
//!       ▼
//!  send to endpoint ──────────────────────► RAW RESPONSE (transparent)
//!       │
//!       ▼
//!  assistant replied?
//!       ├─ yes, wants tool_calls ──► for each: [PreToolUse hook] ─► execute (parallel?)
//!       │                                  │                          │
//!       │                                  │                     [PostToolUse hook] ◄─┘
//!       │                                  └─► append tool results to history ─► loop
//!       └─ no, finished ──► [Stop hook] ──► return final text
//! ```
//!
//! The loop is deliberately thin: the model decides *what* to call, the executor
//! decides *how*, hooks observe *along the way*, and the history is the one piece
//! of state that carries everything forward.

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::chat::{ChatClient, ChatMessage, ChatRequest, ToolCall};
use crate::executor::{execute_many, ExecutionStats};
use crate::hooks::{HookEvent, HookPayload, Hooks};
use crate::tools::ToolRegistry;

/// Configuration for one agent run.
#[derive(Clone)]
pub struct AgentConfig {
    pub model: String,
    pub system_prompt: String,
    pub registry: ToolRegistry,
    pub hooks: Hooks,
    pub client: ChatClient,
    pub parallel_tools: bool,
    pub max_iterations: usize,
}

impl AgentConfig {
    pub fn new(
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        registry: ToolRegistry,
        hooks: Hooks,
        client: ChatClient,
    ) -> Self {
        Self {
            model: model.into(),
            system_prompt: system_prompt.into(),
            registry,
            hooks,
            client,
            parallel_tools: false,
            max_iterations: 10,
        }
    }
}

/// One tool execution within a turn.
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub call: ToolCall,
    pub result: ChatMessage,
}

/// A full round of "send request → model replies". Used to build a transparent
/// transcript of the run.
#[derive(Debug, Clone)]
pub struct IterationTrace {
    pub iteration: usize,
    /// The exact request sent (including the tool list the model saw).
    pub request: Value,
    /// The raw response body.
    pub response: Value,
    /// Whether the model asked to call tools this round.
    pub finish_reason: String,
    /// The tool calls executed (and their results) this round, if any.
    pub executed: Vec<ExecutionStep>,
    /// Batch execution timing, if tools ran.
    pub stats: Option<ExecutionStats>,
}

/// The result of an agent run.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    pub final_text: String,
    pub messages: Vec<ChatMessage>,
    pub traces: Vec<IterationTrace>,
}

/// The live agent: owns the conversation history and drives the loop.
pub struct Agent {
    pub config: AgentConfig,
    messages: Vec<ChatMessage>,
    traces: Vec<IterationTrace>,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            messages: Vec::new(),
            traces: Vec::new(),
        }
    }

    /// Run the loop for one user prompt, returning the final text + full trace.
    pub fn run(&mut self, prompt: &str) -> Result<AgentOutcome> {
        // 1) Observe the user's prompt entering the loop.
        self.config.hooks.fire(
            HookEvent::UserPromptSubmit,
            &HookPayload::UserPromptSubmit {
                prompt: prompt.to_string(),
            },
        );

        // Seed history with the system prompt on the first user turn.
        if self.messages.is_empty() {
            self.messages
                .push(ChatMessage::system(self.config.system_prompt.clone()));
        }
        self.messages.push(ChatMessage::user(prompt));

        let mut iteration = 0usize;

        while iteration < self.config.max_iterations {
            iteration += 1;

            // 2) Build the request blob — messages + the full tool surface.
            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: self.messages.clone(),
                tools: Some(self.config.registry.as_chat_tools()),
                tool_choice: Some(json!("auto")),
                temperature: Some(0.2),
            };

            // 3) Transcribe the request for the trace (the raw wire body).
            let request_value = request.to_json_pretty();

            // 4) Send it.
            let response = self.config.client.complete(&request)?;
            let choice = response
                .choices
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty choices in response"))?
                .clone();
            let finish_reason = choice.finish_reason.unwrap_or_default();

            // 5) Record the assistant message and raw response.
            let assistant = choice.message;
            let raw_response = serde_json::to_value(&response)?;
            self.messages.push(assistant.clone());

            let tool_calls = assistant.tool_calls.clone().unwrap_or_default();

            // Sanity: `finish_reason == "tool_calls"` and non-empty `tool_calls`
            // must agree; prefer the actual call list. If the model decided it is
            // done, we stop here.
            if tool_calls.is_empty() || finish_reason == "stop" {
                let final_text = assistant.content.clone().unwrap_or_default();
                let trace = IterationTrace {
                    iteration,
                    request: request_value,
                    response: raw_response,
                    finish_reason: finish_reason.clone(),
                    executed: Vec::new(),
                    stats: None,
                };
                self.traces.push(trace);

                // 6) Turn is over — fire the Stop hook.
                self.config.hooks.fire(
                    HookEvent::Stop,
                    &HookPayload::Stop {
                        reason: finish_reason,
                    },
                );

                return Ok(AgentOutcome {
                    final_text,
                    messages: self.messages.clone(),
                    traces: self.traces.clone(),
                });
            }

            // 7) The model wants tools. First make the response observable —
            //    this fires *before* any of the advertised tools run, so a
            //    listener can see the raw response (and its tool_calls) first.
            if !tool_calls.is_empty() {
                self.config.hooks.fire(
                    HookEvent::ResponseReceived,
                    &HookPayload::ResponseReceived {
                        response: raw_response.clone(),
                    },
                );
            }

            // 8) Observe each "about to use" event.
            for call in &tool_calls {
                self.config.hooks.fire(
                    HookEvent::PreToolUse,
                    &HookPayload::PreToolUse {
                        tool: call.name.clone(),
                        args: serde_json::from_str(&call.arguments).unwrap_or(Value::Null),
                    },
                );
            }

            // 8) Execute them (parallel or sequential) and observe each result.
            let (results, stats) = execute_many(
                &self.config.registry,
                &tool_calls,
                self.config.parallel_tools,
            );

            let mut steps = Vec::new();
            for (call, result) in tool_calls.iter().zip(results.iter()) {
                steps.push(ExecutionStep {
                    call: call.clone(),
                    result: result.clone(),
                });
                self.config.hooks.fire(
                    HookEvent::PostToolUse,
                    &HookPayload::PostToolUse {
                        tool: call.name.clone(),
                        output: result.content.clone().unwrap_or_default(),
                        is_error: false,
                    },
                );
                self.messages.push(result.clone());
            }

            self.traces.push(IterationTrace {
                iteration,
                request: request_value,
                response: raw_response,
                finish_reason,
                executed: steps,
                stats: Some(stats),
            });
        }

        Err(anyhow::anyhow!(
            "agent exceeded max_iterations ({}) before finishing",
            self.config.max_iterations
        ))
    }

    /// A compact, ordered human-readable transcript of the run, useful for the
    /// "here is what actually happened" demo cell.
    pub fn transcript(&self) -> String {
        let mut out = String::new();
        for t in &self.traces {
            out.push_str(&format!(
                "── iteration {} · finish={} ──────────────────────\n",
                t.iteration, t.finish_reason
            ));
            if let Some(stats) = &t.stats {
                out.push_str(&format!(
                    "   tools: {} ({} in {:.1} ms)\n",
                    stats.n_calls,
                    if stats.parallel {
                        "parallel"
                    } else {
                        "sequential"
                    },
                    stats.elapsed_ms
                ));
            }
            for s in &t.executed {
                out.push_str(&format!(
                    "   · [{}] {} {}\n",
                    s.call.id, s.call.name, s.call.arguments
                ));
                out.push_str(&format!(
                    "       ⇠ {}\n",
                    s.result.content.clone().unwrap_or_default()
                ));
            }
        }
        out
    }

    /// Raw JSON of iteration `i`'s request. Lets a notebook cell print exactly
    /// what a given request advertised.
    pub fn raw_request(&self, i: usize) -> Option<&Value> {
        self.traces.get(i).map(|t| &t.request)
    }
}

// ---------------------------------------------------------------------------
// Sub-agent simulation (used by the SubAgentStart / SubAgentStop demo)
// ---------------------------------------------------------------------------

/// A tiny helper that *models* spawning a sub-agent without standing one up for
/// real: it fires the `SubAgentStart` hook, runs a (shallow) nested loop against
/// the same endpoint, then fires `SubAgentStop`. This keeps the runbook able to
/// demonstrate the sub-agent hooks with any endpoint.
pub fn run_subagent(config: &AgentConfig, name: &str, prompt: &str) -> Result<String> {
    config.hooks.fire(
        HookEvent::SubAgentStart,
        &HookPayload::SubAgentStart {
            name: name.to_string(),
            prompt: prompt.to_string(),
        },
    );

    // A minimal one-shot call for the sub-agent's own "answer".
    let request = ChatRequest {
        model: config.model.clone(),
        messages: vec![
            ChatMessage::system(config.system_prompt.clone()),
            ChatMessage::user(prompt),
        ],
        tools: Some(config.registry.as_chat_tools()),
        tool_choice: Some(json!("auto")),
        temperature: Some(0.2),
    };
    let result = config
        .client
        .complete(&request)
        .map(|r| {
            r.choices
                .first()
                .map(|c| c.message.content.clone().unwrap_or_default())
                .unwrap_or_default()
        })
        .unwrap_or_else(|e| format!("[sub-agent failed: {e}]"));

    config.hooks.fire(
        HookEvent::SubAgentStop,
        &HookPayload::SubAgentStop {
            name: name.to_string(),
            result: result.clone(),
        },
    );

    Ok(result)
}

/// Render the tools a request exposed, for the "here is what the model saw"
/// demo helper.
pub fn tools_section(registry: &ToolRegistry) -> Vec<Map<String, Value>> {
    registry
        .all()
        .into_iter()
        .map(|t| {
            let mut m = Map::new();
            m.insert("name".into(), json!(t.name));
            m.insert("description".into(), json!(t.description));
            m.insert("parameters".into(), t.parameters.clone());
            m
        })
        .collect()
}
