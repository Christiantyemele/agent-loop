//! # agent-loop — a Rust runbook for agent loops.
//!
//! A small, dependency-light implementation of the pieces that make up a tool-
//! calling agent loop, written to be *read* and *run* as a presentation:
//!
//!   - [`chat`]       — message types + an OpenAI-compatible client (raw wire format)
//!   - [`tools`]      — tool definitions (file ops, bash, web, calculator) + a dynamic registry
//!   - [`executor`]   — sequential & parallel tool execution
//!   - [`hooks`]      — the six lifecycle hooks (pretooluse, posttooluse, ...)
//!   - [`agent`]      — the loop that ties it all together
//!
//! The crate compiles standalone and is also imported from the Jupyter
//! notebooks in `notebooks/`.

pub mod agent;
pub mod calc;
pub mod chat;
pub mod executor;
pub mod hooks;
pub mod tools;

pub use agent::{Agent, AgentConfig, AgentOutcome};
pub use chat::{ChatClient, ChatMessage, ChatRequest, ChatTool, Role, ToolCall};
pub use executor::execute_many;
pub use hooks::{HookEvent, HookPayload, Hooks};
pub use tools::{Tool, ToolRegistry, ToolResult};
