//! # Chat: messages and the OpenAI-compatible request / response wire format.
//!
//! Before any "agent" exists there is a loop, and before the loop there is a
//! single request/response exchange with the model. This module owns the types
//! that get serialized onto the wire and parsed back off it, so that we can
//! *show* the raw request and response rather than hiding them behind an
//! abstraction.
//!
//! The wire format is the standard OpenAI Chat Completions format, so this code
//! talks to any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, LM Studio,
//! a corporate gateway, ...). The endpoint is read from the environment:
//!
//!   - `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`)
//!   - `OPENAI_API_KEY` (defaulted to a placeholder for local servers)
//!   - `AGENT_LOOP_MODEL` (e.g. `gpt-4o-mini`, `llama3.1`, ...)
//!
//! Everything here is deliberately transparent: the request builder lets callers
//! dump the exact JSON that is sent, and the response keeps the raw body around.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The role of a message in the conversation, mirroring the OpenAI API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        })
    }
}

/// A single `tool_calls` entry emitted by the assistant.
///
/// The model does **not** "run" a tool — it just emits a declaration that it
/// *wants* to call a tool, with the arguments as a JSON string. Something in
/// the loop has to turn that declaration into an actual tool result. That
/// separation (model-supplies-intent, loop-supplies-execution) is the whole
/// foundation of the agent loop we build later.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// The id the model assigned to this call. The tool result must echo it
    /// back so the model can correlate which result goes with which call.
    pub id: String,
    /// The name of the tool the model wants to invoke.
    pub name: String,
    /// The arguments, as the model serialized them (a JSON *string* on the wire).
    pub arguments: String,
}

/// On the wire a tool call is `{ id, type: "function", function: { name, arguments } }`.
/// We keep the structure flat internally and translate here, so the rest of the
/// code never has to think about the nesting.
impl Serialize for ToolCall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("ToolCall", 3)?;
        s.serialize_field("id", &self.id)?;
        s.serialize_field("type", "function")?;
        let mut f = serde_json::Map::new();
        f.insert(
            "name".to_string(),
            serde_json::Value::String(self.name.clone()),
        );
        f.insert(
            "arguments".to_string(),
            serde_json::Value::String(self.arguments.clone()),
        );
        s.serialize_field("function", &serde_json::Value::Object(f))?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            id: String,
            #[serde(default)]
            function: Option<WireFunction>,
            // Tolerate a flat shape too (name/arguments at top level).
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            arguments: Option<String>,
        }
        #[derive(Deserialize)]
        struct WireFunction {
            name: String,
            arguments: String,
        }
        let w = Wire::deserialize(deserializer)?;
        let (name, arguments) = match w.function {
            Some(f) => (f.name, f.arguments),
            None => (
                w.name
                    .ok_or_else(|| serde::de::Error::custom("tool call missing function.name"))?,
                w.arguments.unwrap_or_default(),
            ),
        };
        Ok(ToolCall {
            id: w.id,
            name,
            arguments,
        })
    }
}

/// One message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// Textual content. Often `null` for assistant messages that only carry
    /// tool calls, and always present for system/user/tool messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Present only on assistant messages that requested tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present only on tool (result) messages; correlates with a `ToolCall.id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// An assistant message that just speaks text (its turn is over).
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// An assistant message that requested one or more tool calls
    /// (its turn continues until the results come back).
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// A tool result message, echoing the id of the call it answers.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// The JSON Schema description of a tool, exactly as the endpoint expects it.
///
/// This is what gets flattened into the `tools` array of the request. The model
/// reads `description` and `parameters` to decide (a) that the tool exists and
/// (b) how to fill in `arguments`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A tool as advertised to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: ToolFunction,
}

impl ChatTool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// A chat completion **request**, mirroring the OpenAI wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

impl ChatRequest {
    /// The exact JSON that will be posted over the wire. Great for a "show me
    /// the request" demo cell.
    pub fn to_json_pretty(&self) -> Value {
        serde_json::to_value(self).expect("ChatRequest is always serializable")
    }
}

/// The response body returned by the endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: Option<String>,
    pub model: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: Option<i64>,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
}

/// A thin client over `reqwest::blocking`. Keeping it blocking keeps the
/// notebook cells simple; real deployments would layer this on an async runtime.
#[derive(Debug, Clone)]
pub struct ChatClient {
    pub base_url: String,
    pub api_key: Option<String>,
    pub client: reqwest::blocking::Client,
}

impl Default for ChatClient {
    fn default() -> Self {
        Self::from_env()
    }
}

impl ChatClient {
    /// Build a client from the environment, honouring standard OpenAI env vars.
    ///
    /// Loads a `.env` file (via `dotenvy`) if present, then reads
    /// `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Real shell environment variables
    /// always take precedence over `.env` values.
    pub fn from_env() -> Self {
        let _ = load_env();
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Explicit construction, handy for pointing at a local endpoint in a demo.
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            client: reqwest::blocking::Client::new(),
        }
    }

    pub fn chat_api_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Send one completion request and return the parsed response.
    ///
    /// Returns an error that includes the raw response text when the server
    /// answers with a non-success status, so demo cells can surface failures.
    pub fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let url = self.chat_api_url();
        let mut rb = self.client.post(&url).json(&request);
        if let Some(key) = &self.api_key {
            rb = rb.header("Authorization", format!("Bearer {key}"));
        }
        let resp = rb.send()?;
        let status = resp.status();
        let body = resp.text()?;

        if !status.is_success() {
            return Err(anyhow!("endpoint returned {status}: {body}"));
        }

        serde_json::from_str(&body).map_err(|e| anyhow!("failed to parse response: {e}\n{body}"))
    }

    /// Send a request and return the *raw* response body as pretty JSON, for
    /// transparency demos where we want to dump exactly what came back.
    pub fn complete_raw(&self, request: &ChatRequest) -> Result<Value> {
        let url = self.chat_api_url();
        let mut rb = self.client.post(&url).json(&request);
        if let Some(key) = &self.api_key {
            rb = rb.header("Authorization", format!("Bearer {key}"));
        }
        let resp = rb.send()?;
        let status = resp.status();
        let body = resp.text()?;
        if !status.is_success() {
            return Err(anyhow!("endpoint returned {status}: {body}"));
        }
        serde_json::from_str(&body).map_err(|e| anyhow!("invalid json: {e}\n{body}"))
    }
}

/// Load a `.env` file from the current directory (or parents) if one exists,
/// using `dotenvy`. Existing real environment variables are never overwritten.
///
/// Safe to call repeatedly: it is idempotent and on subsequent calls simply
/// refreshes against the environment without clobbering anything.
pub fn load_env() -> anyhow::Result<std::path::PathBuf> {
    dotenvy::dotenv().map_err(|e| anyhow::anyhow!("failed to load .env: {e}"))
}

/// Resolve the model name from the environment (used as a default throughout).
pub fn default_model() -> String {
    let _ = load_env();
    std::env::var("AGENT_LOOP_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string())
}

/// Pretty-print a serde Value for terminal/notebook output that is readable.
pub fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| "(unprintable)".into())
}

/// A convenience: build a tiny system+user `ChatRequest` with no tools, mainly
/// for the very first "here is the basic shape" demo.
pub fn simple_request(model: &str, system: &str, user: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage::system(system), ChatMessage::user(user)],
        tools: None,
        tool_choice: None,
        temperature: Some(0.2),
    }
}

/// A compact debug rendering of a message list — used by the loop to log the
/// conversation as it evolves.
pub fn render_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!("[{}] ", m.role));
        if let Some(c) = &m.content {
            out.push_str(c);
        }
        if let Some(tcs) = &m.tool_calls {
            let names: Vec<&str> = tcs.iter().map(|t| t.name.as_str()).collect();
            out.push_str(&format!("→ tools: {}", names.join(", ")));
        }
        out.push('\n');
    }
    out
}
