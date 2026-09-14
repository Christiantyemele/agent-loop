//! # Tools: what the agent can touch.
//!
//! A tool is two things bundled together:
//!
//! 1. **A description** the model can read (name, description, JSON Schema for
//!    parameters). This is what gets serialized into the `tools` array of the
//!    request. The model uses it to *learn the tool exists* and to *fill in the
//!    arguments*.
//! 2. **An executor** — the actual Rust code that runs when the loop decides the
//!    call should happen. The model never runs this; we do.
//!
//! We package the two as a single `Tool` object so that "advertising" and
//! "executing" always travel together. Tools are also **dynamic**: the registry
//! can be extended at runtime, which is exactly how new capabilities get bolted
//! on to a running agent without recompiling it.
//!
//! The built-ins here cover the families from the plan:
//!
//!   - **file operations**: read / write / list files
//!   - **bash**: run a shell command and capture stdout/stderr
//!   - **web search**: ask a search engine (or fetch a URL) for web content
//!   - **calculator**: safely evaluate a numeric expression

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// The outcome of running a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The text content handed back to the model as the tool result message.
    pub output: String,
    /// Whether execution failed. The loop surfaces errors to the model as a
    /// normal tool message so the model can adapt, rather than crashing.
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
        }
    }
    pub fn err(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: true,
        }
    }
}

/// Type of the executor for a tool: takes the parsed arguments JSON object and
/// returns a `ToolResult`.
pub type ToolExecutor = Arc<dyn Fn(&Value) -> Result<ToolResult> + Send + Sync>;

/// A single tool, combining its model-facing description with its executor.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub executor: ToolExecutor,
}

impl Tool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        executor: impl Fn(&Value) -> Result<ToolResult> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            executor: Arc::new(executor),
        }
    }

    /// Convert this tool into its `ChatTool` wire representation for the request.
    pub fn as_chat_tool(&self) -> crate::chat::ChatTool {
        crate::chat::ChatTool::function(
            self.name.clone(),
            self.description.clone(),
            self.parameters.clone(),
        )
    }

    /// Run the executor, converting any panic into an error result so that one
    /// badly-behaved tool doesn't take the whole agent down.
    pub fn run(&self, args: &Value) -> Result<ToolResult> {
        let name = self.name.clone();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.executor)(args)))
            .map_err(|_| anyhow!("tool `{name}` panicked"))?
    }
}

/// Convenience schema builders so tool authors don't hand-write JSON.
pub fn s_required(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

// ---------------------------------------------------------------------------
// Built-in tools
// ---------------------------------------------------------------------------

pub mod file_tools {
    use super::*;
    use std::path::PathBuf;

    /// Read a file from the (sandboxed) working directory and return its contents.
    pub fn read_tool() -> Tool {
        Tool::new(
            "file_read",
            "Read the full contents of a text file at the given absolute or relative path. Returns file contents, or an error if the file does not exist or is not readable.",
            s_required(json!({
                "path": {"type": "string", "description": "Absolute or workspace-relative path of the file to read."}
            }), &["path"]),
            |args| {
                let path = args.get("path").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `path`"))?;
                let p = PathBuf::from(path);
                let contents = std::fs::read_to_string(&p)
                    .with_context(|| format!("read {path}"))?;
                Ok(ToolResult::ok(contents))
            },
        )
    }

    /// Write text to a file, creating parent directories as needed.
    pub fn write_tool() -> Tool {
        Tool::new(
            "file_write",
            "Write the given text content to a file at the given path, overwriting it. Creates parent directories automatically. Returns the path written.",
            s_required(json!({
                "path": {"type": "string", "description": "Path of the file to write."},
                "content": {"type": "string", "description": "Full text content to write to the file."}
            }), &["path", "content"]),
            |args| {
                let path = args.get("path").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `path`"))?;
                let content = args.get("content").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `content`"))?;
                let p = PathBuf::from(path);
                if let Some(parent) = p.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                std::fs::write(&p, content)
                    .with_context(|| format!("write {path}"))?;
                Ok(ToolResult::ok(format!("wrote {path}")))
            },
        )
    }

    /// List entries in a directory.
    pub fn list_tool() -> Tool {
        Tool::new(
            "file_list",
            "List the names of entries inside a directory. Returns one entry name per line.",
            s_required(
                json!({
                    "path": {"type": "string", "description": "Directory to list."}
                }),
                &["path"],
            ),
            |args| {
                let path = args
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `path`"))?;
                let mut entries: Vec<String> = std::fs::read_dir(path)
                    .with_context(|| format!("list {path}"))?
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect();
                entries.sort();
                Ok(ToolResult::ok(entries.join("\n")))
            },
        )
    }
}

pub mod bash_tools {
    use super::*;

    /// Run a shell command via `bash -c` and capture stdout + stderr.
    pub fn run_tool() -> Tool {
        Tool::new(
            "bash_run",
            "Run a shell command through /bin/bash and capture its stdout. If the command exits non-zero, stdout/stderr are returned and the tool is marked as errored.",
            s_required(json!({
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout_secs": {"type": "integer", "description": "Optional per-command timeout in seconds (default 30)."}
            }), &["command"]),
            |args| {
                let command = args.get("command").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `command`"))?;
                let _timeout = args.get("timeout_secs").and_then(Value::as_i64).unwrap_or(30);
                let out = std::process::Command::new("/bin/bash")
                    .arg("-c")
                    .arg(command)
                    .output();
                match out {
                    Ok(o) => {
                        let stdout = String::from_utf8_lossy(&o.stdout);
                        let stderr = String::from_utf8_lossy(&o.stderr);
                        let full = if stderr.is_empty() {
                            stdout.into_owned()
                        } else {
                            format!("{stdout}\n[stderr]\n{stderr}")
                        };
                        let body = if full.is_empty() { "(no output)".to_string() } else { full };
                        if o.status.success() {
                            Ok(ToolResult::ok(body))
                        } else {
                            Ok(ToolResult::err(format!("exit {:?}: {body}", o.status.code())))
                        }
                    }
                    Err(e) => Ok(ToolResult::err(format!("could not spawn bash: {e}"))),
                }
            },
        )
    }
}

pub mod calc_tools {
    use super::*;

    /// A tool that safely evaluates a numeric expression.
    ///
    /// Deliberately **not** an `eval` of arbitrary code: it is a small recursive-
    /// descent parser over a closed grammar (numbers, `+ - * / ^`, parentheses,
    /// a set of named functions and constants). That keeps it safe while still
    /// being useful for the "let the agent do arithmetic" use case.
    pub fn calc_tool() -> Tool {
        Tool::new(
            "calc",
            "Evaluate a numeric expression and return the result. Supports +, -, *, /, ^ (power), parentheses, the constants pi/e/tau, and the functions sqrt, abs, sin, cos, tan, asin, acos, atan, exp, ln, log, floor, ceil, round, min, max. Example expression: (2 + 3) * 4 ^ 2 + sqrt(9).",
            s_required(json!({
                "expression": {"type": "string", "description": "The arithmetic expression to evaluate, e.g. \"(2 + 3) * 4\"."}
            }), &["expression"]),
            |args| {
                let expr = args.get("expression")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `expression`"))?;
                let value = crate::calc::eval(expr)
                    .map_err(|e| anyhow::anyhow!("failed to evaluate {expr:?}: {e}"))?;
                let rendered = if value.fract() == 0.0 && value.abs() < 1e15 {
                    format!("{}", value as i64)
                } else {
                    format!("{value}")
                };
                Ok(ToolResult::ok(rendered))
            },
        )
    }
}

pub mod web_tools {
    use super::*;

    /// Do an HTTP GET against a URL and return the response text. This is the
    /// "web" capability used by the agent to gather external content.
    pub fn fetch_tool() -> Tool {
        Tool::new(
            "web_fetch",
            "Perform an HTTP GET request to the given URL and return the response body as text. Useful for retrieving web pages, APIs, and other remote content the agent cannot see by itself.",
            s_required(json!({
                "url": {"type": "string", "description": "The absolute URL to fetch (https or http)."}
            }), &["url"]),
            |args| {
                let url = args.get("url").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `url`"))?;
                let resp = reqwest::blocking::Client::new()
                    .get(url)
                    .send()
                    .with_context(|| format!("GET {url}"))?;
                let status = resp.status();
                let text = resp.text().unwrap_or_default();
                if status.is_success() {
                    Ok(ToolResult::ok(text))
                } else {
                    Ok(ToolResult::err(format!("HTTP {status}: {text}")))
                }
            },
        )
    }

    /// A search tool. It consults `TAVILY_API_KEY` if present; otherwise it
    /// returns a clear explanation that a search backend is not configured, so
    /// the model can fall back to `web_fetch`.
    pub fn search_tool() -> Tool {
        Tool::new(
            "web_search",
            "Search the web for the given query and return a short list of result titles and URLs. Requires the TAVILY_API_KEY environment variable; when it is absent the tool explains that and the agent should fall back to web_fetch.",
            s_required(json!({
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "Optional number of results to return (default 5)."}
            }), &["query"]),
            |args| {
                let query = args.get("query").and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing `query`"))?;
                let max = args.get("max_results").and_then(Value::as_i64).unwrap_or(5);
                let _ = crate::chat::load_env();
                let key = std::env::var("TAVILY_API_KEY").ok()
                    .filter(|k| !k.is_empty());
                match key {
                    Some(key) => {
                        let body = json!({
                            "query": query,
                            "max_results": max,
                            "search_depth": "basic"
                        });
                        let client = reqwest::blocking::Client::new();
                        let resp = client.post("https://api.tavily.com/search")
                            .header("Authorization", format!("Bearer {key}"))
                            .json(&body)
                            .send()
                            .with_context(|| "tavily request")?;
                        let text = resp.text().unwrap_or_default();
                        Ok(ToolResult::ok(text))
                    }
                    None => Ok(ToolResult::ok(
                        "web_search has no backend configured: TAVILY_API_KEY is not set. \
                         Suggest the agent use web_fetch instead to retrieve a specific URL.",
                    )),
                }
            },
        )
    }
}

/// Return the built-in, "always available" tool set.
pub fn builtin_tools() -> Vec<Tool> {
    vec![
        file_tools::read_tool(),
        file_tools::write_tool(),
        file_tools::list_tool(),
        bash_tools::run_tool(),
        web_tools::fetch_tool(),
        web_tools::search_tool(),
        calc_tools::calc_tool(),
    ]
}

// ---------------------------------------------------------------------------
// Dynamic registry
// ---------------------------------------------------------------------------

/// A mutable registry of tools that can be extended at runtime.
///
/// The interesting property for the demo is that this is a plain in-memory map:
/// we can insert, replace, or remove tools *while the agent is running*. That is
/// what "dynamic tools" means — the tool surface is not fixed at compile time.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    order: Vec<String>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a registry seeded with the built-ins.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        for t in builtin_tools() {
            r.register(t);
        }
        r
    }

    /// Register (or replace) a tool by name.
    pub fn register(&mut self, tool: Tool) {
        if !self.tools.contains_key(&tool.name) {
            self.order.push(tool.name.clone());
        }
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Remove a tool, returning it if it was present.
    pub fn unregister(&mut self, name: &str) -> Option<Tool> {
        self.order.retain(|n| n != name);
        self.tools.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    /// The full list of tools, in registration order.
    pub fn all(&self) -> Vec<&Tool> {
        self.order
            .iter()
            .filter_map(|n| self.tools.get(n))
            .collect()
    }

    /// The tools in wire (ChatTool) form, ready for the request.
    pub fn as_chat_tools(&self) -> Vec<crate::chat::ChatTool> {
        self.all().into_iter().map(|t| t.as_chat_tool()).collect()
    }

    /// Does a tool with this name exist?
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_seeded_with_builtins() {
        let r = ToolRegistry::with_builtins();
        assert!(r.contains("file_read"));
        assert!(r.contains("bash_run"));
        assert!(r.contains("web_search"));
        assert!(r.contains("web_fetch"));
        assert!(r.contains("calc"));
        assert_eq!(r.len(), 7);
    }

    #[test]
    fn file_write_then_read_roundtrip() {
        let registry = ToolRegistry::with_builtins();
        let dir = std::env::temp_dir().join(format!("agent_loop_t_{}", std::process::id()));
        let path = dir.join("notes.txt");
        let path_s = path.to_string_lossy().into_owned();
        let _ = std::fs::create_dir_all(&dir);

        let write = registry.get("file_write").unwrap();
        let r = write.run(&json!({"path": path_s, "content": "hello agent"}));
        assert!(r.is_ok() && !r.unwrap().is_error, "write failed");

        let read = registry.get("file_read").unwrap();
        let r = read.run(&json!({"path": path_s}));
        let res = r.unwrap();
        assert!(!res.is_error);
        assert_eq!(res.output, "hello agent");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bash_runs_and_captures() {
        let registry = ToolRegistry::with_builtins();
        let b = registry.get("bash_run").unwrap();
        let r = b.run(&json!({"command": "echo hello-from-bash"})).unwrap();
        assert!(!r.is_error);
        assert!(r.output.contains("hello-from-bash"));
    }

    #[test]
    fn dynamic_add_and_remove() {
        let mut reg = ToolRegistry::new();
        reg.register(Tool::new(
            "greet",
            "say hi",
            s_required(json!({}), &[]),
            |_| Ok(ToolResult::ok("hi")),
        ));
        assert!(reg.contains("greet"));
        assert_eq!(reg.len(), 1);

        // Replace the handler at runtime.
        reg.register(Tool::new(
            "greet",
            "say hi",
            s_required(json!({}), &[]),
            |_| Ok(ToolResult::ok("hola")),
        ));
        assert_eq!(
            reg.get("greet").unwrap().run(&json!({})).unwrap().output,
            "hola"
        );

        reg.unregister("greet");
        assert!(!reg.contains("greet"));
    }

    #[test]
    fn panic_inside_tool_is_caught() {
        let tool = Tool::new(
            "boom",
            "panics",
            s_required(json!({}), &[]),
            |_| -> Result<ToolResult> {
                panic!("boom");
            },
        );
        let r = tool.run(&json!({}));
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("panicked"));
    }
}
