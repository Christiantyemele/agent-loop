//! # Executor: turning tool *calls* into tool *results*.
//!
//! The model only emits `ToolCall` declarations. The executor is the part of the
//! loop that actually runs them against the `ToolRegistry`. It also owns the
//! two execution strategies we care about for the runbook:
//!
//!   - **Sequential**: run every tool call one at a time, in order.
//!   - **Parallel**: run independent calls on separate threads and collect the
//!     results back in the original order, so the delta is just wall-clock time
//!     (and the notebook can print the timing difference).

use anyhow::Result;
use serde_json::Value;
use std::time::Instant;

use crate::chat::{ChatMessage, ToolCall};
use crate::tools::ToolRegistry;

/// Run a single tool call and wrap the outcome as a tool-result `ChatMessage`.
///
/// Two things happen here that keep the agent from crashing on a bad call:
///   1. If the arguments the model produced do not parse as JSON, we still build
///      a normal tool message (with an error) so the model can fix itself.
///   2. If the tool reports an error, that error is delivered to the model as
///      the tool *content* — the model can then change its plan and retry.
pub fn execute_one(registry: &ToolRegistry, call: &ToolCall) -> ChatMessage {
    let args: Value = match serde_json::from_str(&call.arguments) {
        Ok(v) => v,
        Err(e) => {
            return ChatMessage::tool_result(
                &call.id,
                format!("[invalid JSON arguments] {e}: {}", call.arguments),
            )
        }
    };

    match registry.get(&call.name) {
        Some(tool) => match tool.run(&args) {
            Ok(res) => ChatMessage::tool_result(&call.id, res.output),
            Err(e) => ChatMessage::tool_result(&call.id, format!("[tool error] {e}")),
        },
        None => ChatMessage::tool_result(
            &call.id,
            format!("[unknown tool] `{}` is not registered", call.name),
        ),
    }
}

/// Execute many tool calls, choosing the strategy by `parallel`.
///
/// Returns the results in the **same order** as the input calls, and a small
/// timing report for the demo.
pub fn execute_many(
    registry: &ToolRegistry,
    calls: &[ToolCall],
    parallel: bool,
) -> (Vec<ChatMessage>, ExecutionStats) {
    let started = Instant::now();

    let results: Vec<ChatMessage> = if parallel {
        // `std::thread::scope` lets the child threads borrow `registry` and
        // `calls` without owning them; the scoped handles are joined before this
        // closure returns, so results are simply collected in spawn order.
        std::thread::scope(|scope| {
            let handles: Vec<_> = calls
                .iter()
                .map(|call| scope.spawn(move || execute_one(registry, call)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        ChatMessage::tool_result("?", "[executor] worker thread panicked")
                    })
                })
                .collect()
        })
    } else {
        calls
            .iter()
            .map(|call| execute_one(registry, call))
            .collect()
    };

    let elapsed = started.elapsed();
    (
        results,
        ExecutionStats {
            n_calls: calls.len(),
            parallel,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        },
    )
}

/// Timing metadata from a batch execution.
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub n_calls: usize,
    pub parallel: bool,
    pub elapsed_ms: f64,
}

/// Convenience for a single call, run sequentially.
pub fn execute(registry: &ToolRegistry, call: &ToolCall) -> Result<ChatMessage> {
    Ok(execute_one(registry, call))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{s_required, Tool, ToolResult};
    use serde_json::json;

    fn registry_with_sleep_tools(n: usize) -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        for i in 0..n {
            reg.register(Tool::new(
                format!("sleep_{i}"),
                "sleep then report",
                s_required(json!({"ms": {"type":"integer"}}), &["ms"]),
                move |args| {
                    let ms = args.get("ms").and_then(Value::as_i64).unwrap_or(200);
                    std::thread::sleep(std::time::Duration::from_millis(ms as u64));
                    Ok(ToolResult::ok(format!("slept {ms}ms in tool {i}")))
                },
            ));
        }
        reg
    }

    fn make_calls(n: usize) -> Vec<ToolCall> {
        (0..n)
            .map(|i| ToolCall {
                id: format!("call_{i}"),
                name: format!("sleep_{i}"),
                arguments: r#"{"ms": 300}"#.to_string(),
            })
            .collect()
    }

    #[test]
    fn sequential_is_slower_than_parallel() {
        let reg = registry_with_sleep_tools(4);
        let calls = make_calls(4);

        let (seq, seq_stats) = execute_many(&reg, &calls, false);
        let (par, par_stats) = execute_many(&reg, &calls, true);

        assert_eq!(seq.len(), 4);
        assert_eq!(par.len(), 4);
        // 4 x 300ms sequential should be ~1200ms; parallel ~300ms.
        assert!(
            seq_stats.elapsed_ms > 900.0,
            "sequential too fast: {:?}",
            seq_stats
        );
        assert!(
            par_stats.elapsed_ms < 800.0,
            "parallel too slow: {:?}",
            par_stats
        );
        // Results preserve order in both modes.
        for i in 0..4 {
            assert!(seq[i]
                .content
                .as_ref()
                .unwrap()
                .contains(&format!("tool {i}")));
            assert!(par[i]
                .content
                .as_ref()
                .unwrap()
                .contains(&format!("tool {i}")));
        }
    }

    #[test]
    fn bad_json_arguments_does_not_crash() {
        let reg = registry_with_sleep_tools(1);
        let bad = ToolCall {
            id: "x".into(),
            name: "sleep_0".into(),
            arguments: "not-json".into(),
        };
        let msg = execute_one(&reg, &bad);
        assert!(msg.content.unwrap().contains("invalid JSON"));
    }

    #[test]
    fn unknown_tool_reports_cleanly() {
        let reg = ToolRegistry::new();
        let call = ToolCall {
            id: "y".into(),
            name: "nope".into(),
            arguments: "{}".into(),
        };
        let msg = execute_one(&reg, &call);
        assert!(msg.content.unwrap().contains("unknown tool"));
    }
}
