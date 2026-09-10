//! # Hooks: side effects fired at lifecycle boundaries.
//!
//! An agent loop is a series of events: the user submits a prompt, a tool is
//! about to be used, a tool has just produced a result, a sub-agent starts or
//! stops, and finally the whole turn stops. Hooks are user-registered callbacks
//! that *observe* those events and react to them — logging, telemetry,
//! guardrails, caching, you name it. They sit *around* the loop without changing
//! its core wiring.
//!
//! The six hook events modelled here mirror the well-known coding-agent hook
//! names:
//!
//!   - `PreToolUse`      — fired *before* a tool runs  (could veto / log it)
//!   - `PostToolUse`     — fired *after* a tool returns (log the result)
//!   - `UserPromptSubmit`— fired when the user's message enters the loop
//!   - `Stop`            — fired when the turn ends (finish_reason = stop)
//!   - `SubAgentStart`   — fired when a sub-agent is spawned
//!   - `SubAgentStop`    — fired when a sub-agent finishes
//!
//! Hooks are just `Fn(&HookPayload)` callbacks stored per event. Because they
//! are plain closures, multiple hooks can subscribe to the same event and a
//! hook can be added or removed while the agent is alive — same dynamic spirit
//! as dynamic tools.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// The six lifecycle events the hook system can observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    Stop,
    SubAgentStart,
    SubAgentStop,
}

impl HookEvent {
    /// The canonical string name (used in logs and in the demo).
    pub fn as_str(&self) -> &'static str {
        match self {
            HookEvent::PreToolUse => "pretooluse",
            HookEvent::PostToolUse => "posttooluse",
            HookEvent::UserPromptSubmit => "userpromptsubmit",
            HookEvent::Stop => "stop",
            HookEvent::SubAgentStart => "subagentstart",
            HookEvent::SubAgentStop => "subagentstop",
        }
    }

    pub const ALL: [HookEvent; 6] = [
        HookEvent::PreToolUse,
        HookEvent::PostToolUse,
        HookEvent::UserPromptSubmit,
        HookEvent::Stop,
        HookEvent::SubAgentStart,
        HookEvent::SubAgentStop,
    ];
}

impl fmt::Display for HookEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The payload carried by a hook callback. Each variant corresponds to one
/// event and carries the data that event is naturally about.
#[derive(Debug, Clone)]
pub enum HookPayload {
    PreToolUse {
        tool: String,
        args: Value,
    },
    PostToolUse {
        tool: String,
        output: String,
        is_error: bool,
    },
    UserPromptSubmit {
        prompt: String,
    },
    Stop {
        reason: String,
    },
    SubAgentStart {
        name: String,
        prompt: String,
    },
    SubAgentStop {
        name: String,
        result: String,
    },
}

/// One line that a hook can emit into the shared log. Kept simple on purpose.
#[derive(Debug, Clone)]
pub struct HookLogEntry {
    pub event: HookEvent,
    pub message: String,
}

pub type HookFn = Arc<dyn Fn(&HookPayload) + Send + Sync>;

/// A registry of hook callbacks, one list per event.
#[derive(Default, Clone)]
pub struct Hooks {
    handlers: HashMap<HookEvent, Vec<HookFn>>,
    /// Optional shared log buffer, so the notebook can print the ordered
    /// sequence of every hook that fired.
    log: Option<Arc<Mutex<Vec<HookLogEntry>>>>,
}

impl Hooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the shared log buffer.
    pub fn with_log(mut self) -> Self {
        self.log = Some(Arc::new(Mutex::new(Vec::new())));
        self
    }

    /// Register a callback for an event. Multiple callbacks per event are fine.
    pub fn register(&mut self, event: HookEvent, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.handlers.entry(event).or_default().push(Arc::new(f));
    }

    /// Convenience: register a `PreToolUse` hook.
    pub fn on_pre_tool_use(&mut self, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.register(HookEvent::PreToolUse, f);
    }
    pub fn on_post_tool_use(&mut self, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.register(HookEvent::PostToolUse, f);
    }
    pub fn on_user_prompt_submit(&mut self, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.register(HookEvent::UserPromptSubmit, f);
    }
    pub fn on_stop(&mut self, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.register(HookEvent::Stop, f);
    }
    pub fn on_sub_agent_start(&mut self, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.register(HookEvent::SubAgentStart, f);
    }
    pub fn on_sub_agent_stop(&mut self, f: impl Fn(&HookPayload) + Send + Sync + 'static) {
        self.register(HookEvent::SubAgentStop, f);
    }

    /// Fire an event: run every callback registered for it, append to the log if
    /// enabled. Payload must match the event; mismatch is ignored defensively.
    pub fn fire(&self, event: HookEvent, payload: &HookPayload) {
        if let Some(handlers) = self.handlers.get(&event) {
            for h in handlers {
                h(payload);
            }
        }
        if let Some(log) = &self.log {
            if let Ok(mut l) = log.lock() {
                l.push(HookLogEntry {
                    event,
                    message: event.as_str().to_string(),
                });
            }
        }
    }

    /// Number of all registered callbacks.
    pub fn count(&self) -> usize {
        self.handlers.values().map(|v| v.len()).sum()
    }

    /// Snapshot the shared log, if enabled.
    pub fn log_entries(&self) -> Vec<HookLogEntry> {
        self.log
            .as_ref()
            .map(|l| l.lock().map(|g| g.clone()).unwrap_or_default())
            .unwrap_or_default()
    }

    /// A convenience hook factory used by the demo: a hook that appends a
    /// custom message to the log rather than only the event name.
    pub fn log_hook(mut self, event: HookEvent, message: String) -> Self {
        let log = self.log.clone();
        self.register(event, move |_payload| {
            if let Some(log) = &log {
                if let Ok(mut l) = log.lock() {
                    l.push(HookLogEntry {
                        event,
                        message: message.clone(),
                    });
                }
            }
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn fires_handlers_per_event() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut hooks = Hooks::new().with_log();
        let c = counter.clone();
        hooks.on_pre_tool_use(move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let c2 = counter.clone();
        hooks.on_pre_tool_use(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        });

        hooks.fire(
            HookEvent::PreToolUse,
            &HookPayload::PreToolUse {
                tool: "bash_run".into(),
                args: serde_json::json!({}),
            },
        );
        // No handler, must be a no-op.
        hooks.fire(
            HookEvent::Stop,
            &HookPayload::Stop {
                reason: "stop".into(),
            },
        );

        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(hooks.count(), 2);
    }

    #[test]
    fn log_records_events() {
        let mut hooks = Hooks::new().with_log();
        hooks.on_user_prompt_submit(|_| {});
        hooks.fire(
            HookEvent::UserPromptSubmit,
            &HookPayload::UserPromptSubmit {
                prompt: "hi".into(),
            },
        );
        hooks.fire(
            HookEvent::UserPromptSubmit,
            &HookPayload::UserPromptSubmit {
                prompt: "again".into(),
            },
        );
        let entries = hooks.log_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].event, HookEvent::UserPromptSubmit);
    }

    #[test]
    fn all_event_names_are_distinct() {
        let names: std::collections::HashSet<_> =
            HookEvent::ALL.iter().map(|e| e.as_str()).collect();
        assert_eq!(names.len(), 6);
    }
}
