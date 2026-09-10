use agent_loop::{Agent, AgentConfig, ChatClient, HookPayload, Hooks, ToolRegistry};

fn main() {
    // This binary is a convenience fallback: it runs the same pieces the Jupyter
    // notebooks demonstrate, pointed at whatever OPENAI_BASE_URL you set.

    let client = ChatClient::from_env();
    let registry = ToolRegistry::with_builtins();
    let mut hooks = Hooks::new().with_log();

    hooks.on_pre_tool_use(|p| {
        if let HookPayload::PreToolUse { tool, args } = p {
            println!("[pretooluse] about to call `{tool}` with {args}");
        }
    });
    hooks.on_post_tool_use(|p| {
        if let HookPayload::PostToolUse { tool, output, .. } = p {
            let snippet: String = output.chars().take(80).collect();
            println!("[posttooluse] `{tool}` returned: {snippet}…");
        }
    });

    let mut agent = Agent::new(AgentConfig::new(
        agent_loop::chat::default_model(),
        "You are a helpful coding agent with access to file, bash and web tools.",
        registry,
        hooks,
        client,
    ));
    agent.config.parallel_tools = true;

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Run `echo hello from the agent loop` in the shell, then tell me what it printed."
            .to_string()
    } else {
        prompt
    };

    match agent.run(&prompt) {
        Ok(outcome) => {
            println!("\n{}", "=".repeat(60));
            println!("FINAL ANSWER:\n{}", outcome.final_text);
            println!("\nTRANSCRIPT:\n{}", agent.transcript());
            let log = agent.config.hooks.log_entries();
            println!(
                "HOOK ORDER: {}",
                log.iter()
                    .map(|e| e.event.as_str())
                    .collect::<Vec<_>>()
                    .join(" → ")
            );
        }
        Err(e) => {
            eprintln!("run failed: {e}");
            eprintln!("\nhint: set OPENAI_BASE_URL (and OPENAI_API_KEY / AGENT_LOOP_MODEL) to point at an OpenAI-compatible endpoint, e.g.");
            eprintln!(
                "  OPENAI_BASE_URL=http://localhost:11434/v1 AGENT_LOOP_MODEL=llama3.1 cargo run"
            );
        }
    }
}
