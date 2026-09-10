#!/usr/bin/env python3
"""A tiny OpenAI-compatible chat server for offline rehearsal of the runbook.

Serves `POST /v1/chat/completions` and behaves like a small tool-calling model
so the notebooks can demo the *full* loop without a real model or network:

  * first request   (has the user prompt, no tool results)  -> wants `bash_run`
  * subsequent call (has the tool result in history)        -> final text answer

It also answers plain text for requests that include no tools at all.

Usage:
    python3 scripts/mock_server.py [port]          # default port 8000
Then in another terminal:
    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=sk-mock
    export AGENT_LOOP_MODEL=mock-model
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def tool_call_message(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console clean
        pass

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._chat(body)
        else:
            self._send(404, {"error": {"message": f"unknown path {self.path}"}})

    def _chat(self, body: dict):
        messages = body.get("messages", [])
        has_tools = bool(body.get("tools"))

        # If the last message is a tool result, the "model" now answers.
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            tool_output = last.get("content", "")
            text = (
                f"The bash command produced:\n{tool_output}\n\n"
                "I ran the shell command for you. That is the whole answer."
            )
            resp = [{"role": "assistant", "content": text, "tool_calls": None}]
            finish = "stop"
        elif not has_tools:
            # No tools advertised: just echo a friendly answer.
            prompt = last.get("content", "")
            resp = [
                {
                    "role": "assistant",
                    "content": f"Mock model received your message. (no tools present)",
                    "tool_calls": None,
                }
            ]
            finish = "stop"
            _ = prompt
        else:
            # Tools present, no result yet: request a bash_run call.
            resp = [
                tool_call_message("call_mock_1", "bash_run", {"command": "echo hello from the mock model"})
            ]
            finish = "tool_calls"

        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": resp[0],
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send(200, payload)

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"mock OpenAI-compatible server on http://localhost:{PORT}/v1")
    print("point OPENAI_BASE_URL here, e.g. OPENAI_BASE_URL=http://localhost:%d/v1" % PORT)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
