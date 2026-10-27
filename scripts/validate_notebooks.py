#!/usr/bin/env python3
"""Offline structural + API-drift validation for the agent-loop runbook notebooks.

Reads every `notebooks/*.ipynb` with nbformat and checks:

  1. The file parses as a nbformat v4 notebook.
  2. The kernelspec is the `rust` evcxr kernel (name == "rust").
  3. Every cell is well-formed and every code cell has a non-empty source.
  4. Every `agent_loop::<mod>::<item>` / `use agent_loop::<mod>::{...}`
     reference in the code cells names a module and a public item that actually
     exist in `src/`. This catches wrong imports and renamed/removed APIs, which
     is the notebook->library drift the existing CI (crate-only) never sees.

The check is deliberately independent of the `:dep agent_loop = { path = ... }`
line, which is a machine-specific absolute path that must not be validated.

Usage:
    python3 scripts/validate_notebooks.py [notebooks_dir] [src_dir]
        notebooks_dir  default "notebooks"
        src_dir        default "src"

Exit code 0 on success, 1 if any notebook fails. Prints one human-readable
report block per notebook and a summary line.
"""

import json
import os
import re
import sys


def die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Public API extraction from src/*.rs
# ---------------------------------------------------------------------------

# Symbols we do not treat as API-drift errors: standard-library/aliased items a
# notebook may write that are not defined in our crate.
IGNORED_ITEMS = {"use", "as", "super", "crate", "self"}


def extract_public_api(src_dir: str) -> dict:
    """Return {module_name: set(public_item_names)} parsed from src/*.rs.

    We scan each top-level src/<mod>.rs for `pub fn NAME`, `pub struct NAME`,
    `pub enum NAME`, `pub type NAME`, `pub const NAME`, and `pub use ... as NAME`
    re-exports. Nested `pub mod file_tools { ... }` submodules' items look like
    `pub mod NAME` at the top level; those are captured as separately checkable
    modules via a recursive pass so `agent_loop::tools::file_tools::read_tool`
    resolves.
    """
    modules: dict = {}

    def fn_name(m: re.Match) -> str:
        return m.group(1)

    patterns = [
        re.compile(r"\bpub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bpub\s+struct\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bpub\s+enum\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bpub\s+type\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bpub\s+const\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bpub\s+use\s+[^;]+?\bas\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\bpub\s+mod\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ]

    if not os.path.isdir(src_dir):
        return modules

    for filename in sorted(os.listdir(src_dir)):
        if not filename.endswith(".rs") or filename == "lib.rs" or filename == "main.rs":
            continue
        mod = filename[:-3]
        path = os.path.join(src_dir, filename)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            print(f"warning: cannot read {path}: {e}", file=sys.stderr)
            continue
        names: set = set()
        for pat in patterns:
            for m in pat.finditer(text):
                name = fn_name(m)
                if name:
                    names.add(name)
        modules[mod] = names

    # lib.rs root re-exports (pub use) form the crate-root namespace.
    root: set = set()
    lib_path = os.path.join(src_dir, "lib.rs")
    if os.path.isfile(lib_path):
        try:
            lib_text = open(lib_path, encoding="utf-8").read()
        except OSError:
            lib_text = ""
        # Extract each `pub use ...;` statement body (may be a single item or a
        # `{A, B, C}` brace group) and expand it into re-exported names.
        use_parser = re.compile(
            r"\bpub\s+use\s+(?P<body>[^;]+);", re.MULTILINE
        )
        for m in use_parser.finditer(lib_text):
            body = m.group("body")
            for chunk in body.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                chunk = chunk.split(" as ")[0].strip()
                if chunk.startswith("self"):
                    continue
                # handle `agent::{Agent` -> final segment `{Agent` -> `Agent`
                leaf = chunk.rsplit("::", 1)[-1]
                leaf = leaf.strip("{}")
                if leaf:
                    root.add(leaf)
    modules[""] = root  # crate root namespace

    return modules


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

# agent_loop::chat::pretty  ;  agent_loop::tools::file_tools::read_tool
FULL_PATH = re.compile(r"\bagent_loop::([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)")

# use agent_loop::chat::{ChatRequest, ChatMessage};  and single items:
# use agent_loop::executor::execute_one;
USE_ITEMS = re.compile(
    r"\buse\s+agent_loop::"
    r"((?:[A-Za-z_][A-Za-z0-9_]*::)+)"
    r"(?:\{([^}]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)

# use agent_loop::tools::{Tool, ToolResult, s_required};
USE_ROOT = re.compile(r"\buse\s+agent_loop::\{(.*)\}")


def split_imports(group: str):
    """Split a `{ a, b, c }` import group on commas, trimming whitespace and
    trailing `self`/re-exports."""
    for raw in group.split(","):
        item = raw.strip()
        if not item:
            continue
        # handle `foo as bar`
        item = item.split(" as ")[0].strip()
        if item.startswith("self"):
            continue
        if item:
            yield item


def check_code_cell(source: str, api: dict) -> list:
    """Return a list of (reference, problem) tuples for one code cell source."""
    problems: list = []

    # `use agent_loop::{A, B, ...}`  (crate-root imports)
    for m in USE_ROOT.finditer(source):
        for item in split_imports(m.group(1)):
            if item not in api.get("", set()):
                problems.append((f"use agent_loop::{{{item}}}", item))

    # `use agent_loop::<path>::{A, B}` and `use agent_loop::<path>::Item`
    for m in USE_ITEMS.finditer(source):
        path = m.group(1).rstrip(":")
        parts = [p for p in path.split("::") if p]
        items = split_imports(m.group(2)) if m.group(2) else []
        if m.group(3):
            items.append(m.group(3))
        # Resolve module path: the leaf that names a module is everything up to
        # the final `::`; the final segment may be a module or (for full-path
        # single items) we verify the deepest module exists.
        for item in items:
            if item in IGNORED_ITEMS:
                continue
            mod = resolve_module(parts, api)
            if mod is None:
                problems.append(
                    (f"use agent_loop::{path}::{item}", f"unknown module path {path}")
                )
            elif item not in api.get(mod, set()):
                problems.append(
                    (f"use agent_loop::{path}::{item}", f"{item} not found in {mod}")
                )

    # `agent_loop::mod::item` inline (non-`use`) references, e.g.
    # agent_loop::chat::pretty(...)
    for m in FULL_PATH.finditer(source):
        parts = [p for p in m.group(1).split("::") if p]
        if len(parts) < 2:
            continue
        mod = resolve_module(parts[:-1], api)
        item = parts[-1]
        if mod is None:
            problems.append((m.group(0), f"unknown module path {'::'.join(parts[:-1])}"))
        elif item not in api.get(mod, set()):
            problems.append((m.group(0), f"{item} not found in {mod}"))

    return problems


def resolve_module(parts: list, api: dict):
    """Given the module path parts after `agent_loop::`, return the top-level
    module name whose item set should contain the final item, or None if
    unknown. Submodules (e.g. `tools::file_tools`) are flattened into their
    top-level module's item set, so we match on the longest known prefix."""
    if not parts:
        return ""
    for split in range(1, len(parts) + 1):
        mod = "::".join(parts[:split])
        if mod in api:
            return mod
    return None


# ---------------------------------------------------------------------------
# Notebook validation
# ---------------------------------------------------------------------------


def validate_notebook(path: str, api: dict) -> list:
    """Return a list of problem strings for one notebook. Empty == OK."""
    problems: list = []

    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        return [f"cannot read notebook: {e}"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]

    nbformat_v = data.get("nbformat")
    if nbformat_v != 4:
        problems.append(f"nbformat must be 4 (got {nbformat_v!r})")
    minor = data.get("nbformat_minor", 0)
    if nbformat_v == 4 and minor < 4:
        problems.append(f"nbformat_minor must be >= 4 (got {minor})")

    metadata = data.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    if kernelspec.get("name") != "rust":
        problems.append(f"kernelspec.name must be 'rust' (got {kernelspec.get('name')!r})")
    # language_info.name is the display name (Jupyter writes "Rust"); accept it
    # case-insensitively so generated and hand-edited notebooks both validate.
    lang_name = metadata.get("language_info", {}).get("name")
    if (lang_name or "").lower() != "rust":
        problems.append(f"language_info.name must be 'Rust' (got {lang_name!r})")

    cells = data.get("cells")
    if not isinstance(cells, list):
        return problems + ["missing 'cells' list"]

    code_seen = False
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        if ctype not in ("code", "markdown", "raw"):
            problems.append(f"cell {i}: unknown cell_type {ctype!r}")
            continue
        source = cell.get("source")
        if source is None:
            problems.append(f"cell {i}: missing 'source'")
            continue
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if ctype == "code":
            code_seen = True
            if not source_text.strip():
                problems.append(f"cell {i}: code cell is empty")
            for ref, problem in check_code_cell(source_text, api):
                problems.append(f"cell {i}: {ref} -> {problem}")

    if not code_seen:
        problems.append("notebook has no code cells")

    return problems


def main(argv: list) -> int:
    notebooks_dir = argv[1] if len(argv) > 1 else "notebooks"
    src_dir = argv[2] if len(argv) > 2 else "src"

    if not os.path.isdir(notebooks_dir):
        return die(f"notebooks directory not found: {notebooks_dir}")

    api = extract_public_api(src_dir)
    if not api:
        return die(f"could not extract a public API map from {src_dir}")

    nb_files = sorted(
        f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")
    )
    if not nb_files:
        return die(f"no .ipynb notebooks found in {notebooks_dir}")

    failures = 0
    for nb in nb_files:
        path = os.path.join(notebooks_dir, nb)
        problems = validate_notebook(path, api)
        print(f"== {nb}")
        if problems:
            failures += 1
            for p in problems:
                print(f"   [FAIL] {p}")
        else:
            print("   ok")
        print()

    if failures:
        print(f"FAILED: {failures}/{len(nb_files)} notebook(s) failed validation.")
        return 1
    print(f"OK: all {len(nb_files)} notebooks validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
