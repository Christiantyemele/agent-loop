#!/usr/bin/env python3
"""Execute a notebook with the evcxr kernel and print EVERY output including
the kernel's stderr stream (where evcxr emits Rust compile errors).

Usage:
    python3 scripts/run_nb.py [--fail-on-error] NOTEBOOK.ipynb

--fail-on-error: after execution, exit non-zero if any code cell produced an
`error` output. This turns the script into a strict CI check (the default
behaviour of printing all outputs and exiting 0 is preserved without the flag).
"""
import sys
import nbformat
from nbclient import NotebookClient


def main(argv: list) -> int:
    fail_on_error = False
    args = []
    for a in argv[1:]:
        if a == "--fail-on-error":
            fail_on_error = True
        else:
            args.append(a)
    if len(args) != 1:
        print(
            "usage: run_nb.py [--fail-on-error] NOTEBOOK.ipynb",
            file=sys.stderr,
        )
        return 2

    nbpath = args[0]
    nb = nbformat.read(nbpath, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name="rust",
        timeout=300,
        allow_errors=True,
        # First kernel start can be slow (evcxr compiles its runtime); give it
        # room so a slow start is not mistaken for a cell/setup failure.
        startup_timeout=300,
    )
    client.execute()

    errored = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        print(f"\n===== CELL {i} =====")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                print(out.get("text", ""))
            elif out.get("output_type") == "error":
                ename = out.get("ename")
                evalue = out.get("evalue")
                errored.append((i, ename, evalue))
                print("ERROR:", ename, ":", evalue)
                for line in out.get("traceback", []):
                    print("   ", line)
            elif out.get("output_type") == "execute_result":
                print("".join(out.get("data", {}).get("text/plain", [])))
            elif out.get("output_type") == "display_data":
                print("".join(out.get("data", {}).get("text/plain", [])))

    if fail_on_error and errored:
        print(
            f"\nFAILED: {len(errored)} cell(s) produced errors:"
            + " ".join(f"cell {i}" for i, _, _ in errored),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
