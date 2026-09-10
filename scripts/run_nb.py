#!/usr/bin/env python3
"""Execute a notebook with the evcxr kernel and print EVERY output including
the kernel's stderr stream (where evcxr emits Rust compile errors)."""
import sys
import nbformat
from nbclient import NotebookClient

nbpath = sys.argv[1]
nb = nbformat.read(nbpath, as_version=4)
client = NotebookClient(nb, kernel_name="rust", timeout=300, allow_errors=True)
client.execute()

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    print(f"\n===== CELL {i} =====")
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            print(out.get("text", ""))
        elif out.get("output_type") == "error":
            print("ERROR:", out.get("ename"), ":", out.get("evalue"))
            for line in out.get("traceback", []):
                print("   ", line)
        elif out.get("output_type") == "execute_result":
            print("".join(out.get("data", {}).get("text/plain", [])))
        elif out.get("output_type") == "display_data":
            print("".join(out.get("data", {}).get("text/plain", [])))
