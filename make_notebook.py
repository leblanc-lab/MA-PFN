#!/usr/bin/env python3
"""Regenerate the notebook from the percent-formatted Python source."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "ma_pfn_demo.ipynb"


def main() -> None:
    try:
        import jupytext  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Jupytext is required; install it with `python -m pip install jupytext`."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="ma-pfn-jupyter-") as temp_dir:
        environment = os.environ.copy()
        environment.setdefault("JUPYTER_CONFIG_DIR", temp_dir)
        environment.setdefault("JUPYTER_DATA_DIR", temp_dir)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--from",
                "py:percent",
                "--to",
                "ipynb",
                "--output",
                "ma_pfn_demo.ipynb",
                "ma_pfn_demo.py",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    # nbformat otherwise assigns fresh random cell IDs on every conversion.
    # Stable IDs keep repeated CI runs byte-for-byte reproducible.
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        identity = f"{index}\0{cell['cell_type']}\0{source}".encode()
        cell["id"] = hashlib.sha256(identity).hexdigest()[:12]
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print("Generated ma_pfn_demo.ipynb from ma_pfn_demo.py with Jupytext")


if __name__ == "__main__":
    main()
