#!/usr/bin/env python3
"""Run a pipeline on whichever backend `GSTML_BACKEND` selects, from the checkout.

See `plugins/python/pyml_launch.py` for what the two backends spell differently.
Re-runs itself under the repo's venv when started from another interpreter: the
launcher hands its own site directories to GStreamer and to g2g, neither of
which has a venv of its own.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = (
    VENV / "Scripts" / "python.exe" if os.name == "nt" else VENV / "bin" / "python"
)


def main():
    # sys.prefix, not the executable path: a venv's `python` is a symlink to the
    # base interpreter, so resolving it would say we are already inside.
    if VENV_PYTHON.exists() and Path(sys.prefix) != VENV:
        return subprocess.call([str(VENV_PYTHON), __file__, *sys.argv[1:]])
    sys.path.insert(0, str(ROOT / "plugins" / "python"))
    import pyml_launch

    return pyml_launch.main()


if __name__ == "__main__":
    sys.exit(main())
