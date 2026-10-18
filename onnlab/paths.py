# Made by Ali Nadarloo.
# SPDX-License-Identifier: MIT
"""One writable app home for the source launcher and lightweight Windows EXE."""
import os
from pathlib import Path
import sys


def application_directory():
    if os.environ.get("WAVEFORGE_HOME"):return Path(os.environ["WAVEFORGE_HOME"]).resolve()
    if getattr(sys,"frozen",False):return Path(sys.executable).parent
    return Path(__file__).resolve().parents[1]
