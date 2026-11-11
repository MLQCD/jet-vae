#!/usr/bin/env python
"""Run the compiled cnn_vae visualization app when app.py/core.py are absent."""

import marshal
import sys
import types
from pathlib import Path


VISUALIZE_DIR = Path(__file__).resolve().parent / "visualize"
PYC_DIR = VISUALIZE_DIR / "__pycache__"


def _read_pyc_code(path):
    with open(path, "rb") as f:
        data = f.read()
    return marshal.loads(data[16:])


core = types.ModuleType("core")
core.__file__ = str(VISUALIZE_DIR / "core.py")
core.__package__ = ""
exec(_read_pyc_code(PYC_DIR / "core.cpython-313.pyc"), core.__dict__)
sys.modules["core"] = core

globals()["__file__"] = str(VISUALIZE_DIR / "app.py")
exec(_read_pyc_code(PYC_DIR / "app.cpython-313.pyc"), globals())
