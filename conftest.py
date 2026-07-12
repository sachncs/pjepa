"""pytest configuration for the pjepa project.

Adds the in-tree ``src/`` directory to ``sys.path`` so that tests can
import ``pjepa`` without an editable install. This makes the test
suite runnable from a fresh clone with a single ``pytest`` invocation.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)