"""Guards .env.example against drift: every env var the app reads (backed by
a app.config default) must be documented there, so nothing silently falls
back to a default without a visible, commented reference in .env.example.
"""

from __future__ import annotations

import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"
ENV_EXAMPLE = BACKEND_DIR / ".env.example"

# Matches the literal env var name in os.getenv("NAME", ...) and the local
# _read_bool_env("NAME", ...) helper (app.models). Dynamic lookups (e.g.
# os.getenv(name) where name is a variable) are intentionally not matched --
# there's no literal to check, and none exist in app/ today.
_ENV_READ_PATTERN = re.compile(r"""(?:os\.getenv|_read_bool_env)\(\s*["']([A-Z][A-Z0-9_]*)["']""")

# Matches KEY= at the start of a line, with or without a leading "# " --
# .env.example documents some keys only as a commented-out alternative
# (e.g. TESSERACT_CMD), which still counts as "documented".
_ENV_EXAMPLE_KEY_PATTERN = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _env_vars_read_by_app() -> set[str]:
    names: set[str] = set()
    for path in APP_DIR.rglob("*.py"):
        names.update(_ENV_READ_PATTERN.findall(path.read_text(encoding="utf-8")))
    return names


def _env_vars_documented() -> set[str]:
    return set(_ENV_EXAMPLE_KEY_PATTERN.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


def test_every_env_var_read_by_app_is_documented_in_env_example() -> None:
    undocumented = _env_vars_read_by_app() - _env_vars_documented()
    assert not undocumented, (
        f"env var(s) read by app/ but missing from .env.example: {sorted(undocumented)}"
    )
