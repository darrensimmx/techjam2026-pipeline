"""Proves the GRADED path makes zero network calls -- a static check, not a promise.

If the official harness runs with networking disabled and a call reaches an
external API, every session silently scores zero. There is no traceback to read:
`respond()` catches, the evaluator catches, and the result is a quietly empty
recommendation list.

## Why this is a new module rather than an edit to tests/test_offline.py

`tests/test_offline.py` scans `starter/*.py` only. `starter/` is the superseded
Phase-1 system, deliberately untouched as the historical record, and it stays
covered exactly as it is. But the graded code now lives in `src/`, re-exported
by the repo-root `agent.py` -- and NOTHING currently scans it. That is the
highest-severity silent hole in the migration: the offline guarantee would look
intact on a green CI run while covering none of the code that actually ships.

The two modules are therefore complementary, not duplicates. This one also
covers three module names the older list predates (`asyncio`, `telnetlib`,
`xmlrpc`) and walks `src/` recursively rather than one directory deep.

## What a static import scan does and does not prove

It proves no module in the graded path names a networking package at import
time. It does NOT execute anything, and it cannot see a call made through a
dependency, a dynamically-built module name, or a subprocess. The runtime half
of the guarantee is `scripts/verify_offline_safety.sh`, which revokes networking
with `sandbox-exec` and runs the real evaluator inside it (macOS only; the
Windows equivalent is `docker run --rm --network none`).
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
ENTRY_POINT = ROOT / "agent.py"

BANNED_MODULES = frozenset({
    "socket", "requests", "urllib", "http", "httpx",
    "ftplib", "smtplib", "asyncio", "telnetlib", "xmlrpc",
})

# `importlib.import_module("socket")` and `__import__("socket")` are import
# statements wearing a call's clothes; the AST walk above them sees nothing.
# src/optional_deps.py calls import_module with a VARIABLE, which is fine and
# is exactly why only string literals are inspected here.
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def graded_paths() -> list[Path]:
    """Every Python file on the graded path: all of `src/`, plus the entry point."""
    return sorted(SRC_DIR.rglob("*.py")) + [ENTRY_POINT]


def imported_roots(source: str) -> set[str]:
    """Top-level package name of every import, both statement forms.

    `import urllib.request` and `from urllib.request import urlopen` both reduce
    to `urllib`, so a submodule cannot slip past a top-level ban. Relative
    imports (`from . import x`) have no external root and are skipped.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative: `from .types import X`
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def dynamic_import_literals(source: str) -> set[str]:
    """Root names passed as STRING LITERALS to import_module()/__import__()."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            name = function.attr
        elif isinstance(function, ast.Name):
            name = function.id
        else:
            continue
        if name not in _DYNAMIC_IMPORTERS:
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                roots.add(argument.value.split(".")[0])
    return roots


class TestSrcNoNetwork(unittest.TestCase):
    def test_the_scan_actually_covers_something(self) -> None:
        """`Ran 0 tests ... OK` has a file-level twin: a glob that matches
        nothing passes every assertion below it. Check the count before
        believing green."""
        paths = graded_paths()
        self.assertTrue(SRC_DIR.is_dir(), f"{SRC_DIR} does not exist")
        self.assertTrue(ENTRY_POINT.is_file(), f"{ENTRY_POINT} does not exist")
        self.assertGreaterEqual(
            len(paths), 10,
            f"only {len(paths)} graded file(s) scanned -- the glob is not finding src/")

    def test_graded_path_imports_no_network_modules(self) -> None:
        offenders: list[str] = []
        for path in graded_paths():
            banned = imported_roots(path.read_text(encoding="utf-8")) & BANNED_MODULES
            if banned:
                offenders.append(f"{path.relative_to(ROOT)}: {sorted(banned)}")
        self.assertEqual([], offenders, f"network module(s) imported: {offenders}")

    def test_graded_path_does_not_dynamically_import_network_modules(self) -> None:
        offenders: list[str] = []
        for path in graded_paths():
            banned = dynamic_import_literals(path.read_text(encoding="utf-8")) & BANNED_MODULES
            if banned:
                offenders.append(f"{path.relative_to(ROOT)}: {sorted(banned)}")
        self.assertEqual([], offenders,
                         f"network module(s) imported dynamically: {offenders}")

    def test_the_detector_detects(self) -> None:
        """A scanner that finds nothing because it is broken looks identical to
        a clean tree. Prove both forms and the submodule case actually fire."""
        self.assertIn("socket", imported_roots("import socket"))
        self.assertIn("urllib", imported_roots("import urllib.request"))
        self.assertIn("urllib", imported_roots("from urllib.request import urlopen"))
        self.assertIn("http", imported_roots("from http.client import HTTPConnection"))
        self.assertEqual(set(), imported_roots("from .types import Decode"))
        self.assertIn("socket", dynamic_import_literals("importlib.import_module('socket')"))
        self.assertEqual(set(), dynamic_import_literals("import_module(name)"))


if __name__ == "__main__":
    unittest.main()
