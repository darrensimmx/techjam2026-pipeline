"""Proves the graded path makes zero network calls — a static check, not a
promise. See the planning repo's "Don't score zero": if the official harness
runs with networking disabled and a call reaches an external API, every
session silently scores zero.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

BANNED_MODULES = {"socket", "requests", "urllib", "http", "httpx", "ftplib", "smtplib"}
STARTER_DIR = Path(__file__).parent.parent / "starter"


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


class TestOffline(unittest.TestCase):
    def test_starter_package_imports_no_network_modules(self) -> None:
        for path in STARTER_DIR.glob("*.py"):
            modules = _imported_modules(path.read_text())
            offending = modules & BANNED_MODULES
            self.assertFalse(offending, f"{path.name} imports network module(s): {offending}")


if __name__ == "__main__":
    unittest.main()
