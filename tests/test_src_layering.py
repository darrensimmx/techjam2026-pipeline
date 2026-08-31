"""The architecture, as assertions instead of prose.

Every rule here is stated somewhere in a `src/` docstring. A docstring is a
marker of intent, not a control -- it cannot fail a build. These are the same
rules expressed as things that break when violated.

They are checked with `ast` rather than string matching wherever the property is
structural, so a mention inside a comment or a docstring (there are several,
deliberately) does not read as a violation and a real one cannot hide behind
formatting.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from src.askpolicy import AskState
from src.askpolicy import next_attribute as askpolicy_next_attribute
from src.askyield import next_attribute as askyield_next_attribute
from src.types import ALLOWED_ATTRIBUTES, FORBIDDEN_ASK

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
ENTRY_POINT = ROOT / "agent.py"
PACKAGE = "src"

# Any name that would let a caller un-say something the customer said.
FORBIDDEN_LEDGER_METHODS = re.compile(r"clear|remove|pop|reset|delete|replace", re.IGNORECASE)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def src_dependencies(path: Path) -> set[str]:
    """The `src` submodules this file imports, as bare names ({"types", ...}).

    Covers the four spellings that all mean the same edge:
        import src.slots            from src.slots import X
        from src import slots       from .slots import X
    """
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE + "."):
                    found.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Every file scanned here sits directly in src/, so a single
                # leading dot resolves to the package itself.
                module = f"{PACKAGE}.{node.module}" if node.module else PACKAGE
            else:
                module = node.module or ""
            if module == PACKAGE:
                found.update(alias.name for alias in node.names)
            elif module.startswith(PACKAGE + "."):
                found.add(module.split(".")[1])
    return found


def imports_starter(path: Path) -> set[str]:
    """Names imported from the superseded `starter` package, if any."""
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names
                         if alias.name == "starter" or alias.name.startswith("starter."))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module == "starter" or node.module.startswith("starter."):
                found.add(node.module)
    return found


def function_names(path: Path) -> list[str]:
    return [node.name for node in ast.walk(_tree(path))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def string_constants(path: Path) -> list[str]:
    return [node.value for node in ast.walk(_tree(path))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def graded_paths() -> list[Path]:
    return sorted(SRC_DIR.rglob("*.py")) + [ENTRY_POINT]


class TestSrcLayering(unittest.TestCase):

    def setUp(self) -> None:
        self.assertTrue(SRC_DIR.is_dir(), f"{SRC_DIR} is missing")

    def _require(self, name: str) -> Path:
        path = SRC_DIR / name
        self.assertTrue(path.is_file(), f"{path} is missing -- this check is vacuous without it")
        return path

    # -- slot state must never reach a search query ------------------------

    def test_retrieval_and_overlap_do_not_import_slots_or_session(self) -> None:
        """THE safety property of the whole system.

        Slot state is scheduling-only. If it cannot reach retrieval, a parsing
        bug can corrupt WHAT WE ASK NEXT but can never corrupt WHAT WE SEARCH --
        which is the concrete mechanism behind structured slot parsing measuring
        +0.000000 while the raw verbatim ledger moves the number. The edge is
        what makes that property checkable rather than aspirational.
        """
        for module in ("retrieval.py", "overlap.py"):
            path = self._require(module)
            forbidden = src_dependencies(path) & {"slots", "session"}
            self.assertEqual(
                set(), forbidden,
                f"src/{module} imports {sorted(forbidden)}: slot state must never "
                f"be reachable from the retrieval path")

    # -- the ledger is append-only ----------------------------------------

    def test_ledger_defines_no_erasing_api(self) -> None:
        """"Never erase, not even on intent override" is enforced by the ABSENCE
        of an API, not by a comment.

        The override's `old_value` and `new_value` are both manufactured from
        the SAME target listing (local_evaluator.py:79-86), and `old_value` is
        never added to the evaluator's `disclosed` set -- so the abandoned
        preference still describes the target and can still come back in a later
        disclosure. Erasing it would throw away terms that describe the answer.
        Only the SLOT is cleared on an override; src/slots.py owns that.
        """
        path = self._require("ledger.py")
        offenders = [name for name in function_names(path)
                     if FORBIDDEN_LEDGER_METHODS.search(name)]
        self.assertEqual(
            [], offenders,
            f"src/ledger.py defines {offenders}: the ledger is append-only and must "
            f"expose no way to un-say a disclosure")

    def test_the_forbidden_method_pattern_actually_matches(self) -> None:
        for name in ("clear", "remove_entry", "pop", "reset", "delete_all",
                     "replace", "_clear_query"):
            self.assertTrue(FORBIDDEN_LEDGER_METHODS.search(name), name)
        for name in ("append", "record_segments", "query", "entries", "segments",
                     "distinct_segment_count", "__len__", "__init__"):
            self.assertFalse(FORBIDDEN_LEDGER_METHODS.search(name), name)

    # -- `other` never reaches the wire ------------------------------------

    def test_askpolicy_contains_no_bare_other_literal(self) -> None:
        """`other` bypasses the evaluator's constraint filter entirely
        (`attribute == "other" or ...`, local_evaluator.py:180) and hands back
        any two undisclosed constraints. It is the highest-scoring option on the
        board and is DECLINED, permanently, on judging risk.

        It is absent from src/askpolicy.py by CONSTRUCTION -- ASKABLE is built
        from FIXED_SCHEDULE + HEDGE_ORDER and `_is_valid_ask` subtracts
        FORBIDDEN_ASK -- so a bare string literal is how it would come back.
        Mentions inside comments and docstrings are invisible to this check,
        which is the point of doing it on the AST.
        """
        path = self._require("askpolicy.py")
        offenders = [value for value in string_constants(path) if value == "other"]
        self.assertEqual(
            [], offenders,
            "src/askpolicy.py contains a bare \"other\" string literal; the declined "
            "exploit must stay absent by construction, not by a filter")

    def test_next_attribute_never_returns_none_or_other(self) -> None:
        """The behavioural half of the same rule, over both entry points.

        askyield.next_attribute() is what the pipeline actually calls; it
        delegates to askpolicy today and is the single swap point for Layer 2,
        so it has to hold the same guarantee.
        """
        sendable = ALLOWED_ATTRIBUTES - FORBIDDEN_ASK
        states = [AskState()]
        for turn in range(0, 13):
            states.append(AskState(turn=turn))
            states.append(AskState(turn=turn, retired=set(ALLOWED_ATTRIBUTES)))
            states.append(AskState(turn=turn, asked=list(ALLOWED_ATTRIBUTES),
                                   retired=set(ALLOWED_ATTRIBUTES), disclosed_count=4))
            states.append(AskState(turn=turn, burned="material", disclosed_count=1))
        entry_points = (("askpolicy", askpolicy_next_attribute),
                        ("askyield", askyield_next_attribute))
        for state in states:
            for label, entry_point in entry_points:
                choice = entry_point(state)
                self.assertIsInstance(choice, str, f"{label} returned {choice!r}")
                self.assertIn(choice, sendable,
                              f"{label} returned {choice!r} for turn={state.turn}")

    # -- clean room: no edge to the superseded system ----------------------

    def test_src_imports_nothing_from_starter(self) -> None:
        """`src/` is a clean-room rebuild. `starter/` is the historical record and
        stays untouched -- but untouched only stays true while nothing depends on
        it. Note `src/slots.py` deliberately COPIES the evaluator's
        classify_constraint() rather than importing it, because importing
        evaluator.local_evaluator would drag `from starter.agent import Agent`
        onto the graded path (local_evaluator.py:12).
        """
        paths = graded_paths()
        self.assertGreaterEqual(len(paths), 10,
                                f"only {len(paths)} file(s) scanned -- the glob is broken")
        offenders = [f"{path.relative_to(ROOT)}: {sorted(imports_starter(path))}"
                     for path in paths if imports_starter(path)]
        self.assertEqual([], offenders,
                         f"src/ depends on the superseded starter/: {offenders}")

    # -- the frame decode is self-contained --------------------------------

    def test_frames_imports_nothing_from_src_except_types(self) -> None:
        """Tier 1 is a pure decode of eight f-strings the simulator emits. It
        reads a message and returns a Decode -- no ledger, no slots, no session,
        no retrieval. Keeping the dependency set at {types} is what makes it
        testable in isolation and swappable without touching anything else."""
        path = self._require("frames.py")
        dependencies = src_dependencies(path)
        self.assertEqual(
            set(), dependencies - {"types"},
            f"src/frames.py imports {sorted(dependencies - {'types'})} from src; "
            f"only `types` is allowed")

    def test_the_dependency_reader_reads(self) -> None:
        """All four spellings of the same edge, so the check above cannot pass by
        simply failing to see anything."""
        import tempfile

        cases = {
            "import src.slots": {"slots"},
            "from src.slots import SlotState": {"slots"},
            "from src import slots": {"slots"},
            "from .slots import SlotState": {"slots"},
            "from src.types import Candidate": {"types"},
            "import json": set(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.py"
            for source, expected in cases.items():
                probe.write_text(source, encoding="utf-8")
                self.assertEqual(expected, src_dependencies(probe), source)
            probe.write_text("from starter.agent import Agent", encoding="utf-8")
            self.assertEqual({"starter.agent"}, imports_starter(probe))
            probe.write_text("import starter", encoding="utf-8")
            self.assertEqual({"starter"}, imports_starter(probe))


if __name__ == "__main__":
    unittest.main()
