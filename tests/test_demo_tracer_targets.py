"""Static guards for demo/. Fast, no catalog, no agent.

These exist because the demo observes ``src/pipeline.py`` by wrapping its
private stage functions. That is only safe while two things stay true:

  1. Every patch site still has the shape the recorders assume. A renamed
     helper is obvious; a REORDERED SIGNATURE is not -- ``_assemble(window,
     rest, seen, limit)`` turned into ``(seen, window, rest, limit)`` would pass
     an existence check while silently inverting every provenance label on
     screen. So the check is on parameter names IN ORDER.

  2. demo/ stays read-only over the submission: it never imports the superseded
     starter/, never reaches outside the standard library, and never flips a
     seam flag. Those are properties of the package, so they are checked here
     rather than promised in a docstring.

Registered in .github/workflows/ci.yml job 1.
"""
from __future__ import annotations

import ast
import inspect
import importlib
import unittest
from pathlib import Path

from demo import askrung, tracer

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"

# Flipping any of these is a submission-level decision with a disclosure
# attached (CLAUDE.md). A demo may never do it as a side effect.
SEAM_FLAGS = frozenset((
    "RERANK_ENABLED", "TIER2_ENABLED", "ADAPTIVE_ENABLED",
    "LLM_RERANK_ENABLED", "TIER_15_HEDGE",
))

ALLOWED_TOP_LEVEL_IMPORTS = frozenset((
    # first-party, read-only
    "demo", "src", "evaluator", "scripts", "agent",
    # standard library used across demo/
    "argparse", "ast", "contextlib", "ctypes", "dataclasses", "datetime",
    "importlib", "inspect", "json", "os", "pathlib", "platform", "random",
    "shutil", "sys", "time", "typing", "unittest", "uuid", "__future__",
))


def demo_sources():
    for path in sorted(DEMO_DIR.glob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestPatchTargets(unittest.TestCase):
    """The drift guard."""

    def test_every_target_resolves_with_the_expected_signature(self):
        problems = tracer.verify_targets()
        self.assertEqual(
            [], problems,
            "demo/tracer.py:PATCH_TARGETS has drifted from src/. The demo would "
            "render confident nonsense rather than fail. Problems:\n  "
            + "\n  ".join(str(p) for p in problems))

    def test_parameter_names_are_checked_in_order_not_as_a_set(self):
        """A reordered signature must be rejected, not merely a renamed one."""
        target = next(t for t in tracer.PATCH_TARGETS if t.name == "_assemble")
        self.assertEqual(("window", "rest", "seen", "limit"), target.params)

        module = importlib.import_module(target.module)
        found = tuple(inspect.signature(getattr(module, target.name)).parameters)
        self.assertEqual(target.params, found)
        # Same names, different order -> must be a problem.
        self.assertNotEqual(tuple(sorted(target.params)), found,
                            "the fixture is meaningless if the params are already sorted")

    def test_every_target_has_a_handler(self):
        active = tracer.Tracer()
        for target in tracer.PATCH_TARGETS:
            with self.subTest(target=target.name):
                self.assertTrue(
                    hasattr(active, "_h_" + target.handler),
                    "no handler _h_%s for %s.%s" % (target.handler, target.module,
                                                    target.name))

    def test_handler_names_are_unique(self):
        """src.overlap.gate and src.pipeline._gate must not share a handler."""
        handlers = [t.handler for t in tracer.PATCH_TARGETS]
        self.assertEqual(len(handlers), len(set(handlers)),
                         "two patch targets resolve to the same handler")

    def test_all_nineteen_stages_are_covered(self):
        stages = {t.stage for t in tracer.PATCH_TARGETS}
        for number in range(1, 20):
            self.assertIn(str(number), stages,
                          "stage %d of _run_turn has no patch target" % number)


class TestInstallRestore(unittest.TestCase):

    def test_restore_returns_every_target_to_the_original_object(self):
        originals = {}
        for target in tracer.PATCH_TARGETS:
            module = importlib.import_module(target.module)
            originals[(target.module, target.name)] = getattr(module, target.name)

        active = tracer.Tracer()
        self.assertEqual([], active.install())
        try:
            for target in tracer.PATCH_TARGETS:
                module = importlib.import_module(target.module)
                self.assertIsNot(getattr(module, target.name),
                                 originals[(target.module, target.name)],
                                 "%s was not patched" % target.name)
        finally:
            self.assertTrue(active.restore())

        for target in tracer.PATCH_TARGETS:
            module = importlib.import_module(target.module)
            self.assertIs(getattr(module, target.name),
                          originals[(target.module, target.name)],
                          "%s was not restored to the original object" % target.name)

    def test_context_manager_restores_even_when_the_body_raises(self):
        import src.pipeline as pipeline
        original = pipeline._search

        with self.assertRaises(ValueError):
            with tracer.installed():
                self.assertIsNot(pipeline._search, original)
                raise ValueError("boom")

        self.assertIs(pipeline._search, original)

    def test_wrappers_are_transparent_when_no_turn_is_recording(self):
        """With no active recorder the wrapper is a pass-through."""
        import src.pipeline as pipeline
        with tracer.installed():
            self.assertEqual("hello there", pipeline._normalise("hello   there"))
            self.assertEqual(10, pipeline.clamp_top_k(10))


class TestAskRung(unittest.TestCase):

    def test_fixed_schedule_rung_for_each_of_the_first_seven_turns(self):
        from src.types import FIXED_SCHEDULE

        for turn, expected in enumerate(FIXED_SCHEDULE, start=1):
            snapshot = {"asked": [], "retired": [], "yield_counts": {},
                        "yield_order": [], "burned": None, "burned_reasked": False,
                        "last_ask": None, "turn": turn, "disclosed_count": 0}
            rung, reason, predicted = askrung.derive(snapshot)
            with self.subTest(turn=turn):
                self.assertEqual(askrung.FIXED_SCHEDULE_RUNG, rung)
                self.assertEqual(expected, predicted)
                self.assertIn(expected, reason)

    def test_a_burned_ask_outranks_the_schedule_once_it_is_free(self):
        snapshot = {"asked": ["material"], "retired": [], "yield_counts": {},
                    "yield_order": [], "burned": "material", "burned_reasked": False,
                    "last_ask": "material", "turn": 1, "disclosed_count": 0}
        rung, _, predicted = askrung.derive(snapshot)
        self.assertEqual(askrung.PENDING_REASK, rung)
        self.assertEqual("material", predicted)

    def test_derivation_matches_the_real_policy_across_many_states(self):
        """The whole point of the label: it must agree with what ships."""
        from src.askpolicy import ASKABLE, next_attribute
        from demo.askrung import _rebuild

        cases = []
        for turn in range(1, 11):
            for asked_n in (0, 2, 7, 9):
                for retired_n in (0, 1, 5):
                    for disclosed in (0, 3, 4, 6):
                        cases.append({
                            "asked": list(ASKABLE[:asked_n]),
                            "retired": list(ASKABLE[:retired_n]),
                            "yield_counts": {a: 2 for a in ASKABLE[:asked_n]},
                            "yield_order": list(ASKABLE[:asked_n]),
                            "burned": None, "burned_reasked": False,
                            "last_ask": None, "turn": turn,
                            "disclosed_count": disclosed,
                        })

        mismatches = []
        for snapshot in cases:
            _, _, predicted = askrung.derive(snapshot)
            actual = next_attribute(_rebuild(snapshot))
            if predicted != actual:
                mismatches.append((snapshot, predicted, actual))
        self.assertEqual(
            [], mismatches,
            "demo/askrung.py disagrees with src/askpolicy.py on %d of %d states; "
            "first: %r" % (len(mismatches), len(cases), mismatches[:1]))

    def test_an_unusable_snapshot_yields_unknown_not_a_wrong_label(self):
        for bad in (None, {}, "nonsense", 17):
            rung, _, predicted = askrung.derive(bad)
            with self.subTest(bad=bad):
                if bad == {}:
                    # An empty dict is a real (if empty) state; it must not crash.
                    self.assertIsInstance(rung, str)
                else:
                    self.assertEqual(askrung.UNKNOWN, rung)
                    self.assertIsNone(predicted)


class TestPackageHygiene(unittest.TestCase):
    """demo/ is read-only over the submission. Checked, not promised."""

    def test_demo_never_imports_starter(self):
        for path, tree in demo_sources():
            for node in ast.walk(tree):
                root = ""
                if isinstance(node, ast.Import):
                    root = node.names[0].name.split(".")[0]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                if root == "starter":
                    self.fail("%s imports starter/, the superseded system" % path.name)

    def test_demo_imports_only_stdlib_and_first_party(self):
        offenders = []
        for path, tree in demo_sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    roots = [(node.module or "").split(".")[0]] if node.level == 0 else []
                else:
                    continue
                for root in roots:
                    if root and root not in ALLOWED_TOP_LEVEL_IMPORTS:
                        offenders.append("%s: %s" % (path.name, root))
        self.assertEqual([], offenders,
                         "demo/ must stay standard-library only: %s" % offenders)

    def test_demo_never_assigns_a_seam_flag(self):
        """Enabling a layer is a submission decision, never a demo side effect."""
        offenders = []
        for path, tree in demo_sources():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AugAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    name = (target.attr if isinstance(target, ast.Attribute)
                            else target.id if isinstance(target, ast.Name) else "")
                    if name in SEAM_FLAGS:
                        offenders.append("%s: %s" % (path.name, name))
        self.assertEqual([], offenders,
                         "demo/ assigned a seam flag: %s" % offenders)

    def test_demo_never_calls_setattr_on_a_src_module(self):
        """The tracer patches by design; nothing else may."""
        allowed = {"tracer.py"}
        offenders = []
        for path, tree in demo_sources():
            if path.name in allowed:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "setattr"):
                    offenders.append(path.name)
        self.assertEqual([], offenders,
                         "only demo/tracer.py may patch: %s" % offenders)


class TestRepublishedConstants(unittest.TestCase):
    """The backend never imports src/, so run_open must carry the truth."""

    def test_constants_match_the_live_values(self):
        from demo.frontend import constants
        from src.askpolicy import ASKABLE
        from src.pipeline import OVERRIDE_SUPPRESS_MAX_TURN
        from src.types import (CARD_CAPACITY, DISCLOSURE_CAP, FIXED_SCHEDULE,
                               HEDGE_ORDER, MAX_QUERY_TERMS, MAX_TURNS, POOL_SIZE,
                               RERANK_WINDOW)
        published = constants()
        self.assertEqual(int(POOL_SIZE), published["POOL_SIZE"])
        self.assertEqual(int(RERANK_WINDOW), published["RERANK_WINDOW"])
        self.assertEqual(int(MAX_TURNS), published["MAX_TURNS"])
        self.assertEqual(int(MAX_QUERY_TERMS), published["MAX_QUERY_TERMS"])
        self.assertEqual(int(DISCLOSURE_CAP), published["DISCLOSURE_CAP"])
        self.assertEqual(int(CARD_CAPACITY), published["CARD_CAPACITY"])
        self.assertEqual(int(OVERRIDE_SUPPRESS_MAX_TURN),
                         published["OVERRIDE_SUPPRESS_MAX_TURN"])
        self.assertEqual(list(FIXED_SCHEDULE), published["FIXED_SCHEDULE"])
        self.assertEqual(list(HEDGE_ORDER), published["HEDGE_ORDER"])
        self.assertEqual(list(ASKABLE), published["ASKABLE"])


class TestTraceTransport(unittest.TestCase):

    def test_reader_buffers_a_partial_trailing_line(self):
        import json
        import tempfile

        directory = tempfile.mkdtemp()
        path = Path(directory) / "partial.jsonl"
        complete = json.dumps({"v": 1, "type": "turn", "seq": 0}) + "\n"
        path.write_text(complete + '{"v": 1, "type": "tu', encoding="utf-8")

        reader = trace_reader(path)
        first = reader.read_new()
        self.assertEqual(1, len(first), "the fragment must not be parsed yet")
        self.assertEqual(0, reader.malformed, "a fragment is not a malformed line")

        with open(path, "a", encoding="utf-8") as handle:
            handle.write('rn", "seq": 1}\n')
        second = reader.read_new()
        self.assertEqual(1, len(second))
        self.assertEqual(1, second[0]["seq"])
        self.assertEqual(0, reader.malformed)

    def test_a_malformed_line_is_counted_and_skipped_not_fatal(self):
        import json
        import tempfile

        path = Path(tempfile.mkdtemp()) / "bad.jsonl"
        path.write_text(
            json.dumps({"v": 1, "type": "turn", "seq": 0}) + "\n"
            + "{not json at all\n"
            + json.dumps({"v": 1, "type": "turn", "seq": 1}) + "\n",
            encoding="utf-8")
        reader = trace_reader(path)
        records = reader.read_all()
        self.assertEqual([0, 1], [r["seq"] for r in records])
        self.assertEqual(1, reader.malformed)


def trace_reader(path):
    from demo.trace import TraceReader
    return TraceReader(path)


if __name__ == "__main__":
    unittest.main()
