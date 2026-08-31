"""WS-C: the slot state, the classifier mirror, and the override DIFF.

The classifier tests do NOT hand-copy expected labels. They import the
evaluator's own classify_constraint() and assert byte-equal agreement, so if the
vendored harness is ever re-vendored with a different branch order this fails
here instead of silently desynchronising the ask schedule.
"""
from __future__ import annotations

import ast
import random
import unittest
from pathlib import Path

from evaluator.local_evaluator import classify_constraint
from src.slots import LABELS, SlotState, apply_override, classify_local

# A table wide enough to cover every branch AND the collisions between them.
# The comments name the branch that must win; the assertions never use them --
# the evaluator is the oracle.
CLASSIFIER_TABLE: tuple[str, ...] = (
    # budget, first branch: literal token, and the three numeric forms
    "budget around $45.00",
    "budget friendly pick",
    "under 40 dollars",
    "<=25 usd",
    "$30",
    # material -- a plain substring test in the original, not word-boundary
    "leather upper",
    "100% cotton blend",
    "silky smooth finish",          # "silk" inside "silky"
    "woolen mittens",               # "wool" inside "woolen"
    "fabric type: mesh",
    "polyester shell",
    "nylon ripstop",
    "spandex waistband",
    "rayon lining",
    # color
    "color: black",
    "bright red trim",
    "pink and white stripes",
    "green",
    "greenhouse",                   # substring again
    "blue",
    # size
    "size 10 medium",
    "wide width available",
    "narrow toe box",
    "sizing runs small",
    # style
    "department: mens",
    "slim fit",
    "long sleeve shirt",
    "crew neck",
    "network cable",                # "neck" inside "network": style, not use_case
    # use_case
    "designed for hiking",
    "running shoes",
    "gym bag",
    "winter warmth",
    "outdoor use",
    "work boots",
    "workwear jacket",              # "work" inside "workwear"
    # feature -- the fallthrough
    "lightweight and durable",
    "machine washable",
    "Waterproof Membrane",
    "underwear",                    # "under" with no digit is NOT a budget
    "",
    # branch-order collisions: the earlier branch must win
    "budget around $40 for a black leather belt",   # budget beats color/material
    "black leather boots",                          # material beats color
    "wide leather belt",                            # material beats size
    "black size 10",                                # color beats size
    "size 10 for hiking",                           # size beats use_case
    "slim fit for running",                         # style beats use_case
)

# Every trigger token in the evaluator's own branches, plus filler that hits
# none of them. Used to fuzz the branch ORDER, which a hand table can miss.
_VOCAB: tuple[str, ...] = (
    "budget", "$25", "under 30", "<=15",
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
    "color", "black", "white", "blue", "red", "pink", "green",
    "size", "sizing", "width", "wide", "narrow",
    "department", "style", "fit", "sleeve", "neck",
    "hiking", "running", "gym", "winter", "outdoor", "work",
    "lightweight", "durable", "packable", "the", "with", "and", "12", "premium",
)


class TestClassifyLocalMirrorsEvaluator(unittest.TestCase):
    """classify_local() is a copy of classify_constraint(); prove the copy."""

    def test_table_agrees_with_evaluator(self) -> None:
        for value in CLASSIFIER_TABLE:
            with self.subTest(value=value):
                self.assertEqual(classify_local(value), classify_constraint(value))

    def test_table_only_ever_returns_the_seven_labels(self) -> None:
        for value in CLASSIFIER_TABLE:
            with self.subTest(value=value):
                label = classify_local(value)
                self.assertIn(label, LABELS)
                self.assertNotEqual(label, "other")

    def test_fuzzed_phrases_agree_with_evaluator(self) -> None:
        rng = random.Random(20260831)
        for _ in range(2000):
            phrase = " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(1, 6)))
            if rng.random() < 0.3:
                phrase = phrase.upper()
            with self.subTest(phrase=phrase):
                self.assertEqual(classify_local(phrase), classify_constraint(phrase))

    def test_never_raises_on_non_strings(self) -> None:
        for value in (None, 12, 4.5, [], {}, object(), b"leather"):
            with self.subTest(value=repr(value)):
                self.assertEqual(classify_local(value), "feature")  # type: ignore[arg-type]


class TestSlotStateFill(unittest.TestCase):
    """fill() returns True only on a genuine contradiction."""

    def setUp(self) -> None:
        self.slots = SlotState()

    def test_first_fill_is_never_a_contradiction(self) -> None:
        self.assertFalse(self.slots.fill("color", "blue"))
        self.assertEqual(self.slots.get("color"), "blue")

    def test_identical_value_is_not_a_contradiction(self) -> None:
        self.slots.fill("color", "blue")
        self.assertFalse(self.slots.fill("color", "blue"))

    def test_case_variant_is_not_a_contradiction(self) -> None:
        self.slots.fill("color", "blue")
        self.assertFalse(self.slots.fill("color", "BLUE"))
        self.assertFalse(self.slots.fill("color", "Blue"))

    def test_whitespace_variant_is_not_a_contradiction(self) -> None:
        self.slots.fill("color", "blue")
        self.assertFalse(self.slots.fill("color", "   blue "))
        self.assertFalse(self.slots.fill("color", "\tblue\n"))

    def test_different_value_is_a_contradiction(self) -> None:
        self.slots.fill("color", "blue")
        self.assertTrue(self.slots.fill("color", "red"))

    def test_contradiction_keeps_the_newest_value(self) -> None:
        self.slots.fill("color", "blue")
        self.slots.fill("color", "red")
        self.assertEqual(self.slots.get("color"), "red")

    def test_a_different_attribute_never_collides(self) -> None:
        self.slots.fill("color", "blue")
        self.assertFalse(self.slots.fill("material", "leather"))
        self.assertEqual(self.slots.as_dict(), {"color": "blue", "material": "leather"})

    def test_empty_value_is_a_no_op(self) -> None:
        self.slots.fill("color", "blue")
        self.assertFalse(self.slots.fill("color", "   "))
        self.assertEqual(self.slots.get("color"), "blue")

    def test_bad_types_never_raise(self) -> None:
        for attribute, value in ((None, "blue"), ("color", None), (3, 4), ([], {}), ("", "")):
            with self.subTest(pair=(repr(attribute), repr(value))):
                self.assertFalse(self.slots.fill(attribute, value))  # type: ignore[arg-type]
        self.assertIsNone(self.slots.get(None))                      # type: ignore[arg-type]
        self.slots.clear(None)                                       # type: ignore[arg-type]

    def test_clear_removes_one_slot_only(self) -> None:
        self.slots.fill("color", "blue")
        self.slots.fill("material", "leather")
        self.slots.clear("color")
        self.assertIsNone(self.slots.get("color"))
        self.assertEqual(self.slots.get("material"), "leather")
        self.slots.clear("color")  # idempotent
        self.assertEqual(self.slots.filled(), ("material",))

    def test_as_dict_is_a_copy(self) -> None:
        self.slots.fill("color", "blue")
        snapshot = self.slots.as_dict()
        snapshot["color"] = "tampered"
        self.assertEqual(self.slots.get("color"), "blue")


class TestApplyOverride(unittest.TestCase):
    """G5: classify -> DIFF -> clear exactly one slot."""

    def setUp(self) -> None:
        self.slots = SlotState()
        self.slots.fill("material", "leather")
        self.slots.fill("color", "black")
        self.slots.fill("use_case", "hiking")

    def test_clears_only_the_contradicting_slot_and_names_it(self) -> None:
        self.assertEqual(apply_override(self.slots, "nylon ripstop shell"), "material")
        self.assertIsNone(self.slots.get("material"))
        self.assertEqual(self.slots.get("color"), "black")
        self.assertEqual(self.slots.get("use_case"), "hiking")

    def test_identical_value_clears_nothing_and_returns_none(self) -> None:
        self.assertIsNone(apply_override(self.slots, "leather"))
        self.assertEqual(self.slots.as_dict(),
                         {"material": "leather", "color": "black", "use_case": "hiking"})

    def test_case_and_whitespace_variants_clear_nothing(self) -> None:
        for value in ("LEATHER", "  leather  ", "Leather"):
            with self.subTest(value=value):
                self.assertIsNone(apply_override(self.slots, value))
                self.assertEqual(self.slots.get("material"), "leather")

    def test_unfilled_class_returns_none_and_touches_nothing(self) -> None:
        before = self.slots.as_dict()
        self.assertIsNone(apply_override(self.slots, "budget around $60"))
        self.assertEqual(self.slots.as_dict(), before)

    def test_returned_name_is_the_classified_label(self) -> None:
        self.assertEqual(apply_override(self.slots, "a bright red panel"), "color")
        self.assertEqual(classify_local("a bright red panel"), "color")
        self.assertIsNone(self.slots.get("color"))

    def test_bad_input_never_raises(self) -> None:
        self.assertIsNone(apply_override(None, "nylon"))          # type: ignore[arg-type]
        self.assertIsNone(apply_override("not a slotstate", "nylon"))  # type: ignore[arg-type]
        self.assertIsNone(apply_override(self.slots, None))       # type: ignore[arg-type]
        self.assertIsNone(apply_override(self.slots, "   "))
        self.assertIsNone(apply_override(self.slots, 42))         # type: ignore[arg-type]
        self.assertEqual(len(self.slots.filled()), 3)


class TestSlotsLayering(unittest.TestCase):
    """The safety property, asserted locally as well as by the shared AST test.

    Slot state decides what to ASK. It must never be able to reach retrieval,
    so a parsing bug can corrupt scheduling and nothing else.
    """

    def test_module_imports_stdlib_only(self) -> None:
        source = Path("src/slots.py").read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        # Checked on the parse tree, not the text: the module's own docstring
        # names src.retrieval and src.overlap while explaining why it must not
        # import them, and a substring check cannot tell prose from an import.
        self.assertTrue(imported <= {"re", "__future__"}, f"unexpected imports: {imported}")
        self.assertNotIn("src", imported)


if __name__ == "__main__":
    unittest.main()
