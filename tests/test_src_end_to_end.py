"""End-to-end assembly proof for the `src/` system.

This is the only test in the suite that drives the REAL vendored evaluator over
the REAL submission entry point (`agent.py`) with every turn instrumented. Every
other src/ test proves one module in isolation; this one proves they compose.

## Why a synthetic catalog and not tests/fixtures/catalog.jsonl

The fixture has SIX products against `top_k=10`, so any query matching a single
term returns the whole fixture -- `tests/test_evaluator_smoke` passes even with a
query-blind ranker. `tests/synthetic.py` builds 250 lexically varied products
plus one planted rarity into a tempdir instead, so a ranker that ignores the
query genuinely fails here.

## Why the hit-rate floor is the load-bearing assertion

`evaluate()` swallows every exception into an empty response
(local_evaluator.py:239-244) and zeroes a schema-invalid dict just as silently.
There is no traceback, no warning, no crash -- a systematically broken agent
looks exactly like a working one except for the number. So the hit rate is the
ONLY channel through which "the assembly is broken" reaches this test. The
targets here are lexically distinctive (a synthetic title is
"{color} {material} {item}", and the disclosed constraints are lifted from the
target's own listing by the evaluator's `intent_card()` fallback), so a working
agent should find nearly all of them. The floor is set at 0.75 rather than 1.0
so the gate is not flaky.

Everything else in this module is an invariant the score alone cannot see: a
null ask, a short Top-10, a repeated recommendation and a missing `usage` block
all cost points quietly without ever failing a run.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    intent_card,
)

from agent import Agent  # the submission entry point, deliberately not src.agent
from tests import synthetic

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads((ROOT / "docs" / "agent_api_contract.json").read_text(encoding="utf-8"))
TURN_RESPONSE = CONTRACT["turn_response"]

HIT_RATE_FLOOR = 0.75
TOP_K = 10

# local_evaluator.py:171 -- the reply the simulated customer sends when we hand
# back a null ask_attribute. We never send one, so this must never appear.
NULL_NUDGE_PREFIX = "those options are not quite right yet"
# local_evaluator.py:85 -- the override message, which lands on turn 3 or 4.
OVERRIDE_PREFIX = "actually, ignore my earlier preference"


# --------------------------------------------------------------------------
# A JSON-Schema subset validator, driven by docs/agent_api_contract.json rather
# than by a transcription of it. Restating the schema in Python is how a check
# ends up self-confirming against the wrong contract.
#
# The one thing a hand-rolled checker usually drops is `additionalProperties:
# false`, which the contract applies at BOTH levels -- the response object and
# each recommendation object -- so it is enforced generically below.
# --------------------------------------------------------------------------

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def schema_errors(value: object, schema: dict, path: str = "response") -> list[str]:
    """Every way `value` violates `schema`. Empty list means valid."""
    errors: list[str] = []

    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        if not any(_TYPE_CHECKS.get(name, lambda _v: True)(value) for name in names):
            return [f"{path}: {value!r} is not of type {declared}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: {value!r} != const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} outside enum {schema['enum']}")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(f"{path}: string shorter than minLength {schema['minLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in schema:
        if value < schema["minimum"]:
            errors.append(f"{path}: {value} below minimum {schema['minimum']}")

    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items exceeds maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, item_schema, f"{path}[{index}]"))

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required property {required!r}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                errors.append(
                    f"{path}: unknown key(s) {unknown} -- additionalProperties is false")
        for name, subschema in properties.items():
            if name in value and isinstance(subschema, dict):
                errors.extend(schema_errors(value[name], subschema, f"{path}.{name}"))

    return errors


# --------------------------------------------------------------------------
# The recorder. Same shape as scripts/verify_offline_safety.py's
# InstrumentedAgent: it OBSERVES and re-raises, so a missing guard surfaces as a
# failure here instead of being masked by the instrumentation itself.
# --------------------------------------------------------------------------

class RecordingAgent(Agent):
    """Captures every reset()/respond() call and its response, in order."""

    def __init__(self, catalog_path) -> None:
        super().__init__(catalog_path)
        self.reset_order: list[str] = []
        self.records: list[dict] = []
        self.raises: list[str] = []

    def reset(self, session_id, user_profile) -> None:
        self.reset_order.append(str(session_id))
        super().reset(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        try:
            response = super().respond(session_id, user_message, turn, top_k)
        except Exception as error:
            self.raises.append(
                f"session {session_id} turn {turn}: {type(error).__name__}: {error}")
            raise
        self.records.append({
            "session_id": str(session_id),
            "turn": turn,
            "user_message": user_message if isinstance(user_message, str) else "",
            "response": response,
        })
        return response


def close_index(agent: object) -> None:
    """Release the agent's in-memory sqlite index.

    The graded path never calls this -- one index lives for the whole run -- but
    a test suite builds a dozen of them and an unclosed sqlite3.Connection is a
    ResourceWarning each time. src/retrieval.py provides close() for exactly
    this. Never raises: a cleanup failure must not mask a real test failure.
    """
    index = getattr(getattr(agent, "_deps", None), "index", None)
    closer = getattr(index, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _asins(record: dict) -> list[str]:
    response = record["response"]
    if not isinstance(response, dict):
        return []
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        return []
    return [item.get("parent_asin") for item in recommendations
            if isinstance(item, dict) and isinstance(item.get("parent_asin"), str)]


def _where(record: dict, scenario: str) -> str:
    return f"[{scenario} session, turn {record['turn']}]"


class SrcEndToEnd(unittest.TestCase):
    """One session per scenario type, driven by the real evaluator."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.catalog_path = tmp / "catalog.jsonl"
        cls.products = synthetic.build_catalog(
            cls.catalog_path, n=250, planted=[synthetic.rare_product()])
        cls.samples = synthetic.build_samples(cls.products)
        catalog_ids, categories, products = catalog_index(cls.catalog_path)

        cls.agent = RecordingAgent(cls.catalog_path)
        cls.degraded = bool(getattr(cls.agent, "degraded", False))
        cls.result = evaluate(cls.agent, cls.samples, catalog_ids, categories, products)

        # evaluate() runs sessions strictly one at a time -- reset, then every
        # respond, before the next sample -- so reset_order lines up 1:1 with
        # `samples` and gives every recorded turn its scenario type.
        scenario_of = {
            session_id: sample["scenario_type"]
            for session_id, sample in zip(cls.agent.reset_order, cls.samples)
        }
        grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
        for record in cls.agent.records:
            scenario = scenario_of.get(record["session_id"], "unknown")
            grouped.setdefault(scenario, []).append(record)
        cls.by_scenario = grouped

    @classmethod
    def tearDownClass(cls) -> None:
        close_index(cls.agent)
        cls._tmp.cleanup()

    # -- preconditions -----------------------------------------------------

    def test_catalog_built_so_the_agent_is_not_degraded(self) -> None:
        """A degraded agent scores 0.0 while raising nothing. Every assertion
        below would then fail for that one reason, so name it first."""
        self.assertFalse(
            self.degraded,
            "Agent.degraded is True against a valid 251-product synthetic catalog: "
            "the index did not build, so every other failure in this module is "
            "downstream of that one.")

    def test_respond_never_raised(self) -> None:
        self.assertEqual([], self.agent.raises,
                         "respond() raised; the evaluator would swallow this into a "
                         "silent zero in a real run")

    def test_every_session_produced_turns(self) -> None:
        self.assertEqual(
            sorted(sample["scenario_type"] for sample in self.samples),
            sorted(self.by_scenario),
            "one session per scenario type must have run")
        for scenario, records in self.by_scenario.items():
            self.assertTrue(records, f"{scenario} session recorded no turns")

    # -- 1. the score gate -------------------------------------------------

    def test_sample_count_matches(self) -> None:
        self.assertEqual(len(self.samples), self.result["sample_count"])

    def test_hit_rate_clears_the_floor(self) -> None:
        # The evaluator swallows exceptions and schema violations into silence,
        # so a low hit rate here is the ONLY way a systematically broken agent
        # shows up at all. Do not lower this floor to make the gate green.
        self.assertGreaterEqual(
            self.result["hit_rate_at_10"], HIT_RATE_FLOOR,
            f"hit@10 {self.result['hit_rate_at_10']} < {HIT_RATE_FLOOR} against a "
            f"synthetic catalog whose targets are lexically distinctive. "
            f"scenario_metrics={self.result['scenario_metrics']}")

    # -- 2. the per-turn invariants ---------------------------------------

    def test_ask_attribute_is_never_null_and_never_other(self) -> None:
        """A null ask draws the evaluator's null_nudge, which the ledger drops --
        the turn teaches nothing. `other` bypasses the constraint filter
        (local_evaluator.py:180) and is permanently declined on judging risk;
        note the evaluator silently rewrites ANY off-enum value to `other`
        (:172-173), so a typo switches the declined exploit on by accident."""
        for scenario, records in self.by_scenario.items():
            for record in records:
                attribute = record["response"].get("ask_attribute") \
                    if isinstance(record["response"], dict) else None
                self.assertIsNotNone(
                    attribute, f"{_where(record, scenario)} sent ask_attribute=None")
                self.assertNotEqual(
                    "other", attribute,
                    f"{_where(record, scenario)} sent ask_attribute='other'")

    def test_top_ten_is_never_short(self) -> None:
        """Exactly ten, on every turn -- including late turns when the fresh
        candidate pool has drained. shown.partition() returns (fresh, seen) and
        the pipeline emits `fresh + seen` truncated to k precisely so that
        re-showing a proven-wrong product keeps the list full instead of the
        list going short."""
        for scenario, records in self.by_scenario.items():
            for record in records:
                self.assertEqual(
                    TOP_K, len(_asins(record)),
                    f"{_where(record, scenario)} returned "
                    f"{len(_asins(record))} recommendations, expected {TOP_K}")

    def test_no_repeated_parent_asin_within_a_session(self) -> None:
        """The evaluator ends a session the instant our list holds the target, so
        anything still on screen in a running session is CONFIRMED wrong.

        The documented exception is the override restore: in an intent_override
        session the evaluator's hit check is switched off until the override
        lands (`override_applied` starts False, local_evaluator.py:234), so
        early picks are NOT confirmed wrong and go back in play. Turns from the
        override onward are recorded normally and must not repeat each other.

        MEASURED VACUITY: against a working agent this check barely fires. Three
        of the four sessions hit on turn 1 and have no second turn to repeat
        into, and the override session's window opens on its last turn. A
        mutation that returns turn 1's ten on every turn does NOT fail here --
        it fails SrcDrainedPoolInvariants below, which is where this property is
        really held. Keep both: this one guards the override exception, that one
        guards the rule.
        """
        for scenario, records in self.by_scenario.items():
            window = records
            if scenario == "intent_override":
                start = next((index for index, record in enumerate(records)
                              if record["user_message"].lower().startswith(OVERRIDE_PREFIX)),
                             None)
                self.assertIsNotNone(
                    start, "intent_override session never received the override message")
                window = records[start:]
            first_seen: dict[str, int] = {}
            for record in window:
                for parent_asin in _asins(record):
                    previous = first_seen.get(parent_asin)
                    self.assertIsNone(
                        previous,
                        f"{_where(record, scenario)} repeats {parent_asin}, already "
                        f"shown on turn {previous}")
                    first_seen[parent_asin] = record["turn"]

    def test_override_restores_products_shown_before_it(self) -> None:
        """restore_all(): the picks on the override turn re-include products from
        turn 1. Those were never confirmed wrong -- excluding them forfeits the
        session permanently."""
        records = self.by_scenario.get("intent_override") or []
        self.assertTrue(records, "no intent_override session was recorded")
        override_records = [record for record in records
                            if record["user_message"].lower().startswith(OVERRIDE_PREFIX)]
        self.assertTrue(
            override_records,
            "intent_override session never received the override message; "
            f"messages seen: {[record['user_message'][:60] for record in records]}")
        override_record = override_records[0]
        self.assertIn(override_record["turn"], (3, 4),
                      "behavior_for() places the override on turn 3 or 4 "
                      "(local_evaluator.py:82)")
        turn_one = set(_asins(records[0]))
        restored = turn_one & set(_asins(override_record))
        self.assertTrue(
            restored,
            f"turn {override_record['turn']} shares no product with turn 1: the "
            f"pre-override shown-set was not restored, so those sessions are "
            f"forfeited. turn1={sorted(turn_one)}")

    # -- 3. the wire contract ---------------------------------------------

    def test_every_response_validates_against_the_contract(self) -> None:
        """Read from docs/agent_api_contract.json, including
        `additionalProperties: false` at the response AND recommendation level."""
        failures: list[str] = []
        for scenario, records in self.by_scenario.items():
            for record in records:
                for error in schema_errors(record["response"], TURN_RESPONSE):
                    failures.append(f"{_where(record, scenario)} {error}")
        self.assertEqual([], failures[:20],
                         f"{len(failures)} schema violation(s); first 20 shown")

    def test_unknown_keys_are_rejected_by_the_validator_itself(self) -> None:
        """The validator has to actually enforce additionalProperties, or every
        test above it is vacuous. Two negative probes, one per level."""
        self.assertTrue(schema_errors(
            {"message": "", "ask_attribute": None, "recommendations": [], "surprise": 1},
            TURN_RESPONSE))
        self.assertTrue(schema_errors(
            {"message": "", "ask_attribute": None,
             "recommendations": [{"parent_asin": "A", "why": "because"}]},
            TURN_RESPONSE))
        self.assertEqual([], schema_errors(
            {"message": "", "ask_attribute": "color",
             "recommendations": [{"parent_asin": "A", "score": 1.0}],
             "usage": {"prompt_tokens": 0, "completion_tokens": 0}},
            TURN_RESPONSE))

    # -- 4. usage ----------------------------------------------------------

    def test_usage_present_with_non_negative_integer_counts(self) -> None:
        """The schema makes `usage` optional; this system always sends it, so a
        future LLM layer needs no new plumbing to be tracked."""
        for scenario, records in self.by_scenario.items():
            for record in records:
                response = record["response"]
                self.assertIsInstance(response, dict, _where(record, scenario))
                usage = response.get("usage")
                self.assertIsInstance(
                    usage, dict, f"{_where(record, scenario)} has no usage block")
                for name in ("prompt_tokens", "completion_tokens"):
                    value = usage.get(name)
                    self.assertIsInstance(
                        value, int, f"{_where(record, scenario)} usage.{name}={value!r}")
                    self.assertNotIsInstance(
                        value, bool, f"{_where(record, scenario)} usage.{name} is a bool")
                    self.assertGreaterEqual(
                        value, 0, f"{_where(record, scenario)} usage.{name}={value}")

    # -- 5. the free canary ------------------------------------------------

    def test_null_nudge_frame_never_fires(self) -> None:
        """We never send a null ask, so the simulated customer can never emit
        "Those options are not quite right yet" (local_evaluator.py:171).
        Seeing it means a null reached the wire on the turn before -- a free
        end-to-end check on the never-send-null rule, read off the messages the
        evaluator fed back."""
        for scenario, records in self.by_scenario.items():
            for record in records:
                self.assertFalse(
                    record["user_message"].lower().startswith(NULL_NUDGE_PREFIX),
                    f"{_where(record, scenario)} received the null_nudge reply, so "
                    f"the previous turn sent ask_attribute=None")

        # Second, independent reading of the same fact, when the session
        # aggregate is reachable. frame_counts is diagnostics-only and never
        # read by policy (src/session.py), which is exactly what makes it safe
        # to assert on.
        sessions = getattr(self.agent, "_sessions", None)
        if isinstance(sessions, dict):
            for session_id, session in sessions.items():
                counts = getattr(session, "frame_counts", None)
                if isinstance(counts, dict):
                    self.assertEqual(
                        0, counts.get("null_nudge", 0),
                        f"session {session_id} decoded a null_nudge frame")


class SrcDrainedPoolInvariants(unittest.TestCase):
    """The same per-turn invariants, on a session that actually reaches turn 10.

    A WORKING agent makes the end-to-end class above weaker, not stronger: three
    of its four sessions hit on turn 1, the evaluator breaks immediately
    (local_evaluator.py:252-255), and the invariants that only bite late -- a
    drained fresh pool, the free-turn ask ladder at turns 8-10 -- are never
    reached. Any of them could be broken with every assertion above still green.

    So this drives the same public `respond()` API for all ten turns against a
    25-product catalog: smaller than 10 turns x top_k=10, so the fresh pool
    PROVABLY drains and `fresh + seen` is the only way the list stays full. The
    customer side is the evaluator's own `initial_message()` / `customer_reply()`
    -- nothing here reimplements the scoring loop, it only declines to stop the
    conversation at the point a scored session would have stopped.
    """

    CATALOG_SIZE = 25

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        catalog_path = Path(cls._tmp.name) / "catalog.jsonl"
        products = synthetic.build_catalog(catalog_path, n=cls.CATALOG_SIZE)
        target = products[0]
        sample = {
            "sample_id": "drain_0000",
            "scenario_type": "buying",
            "intent_card": intent_card(target),
            "user_profile": synthetic.profile_for(0),
        }

        cls.agent = RecordingAgent(catalog_path)
        session_id = "drain-session"
        cls.agent.reset(session_id, sample["user_profile"])

        disclosed: set[str] = set()
        boundary_used = False
        message = initial_message(
            sample, coarse_category(target.get("categories") or []), disclosed)
        for turn in range(1, MAX_TURNS + 1):
            response = cls.agent.respond(session_id, message, turn, TOP_K)
            message, boundary_used = customer_reply(
                sample, response.get("ask_attribute") if isinstance(response, dict) else None,
                disclosed, boundary_used)
        cls.records = cls.agent.records

    @classmethod
    def tearDownClass(cls) -> None:
        close_index(cls.agent)
        cls._tmp.cleanup()

    def test_all_ten_turns_ran(self) -> None:
        self.assertEqual(MAX_TURNS, len(self.records))

    def test_the_pool_really_drained(self) -> None:
        """Evidence that the test below is not vacuous: with 25 products and ten
        distinct picks a turn, by turn 3 there is nothing fresh left, so some
        turn MUST re-show something. If this fails the catalog is too big and
        the never-short assertion never met the case it was written for."""
        seen: set[str] = set()
        repeated = False
        for record in self.records:
            picks = set(_asins(record))
            if picks & seen:
                repeated = True
            seen |= picks
        self.assertTrue(
            repeated,
            f"no turn re-showed a product across {MAX_TURNS} turns of a "
            f"{self.CATALOG_SIZE}-product catalog, so the pool never drained")

    def test_top_ten_is_never_short_even_when_drained(self) -> None:
        for record in self.records:
            self.assertEqual(
                TOP_K, len(_asins(record)),
                f"turn {record['turn']} returned {len(_asins(record))} of {TOP_K} "
                f"against a drained pool: `fresh + seen` must keep the list full, "
                f"because re-showing a proven-wrong product costs nothing and a "
                f"short list costs a rank")

    def test_the_shown_set_works_before_the_pool_drains(self) -> None:
        """Turns 1 and 2 have twenty distinct products available, so they must
        not overlap. Without this, `test_top_ten_is_never_short` would also pass
        for an agent that simply returns the same ten every turn."""
        first, second = set(_asins(self.records[0])), set(_asins(self.records[1]))
        self.assertEqual(
            set(), first & second,
            f"turns 1 and 2 overlap on {sorted(first & second)} while "
            f"{self.CATALOG_SIZE - TOP_K} unshown products remained")

    def test_ask_attribute_survives_the_free_turns(self) -> None:
        """Turns 8-10 leave the fixed seven-slot schedule and run the free-turn
        fallthrough, which is where a null or an off-enum value would first
        appear. The end-to-end class never gets here."""
        for record in self.records:
            attribute = record["response"].get("ask_attribute")
            self.assertIsNotNone(attribute, f"turn {record['turn']} sent None")
            self.assertNotEqual("other", attribute, f"turn {record['turn']} sent 'other'")

    def test_no_null_nudge_across_ten_turns(self) -> None:
        for record in self.records:
            self.assertFalse(
                record["user_message"].lower().startswith(NULL_NUDGE_PREFIX),
                f"turn {record['turn']} received the null_nudge reply")

    def test_every_turn_still_validates_against_the_contract(self) -> None:
        failures = [f"turn {record['turn']}: {error}"
                    for record in self.records
                    for error in schema_errors(record["response"], TURN_RESPONSE)]
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
