"""The hostile-input matrix for src/agent.py.  [WS-A OWNS]

What this file is for. The evaluator never reports a failure -- it swallows an
exception into an empty response (local_evaluator.py:239-244) and replaces a
schema-invalid dict just as silently. So the only observable difference between
"working" and "scoring zero" is the response itself. Every cell of the matrix
below therefore asserts exactly two things: no exception escaped, and the
response validates against docs/agent_api_contract.json.

The three severities, which is why construction gets its own tests:
  respond()  raising -> one turn lost, silently.
  reset()    raising -> the whole 200-session run dies (:228, unwrapped).
  __init__() raising -> the whole 200-session run dies (:306, unwrapped).

These tests deliberately do NOT depend on retrieval working. src/pipeline.py and
src/retrieval.py are being built in parallel and are inert skeletons today; an
assertion on ranking here would be an assertion about somebody else's unfinished
file. What IS asserted is the wiring: that the index object reaches Deps, that
`degraded` reports the index's real state, and that a turn reaches run_turn
rather than falling into respond()'s except clause.

data/catalog.jsonl is never read. Catalogs come from tests/synthetic.py into a
temp directory, and the no-argument construction test chdirs somewhere the
default relative path does not resolve.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path

from src.agent import Agent
from src.contract import clamp_top_k, empty_response
from src.types import FORBIDDEN_ASK, MAX_RECOMMENDATIONS
from tests.synthetic import build_catalog, profile_for, rare_product
from tests.test_src_contract import assert_valid_turn_response

# --------------------------------------------------------------------------
# The matrix axes.
# --------------------------------------------------------------------------

SESSION_IDS: tuple[object, ...] = (
    "",                     # minLength 1 in the request schema -- we get it anyway
    "s" * 10_000,           # a very long string
    None,
    123,
    [1, 2, 3],              # unhashable: str() is what saves the dict lookup
    {"a": 1},               # unhashable
)

PROFILES: tuple[object, ...] = (
    None,
    {},
    [],
    "a string, not a profile",
    profile_for(0),         # the well-formed five-field profile
)

MESSAGES: tuple[object, ...] = (
    None,
    "",
    123,
    "NEAR NOT AND OR",      # every one of these is an FTS5 MATCH operator
    'a" OR "b',             # an unbalanced quote closes and reopens a phrase
    "\x00\x01",             # control bytes
    "z" * 20_000,
)

TOP_KS: tuple[object, ...] = (-1, 0, 10, 10 ** 12, True, "10", 3.7, None)

TURNS: tuple[object, ...] = (1, 5, 10, 0, -1, 11, 10 ** 9, None, "3", 3.7, True)

MALFORMED_CATALOG = (
    b'{"parent_asin": "M1", "title": "unterminated\n'
    b'not json at all\n'
    b'{"parent_asin": }\n'
    b'[]\n'
    b'{}\n'
    b'null\n'
    b'{"parent_asin": ["not", "a", "string"], "title": {"nested": "wrong"}}\n'
    b'\xff\xfe not valid utf-8\n'
    b'\n'
)


class _CatalogFixtures(unittest.TestCase):
    """Builds the five catalogs once and shares one Agent per catalog.

    One Agent per catalog rather than one per cell: construction is the expensive
    half, `_sessions` accumulating across the sweep is exactly what happens in a
    real 200-session run, and reset() is called per cell anyway.
    """

    root: Path
    catalogs: dict[str, Path]
    agents: dict[str, Agent]

    @classmethod
    def setUpClass(cls) -> None:
        holder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(holder.cleanup)
        cls.root = Path(holder.name)

        nonexistent = cls.root / "no-such-dir" / "nope.jsonl"

        directory = cls.root / "a_directory"
        directory.mkdir()

        empty = cls.root / "empty.jsonl"
        empty.write_bytes(b"")

        malformed = cls.root / "malformed.jsonl"
        malformed.write_bytes(MALFORMED_CATALOG)

        valid = cls.root / "valid.jsonl"
        build_catalog(valid, n=250, planted=[rare_product()])

        cls.catalogs = {
            "nonexistent": nonexistent,
            "directory": directory,
            "empty file": empty,
            "malformed json": malformed,
            "valid synthetic": valid,
        }
        cls.agents = {label: Agent(str(path)) for label, path in cls.catalogs.items()}

    def assert_ok(self, response: object, top_k: object) -> None:
        """The two assertions every cell makes, plus the schema's own maxItems.

        There is deliberately no assertion that len(recommendations) <= top_k:
        truncation is src/pipeline.py's contract, not this file's. What the wire
        boundary owes is maxItems, and that the payload is a real dict.
        """
        self.assertIsInstance(response, dict)
        assert_valid_turn_response(self, response)
        self.assertLessEqual(len(response["recommendations"]), MAX_RECOMMENDATIONS)
        self.assertIsInstance(clamp_top_k(top_k), int)


class TestConstruction(_CatalogFixtures):
    """__init__ is unwrapped at local_evaluator.py:306. A raise here is the
    whole-run killer, so every catalog shape must degrade instead."""

    def test_never_raises_for_any_catalog(self) -> None:
        variants: list[tuple[str, object]] = []
        for label, path in self.catalogs.items():
            variants.append((f"{label} (str)", str(path)))
            variants.append((f"{label} (Path)", path))
        variants.extend([
            ("None", None),
            ("int", 123),
            ("bytes", b"/tmp/not-a-catalog.jsonl"),
            ("empty string", ""),
            ("a list", ["data/catalog.jsonl"]),
            ("null byte in path", "data/\x00catalog.jsonl"),
        ])
        for label, path in variants:
            with self.subTest(label):
                try:
                    agent = Agent(path)  # type: ignore[arg-type]
                except Exception as error:  # pragma: no cover - the failure we test for
                    self.fail(f"Agent({path!r}) raised {type(error).__name__}: {error}")
                self.assertIsInstance(agent.degraded, bool)

    def test_catalog_path_stays_the_first_positional_parameter(self) -> None:
        # The evaluator calls Agent(args.catalog) positionally and the submission
        # harness may call Agent(). Both are frozen; a keyword-only parameter or a
        # dropped default would break one of them at import-adjacent time.
        parameters = list(inspect.signature(Agent.__init__).parameters.values())
        self.assertEqual(parameters[0].name, "self")
        first = parameters[1]
        self.assertEqual(first.name, "catalog_path")
        self.assertEqual(first.default, "data/catalog.jsonl")
        self.assertIn(first.kind, (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD))

    def test_constructs_with_no_argument_when_the_catalog_is_absent(self) -> None:
        # chdir somewhere the default RELATIVE path cannot resolve. This is both
        # the missing-catalog case and the reason we never touch the real
        # 60 MB data/catalog.jsonl from a test.
        with tempfile.TemporaryDirectory() as empty_cwd:
            previous = os.getcwd()
            os.chdir(empty_cwd)
            try:
                self.assertFalse(Path("data/catalog.jsonl").exists())
                agent = Agent()
            finally:
                os.chdir(previous)
        self.assertTrue(agent.degraded)
        agent.reset("s1", profile_for(1))
        self.assert_ok(agent.respond("s1", "waterproof leather boots", 1, 10), 10)

    def test_constructs_with_no_argument_when_the_default_path_resolves(self) -> None:
        # Proves the default is actually used rather than ignored: same call,
        # a catalog present at the relative default, and the index gets wired.
        with tempfile.TemporaryDirectory() as cwd:
            build_catalog(Path(cwd) / "data" / "catalog.jsonl", n=40)
            previous = os.getcwd()
            os.chdir(cwd)
            try:
                agent = Agent()
            finally:
                os.chdir(previous)
        self.assertIsNotNone(agent._deps)
        self.assertIsNotNone(agent._deps.index)

    def test_accepts_catalog_path_as_a_keyword(self) -> None:
        agent = Agent(catalog_path=self.catalogs["valid synthetic"])
        self.assertIsInstance(agent.degraded, bool)


class TestDegraded(_CatalogFixtures):
    def test_nonexistent_catalog_is_degraded_and_still_answers(self) -> None:
        agent = Agent(str(self.root / "no-such-dir" / "nope.jsonl"))
        self.assertIs(agent.degraded, True)
        agent.reset("s1", profile_for(2))
        response = agent.respond("s1", "I'm looking for boots.", 1, 10)
        self.assert_ok(response, 10)
        # Non-empty message proves run_turn was reached: respond()'s except
        # clause returns empty_response(), whose message is "". This is the
        # difference between "degraded" and "silently scoring zero every turn".
        self.assertNotEqual(response["message"], "")
        self.assertNotEqual(response, empty_response())

    def test_degraded_is_a_bool_for_every_catalog(self) -> None:
        for label, agent in self.agents.items():
            with self.subTest(label):
                self.assertIsInstance(agent.degraded, bool)

    def test_degraded_mirrors_the_index_state(self) -> None:
        # The wiring assertion, deliberately phrased so it holds both before and
        # after src/retrieval.py is implemented: degraded is true exactly when
        # there is no index or the index it built holds nothing.
        for label, agent in self.agents.items():
            with self.subTest(label):
                index = agent._deps.index
                expected = True if index is None else bool(index.is_empty())
                self.assertIs(agent.degraded, expected)

    def test_a_valid_catalog_wires_an_index_into_deps(self) -> None:
        agent = self.agents["valid synthetic"]
        self.assertIsNotNone(agent._deps)
        self.assertIsNotNone(
            agent._deps.index,
            "a readable 250-product catalog must produce an index object; "
            "whether that index can rank is src/retrieval.py's test, not this one",
        )

    def test_a_failing_optional_layer_cannot_degrade_the_agent(self) -> None:
        # rerank and semantic are Layer 3. Each owns its own except clause, so a
        # loader that throws must cost its own layer and nothing else.
        import src.agent as agent_module

        original_rerank = agent_module.load_reranker
        original_semantic = agent_module.load_semantic_decoder

        def explode() -> object:
            raise RuntimeError("optional layer failed to load")

        agent_module.load_reranker = explode          # type: ignore[assignment]
        agent_module.load_semantic_decoder = explode  # type: ignore[assignment]
        try:
            agent = Agent(str(self.catalogs["valid synthetic"]))
        finally:
            agent_module.load_reranker = original_rerank
            agent_module.load_semantic_decoder = original_semantic

        self.assertIsNone(agent._deps.reranker)
        self.assertIsNone(agent._deps.semantic)
        self.assertIsNotNone(agent._deps.index)  # the index survived both failures
        agent.reset("s1", profile_for(3))
        self.assert_ok(agent.respond("s1", "leather boots", 1, 10), 10)

    def test_a_failing_index_cannot_stop_the_optional_layers(self) -> None:
        import src.agent as agent_module

        original_index = agent_module.Bm25Index

        def explode(_path: object) -> object:
            raise RuntimeError("catalog unreadable")

        agent_module.Bm25Index = explode  # type: ignore[assignment]
        try:
            agent = Agent(str(self.catalogs["valid synthetic"]))
        finally:
            agent_module.Bm25Index = original_index

        self.assertTrue(agent.degraded)
        self.assertIsNone(agent._deps.index)
        self.assertIsNotNone(agent._deps.reranker)
        self.assertIsNotNone(agent._deps.semantic)
        agent.reset("s1", profile_for(4))
        self.assert_ok(agent.respond("s1", "leather boots", 1, 10), 10)


class TestHostileMatrix(_CatalogFixtures):
    """Nothing in here may raise, and nothing may come back schema-invalid."""

    def test_catalog_x_session_id_x_profile(self) -> None:
        for catalog_label, agent in self.agents.items():
            for session_id in SESSION_IDS:
                for profile in PROFILES:
                    label = f"{catalog_label} | id={session_id!r:.40} | profile={type(profile).__name__}"
                    with self.subTest(label):
                        try:
                            self.assertIsNone(agent.reset(session_id, profile))  # type: ignore[arg-type]
                            response = agent.respond(session_id, "I need boots.", 1, 10)  # type: ignore[arg-type]
                        except Exception as error:  # pragma: no cover
                            self.fail(f"{label} raised {type(error).__name__}: {error}")
                        self.assert_ok(response, 10)

    def test_catalog_x_message_x_top_k(self) -> None:
        for catalog_label, agent in self.agents.items():
            for message in MESSAGES:
                for top_k in TOP_KS:
                    label = f"{catalog_label} | msg={message!r:.40} | top_k={top_k!r}"
                    with self.subTest(label):
                        agent.reset("sweep", profile_for(5))
                        try:
                            response = agent.respond("sweep", message, 1, top_k)  # type: ignore[arg-type]
                        except Exception as error:  # pragma: no cover
                            self.fail(f"{label} raised {type(error).__name__}: {error}")
                        self.assert_ok(response, top_k)

    def test_catalog_x_turn(self) -> None:
        for catalog_label, agent in self.agents.items():
            for turn in TURNS:
                with self.subTest(f"{catalog_label} | turn={turn!r}"):
                    agent.reset("turns", profile_for(6))
                    try:
                        response = agent.respond("turns", "leather boots", turn, 10)  # type: ignore[arg-type]
                    except Exception as error:  # pragma: no cover
                        self.fail(f"turn={turn!r} raised {type(error).__name__}: {error}")
                    self.assert_ok(response, 10)

    def test_reset_never_raises_and_always_returns_none(self) -> None:
        agent = self.agents["valid synthetic"]
        for session_id in SESSION_IDS:
            for profile in PROFILES:
                with self.subTest(f"id={session_id!r:.40} profile={type(profile).__name__}"):
                    try:
                        self.assertIsNone(agent.reset(session_id, profile))  # type: ignore[arg-type]
                    except Exception as error:  # pragma: no cover
                        self.fail(f"reset raised {type(error).__name__}: {error}")

    def test_an_object_whose_str_raises_does_not_take_anything_down(self) -> None:
        class Hostile:
            def __str__(self) -> str:
                raise RuntimeError("__str__ is not safe")

            def __repr__(self) -> str:
                return "<Hostile>"

        agent = self.agents["valid synthetic"]
        hostile = Hostile()
        self.assertIsNone(agent.reset(hostile, profile_for(7)))  # type: ignore[arg-type]
        response = agent.respond(hostile, "boots", 1, 10)  # type: ignore[arg-type]
        # respond() cannot key the session, so it takes its except clause. The
        # requirement is that it returns the empty form rather than raising.
        self.assertEqual(response, empty_response())
        self.assert_ok(response, 10)


class TestSessionLifecycle(_CatalogFixtures):
    def _agent(self) -> Agent:
        return Agent(str(self.catalogs["valid synthetic"]))

    def test_respond_without_a_prior_reset_returns_a_valid_response(self) -> None:
        agent = self._agent()
        response = agent.respond("never-reset", "I need waterproof boots.", 1, 10)
        self.assert_ok(response, 10)
        self.assertIn("never-reset", agent._sessions)

    def test_reset_replaces_rather_than_merges(self) -> None:
        agent = self._agent()
        agent.reset("shopper", profile_for(1))
        first = agent._sessions["shopper"]

        # Dirty the session the way a run of turns would. The ledger and shown
        # skeletons are inert today, so seed the containers directly too --
        # otherwise this test would pass vacuously right now and only start
        # meaning something after WS-B and WS-E land.
        first.ledger.append("verbatim reply from shopper one")
        first.ledger.record_segments(("leather",))
        first.shown.record(("A1", "A2"))
        if hasattr(first.ledger, "_entries"):
            first.ledger._entries.append("seeded reply")
        if hasattr(first.shown, "_shown"):
            first.shown._shown.add("SEEDED")
        self.assertTrue(
            len(first.ledger) or len(first.shown),
            "seeding no longer reaches the session; update this test rather "
            "than letting it assert nothing",
        )

        agent.reset("shopper", profile_for(2))
        second = agent._sessions["shopper"]

        self.assertIsNot(second, first, "reset must replace the Session, not reuse it")
        self.assertIsNot(second.ledger, first.ledger)
        self.assertIsNot(second.shown, first.shown)
        self.assertEqual(len(second.ledger), 0)
        self.assertEqual(second.ledger.query, "")
        self.assertEqual(second.ledger.entries, ())
        self.assertEqual(len(second.shown), 0)
        self.assertEqual(second.slots.filled(), ())
        self.assertEqual(second.asks.asked, [])
        self.assertEqual(second.profile, profile_for(2))

    def test_two_session_ids_do_not_share_state(self) -> None:
        agent = self._agent()
        agent.reset("a", profile_for(1))
        agent.reset("b", profile_for(2))
        first, second = agent._sessions["a"], agent._sessions["b"]
        self.assertIsNot(first, second)
        self.assertIsNot(first.ledger, second.ledger)
        self.assertIsNot(first.shown, second.shown)
        self.assertIsNot(first.slots, second.slots)
        self.assertIsNot(first.asks, second.asks)

    def test_unhashable_ids_collapse_onto_their_string_form(self) -> None:
        agent = self._agent()
        agent.reset([1, 2, 3], profile_for(1))  # type: ignore[arg-type]
        self.assertIn("[1, 2, 3]", agent._sessions)
        self.assert_ok(agent.respond([1, 2, 3], "boots", 1, 10), 10)  # type: ignore[arg-type]

    def test_a_full_ten_turn_session_stays_valid(self) -> None:
        agent = self._agent()
        agent.reset("full", profile_for(3))
        replies = [
            "I'm looking for boots. A key requirement is: leather.",
            "For that, what matters is: waterproof; ankle support.",
            "I don't have a preference for color; please use your judgment.",
            "Actually, ignore my earlier preference. What I need is: wool.",
            "I don't have an additional preference for size.",
            "Those options are not quite right yet. Ask me about one specific attribute.",
        ]
        for turn in range(1, 11):
            response = agent.respond("full", replies[(turn - 1) % len(replies)], turn, 10)
            with self.subTest(turn=turn):
                self.assert_ok(response, 10)

    def test_responses_are_fresh_dicts_not_shared_objects(self) -> None:
        agent = self._agent()
        agent.reset("fresh", profile_for(4))
        first = agent.respond("fresh", "leather boots", 1, 10)
        first["usage"]["prompt_tokens"] = 999_999
        first["recommendations"].append({"parent_asin": "MUTATED"})
        second = agent.respond("fresh", "leather boots", 2, 10)
        self.assertIsNot(first, second)
        self.assertIsNot(first["usage"], second["usage"])
        self.assertIsNot(first["recommendations"], second["recommendations"])
        self.assertNotEqual(second["usage"]["prompt_tokens"], 999_999)
        self.assert_ok(second, 10)

    def test_a_turn_reaches_the_pipeline_rather_than_the_except_clause(self) -> None:
        # empty_response()'s message is "", so a non-empty message is the one
        # cheap signal that respond() did not silently swallow an exception.
        agent = self._agent()
        agent.reset("reached", profile_for(5))
        response = agent.respond("reached", "I'm looking for leather boots.", 1, 10)
        self.assert_ok(response, 10)
        self.assertNotEqual(
            response["message"], "",
            "respond() returned the empty form for a well-formed turn, which "
            "means an exception was swallowed somewhere below it",
        )

    def test_never_sends_the_forbidden_ask(self) -> None:
        # `other` is schema-valid and permanently declined (src/types.py). The
        # trap is that any value outside ALLOWED_ATTRIBUTES is silently rewritten
        # to `other` by the evaluator, so a typo switches it on by accident.
        agent = self._agent()
        agent.reset("policy", profile_for(6))
        for turn in range(1, 11):
            response = agent.respond("policy", "I need boots.", turn, 10)
            with self.subTest(turn=turn):
                self.assertNotIn(response["ask_attribute"], FORBIDDEN_ASK)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
