"""End-to-end proof that the demo observes the agent without changing it.

The load-bearing test is :class:`TracingDoesNotChangeBehaviour`. The demo works
by wrapping ``src/pipeline.py``'s private stage functions, and the one way that
can lie is if a recorder raises: ``run_turn``'s outer except (pipeline.py:105)
would swallow it into ``_degraded_plan()`` and the demo would silently render a
DIFFERENT agent than the one being scored. So we run the same sessions with and
without the tracer installed and demand byte-identical output.

The second load-bearing test is :class:`DriverMatchesTheRealEvaluator`. The demo
copies the vendored drive loop because ``evaluate()`` has no per-turn seam; a
copy can drift, and a drifted copy shows a conversation the scorer would never
produce. Both loops are deterministic, so equality is checkable.

Everything runs against ``tests/synthetic.py``'s generated catalog -- no
``data/catalog.jsonl`` needed, so this is CI-safe.

Registered in .github/workflows/ci.yml job 2.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import synthetic

from demo import driver, trace, tracer

ROOT = Path(__file__).resolve().parent.parent


def close_index(agent) -> None:
    """Release the sqlite connection so temp dirs can be removed on Windows."""
    index = getattr(getattr(agent, "_deps", None), "index", None)
    try:
        index.close()
    except Exception:
        pass


def drive(agent, samples, catalog_ids, categories, products, on_turn=None):
    """Run every sample and collect the per-turn observable output."""
    transcript = []
    results = []
    for sample in samples:
        turns = []

        def collect(payload, sink=turns):
            sink.append((payload["turn"],
                         tuple(payload["ranked"]),
                         payload["response"].get("ask_attribute"),
                         payload["response"].get("message")))
            if on_turn is not None:
                on_turn(payload)

        results.append(driver.run_session(
            agent, sample, catalog_ids, categories, products,
            on_turn=collect, session_id="fixed_" + sample["sample_id"]))
        transcript.append((sample["sample_id"], tuple(turns)))
    return transcript, results


class DemoFixture(unittest.TestCase):
    """A synthetic catalog and one sample per scenario."""

    CATALOG_SIZE = 250

    @classmethod
    def setUpClass(cls) -> None:
        from evaluator.local_evaluator import catalog_index

        cls._tmp = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls._tmp.name) / "catalog.jsonl"
        cls.products_list = synthetic.build_catalog(
            cls.catalog_path, n=cls.CATALOG_SIZE,
            planted=[synthetic.rare_product()])
        cls.samples = synthetic.build_samples(cls.products_list)
        cls.catalog_ids, cls.categories, cls.products = catalog_index(cls.catalog_path)

        cls.dataset_path = Path(cls._tmp.name) / "samples.jsonl"
        with cls.dataset_path.open("w", encoding="utf-8") as handle:
            for sample in cls.samples:
                handle.write(json.dumps(sample) + "\n")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


class TracingDoesNotChangeBehaviour(DemoFixture):
    """The proof the whole approach rests on."""

    def test_traced_and_untraced_runs_are_identical(self):
        from agent import Agent

        agent = Agent(str(self.catalog_path))
        self.addCleanup(close_index, agent)
        self.assertFalse(agent.degraded, "synthetic catalog failed to index")

        clean, clean_results = drive(
            agent, self.samples, self.catalog_ids, self.categories, self.products)

        active = tracer.Tracer()
        self.assertEqual([], active.install())
        try:
            traced, traced_results = drive(
                agent, self.samples, self.catalog_ids, self.categories, self.products)
        finally:
            self.assertTrue(active.restore(), "patches were not fully restored")

        self.assertEqual(
            clean, traced,
            "installing the tracer changed the agent's output -- a recorder "
            "probably raised inside a stage and run_turn swallowed it into "
            "_degraded_plan()")
        self.assertEqual(
            [(r["hit"], r["first_hit_turn"], r["best_rank"]) for r in clean_results],
            [(r["hit"], r["first_hit_turn"], r["best_rank"]) for r in traced_results])

    def test_a_recorder_that_always_raises_cannot_change_the_output(self):
        """The guard, exercised rather than asserted."""
        from agent import Agent

        agent = Agent(str(self.catalog_path))
        self.addCleanup(close_index, agent)
        clean, _ = drive(agent, self.samples, self.catalog_ids,
                         self.categories, self.products)

        class Sabotaged(tracer.Tracer):
            pass

        def explode(self, rec, args, result, before, pre):
            raise RuntimeError("recorder is broken")

        for target in tracer.PATCH_TARGETS:
            setattr(Sabotaged, "_h_" + target.handler, explode)

        active = Sabotaged()
        self.assertEqual([], active.install())
        try:
            broken, _ = drive(agent, self.samples, self.catalog_ids,
                              self.categories, self.products)
        finally:
            self.assertTrue(active.restore())

        self.assertEqual(clean, broken,
                         "a failing recorder changed the agent's answers")
        self.assertFalse(active.degraded_plan_fired,
                         "_degraded_plan fired -- the recorder escaped its guard")


class DriverMatchesTheRealEvaluator(DemoFixture):

    def test_same_hit_turn_and_rank_as_the_vendored_loop(self):
        from agent import Agent
        from evaluator.local_evaluator import evaluate

        agent = Agent(str(self.catalog_path))
        self.addCleanup(close_index, agent)

        reference = evaluate(agent, self.samples, self.catalog_ids,
                             self.categories, self.products)
        expected = {r["sample_id"]: (r["hit"], r["first_hit_turn"], r["best_rank"])
                    for r in reference["sessions"]}

        _, results = drive(agent, self.samples, self.catalog_ids,
                           self.categories, self.products)
        actual = {r["sample_id"]: (r["hit"], r["first_hit_turn"], r["best_rank"])
                  for r in results}

        self.assertEqual(
            expected, actual,
            "demo/driver.py has drifted from evaluator/local_evaluator.py:226-276 "
            "-- the demo would show a conversation the scorer never produces")


class TraceStream(DemoFixture):
    """Schema and coverage of what the frontend writes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.records = run_frontend(cls, trace.LEAKY)
        cls.turns = [r for r in cls.records if r["type"] == trace.TURN]

    def test_every_record_is_well_formed_and_seq_is_monotonic(self):
        previous = -1
        for record in self.records:
            self.assertEqual(trace.SCHEMA_VERSION, record["v"])
            self.assertIn(record["type"], trace.RECORD_TYPES)
            for field in ("ts", "run_id", "seq"):
                self.assertIn(field, record)
            self.assertGreater(record["seq"], previous, "seq must strictly increase")
            previous = record["seq"]

    def test_exactly_one_run_open_and_one_run_close(self):
        for kind in (trace.RUN_OPEN, trace.RUN_CLOSE):
            self.assertEqual(1, sum(1 for r in self.records if r["type"] == kind), kind)
        opens = [r for r in self.records if r["type"] == trace.SESSION_OPEN]
        closes = [r for r in self.records if r["type"] == trace.SESSION_CLOSE]
        self.assertEqual(len(self.samples), len(opens))
        self.assertEqual(len(opens), len(closes))

    def test_the_run_restored_everything_and_recorded_cleanly(self):
        close = next(r for r in self.records if r["type"] == trace.RUN_CLOSE)
        self.assertTrue(close["patch_restore_ok"])
        self.assertEqual(0, close["tracer_record_errors"])
        self.assertFalse(close["degraded_plan_fired"],
                         "_degraded_plan fired: the trace is not the scored agent")

    def test_all_nineteen_stages_are_observed_on_every_turn(self):
        """A stage that stops being observed fails here, not silently on screen."""
        checks = {
            "1 normalise": lambda t: t["input"].get("normalised") is not None,
            "2 clamp_turn": lambda t: t["input"].get("turn_clamped") is not None,
            "3 tier1": lambda t: t["decode"].get("tier1") is not None,
            "4 tier2": lambda t: t["decode"].get("tier2_ran") is not None,
            "5 note": lambda t: t["state"].get("frame_counts") is not None,
            "6 override_guard": lambda t: t["state"].get("override_guard"),
            "7 ledger": lambda t: t["state"].get("ledger"),
            "8 slots": lambda t: t["state"].get("slots"),
            "9 ask_bookkeeping": lambda t: t["state"].get("ask_bookkeeping"),
            "10 query": lambda t: t["retrieval"].get("query") is not None,
            "11 search": lambda t: t["retrieval"].get("pool_size") is not None,
            "12 partition": lambda t: t["partition"].get("is_true_partition") is not None,
            "13 hydrate": lambda t: t["window"].get("hydrated_count") is not None,
            "14 rerank": lambda t: t["window"].get("rerank_name") is not None,
            "15 gate": lambda t: t["window"].get("gate_changed_order") is not None,
            "16 assemble": lambda t: t["picks"].get("provenance") is not None,
            "17 record": lambda t: t["picks"].get("shown_after") is not None,
            "18 choose": lambda t: t["ask"].get("final"),
            "18a policy": lambda t: t["ask"].get("policy_return"),
            "19 message": lambda t: t["ask"].get("message"),
        }
        for turn in self.turns:
            for name, check in checks.items():
                with self.subTest(sample=turn["sample_id"], turn=turn["turn"], stage=name):
                    self.assertTrue(check(turn), "stage %s was not observed" % name)

    def test_the_window_split_cross_check_holds_on_every_turn(self):
        for turn in self.turns:
            with self.subTest(sample=turn["sample_id"], turn=turn["turn"]):
                self.assertTrue(
                    turn["window"]["split_consistent"],
                    "the derived fresh[:RERANK_WINDOW] split disagrees with what "
                    "_hydrate and _assemble were handed")

    def test_the_derived_ask_rung_agrees_on_every_turn(self):
        for turn in self.turns:
            ask = turn["ask"]
            with self.subTest(sample=turn["sample_id"], turn=turn["turn"]):
                self.assertTrue(
                    ask["rung_agrees"],
                    "demo/askrung.py predicted %r but the policy returned %r "
                    "(rung %s)" % (ask.get("rung_predicted_attribute"),
                                   ask.get("policy_return"), ask.get("rung")))

    def test_picks_carry_provenance_and_a_score_for_every_entry(self):
        for turn in self.turns:
            picks = turn["picks"]
            self.assertEqual(len(picks["parent_asins"]), len(picks["provenance"]))
            self.assertEqual(len(picks["parent_asins"]), len(picks["scores"]))
            for i, source in enumerate(picks["provenance"]):
                with self.subTest(sample=turn["sample_id"], turn=turn["turn"], i=i):
                    self.assertNotEqual("?", source, "a pick had no known origin")

    def test_the_target_is_always_serialised_even_beyond_the_pool_cap(self):
        """Without the must-include set the ground-truth row renders with no score."""
        for turn in self.turns:
            outcome = turn["outcome"]
            pool = turn["retrieval"]["pool"]
            if not turn["retrieval"]["pool_size"]:
                continue
            ids = {row["parent_asin"] for row in pool}
            if outcome["target_pool_rank"] is not None:
                with self.subTest(sample=turn["sample_id"], turn=turn["turn"]):
                    self.assertIn(outcome["target_parent_asin"], ids)
            for asin in turn["picks"]["parent_asins"]:
                self.assertIn(asin, ids, "a pick was dropped from the serialised pool")


class DrainedPoolTurns(unittest.TestCase):
    """Cover the free-turn ladder, which only fires past turn 7.

    A scored session stops the moment the target is found, and against a
    synthetic catalog that is almost always turn 1-4 -- so no scored session
    ever reaches the ladder. This drives all ten turns REGARDLESS of hits,
    exactly as ``tests/test_src_end_to_end.py:462-490`` does and for the same
    reason: it is not scoring anything, it is exercising a code path.

    The 25-product catalog is smaller than 10 turns x top_k=10, so the fresh
    pool provably drains too.
    """

    CATALOG_SIZE = 25

    def test_rung_and_split_hold_through_turn_ten(self):
        from evaluator.local_evaluator import (MAX_TURNS, TOP_K, coarse_category,
                                               customer_reply, initial_message,
                                               intent_card)

        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.jsonl"
            products = synthetic.build_catalog(catalog_path, n=self.CATALOG_SIZE)
            target = products[0]
            sample = {
                "sample_id": "drain_0000",
                "scenario_type": "buying",
                "intent_card": intent_card(target),
                "user_profile": synthetic.profile_for(0),
            }

            active = tracer.Tracer()
            self.assertEqual([], active.install())
            try:
                agent = tracer.make_tracing_agent(str(catalog_path), active)
                self.addCleanup(close_index, agent)
                agent.reset("drain-session", sample["user_profile"])

                disclosed: set = set()
                boundary_used = False
                message = initial_message(
                    sample, coarse_category(target.get("categories") or []), disclosed)
                for turn in range(1, MAX_TURNS + 1):
                    response = agent.respond("drain-session", message, turn, TOP_K)
                    message, boundary_used = customer_reply(
                        sample, response.get("ask_attribute"), disclosed, boundary_used)
                records = [r.finalise("", 25) for r in agent.turns]
            finally:
                self.assertTrue(active.restore())

        self.assertEqual(MAX_TURNS, len(records), "not every turn was recorded")

        schedule_length = 7
        deep = [(i + 1, r) for i, r in enumerate(records) if i + 1 > schedule_length]
        self.assertTrue(deep, "no free turns were reached")

        for number, record in records_with_numbers(records):
            with self.subTest(turn=number):
                self.assertTrue(record["ask"]["rung_agrees"],
                                "derived %r != policy %r at turn %d"
                                % (record["ask"]["rung_predicted_attribute"],
                                   record["ask"]["policy_return"], number))
                self.assertTrue(record["window"]["split_consistent"])
                self.assertEqual(0, record["tracer"]["record_errors"])

        rungs = {record["ask"]["rung"] for _, record in deep}
        self.assertTrue(
            rungs - {"1-fixed-schedule"},
            "every free turn still reported the fixed schedule: %s" % rungs)

    def test_the_pool_really_drained(self):
        """If the pool never drains, the free-turn assertions prove nothing."""
        from evaluator.local_evaluator import (MAX_TURNS, TOP_K, coarse_category,
                                               customer_reply, initial_message,
                                               intent_card)

        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.jsonl"
            products = synthetic.build_catalog(catalog_path, n=self.CATALOG_SIZE)
            target = products[0]
            sample = {"sample_id": "drain_0001", "scenario_type": "buying",
                      "intent_card": intent_card(target),
                      "user_profile": synthetic.profile_for(0)}

            active = tracer.Tracer()
            self.assertEqual([], active.install())
            try:
                agent = tracer.make_tracing_agent(str(catalog_path), active)
                self.addCleanup(close_index, agent)
                agent.reset("drain2", sample["user_profile"])
                disclosed: set = set()
                boundary_used = False
                message = initial_message(
                    sample, coarse_category(target.get("categories") or []), disclosed)
                for turn in range(1, MAX_TURNS + 1):
                    response = agent.respond("drain2", message, turn, TOP_K)
                    message, boundary_used = customer_reply(
                        sample, response.get("ask_attribute"), disclosed, boundary_used)
                records = [r.finalise("", 25) for r in agent.turns]
            finally:
                self.assertTrue(active.restore())

        last = records[-1]
        self.assertGreater(last["partition"]["seen_count"], 0,
                           "nothing was ever marked seen -- the pool did not drain")
        # The invariant that matters: the list is still full even though the
        # fresh pool is exhausted.
        self.assertEqual(10, len(last["picks"]["parent_asins"]),
                         "the picks went short on a drained pool")


def records_with_numbers(records):
    return [(i + 1, r) for i, r in enumerate(records)]


class BracketLabelling(DemoFixture):
    """A number on a filmed screen must never appear without its bracket."""

    def test_card_source_records_the_arm_that_actually_ran(self):
        for name in (trace.LEAKY, trace.SCRUBBED):
            records = run_frontend(self, name)
            opens = [r for r in records if r["type"] == trace.SESSION_OPEN]
            self.assertTrue(opens)
            for record in opens:
                with self.subTest(bracket=name):
                    self.assertEqual(trace.CARD_SOURCE[name],
                                     record["hidden_card"]["source"])

    def test_the_evaluator_patch_is_always_restored(self):
        from evaluator import local_evaluator

        before = local_evaluator.intent_card
        run_frontend(self, trace.SCRUBBED)
        self.assertIs(before, local_evaluator.intent_card,
                      "bracket() left evaluator/ monkeypatched")

    def test_no_rendered_score_line_omits_its_bracket(self):
        records = run_frontend(self, trace.LEAKY)
        rendered = render_backend(records)
        offenders = []
        for line in rendered.splitlines():
            lowered = line.lower()
            if "score" in lowered and any(ch.isdigit() for ch in line):
                if "[leaky]" not in lowered and "[scrubbed]" not in lowered:
                    offenders.append(line.strip())
        self.assertEqual([], offenders,
                         "a score line carried no bracket tag: %s" % offenders[:3])

    def test_ground_truth_is_never_marked_visible_to_the_agent(self):
        records = run_frontend(self, trace.LEAKY)
        for record in records:
            if record["type"] == trace.SESSION_OPEN:
                self.assertFalse(record["ground_truth"]["visible_to_agent"])
                self.assertFalse(record["hidden_card"]["visible_to_agent"])


class CasesMode(DemoFixture):
    """The default demo unit is one session per scenario, not one long session."""

    def test_cases_mode_runs_one_session_per_scenario_in_order(self):
        records = run_frontend(self, trace.LEAKY, cases=True)
        opens = [r for r in records if r["type"] == trace.SESSION_OPEN]

        from demo.frontend import CASES
        expected = [scenario for scenario, _, _ in CASES]
        self.assertEqual(expected, [r["scenario_type"] for r in opens],
                         "cases did not run one per scenario, in CASES order")

        for index, record in enumerate(opens, start=1):
            with self.subTest(case=index):
                self.assertEqual(index, record["case_index"])
                self.assertEqual(len(expected), record["case_total"])
                self.assertTrue(record["case_note"], "a case carried no description")

    def test_cases_is_the_default_but_explicit_selection_wins(self):
        from demo import frontend

        default_args = frontend.parse_args(["--bracket", "leaky"])
        self.assertTrue(frontend.wants_cases(default_args))

        for override in (["--sample-id", "x"], ["--scenario", "buying"],
                         ["--sessions", "1"], ["--no-cases"]):
            args = frontend.parse_args(["--bracket", "leaky"] + override)
            with self.subTest(override=override):
                self.assertFalse(frontend.wants_cases(args),
                                 "%s should turn cases mode off" % override)

    def test_a_missing_curated_id_falls_back_to_its_scenario(self):
        """A changed public_set.jsonl must degrade, not crash."""
        from demo.frontend import CASES, choose_cases

        # Same scenarios, none of the curated ids present.
        samples = [{"sample_id": "other_%d" % i, "scenario_type": scenario}
                   for i, (scenario, _, _) in enumerate(CASES)]
        chosen = choose_cases(samples)
        self.assertEqual([c[0] for c in CASES],
                         [s["scenario_type"] for s, _ in chosen])

    def test_no_known_scenario_at_all_exits_cleanly(self):
        from demo.frontend import choose_cases

        with self.assertRaises(SystemExit):
            choose_cases([{"sample_id": "z", "scenario_type": "not_a_scenario"}])


class TopCut(unittest.TestCase):
    """--top trims the list, but never hides the target."""

    def _render(self, ranked, target, top):
        import io
        from contextlib import redirect_stdout

        from demo import ansi, frontend

        ansi.configure(no_color=True)
        payload = {
            "turn": 1, "user_message": "hello", "target": target,
            "response": {"message": "hi", "ask_attribute": "material"},
            "ranked": ranked, "target_rank": (ranked.index(target) + 1
                                              if target in ranked else None),
            "hit_counted": target in ranked, "hit_suppressed_by_override": False,
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            frontend.render_turn(100, payload, {}, trace.LEAKY, top)
        return buffer.getvalue()

    def test_target_below_the_cut_is_still_printed(self):
        ranked = ["A%d" % i for i in range(10)]
        out = self._render(ranked, "A9", top=5)
        self.assertIn("A9", out, "the target was cut from its own demo")
        self.assertIn("TARGET", out)
        self.assertIn("...", out, "no elision marker before the out-of-order row")
        self.assertIn("10", out, "the target's real rank was not shown")

    def test_the_cut_actually_trims(self):
        ranked = ["A%d" % i for i in range(10)]
        out = self._render(ranked, "A0", top=3)
        self.assertIn("A2", out)
        self.assertNotIn("A7", out, "--top did not trim")

    def test_top_ten_shows_everything(self):
        ranked = ["A%d" % i for i in range(10)]
        out = self._render(ranked, "A0", top=10)
        for asin in ranked:
            self.assertIn(asin, out)
        self.assertNotIn("...", out)


class PacingIsInertWhenPiped(DemoFixture):
    """Not cosmetic: without this the demo tests would take minutes."""

    def test_configure_refuses_to_pace_a_non_tty(self):
        import io

        from demo import pacing

        self.assertFalse(pacing.configure(0.5, stream=io.StringIO()))
        self.assertFalse(pacing.enabled())

    def test_zero_delay_disables_pacing_even_on_a_tty(self):
        import io

        from demo import pacing

        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        self.assertTrue(pacing.configure(0.01, stream=FakeTTY()))
        self.assertFalse(pacing.configure(0, stream=FakeTTY()))

    def test_say_splits_multi_line_blocks(self):
        """A straight print -> say swap must pace pre-joined blocks correctly."""
        import io
        from contextlib import redirect_stdout

        from demo import pacing

        pacing.configure(0, stream=io.StringIO())
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pacing.say("a\nb\nc")
        self.assertEqual(["a", "b", "c"], buffer.getvalue().splitlines())

    def test_a_full_captured_frontend_run_stays_fast(self):
        """The guard that keeps CI honest, measured rather than assumed."""
        import time as clock

        started = clock.perf_counter()
        run_frontend(self, trace.LEAKY)
        elapsed = clock.perf_counter() - started
        self.assertLess(
            elapsed, 10.0,
            "a captured run took %.1fs -- pacing is not being disabled under "
            "redirected stdout, and CI will crawl" % elapsed)


class BackendIndependence(DemoFixture):
    """The renderer must not need src/ -- that is why run_open republishes."""

    def test_renders_with_src_unimportable(self):
        records = run_frontend(self, trace.LEAKY)
        path = write_trace(self, records)

        script = (
            "import builtins, sys\n"
            "real = builtins.__import__\n"
            "def blocked(name, *a, **k):\n"
            "    if name == 'src' or name.startswith('src.'):\n"
            "        raise ImportError('src blocked')\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = blocked\n"
            "from demo import backend\n"
            "raise SystemExit(backend.main(['--replay', sys.argv[1], '--no-color',\n"
            "                               '--width', '100', '--speed', '1000']))\n"
        )
        # encoding: without it the PARENT decodes with the locale encoding too,
        # so the box-drawing frame the child now emits as UTF-8 would come back
        # as mojibake. The ASCII assertions below would still pass, which is
        # the wrong reason to be green.
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
        self.assertEqual(0, result.returncode, result.stderr[-2000:])
        for block in ("INPUT & DECODE", "QUERY & POOL", "PICKS", "GROUND TRUTH"):
            self.assertIn(block, result.stdout)


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------

def run_frontend(case, bracket: str, cases: bool = False) -> list:
    """Run the real frontend over the fixture and return the parsed trace.

    Plain ``assert`` rather than ``case.assertEqual``: this is called from
    ``setUpClass`` too, where ``case`` is the class and the assertion helpers
    are unbound.
    """
    import io
    from contextlib import redirect_stdout

    from demo import frontend

    run_dir = Path(tempfile.mkdtemp())
    argv = [
        "--bracket", bracket,
        "--catalog", str(case.catalog_path),
        "--dataset", str(case.dataset_path),
        "--delay", "0",
        "--line-delay", "0",
        "--top", "10",
        "--no-color",
        "--run-dir", str(run_dir),
    ]
    # --sessions is an explicit selection and turns cases mode off, so the two
    # are mutually exclusive here on purpose.
    argv += ["--cases"] if cases else ["--sessions", str(len(case.samples))]

    # The frontend renders a full transcript; swallow it so the test output
    # stays readable. The trace file is what we are checking, not the screen.
    with redirect_stdout(io.StringIO()):
        code = frontend.main(argv)
    assert code == 0, "frontend exited %s" % code

    path = trace.discover_run(run_dir)
    assert path is not None, "frontend wrote no trace"
    reader = trace.TraceReader(path)
    records = reader.read_all()
    assert reader.malformed == 0, "the frontend wrote malformed JSON"
    return records


def write_trace(case, records: list) -> Path:
    path = Path(tempfile.mkdtemp()) / "trace.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def render_backend(records: list) -> str:
    """Render in-process, capturing stdout."""
    import io
    from contextlib import redirect_stdout

    from demo import ansi, backend

    ansi.configure(no_color=True)
    renderer = backend.Renderer(100)
    args = backend.parse_args(["--no-color", "--width", "100"])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        for record in records:
            backend.dispatch(renderer, record, args)
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
