"""Does the LLM escalation SURVIVE the real multi-turn pipeline, and what does it cost?

Arm B. The companion to `followup_llmrr_esci.py`, and it answers a different
question -- do not read one for the other.

    ESCI (Arm A)   600 human queries, single turn. Does the layer RECOVER
                   relevance that BM25 + CE lose on vague wording?
    here (Arm B)   30 stratified public sessions, up to 10 turns each. Does the
                   layer INTEGRATE -- survive the ledger, the shown-set, the
                   override guard and the overlap gate -- and what does a turn
                   cost against an unpublished organizer timeout?

WHY THE STRATA COME FROM HERE AND NOT FROM ESCI
    `scenario_type` -- buying / browsing / boundary / intent_override -- is a
    field on `data/public_set.jsonl` and a set of branches inside the
    organizer's simulated customer. ESCI has none of it:
    `esci_public_schema.py` writes the constant `"buying"` for all 600 samples
    and says why -- "ESCI has no dialogue, so browsing / boundary /
    intent_override have no counterpart and none is invented." A stratified
    ESCI sample on those four labels would be fabricated.

    Sampled 12 / 12 / 4 / 2 against a population of 80 / 80 / 30 / 10, i.e.
    40 / 40 / 15 / 5 percent. NOTE `difficulty_bucket` carries ZERO independent
    information -- it is a perfect function of `scenario_type` across all 200
    sessions (buying->easy, browsing->medium, boundary->medium,
    intent_override->hard), so stratifying on scenario already stratifies on
    difficulty and reporting both would be reporting one split twice.

WHAT THIS CAN AND CANNOT CONCLUDE. READ THIS BEFORE QUOTING A NUMBER.
    It CANNOT conclude that the layer helps. The vendored simulator has no real
    `intent_card`, so it builds the customer's hidden preferences out of the
    TARGET PRODUCT'S OWN LISTING: 94.5% of disclosed constraint strings are
    verbatim substrings of the target's indexed text. The low-overlap branch
    this layer exists for is therefore ~5.5% of the local set BY CONSTRUCTION.
    Expect a delta indistinguishable from zero under `--bracket leaky`
    regardless of how good the reranker is. That is a property of the
    simulator, not evidence about the model. Never quote a number without its
    bracket.

    It CAN conclude that the layer integrates: that the permutation contract
    survives two independent guards, that the fallback fires on an API failure,
    how often the overlap gate overrides the model, and what a turn really
    costs in wall clock and tokens.

WHY THIS RUNS LIVE RATHER THAN CAPTURE-AND-REPLAY
    `bakeoff/capture.py` can replay arms off one capture because `starter/` has
    no shown-set. `src/` does: `_record(session, picks)` fills it every turn and
    `_partition` reads it, so a reranker that changes WHICH TEN are shown
    changes the NEXT turn's candidate window. A capture is exact only for the
    arm it was captured under. Running live costs one pass per arm and is
    exact; the response cache in `followup_llmrr_esci.py` makes a repeat free.

NOTHING HERE SHIPS. `src/` is not modified: the reranker is injected by
rebuilding the frozen `Deps` on an already-constructed `Agent`.
`LLM_RERANK_ENABLED` stays False and `MODEL_BUILDERS` stays empty. Enabling the
layer is a submission-level decision with a disclosure attached, not a line edit.

    # offline, free -- the fidelity check and the control
    python bakeoff\\followup_llmrr_publicset.py --arms none --bracket both
    # paid
    python bakeoff\\followup_llmrr_publicset.py --arms llm --encodings indices --bracket leaky
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from scripts.evaluate_src import bracket                                   # noqa: E402
from bakeoff.llmrr_contract import (                                       # noqa: E402
    ENCODINGS, LISTING_CHARS, RETURN_K, apply_indices, cache_key,
    check_indices, check_permutation, phrase_overlap,
)
from bakeoff.followup_llmrr_esci import (                                  # noqa: E402
    ARMS, DISPATCH, append_response, load_responses, set_deadline,
)

QUOTA = {"buying": 12, "browsing": 12, "intent_override": 4, "boundary": 2}
SEED = 20260901

# The per-call deadline and the per-session budget. Neither exists in `src/`:
# `safe_rerank` and `_rerank` have NO time budget, and `load_reranker` has none
# either -- it used to take a `timeout_s` that `_load_checkpoint` accepted and
# ignored, and that parameter was deleted on 1 Sep 2026 rather than left to
# imply a budget it never applied. A sync SDK call cannot be interrupted from
# the calling thread, so the client timeout IS the mechanism -- and
# `max_retries=0` is the load-bearing half, because the SDK default of 2 means
# a worst case of 3x the deadline on a single turn.
CALL_DEADLINE_S = 6.0
SESSION_BUDGET_S = 20.0


def stratified(samples: list[dict], quota: dict, seed: int) -> list[dict]:
    """Seeded, and the chosen sample_ids go in the artifact -- so the sample
    survives a later change to this function."""
    buckets = collections.defaultdict(list)
    for row in samples:
        buckets[row["scenario_type"]].append(row)
    rng = random.Random(seed)
    picked = []
    for scenario in sorted(quota):
        pool = sorted(buckets[scenario], key=lambda r: r["sample_id"])
        want = quota[scenario]
        if len(pool) < want:
            raise SystemExit(f"only {len(pool)} {scenario} sessions, need {want}")
        picked += rng.sample(pool, want)
    return sorted(picked, key=lambda r: r["sample_id"])


class InstrumentedReranker:
    """Sits in `deps.reranker` and IS the measurement.

    It receives exactly what the shipped seam receives -- the ledger query and
    the hydrated top-RERANK_WINDOW fresh window -- and must return an exact
    permutation, which `safe_rerank` and `_same_multiset_or_original` then check
    independently. Returning `list(candidates)` is behaviourally identical to
    `NullReranker`, which is what makes the `none` arm a true control.
    """

    name = "instrumented"

    def __init__(self, mode, arm=None, encoding=None, cache=None, ce=None):
        self.mode = mode                # none | ce | llm
        self.arm = arm
        self.encoding = encoding
        self.cache = cache if cache is not None else {}
        self.ce = ce
        self.turns: list[dict] = []
        self.session = None
        self.session_spent = 0.0
        self.usage = {"input": 0, "output": 0, "cache_read": 0}
        self.contract_failures: dict[str, int] = {}
        self.call_failures: dict[str, int] = {}
        self.latencies: list[float] = []
        self.ce_latencies: list[float] = []
        self.budget_exhausted: set = set()
        self.escalations = 0
        self.gate_overrides = 0

    def new_session(self, session_id):
        self.session = session_id
        self.session_spent = 0.0

    # -- the seam --------------------------------------------------------
    def rerank(self, query, candidates):
        window = list(candidates)
        if not window:
            return window
        record = {"session": self.session, "query": query,
                  "window": len(window),
                  "incoming": [c.parent_asin for c in window[:RETURN_K]]}

        # The gate signal, computed here rather than smuggled out of the
        # session: `rerank(query, candidates)` is handed no session and no
        # segments, and widening that frozen signature to get them would be a
        # change to the shipped contract. The query IS the ledger
        # (`pipeline._query_for`), so phrase overlap between query and listings
        # is a faithful proxy -- and it is the SAME function Arm A bands on,
        # which is the only reason the two arms' "vague" means one thing.
        record["overlap"] = round(
            phrase_overlap(query, [c.text for c in window[:RETURN_K]]), 4)

        order = window
        if self.mode in ("ce", "llm") and self.ce is not None:
            started = time.time()
            order = self.ce(query, order)
            self.ce_latencies.append(time.time() - started)
        if self.mode == "llm":
            order = self._escalate(query, order)

        record["outgoing"] = [c.parent_asin for c in order[:RETURN_K]]
        self.turns.append(record)
        return order

    def _escalate(self, query, window):
        if self.session_spent >= SESSION_BUDGET_S:
            self.budget_exhausted.add(self.session)
            return window
        head = [c.parent_asin for c in window]
        candidates = [(c.parent_asin, c.text[:LISTING_CHARS]) for c in window]
        prompt = self.encoding["build_prompt"](query, candidates)
        key = cache_key(self.arm["model"], self.encoding_name,
                        self.encoding["instruction"], prompt)
        row = self.cache.get(key)
        try:
            if row is not None:
                payload, secs, usage = row["payload"], None, row["usage"]
            else:
                payload, secs, usage = DISPATCH[self.arm["provider"]](
                    self.arm, self.encoding, prompt)
                append_response({"key": key, "arm": self.arm["model"],
                                 "model": self.arm["model"],
                                 "encoding": self.encoding_name, "depth": len(head),
                                 "query": query, "payload": payload,
                                 "latency_s": secs, "usage": usage,
                                 "ts": time.time()})
                self.cache[key] = {"payload": payload, "usage": usage}
            if secs is not None:
                self.latencies.append(secs)
                self.session_spent += secs
            for k in self.usage:
                self.usage[k] += usage.get(k, 0)
            self.escalations += 1

            ranking = payload.get("ranking")
            if self.encoding_name == "indices":
                reason = check_indices(ranking, len(head))
                ordered_ids = None if reason else apply_indices(ranking, head)
            else:
                reason = check_permutation(ranking, head)
                ordered_ids = None if reason else list(ranking)
            if reason:
                self.contract_failures[reason] = self.contract_failures.get(reason, 0) + 1
                return window
            by_id = {c.parent_asin: c for c in window}
            return [by_id[a] for a in ordered_ids]
        except Exception as exc:                       # noqa: BLE001
            kind = type(exc).__name__
            self.call_failures[kind] = self.call_failures.get(kind, 0) + 1
            return window                              # the shipped fallback


class InstrumentedAgent:
    """Forwards to the real `Agent`, and tells the reranker when a session starts.

    The per-session budget needs a session boundary, and the frozen `Reranker`
    signature carries none. Wrapping `reset` is how it gets one without touching
    `src/`. `respond` is passed straight through, so the evaluator drives the
    genuine agent -- nothing about the graded path is simulated here.
    """

    def __init__(self, agent, recorder):
        self._agent = agent
        self._recorder = recorder

    def reset(self, session_id, user_profile):
        self._recorder.new_session(session_id)
        return self._agent.reset(session_id, user_profile)

    def respond(self, session_id, message, turn, top_k):
        before = len(self._recorder.turns)
        response = self._agent.respond(session_id, message, turn, top_k)
        # Pair what the reranker HANDED BACK with what actually reached the
        # wire. They are not the same list: `src/overlap.py::gate` runs AFTER
        # `_rerank` (pipeline.py:140) as a stable sort on (-overlap,
        # incoming_index), so it can reorder on top of the reranker -- and does.
        # The reranker cannot see this from inside its own call, which is why
        # the comparison has to happen out here.
        if len(self._recorder.turns) > before:
            shipped = [r.get("parent_asin") for r
                       in (response.get("recommendations") or [])]
            self._recorder.turns[-1]["shipped"] = shipped[:RETURN_K]
        return response


def load_cross_encoder():
    """The real bundled-style CE, run from `.venv`. `src/rerank.py` stays inert
    throughout -- this runs in bakeoff/, never on the graded path."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                         max_length=256, device="cpu")

    def score(query, window):
        pairs = [[query, c.text[:LISTING_CHARS]] for c in window]
        values = model.predict(pairs, batch_size=64, show_progress_bar=False)
        return [c for _, c in sorted(zip(values, window),
                                     key=lambda p: -float(p[0]))]
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--arms", default="none",
                        help="comma-separated: none (control) | ce | llm")
    parser.add_argument("--model", default="haiku-4.5", choices=sorted(ARMS))
    parser.add_argument("--encodings", default="indices",
                        help="comma-separated: " + ", ".join(ENCODINGS))
    parser.add_argument("--bracket", choices=("leaky", "scrubbed", "both"),
                        default="both")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="bakeoff/results-llmrr-publicset.json")
    args = parser.parse_args()

    from agent import Agent           # the repo-root entry point -- src/, not starter/

    everything = load_jsonl(args.dataset)
    scale = args.sample / sum(QUOTA.values())
    quota = {k: max(1, round(v * scale)) for k, v in QUOTA.items()} \
        if args.sample != sum(QUOTA.values()) else dict(QUOTA)
    samples = stratified(everything, quota, args.seed)
    catalog_ids, categories, products = catalog_index(args.catalog)

    print("sample: %d sessions %s" % (len(samples), dict(
        collections.Counter(s["scenario_type"] for s in samples))))
    print("difficulty (a perfect function of scenario -- not a second axis): %s"
          % dict(collections.Counter(s["difficulty_bucket"] for s in samples)))

    ce = None
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if any(a in ("ce", "llm") for a in arms):
        print("loading cross-encoder (needs .venv: torch + sentence-transformers)")
        ce = load_cross_encoder()

    set_deadline(CALL_DEADLINE_S)
    cache = load_responses()
    brackets = ("leaky", "scrubbed") if args.bracket == "both" else (args.bracket,)
    report = {"sample_ids": [s["sample_id"] for s in samples],
              "seed": args.seed, "quota": quota,
              "scenario_counts": dict(collections.Counter(
                  s["scenario_type"] for s in samples)),
              "model": ARMS[args.model]["model"],
              "call_deadline_s": CALL_DEADLINE_S,
              "session_budget_s": SESSION_BUDGET_S,
              "runs": {}}

    for arm in arms:
        encodings = [None] if arm != "llm" else [
            e.strip() for e in args.encodings.split(",") if e.strip()]
        for encoding_name in encodings:
            for bracket_name in brackets:
                label = "%s%s@%s" % (arm, "/" + encoding_name if encoding_name else "",
                                     bracket_name)
                print("\n=== %s ===" % label, flush=True)
                recorder = InstrumentedReranker(
                    mode=arm, arm=ARMS[args.model], cache=cache, ce=ce,
                    encoding=ENCODINGS[encoding_name] if encoding_name else None)
                recorder.encoding_name = encoding_name

                agent = Agent(args.catalog)
                if getattr(agent, "degraded", False):
                    raise SystemExit("agent.degraded -- data/catalog.jsonl is missing")
                agent._deps = dataclasses.replace(agent._deps, reranker=recorder)

                started = time.time()
                with bracket(bracket_name):
                    outcome = evaluate(InstrumentedAgent(agent, recorder),
                                       samples, catalog_ids, categories, products)
                elapsed = time.time() - started

                # Two different counts, and conflating them is the easy mistake.
                #   moved   the reranker changed the top pick it was handed
                #   gated   the overlap gate then changed it AGAIN before the
                #           wire. A layer whose ordering the gate routinely
                #           erases is not the layer the architecture draws.
                moved = sum(1 for t in recorder.turns
                            if t["outgoing"][:1] != t["incoming"][:1])
                gated = sum(1 for t in recorder.turns
                            if t.get("shipped") is not None
                            and t["shipped"][:1] != t["outgoing"][:1])
                gate_seen = sum(1 for t in recorder.turns if t.get("shipped") is not None)
                ordered = sorted(recorder.latencies)
                ce_ordered = sorted(recorder.ce_latencies)
                report["runs"][label] = {
                    "technical_score": outcome["recommended_technical_score"],
                    "hit_rate_at_10": outcome["hit_rate_at_10"],
                    "mrr": outcome["mrr"], "mttc": outcome["mttc"],
                    "efficiency": outcome["efficiency"],
                    "scenario_metrics": outcome["scenario_metrics"],
                    "reported_token_usage": outcome["reported_token_usage"],
                    "turns_reranked": len(recorder.turns),
                    "escalations": recorder.escalations,
                    "reranker_moved_top": moved,
                    "gate_overrode_reranker": gated,
                    "turns_with_shipped_list": gate_seen,
                    "overlap_p50": (sorted(t["overlap"] for t in recorder.turns)
                                    [len(recorder.turns) // 2]
                                    if recorder.turns else None),
                    "llm_latency_s": _pcts(ordered),
                    "ce_latency_s": _pcts(ce_ordered),
                    "contract_failures": recorder.contract_failures,
                    "call_failures": recorder.call_failures,
                    "llm_usage": dict(recorder.usage),
                    "budget_exhausted_sessions": len(recorder.budget_exhausted),
                    "wall_clock_s": round(elapsed, 1),
                }
                r = report["runs"][label]
                print("  score %.6f  hit@10 %.4f  mrr %.6f  mttc %.3f"
                      % (r["technical_score"], r["hit_rate_at_10"],
                         r["mrr"], r["mttc"]))
                print("  turns reranked %d | reranker moved the top pick on %d"
                      % (r["turns_reranked"], moved))
                print("  overlap gate then overrode it on %d of %d turns "
                      "that reached the wire" % (gated, gate_seen))
                if recorder.escalations:
                    print("  escalations %d | contract %s | call %s"
                          % (recorder.escalations, recorder.contract_failures
                             or "{}", recorder.call_failures or "{}"))
                    print("  tokens %s" % dict(recorder.usage))

    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote " + str(out))
    print("NOT comparable to results_src.md's 200-session references -- this is "
          "%d sessions. Do not append a row there." % len(samples))
    print("Under --bracket leaky a ~0 delta is the PREDICTED result: the "
          "simulator copies the target's listing, so the low-overlap branch "
          "this layer gates on is ~5.5% of the set by construction.")


def _pcts(ordered: list[float]) -> dict:
    if not ordered:
        return {"p50": None, "p95": None, "max": None, "n": 0}
    return {"p50": round(ordered[len(ordered) // 2], 3),
            "p95": round(ordered[min(int(len(ordered) * 0.95),
                                     len(ordered) - 1)], 3),
            "max": round(max(ordered), 3), "n": len(ordered)}


if __name__ == "__main__":
    main()
