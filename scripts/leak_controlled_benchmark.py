"""Dev-only: measures how much of Phase 1's local score comes from a leak in
the vendored evaluator's customer simulator, rather than from the agent
actually working out customer intent.

## The leak

`data/public_set.jsonl` never carries a real `intent_card` / `behavior` (the
organizer's held-out per-session customer data) -- verified: 0/200 rows have
either field. `materialize_hidden_fields()` in the vendored evaluator falls
back, for every session, to `intent_card()`, which builds the simulated
customer's "hidden" preferences by lifting sentences straight out of the
*target product's own listing* -- full feature bullets, detail values, title
fragments. `customer_reply()` then recites those sentences back to the agent
turn by turn. Measured: 94% of disclosed constraint strings are exact
substrings of the target's own indexed text.

A frozen agent that only ever looks at turn 1 and ignores every later
disclosure already scores close to the organizer's own published baseline
(hit@10 0.185 vs. the official 0.125). Nearly the entire local Phase 1 gain
(hit@10 0.80, TechnicalScore ~0.69-0.72) comes from turns 2-10 -- exactly the
turns where the simulator reads out more of the answer's own spec sheet. A
BM25 agent doesn't need to understand the customer when the customer is
quoting the product page it's trying to find.

## What this script does

Runs the real, unmodified evaluator scoring loop (`evaluate()`,
`catalog_index()`, session/turn/hit/MRR/MTTC logic -- none of that changes)
twice against the same catalog and public set:

  1. **leaky**  -- the organizer's own vendored `intent_card()`, unpatched.
  2. **scrubbed** -- `intent_card()` monkeypatched (only that one function;
     `customer_reply` / `initial_message` / `behavior_for` are untouched and
     work unmodified on the new card) to disclose only atomic attribute
     values -- a material word, a color word, a short structured detail
     value (size/fit/style/use_case/department/occasion/season), a budget
     number -- through the same style of templated phrasing the vendored
     evaluator already uses elsewhere. It never reuses a multi-word span
     copied from `features`, `description`, or `title`.

  `evaluator/local_evaluator.py` itself is never edited -- README says it's
  vendored, never edited, and this script honors that: the patch is applied
  to the imported module object at runtime, in this process, and is restored
  before exit.

This is a bracket, not a replacement number. Real customers plausibly do say
"I want it in cotton" -- that's legitimate, not a leak. What they don't do is
recite "Seam sealed waterproof full grain leather and suede upper with mesh
tongue for lasting breathability" verbatim. So `leaky` is an upper bound on
what a purely keyword-matching agent can score locally, `scrubbed` is a lower
bound, and the organizer's real held-out evaluator -- presumably backed by
genuine customer profiles rather than text extracted from the answer -- should
land somewhere between them.

Also reports leak_ngram_overlap: the fraction of 3-grams in each session's
final accumulated ledger (the actual retrieval query, built by the real
`starter.ledger.SessionState`) that appear verbatim in the target product's
own indexed text. A number close to 0 means the query and the answer share
almost no phrasing; close to 1 means the query is close to a copy of the
answer.

Run via: python -m scripts.leak_controlled_benchmark
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import evaluator.local_evaluator as local_evaluator
from evaluator.local_evaluator import (
    MATERIAL_RE,
    COLOR_RE,
    _clean_constraint,
    catalog_index,
    intent_card as _vendored_intent_card,
    load_jsonl,
    searchable_text,
)
from starter.agent import Agent
from starter.retrieval import _terms

# Detail keys a real customer could plausibly name unprompted -- short,
# structured attribute values, not prose. Substring match on the lowercased
# key, deliberately excluding provenance/manufacturing keys like "Country of
# Origin", "Item model number", "Manufacturer", "Care instructions".
_NAMEABLE_DETAIL_KEYS = (
    "size", "fit", "style", "department", "sleeve", "neck",
    "occasion", "season", "use_case", "use case",
)
_MAX_DETAIL_VALUE_WORDS = 3
_MAX_DETAIL_VALUE_CHARS = 25
_NGRAM_N = 3


def intent_card_scrubbed(product: dict, limit: int = 180) -> dict:
    """Leak-controlled replacement for evaluator.local_evaluator.intent_card.

    Only atomic attribute *values* go in, never a multi-word span lifted from
    `features`, `description`, or `title`. See module docstring for why.
    """
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    corpus = searchable_text(product)
    candidates: list[str] = []

    material = MATERIAL_RE.search(corpus)
    if material:
        candidates.append(material.group(1).lower())
    color = COLOR_RE.search(corpus)
    if color:
        candidates.append(f"color: {color.group(1).lower()}")

    details = product.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            key_lower = str(key).lower()
            if not any(marker in key_lower for marker in _NAMEABLE_DETAIL_KEYS):
                continue
            value_text = str(value).strip()
            if not value_text or len(value_text) > _MAX_DETAIL_VALUE_CHARS:
                continue
            if len(value_text.split()) > _MAX_DETAIL_VALUE_WORDS:
                continue
            candidates.append(f"{key_lower}: {value_text.lower()}")

    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        # Rare fallback (no material/color/detail/price match at all): the
        # vendored version falls back to the full title, which can itself be
        # a fairly unique multi-word span. Cap it instead of reusing it whole.
        cleaned = [" ".join(title.split()[:4]) or title]

    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


class LedgerSpyAgent(Agent):
    """Same Agent, plus a record of each session's final accumulated ledger
    text -- the actual retrieval query -- for the leak measurement below.
    evaluate() runs sessions strictly one at a time (reset, then every
    respond, before the next sample), so appending on reset() and overwriting
    the last slot on every respond() keeps this aligned 1:1 with `samples`."""

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.ledger_by_order: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.ledger_by_order.append("")

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        result = super().respond(session_id, user_message, turn, top_k)
        state = self._sessions.get(str(session_id))
        if state is not None and self.ledger_by_order:
            self.ledger_by_order[-1] = state.disclosed_constraints
        return result


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_leak_ratio(ledger_text: str, target_text: str, n: int = _NGRAM_N) -> float:
    """Fraction of the ledger's own n-grams that also appear in the target's
    indexed text. 0 = no shared phrasing, 1 = ledger reads like a copy."""
    ledger_grams = _ngrams(_terms(ledger_text), n)
    if not ledger_grams:
        return 0.0
    target_grams = _ngrams(_terms(target_text), n)
    return len(ledger_grams & target_grams) / len(ledger_grams)


def run_condition(label: str, catalog_path: str, samples: list[dict], catalog_ids, categories, products) -> dict:
    agent = LedgerSpyAgent(catalog_path)
    result = local_evaluator.evaluate(agent, samples, catalog_ids, categories, products)

    leak_ratios = []
    for sample, ledger_text in zip(samples, agent.ledger_by_order):
        target = str(sample["ground_truth"]["parent_asin"])
        product = products.get(target)
        if product is None or not ledger_text:
            continue
        leak_ratios.append(ngram_leak_ratio(ledger_text, searchable_text(product)))
    mean_leak = sum(leak_ratios) / len(leak_ratios) if leak_ratios else 0.0

    return {
        "label": label,
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "recommended_technical_score": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
        "leak_ngram_overlap_mean": round(mean_leak, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results_leak_controlled.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    print(f"Running both conditions over {len(samples)} sessions, catalog={args.catalog}\n")

    leaky = run_condition("leaky (vendored intent_card, unpatched)", args.catalog, samples, catalog_ids, categories, products)

    local_evaluator.intent_card = intent_card_scrubbed
    try:
        scrubbed = run_condition("scrubbed (atomic attribute values only)", args.catalog, samples, catalog_ids, categories, products)
    finally:
        local_evaluator.intent_card = _vendored_intent_card  # restore -- never leave the vendored module patched

    def row(cond: dict) -> str:
        return (f"{cond['label']:<42} hit@10 {cond['hit_rate_at_10']:.3f}   "
                f"mrr {cond['mrr']:.3f}   mttc {cond['mttc']:.2f}   "
                f"score {cond['recommended_technical_score']:.4f}   "
                f"leak(3-gram) {cond['leak_ngram_overlap_mean']:.3f}")

    print(row(leaky))
    print(row(scrubbed))
    print()
    print(f"{'delta (leaky - scrubbed)':<42} "
          f"hit@10 {leaky['hit_rate_at_10'] - scrubbed['hit_rate_at_10']:+.3f}   "
          f"score {leaky['recommended_technical_score'] - scrubbed['recommended_technical_score']:+.4f}")
    print()
    print("Read this as a bracket, not a replacement number: leaky is an upper")
    print("bound (a keyword-matching agent gets free credit from the simulator")
    print("quoting the answer's own listing), scrubbed is a lower bound (real")
    print("customers plausibly do name atomic facts like material/color/size).")
    print("The organizer's real held-out evaluator should land between them.")

    Path(args.output).write_text(
        json.dumps({"leaky": leaky, "scrubbed": scrubbed}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
