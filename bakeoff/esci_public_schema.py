"""Emit the ESCI real-query set in `data/public_set.jsonl`'s schema.

So the real-language evaluation drops into the same tooling as the organizer's
public set -- `load_jsonl`, `catalog_index`, and `evaluate()` itself -- instead
of needing a parallel code path.

The trap this is built around
-----------------------------
`local_evaluator.py:204-206`:

    if "intent_card" in sample and "behavior" in sample:
        return sample["intent_card"], sample["behavior"]

Those two keys are **absent** from the organizer's `public_set.jsonl`, which is
exactly why local scores are inflated: with them missing the evaluator falls
through to `intent_card(product)` (:52-71) and builds the simulated customer's
"hidden" preferences out of *the target product's own listing*, then recites
them back turn by turn.

Emit ESCI in the public set's literal shape and that fallback fires: the
evaluator would throw the human query away and regenerate a copy-pasted one from
the target's listing -- deleting the entire reason this dataset exists. So these
two keys are written deliberately, and they are the one intentional departure
from a byte-identical schema. Every other key matches `public_set.jsonl` exactly.

What each field is, and where it comes from
-------------------------------------------
| field | source |
|---|---|
| `sample_id` | synthesised, `esci_0001`-style, stable under the seed |
| `ground_truth.parent_asin` | ESCI, first Exact-labelled product for the query (sorted) |
| `scenario_type` | **`buying`** -- the only honest mapping. An ESCI query states a requirement up front, which is what `buying` means in `initial_message:156`. ESCI has no dialogue, so `browsing` / `boundary` / `intent_override` have no counterpart and none is invented. |
| `intent_card.hard_constraints` | ESCI, the human query verbatim -- this is the payload |
| `intent_card.soft_preferences` | `[]`. ESCI gives one query per session, not a disclosure ladder. Turns 2+ therefore return *"I don't have an additional preference for X"*, so a session is a single human query with no accumulation -- i.e. exactly the turn-1 condition, which is where dense retrieval was predicted to help most. |
| `category_bucket` / `difficulty_bucket` | `"unknown"`. Organizer metadata with no ESCI counterpart. Guessing them would be fabrication, and `category_bucket` is a constant `"clothing"` across all 200 public samples anyway. |
| `user_profile` | structurally valid, explicitly unsourced. ESCI has no user model; `preference_tags` is empty rather than invented, and `summary` says so in words so it cannot be quoted as if it were real. |

One distortion, stated rather than hidden: `initial_message` wraps every opener
in `f"I'm looking for {category}. A key requirement is: {constraint}."`, and
ESCI carries no category, so `coarse_category([])` resolves to the constant
`"clothing item"`. Every ESCI sample therefore gains the same handful of frame
tokens. It is constant across samples, so it shifts all arms equally and cannot
flip a BM25-vs-dense ordering -- but the retrieval-only path in
`part5_realqueries.py` has no such frame, and remains the primary measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "bakeoff" / "cache"
OUT_DIR = ROOT / "evaluation-data" / "esci"

UNSOURCED_PROFILE = {
    "average_prior_rating": None,
    "preference_tags": [],
    "purchase_frequency": "unknown",
    "rating_style": "unknown",
    "summary": "Not sourced from ESCI; this dataset carries no user model. "
               "Do not quote this profile as evidence of anything.",
}


def main() -> None:
    blob = json.loads((CACHE / "esci_queries.json").read_text(encoding="utf-8"))
    entries = blob["queries"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUT_DIR / "esci_public_set.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for position, entry in enumerate(entries, 1):
            query = entry["query"]
            handle.write(json.dumps({
                "category_bucket": "unknown",
                "difficulty_bucket": "unknown",
                "ground_truth": {"parent_asin": entry["targets"][0]},
                "sample_id": f"esci_{position:04d}",
                "scenario_type": "buying",
                "user_profile": dict(UNSOURCED_PROFILE),
                # Not in the organizer's public_set.jsonl. Deliberate -- see the
                # module docstring. Without these two the evaluator rebuilds the
                # customer's hidden preferences from the target's own listing and
                # the human query is silently discarded.
                "intent_card": {
                    "target_category": "unknown",
                    "hard_constraints": [query],
                    "soft_preferences": [],
                },
                "behavior": {"scenario_type": "buying"},
            }) + "\n")

    # Prove the round trip: the human query must survive into the opener.
    from evaluator.local_evaluator import (
        coarse_category, initial_message, load_jsonl, materialize_hidden_fields,
    )
    samples = load_jsonl(path)
    assert len(samples) == len(entries), (len(samples), len(entries))
    checked = 0
    for sample in samples:
        card, _behavior = materialize_hidden_fields(sample, {})
        disclosed: set[str] = set()
        opener = initial_message({**sample, "intent_card": card},
                                 coarse_category([]), disclosed)
        assert card["hard_constraints"][0] in opener, sample["sample_id"]
        checked += 1
    print(f"wrote {path}")
    print(f"  {len(samples)} samples, public-set schema + intent_card/behavior")
    print(f"  round-trip verified: human query reaches the opener in {checked}/{checked}")
    print(f"  example opener: "
          f"{initial_message({**samples[0], 'intent_card': samples[0]['intent_card']}, coarse_category([]), set())}")
    print("\nUse with the ESCI corpus, not data/catalog.jsonl:")
    print("  python -m evaluator.local_evaluator \\")
    print("    --catalog bakeoff/cache/esci_catalog.jsonl \\")
    print(f"    --dataset {path.relative_to(ROOT)} --output results_esci.json")


if __name__ == "__main__":
    main()
