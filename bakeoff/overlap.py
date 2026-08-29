"""Verbatim-overlap rate per session -- the mechanism Part 5 correlates against.

The retrieval one-pager rests on a single number: "the 84% verbatim-substring
stat is decisive: when the query literally contains text copied out of the
target document, exact term matching is close to the ideal algorithm". That
number has never been recomputed in this repo, and every argument for BM25 and
against dense is downstream of it. So recompute it from the evaluator's own
`intent_card` (local_evaluator.py:52-71) rather than quoting it.

Two rates are reported because they answer different questions:

  string-level  -- of the constraint strings the simulator discloses, how many
                   appear verbatim in the target's own searchable text?
  token-level   -- of the query tokens BM25 actually sees, how many appear in
                   the target's searchable text at all?

The string rate is the one the one-pager quotes. The token rate is the one that
governs whether BM25 can score the target highly, and it is the one Part 5
correlates with per-arm gain.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, load_jsonl, materialize_hidden_fields, searchable_text,
)
from starter.retrieval import _terms  # noqa: E402


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    catalog_path = ROOT / "data" / "catalog.jsonl"
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    _ids, _cats, products = catalog_index(catalog_path)
    records = {r["sample_id"]: r
               for r in json.loads((ROOT / "bakeoff" / "trajectories-current.json")
                                   .read_text(encoding="utf-8"))}

    rows = []
    verbatim_strings = total_strings = 0
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, _behavior = materialize_hidden_fields(sample, products)
        corpus = normalise(searchable_text(products[target]))

        constraints = [str(v) for v in card.get("hard_constraints", [])] + \
                      [str(v) for v in card.get("soft_preferences", [])]
        hits = [normalise(c) in corpus for c in constraints]
        verbatim_strings += sum(hits)
        total_strings += len(hits)

        # Token view: the final accumulated query the agent actually issued.
        final_query = records[sample["sample_id"]]["queries"][-1]
        query_tokens = set(_terms(final_query))
        corpus_tokens = set(_terms(corpus))
        covered = query_tokens & corpus_tokens
        rows.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "constraint_strings": len(hits),
            "constraint_strings_verbatim": sum(hits),
            "string_verbatim_rate": round(sum(hits) / len(hits), 4) if hits else None,
            "query_tokens": len(query_tokens),
            "token_coverage": round(len(covered) / len(query_tokens), 4) if query_tokens else 0.0,
        })

    print(f"constraint strings, all sessions : {total_strings}")
    print(f"  verbatim in target listing     : {verbatim_strings} "
          f"({verbatim_strings / total_strings:.1%})")

    coverages = sorted(r["token_coverage"] for r in rows)
    n = len(coverages)
    print(f"\nquery-token coverage of the target listing, per session (n={n}):")
    for label, value in [("min", coverages[0]), ("p10", coverages[n // 10]),
                         ("median", coverages[n // 2]), ("p90", coverages[9 * n // 10]),
                         ("max", coverages[-1])]:
        print(f"  {label:<7} {value:.3f}")
    print(f"  mean    {sum(coverages) / n:.3f}")

    out = ROOT / "bakeoff" / "results-overlap.json"
    out.write_text(json.dumps({
        "constraint_strings_total": total_strings,
        "constraint_strings_verbatim": verbatim_strings,
        "string_verbatim_rate": round(verbatim_strings / total_strings, 4),
        "per_session": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
