"""Is the phrase-query gain real retrieval, or is it riding the simulator leak?

The recommendation to ship phrase queries carries a caveat that was asserted and
not measured: "this gain leans on the 94.5% verbatim property, so by Part 5's
logic it would help less on real human queries." That is a testable claim and it
should not go into a write-up untested -- it is exactly the sort of plausible
mechanism story this bake-off has already had to retract once.

Same transformation, real queries. On the public set the phrase clause is the
disclosed constraint string; here it is the customer's own query, which is the
honest analogue -- a human types one utterance, and the question is whether
matching it as a phrase beats matching its terms independently.

Corpus, index and weights are the ESCI benchmark from part5_realqueries.py, so
the only thing that varies against that baseline is the query expression.

Three outcomes, all publishable:
  * phrase helps roughly as much here      -> the gain is ordinary IR, ship it
                                              and say so
  * phrase helps much less here            -> the gain is leak-dependent; still
                                              ship it, but the write-up must say
                                              it will not transfer
  * phrase hurts here                      -> it is a harness exploit and the
                                              recommendation needs revisiting
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.bm25_scores import ScoringIndex  # noqa: E402
from bakeoff.followup_phrase import _fts_phrase  # noqa: E402
from bakeoff.part5_realqueries import first_rank  # noqa: E402
from starter.retrieval import _terms  # noqa: E402

CACHE = ROOT / "bakeoff" / "cache"
DEPTH = 100


def main() -> None:
    blob = json.loads((CACHE / "esci_queries.json").read_text(encoding="utf-8"))
    entries = blob["queries"]
    queries = [e["query"] for e in entries]
    targets = [set(e["targets"]) for e in entries]
    index = ScoringIndex(CACHE / "esci_catalog.jsonl")

    def unigram(q: str) -> str:
        return " OR ".join(f'"{t}"' for t in list(dict.fromkeys(_terms(q)))[:40])

    def phrase_plus(q: str) -> str:
        p = _fts_phrase(q)
        u = unigram(q)
        return f"{p} OR {u}" if p and u else (p or u)

    def phrase_only(q: str) -> str:
        return _fts_phrase(q) or unigram(q)

    print(f"ESCI: {len(entries)} human queries, "
          f"{sum(1 for _ in (CACHE / 'esci_catalog.jsonl').open(encoding='utf-8'))} products")
    print(f"\n{'arm':<22} {'R@10':>8} {'R@50':>8} {'MRR@10':>8}   vs unigram")

    rows = {}
    baseline = None
    for label, builder in (("unigram OR (shipped)", unigram),
                           ("phrases + unigrams", phrase_plus),
                           ("phrases only", phrase_only)):
        ranks = []
        for query, target in zip(queries, targets):
            expression = builder(query)
            try:
                ranked = [a for a, _ in index.search_expression(expression, DEPTH)] \
                    if expression else []
            except Exception:
                ranked = []
            ranks.append(first_rank(ranked, target, DEPTH))
        row = {
            "recall@10": round(sum(1 for r in ranks if r and r <= 10) / len(ranks), 4),
            "recall@50": round(sum(1 for r in ranks if r and r <= 50) / len(ranks), 4),
            "mrr@10": round(sum(1.0 / r for r in ranks if r and r <= 10) / len(ranks), 4),
        }
        rows[label] = row
        if baseline is None:
            baseline = row
            delta = ""
        else:
            delta = (f"  R@10 {row['recall@10'] - baseline['recall@10']:+.4f}"
                     f"  MRR {row['mrr@10'] - baseline['mrr@10']:+.4f}")
        print(f"{label:<22} {row['recall@10']:>8} {row['recall@50']:>8} "
              f"{row['mrr@10']:>8}{delta}")

    out = ROOT / "bakeoff" / "results-followup-phrase-esci.json"
    out.write_text(json.dumps({"source": blob["source"], "queries": len(entries),
                               "arms": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
