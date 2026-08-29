"""Follow-up A -- phrase queries. Does exact-phrase matching beat a bag of terms?

Why this is worth measuring before any reranker
-----------------------------------------------
`overlap.py` measures 94.5% of the simulator's disclosed constraint strings as
verbatim substrings of the target's own listing. The shipped retriever throws
that away: `starter/retrieval.py:70-73` splits the accumulated ledger into <=40
unique stopword-filtered unigrams and ORs them, so a disclosed constraint like
"Lightweight dangle earrings, hoops measure approximately 2 inches" matches any
product containing "lightweight". The single most reliable property of this
data is discarded at the last step before scoring.

FTS5 supports phrase queries natively, so testing this costs one more BM25 pass
and no new dependency. It is also the only lever in this bake-off that could
make the system *simpler* rather than more complex.

Two further reasons to run it first: our BM25 configuration is the organizer's
starter verbatim -- `competition-source/starter/agent.py:93` and
`starter/retrieval.py:76` carry the identical weight vector -- so nothing about
retrieval has ever been tuned; and a retrieval gain compounds with reranking
rather than competing with it, since a better shortlist is a better input to
any reranker.

Recovering the disclosed strings without another evaluator run
--------------------------------------------------------------
`SessionState.record_message` appends with a single space, so the cached
per-turn queries are strict prefixes of one another and each turn's appended
message is `queries[t][len(queries[t-1]):]`. A content-free turn appends
nothing and yields an empty increment. So the messages are recoverable from
`trajectories-*.json` directly -- no re-run, and no risk of the reconstruction
drifting from what was actually measured.

Frames are decoded with the evaluator's own literal templates
(`local_evaluator.py:154-185`), which is the Tier 1 regex spine hard-rules A3
already sanctions -- not a new parsing component.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.bm25_scores import ScoringIndex  # noqa: E402
from bakeoff.simulate import (  # noqa: E402
    MAX_TURNS, bootstrap_delta_ci, load_trajectories, paired, play, score,
)
from starter.retrieval import _terms  # noqa: E402

CACHE = ROOT / "bakeoff" / "cache"
DEPTH = 100

# local_evaluator.py:154-185. Ordered: the more specific opener first, because
# frame 2 ("I'm looking for {cat}. {old_value}") is identified by exclusion and
# would otherwise swallow frame 1.
_FRAMES = (
    re.compile(r"^i'm looking for .+?\. a key requirement is: (?P<c>.+)\.$", re.I),
    re.compile(r"^for that, what matters is: (?P<c>.+)\.$", re.I),
    re.compile(r"^actually, ignore my earlier preference\. what i need is: (?P<c>.+)\.$", re.I),
    re.compile(r"^i'm looking for .+?, but i'm still exploring\.$", re.I),   # no payload
    re.compile(r"^i'm looking for [^.]+\. (?P<c>.+)$", re.I),                # frame 2, last
)


def constraints_from(message: str) -> list[str]:
    """The disclosed constraint strings inside one simulator message."""
    text = message.strip()
    for frame in _FRAMES:
        m = frame.match(text)
        if not m:
            continue
        payload = m.groupdict().get("c")
        if not payload:
            return []
        # "For that, what matters is: A; B." joins up to two constraints.
        return [p.strip() for p in payload.split(";") if p.strip()]
    return []


def disclosed_per_turn(record: dict) -> list[list[str]]:
    """Cumulative list of disclosed constraint strings at each turn."""
    out, running, previous = [], [], ""
    for turn in range(MAX_TURNS):
        query = record["queries"][turn]
        increment = query[len(previous):].strip() if query.startswith(previous) else query
        if increment:
            running = running + constraints_from(increment)
        previous = query
        out.append(list(running))
    return out


def _fts_phrase(value: str) -> str:
    """FTS5 phrase literal. Doubles embedded quotes; drops anything with no
    indexable token left, which would make the whole expression a syntax error."""
    tokens = _terms(value)
    if not tokens:
        return ""
    return '"' + " ".join(tokens) + '"'


def build_query(constraints: list[str], fallback: str, mode: str) -> str:
    phrases = [p for p in (_fts_phrase(c) for c in constraints) if p]
    unigrams = [f'"{t}"' for t in list(dict.fromkeys(_terms(fallback)))[:40]]
    if mode == "phrase_only":
        clauses = phrases or unigrams
    elif mode == "phrase_plus":
        clauses = phrases + unigrams
    else:
        clauses = unigrams
    return " OR ".join(clauses)


def main() -> None:
    ledger = sys.argv[1] if len(sys.argv) > 1 else "current"
    records = load_trajectories(ledger)
    index = ScoringIndex(ROOT / "data" / "catalog.jsonl")

    baseline = play(records, lambda r, t: r["bm25"][t - 1][:10])
    base = score(baseline)
    print(f"\n{'=' * 92}\nFOLLOW-UP A -- PHRASE QUERIES   (ledger: {ledger})\n{'=' * 92}")
    print(f"{'arm':<28} {'Tech':>9} {'delta':>10} {'Hit@10':>8} {'MRR':>9} {'MTTC':>7} "
          f"{'Eff':>8} {'win/loss/tie':>15}")
    print(f"{'P0  unigram OR (shipped)':<28} {base['technical_score']:>9} {'--':>10} "
          f"{base['hit_rate_at_10']:>8} {base['mrr']:>9} {base['mttc']:>7} "
          f"{base['efficiency']:>8} {'--':>15}")

    disclosed = {r["sample_id"]: disclosed_per_turn(r) for r in records}
    extracted = sum(len(d[-1]) for d in disclosed.values())
    print(f"\n(recovered {extracted} disclosed constraint strings across "
          f"{len(records)} sessions)\n")

    results = {}
    for mode, label in (("phrase_plus", "P1  phrases + unigrams"),
                        ("phrase_only", "P2  phrases only")):
        cache: dict[str, list[str]] = {}
        t0 = time.time()
        for record in records:
            for turn in range(1, MAX_TURNS + 1):
                key = f"{record['sample_id']}|{turn}"
                expression = build_query(disclosed[record["sample_id"]][turn - 1],
                                         record["queries"][turn - 1], mode)
                if not expression:
                    cache[key] = []
                    continue
                try:
                    cache[key] = [a for a, _ in index.search_expression(expression, DEPTH)]
                except Exception:
                    cache[key] = record["bm25"][turn - 1][:DEPTH]
        arm = play(records, lambda r, t: cache[f"{r['sample_id']}|{t}"][:10])
        s, p = score(arm), paired(baseline, arm)
        ci = bootstrap_delta_ci(baseline, arm)
        results[mode] = {"score": s, "paired": p, "delta_ci": ci,
                         "seconds": round(time.time() - t0, 1)}
        print(f"{label:<28} {s['technical_score']:>9} "
              f"{round(s['technical_score'] - base['technical_score'], 6):>10} "
              f"{s['hit_rate_at_10']:>8} {s['mrr']:>9} {s['mttc']:>7} {s['efficiency']:>8} "
              f"{str(p['wins']) + '/' + str(p['losses']) + '/' + str(p['ties']):>15}")

    print("\n-- paired bootstrap --")
    for mode, r in results.items():
        print(f"  {mode:<12} {json.dumps(r['delta_ci'])}")

    out = ROOT / "bakeoff" / f"results-followup-phrase-{ledger}.json"
    out.write_text(json.dumps({"ledger": ledger, "baseline": base,
                               "arms": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
