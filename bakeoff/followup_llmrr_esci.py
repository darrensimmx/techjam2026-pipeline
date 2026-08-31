"""Does an LLM escalation recover what BM25 + CE lose on vague human queries?

The three-arm probe the model-choice question actually turns on. Follow-up D
measured the cross-encoder on real human queries; this measures what sits above
it -- the `LLMRR` node from the docs repo's architecture v5, which fires only on
the low-overlap branch and reorders the CE's shortlist.

Why ESCI and not a paraphrase set we write ourselves: perturbation means
inventing the distribution that decides the answer, which is the trap Part 5
caught and Follow-up C repeated. ESCI's queries were typed by Amazon customers,
so their vagueness is real rather than authored here. The cost is that ESCI is
single-turn, so this measures turn-1 retrieval, NOT the conversational pipeline.
Say so when quoting any number out of this file.

WHAT IS MEASURED
    Input to every arm is identical: the CE-ordered shortlist from
    `cache/ce-esci-top50.json` when it exists, else `ce-esci-top20.json`
    (Follow-up D's persisted scores). The baseline is that ordering untouched --
    which is also the layer's documented fallback, so a negative delta here is a
    direct argument against shipping it.

    WHICH CE CACHE RAN IS RECORDED IN THE OUTPUT, and it matters. With only the
    top-20 cache the depth-50 shortlist is CE-sorted top-20 + raw BM25 ranks
    21-50, so the CE baseline at depth 20 and depth 50 come out BIT-IDENTICAL
    (only the CE-sorted first 20 can reach a top-10). The LLM at depth 50 is
    then largely being asked to find the target hiding at ranks 21-50 -- work a
    real CE@50 would already have done -- and any gain it books is credit taken
    from the cross-encoder. Run `followup_ce_esci.py --depth 50` first.

    Two OUTPUT ENCODINGS run as arms, per `report.md`: `permutation` (every ASIN
    returned, reordered, plus a rationale -- as shipped) and `indices` (ten
    integer positions, no rationale -- the recommendation whose accuracy cost
    was never measured). Their guards differ in shape and their failure kinds
    are counted apart, which is the only way that comparison is readable.

    Contract failures (the MODEL broke the output contract) and call failures
    (the API or the network) are separate counters. Summing them makes a
    hallucinated id indistinguishable from a rate limit.

    Every paid response is cached to `cache/llmrr-responses.jsonl`, keyed by a
    hash of the exact rendered call, so an interrupted sweep resumes instead of
    paying twice. A cached replay carries NO wall clock and is excluded from the
    latency percentiles.

TWO OFFLINE FAKE ARMS EXIST, AND SHOULD BE RUN BEFORE EVERY PAID SWEEP
    `--arms echo,reverse` needs no key and no network. `echo` must reproduce the
    CE baseline to the last digit at every depth and encoding; `reverse` must be
    measurably worse. If echo drifts, the arm path and the baseline path are not
    sharing a scoring route. If reverse does not, the model's output is not
    reaching the metric -- and a real arm would then report the baseline back as
    its own score, which is the most expensive silent failure available here.

    Segmentation is by verbatim overlap between the query and the candidate
    listings, because that IS the gate (`overlap.py`, architecture v5's OVERLAP
    node). Reported in TERCILES -- vague / mid / literal -- not a binary split,
    so you can see whether an arm's lead SCALES with vagueness. A flat profile
    means the win is not coming from the mechanism the layer is argued for.
    `vague` is the headline band; the others confirm the gate is worth having.

    Rationales are captured verbatim and sampled into the report, because the
    question is not only "does it move the target up" but "is the reasoning the
    kind that would generalise" -- a win on a bad rationale is a coin flip that
    landed well.

LATENCY IS A FIRST-CLASS RESULT, NOT A FOOTNOTE
    Per-call wall clock is recorded for every arm. The shipped CE already costs
    ~1.2 s/turn on a single rig and that is flagged as timeout/DQ risk, so an
    arm that wins on MRR and costs 15 s is a losing arm. Public leaderboard
    latency for these models is quoted at HIGH effort and runs to tens of
    seconds; every arm here is therefore pinned to its minimum-reasoning mode.
    Do not raise effort to chase a number without re-reporting latency.

THIS PRODUCES NO TechnicalScore. It is not the competition task.

    # offline, free -- the harness's own regression tests
    python bakeoff\\followup_llmrr_esci.py --arms "" --limit 0
    python bakeoff\\followup_llmrr_esci.py --arms echo,reverse --limit 0

    # paid -- smoke first, and confirm both failure columns are zero
    python bakeoff\\followup_llmrr_esci.py --arms haiku-4.5 --depths 20,50 --limit 40
    python bakeoff\\followup_llmrr_esci.py --arms haiku-4.5 --depths 20,50 --limit 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.dense import catalog_documents
from bakeoff.part5_realqueries import first_rank
# The contract -- output encodings, guards, banding, cache key, metrics. Kept in
# its own standard-library-only module so `tests/` can import it: this file
# reaches `bakeoff.dense`, which imports numpy at module scope, and CI installs
# only the comments-only `requirements.txt`.
from bakeoff.llmrr_contract import (
    ENCODINGS, LISTING_CHARS, RETURN_K, apply_indices, band, cache_key,
    check_indices, check_permutation, metrics, normalise, phrase_overlap,
    tercile_cuts,
)

CACHE = ROOT / "bakeoff" / "cache"
CORPUS = CACHE / "esci_catalog.jsonl"
# Append-only JSONL, not a rewritten dict: a crash mid-write costs one line
# rather than every answer paid for so far. `followup_ce_esci.py:50` already
# paid for this lesson once ("the same mistake Part 4 made, and it cost a
# re-run") and this probe did not inherit it.
RESPONSES = CACHE / "llmrr-responses.jsonl"

# Escalation window sizes to sweep. This is a MODEL DISCRIMINATOR, not a config
# knob, and it is the reason the sweep exists:
#
#   Recall argues for a WIDE window. Measured on all 600 ESCI queries, the
#   oracle headroom on the `vague` band (ceiling minus the CE baseline) is
#   0.0 at top-10, 1.83 points at top-20 and 9.13 points at top-50. At top-10
#   it is zero BY CONSTRUCTION -- reranking ten items you then return all ten
#   of cannot change R@10, only MRR.
#
#   Listwise degradation argues for a NARROW one. The reranking literature is
#   consistent that quality falls off as the window grows (O(n^2) attention over
#   the candidate block); RankZephyr covers a top-100 with a window of 20 and a
#   stride of 10 rather than one wide call.
#
# Which force wins is a property of the model, and no public leaderboard
# measures it. That is what this sweep is for. NOTE this supersedes
# `llm-escalation-proposal.md`'s "escalate over the top-10 or top-20, not the
# top-50", which was asserted without measurement.
DEFAULT_DEPTHS = (10, 20, 50)

# Reported in this order. "vague" is the headline: least verbatim overlap.
SLICES = ("all", "vague", "mid", "literal")

# Ceiling on what ANY reranker can do, at any window size: 12.6% of low-overlap
# queries have the target beyond BM25 rank 50 (2.1% at 51-100, 10.5% past 100),
# against 3.2% on the high-overlap slice. Those are a RECALL failure and this
# layer cannot touch them -- the scope limit the proposal already writes down.
# Report the reachable ceiling beside every arm so no gain is read as larger
# than the headroom that exists.


# --- Arms -------------------------------------------------------------------
# Model IDs and the minimum-reasoning switch for each provider. The Anthropic
# row is verified against the current model table; the OpenAI and Google rows
# name models released after this file's author's knowledge cutoff and their
# exact parameter spelling MUST be confirmed against provider docs before the
# first paid run. They are isolated here so that is a one-line change.
ARMS = {
    "haiku-4.5": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        # Haiku 4.5 REJECTS output_config.effort -- it errors. Omitting
        # `thinking` on a pre-4.6 model means no thinking, which is the fast
        # path we want. Do not "fix" this by adding an effort dial.
        "params": {},
    },
    "gemini-3.7-flash": {
        "provider": "google",
        "model": "gemini-3.7-flash",                 # VERIFY before paid run
        "params": {"thinking_budget": 0},            # VERIFY: non-reasoning mode
    },
    "gpt-5.6-luna": {
        "provider": "openai",
        "model": "gpt-5.6-luna",                     # VERIFY before paid run
        "params": {"reasoning_effort": "minimal"},   # VERIFY
    },
    # --- Offline fakes. No network, no key, no spend. ------------------------
    # These are the harness's own regression tests, and they are cheap enough
    # to run before every paid sweep:
    #
    #   echo     returns the input order untouched. Its metrics MUST equal the
    #            CE baseline to the last digit, at every depth and encoding. If
    #            they do not, the LLM path and the baseline path are not sharing
    #            a scoring route, and every arm below is measuring two things.
    #   reverse  returns the exact reverse. Its MRR@10 MUST be measurably worse.
    #            If it is not, the model's output is not reaching the metric --
    #            the quietest and most expensive failure available here, because
    #            a real arm would then report the baseline back as its own score.
    "echo": {"provider": "local", "model": "echo", "params": {"mode": "echo"}},
    "reverse": {"provider": "local", "model": "reverse", "params": {"mode": "reverse"}},
}

# The instruction, schema, prompt builder and guard for each encoding all live
# in `llmrr_contract.ENCODINGS`. Two arms, per `report.md` section 6:
#
#   permutation  every ASIN returned, reordered, plus a rationale -- as shipped
#   indices      10 integer positions, no rationale -- section 4's recommendation
#
# PROMPT CACHING CANNOT FIRE HERE, and that is structural rather than a tuning
# miss. Haiku 4.5's minimum cacheable prefix is 4096 tokens (the Opus 4.6 /
# Opus 4.5 / Haiku 4.5 tier; the Sonnet tier is 1024). The instruction is ~190
# tokens -- 21x under the floor -- so `cache_read` will read 0 forever. The
# `cache_control` marker below is kept only so the zero is attributable; do not
# read it as a bug, and do not quote `report.md`'s "8.7% of input cost" saving
# for this arm, which is 0%.


# --- Providers --------------------------------------------------------------
# Each returns (payload_dict, seconds, usage_dict) or raises. Callers treat ANY
# exception as "keep the CE order", which is the layer's shipped fallback.
#
# CLIENTS ARE BUILT ONCE, not per call. The previous shape constructed a fresh
# client inside the request path, which re-reads credentials and re-establishes
# a connection pool on every one of thousands of calls -- and charges the setup
# to the latency sample, which is the one number this layer is judged on.
_CLIENTS: dict = {}


DEADLINE_S = 30.0    # Arm B lowers this to a per-turn budget; see set_deadline.


def set_deadline(seconds: float) -> None:
    """Arm B runs inside a turn loop and needs a far tighter deadline than a
    batch sweep does. Changing it drops any client built at the old one."""
    global DEADLINE_S
    DEADLINE_S = seconds
    _CLIENTS.pop("anthropic", None)


def _anthropic_client():
    if "anthropic" not in _CLIENTS:
        import anthropic
        # max_retries=0 is the load-bearing half: the SDK default of 2 means a
        # worst case of 3x the timeout on a single turn, against an organizer
        # per-turn timeout that has never been published.
        _CLIENTS["anthropic"] = anthropic.Anthropic(
            timeout=anthropic.Timeout(DEADLINE_S, connect=5.0), max_retries=0)
    return _CLIENTS["anthropic"]


def call_anthropic(arm, encoding, prompt):
    client = _anthropic_client()
    started = time.time()
    response = client.messages.create(
        model=arm["model"],
        max_tokens=2048,
        system=[{"type": "text", "text": encoding["instruction"],
                 "cache_control": {"type": "ephemeral"}}],
        output_config={"format": {"type": "json_schema",
                                  "schema": encoding["schema"]}},
        messages=[{"role": "user", "content": prompt}],
        **arm["params"],
    )
    elapsed = time.time() - started
    payload = json.loads("".join(b.text for b in response.content if b.type == "text"))
    usage = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
        "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    return payload, elapsed, usage


def call_openai(arm, encoding, prompt):
    if "openai" not in _CLIENTS:
        from openai import OpenAI
        _CLIENTS["openai"] = OpenAI(timeout=30.0, max_retries=0)
    client = _CLIENTS["openai"]
    started = time.time()
    response = client.chat.completions.create(
        model=arm["model"],
        messages=[{"role": "system", "content": encoding["instruction"]},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "ranking", "schema": encoding["schema"], "strict": True}},
        **arm["params"],
    )
    elapsed = time.time() - started
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return payload, elapsed, {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "cache_read": getattr(details, "cached_tokens", 0) or 0,
    }


def call_google(arm, encoding, prompt):
    # NOTE: `google.genai` is the current SDK and is NOT installed here; only
    # the legacy `google.generativeai` is. `pip install google-genai` before
    # running this arm, and confirm the model id and the non-reasoning
    # parameter spelling against provider docs first -- both are unverified.
    if "google" not in _CLIENTS:
        from google import genai
        _CLIENTS["google"] = genai.Client()
    from google.genai import types

    client = _CLIENTS["google"]
    started = time.time()
    response = client.models.generate_content(
        model=arm["model"],
        contents=encoding["instruction"] + "\n\n" + prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=encoding["schema"],
            thinking_config=types.ThinkingConfig(
                thinking_budget=arm["params"].get("thinking_budget", 0)),
        ),
    )
    elapsed = time.time() - started
    payload = json.loads(response.text)
    meta = response.usage_metadata
    return payload, elapsed, {
        "input": meta.prompt_token_count,
        "output": meta.candidates_token_count,
        "cache_read": getattr(meta, "cached_content_token_count", 0) or 0,
    }


def call_local(arm, encoding, prompt):
    """The offline fakes. Parses ids back out of the rendered prompt, exactly as
    a model reading that prompt would have to -- so this exercises the real
    prompt builder and the real guard, not a shortcut around them."""
    ids = [line[1:line.index("] ")] for line in prompt.split("\n")
           if line.startswith("[") and "] " in line]
    if encoding["has_rationale"]:
        order = ids if arm["params"]["mode"] == "echo" else list(reversed(ids))
        return {"ranking": order, "reason": "offline fake"}, 0.0, {}
    positions = list(range(len(ids)))
    if arm["params"]["mode"] == "reverse":
        positions = list(reversed(positions))
    return {"ranking": positions[:RETURN_K]}, 0.0, {}


DISPATCH = {"anthropic": call_anthropic, "openai": call_openai,
            "google": call_google, "local": call_local}


# `metrics` and `phrase_overlap` now live in `llmrr_contract`, so Arm B bands by
# the identical function -- otherwise "vague" would mean two different things in
# the two arms and the results could not be read together.


# --- Response cache ---------------------------------------------------------

def load_responses() -> dict:
    """Every cached call, keyed by the sha256 of its exact rendered inputs."""
    if not RESPONSES.exists():
        return {}
    cached: dict = {}
    with RESPONSES.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue        # a torn final line from a killed run; skip it
            cached[row["key"]] = row          # last write wins
    return cached


def append_response(row: dict) -> None:
    RESPONSES.parent.mkdir(parents=True, exist_ok=True)
    with RESPONSES.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma-separated subset of: " + ", ".join(ARMS))
    parser.add_argument("--encodings", default="permutation,indices",
                        help="output contracts to sweep: " + ", ".join(ENCODINGS))
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                        help="escalation window sizes, comma-separated")
    parser.add_argument("--limit", type=int, default=25,
                        help="queries to score (default 25 -- a smoke run). "
                             "Use --limit 0 for the full set.")
    parser.add_argument("--out", default="bakeoff/results-followup-llmrr-esci.json")
    args = parser.parse_args()

    depths = tuple(int(d) for d in args.depths.split(",") if d.strip())
    encoding_names = [e.strip() for e in args.encodings.split(",") if e.strip()]

    blob = json.loads((CACHE / "esci_queries.json").read_text(encoding="utf-8"))
    entries = blob["queries"]
    bm25 = json.loads((CACHE / "esci-bm25.json").read_text(encoding="utf-8"))

    # Prefer the wider CE cache when it exists, and RECORD WHICH ONE RAN.
    #
    # This matters more than it looks. With only the top-20 cache, the depth-50
    # shortlist is CE-sorted top-20 + raw BM25 ranks 21-50 -- so the CE baseline
    # at depth 20 and depth 50 come out BIT-IDENTICAL (only the CE-sorted first
    # 20 can reach a top-10, so appending an unranked tail cannot move
    # first_rank at k=10). The LLM at depth 50 is then largely being asked to
    # "find the target hiding at ranks 21-50" -- precisely the work a real CE@50
    # would already have done -- and any gain it books is credit taken from the
    # cross-encoder. Re-run `followup_ce_esci.py --depth 50` before reading a
    # depth-50 number as a result.
    ce_wide = CACHE / "ce-esci-top50.json"
    ce_path = ce_wide if ce_wide.exists() else CACHE / "ce-esci-top20.json"
    cross_encoder = json.loads(ce_path.read_text(encoding="utf-8"))
    ce_depth = 50 if ce_path == ce_wide else 20

    asins, documents = catalog_documents(CORPUS)
    text = dict(zip(asins, documents))
    if args.limit:
        entries = entries[:args.limit]

    def shortlist_at(depth: int) -> dict[str, list[str]]:
        """CE order within the scored window, BM25 order beyond it."""
        per_depth: dict[str, list[str]] = {}
        for entry in entries:
            query = entry["query"]
            full = [a for a, _ in bm25.get(query, [])][:depth]
            scored, tail = full[:ce_depth], full[ce_depth:]
            per_depth[query] = sorted(
                scored, key=lambda a: -cross_encoder.get(query + "\x1f" + a, 0.0)) + tail
        return per_depth

    shortlists: dict[int, dict[str, list[str]]] = {d: shortlist_at(d) for d in depths}

    # The gate: does the shopper's wording appear literally in the listings we
    # retrieved? Phrase-level, per phrase_overlap's docstring. Computed against
    # the returned window only, so it does not move with the sweep.
    #
    # Built from its OWN RETURN_K pass rather than indexing `shortlists`, which
    # is only populated for the swept depths -- `--depths 20,50` would otherwise
    # KeyError on a hardcoded shortlists[RETURN_K]. Banding is depth-independent
    # by construction, which is also why band sizes must not move when --depths
    # changes; the committed baseline artifact is the regression test.
    gate_shortlist = shortlists.get(RETURN_K) or shortlist_at(RETURN_K)
    gate = {}
    for entry in entries:
        query = entry["query"]
        listings = [normalise(text.get(a, ""))
                    for a in gate_shortlist[query][:RETURN_K]]
        gate[query] = phrase_overlap(query, listings)

    # TERCILES, not a binary split. Criterion (b) is not "does it help on vague
    # queries" but "does its help SCALE with vagueness" -- a layer justified by
    # lexical blindness should widen its lead monotonically as overlap falls. A
    # flat profile across bands means the win, if any, is not coming from the
    # mechanism the layer is argued for.
    cuts = tercile_cuts(list(gate.values()))

    def band_of(query: str) -> str:
        return band(gate[query], cuts)

    sizes = {b: sum(1 for q in gate if band_of(q) == b)
             for b in ("vague", "mid", "literal")}
    if min(sizes.values()) == 0:
        print("WARNING: the overlap banding is degenerate -- {}. The bands below "
              "are not a segmentation.".format(sizes), flush=True)

    report = {"source": blob["source"], "queries": len(entries),
              "depths": list(depths), "encodings": encoding_names,
              "ce_cache": ce_path.name, "ce_depth": ce_depth,
              "overlap_cuts": [round(c, 4) for c in cuts], "band_sizes": sizes,
              "reachable_ceiling": {}, "arms": {}}

    # The ceiling: how often the target is inside the window at all. No reranker
    # of any kind can beat this, so every gain below must be read against it.
    # Free, offline, no model.
    for depth in depths:
        per_slice: dict[str, list] = {s: [] for s in SLICES}
        for entry in entries:
            query, targets = entry["query"], set(entry["targets"])
            inside = first_rank(shortlists[depth][query], targets, depth) is not None
            per_slice["all"].append(inside)
            per_slice[band_of(query)].append(inside)
        report["reachable_ceiling"][depth] = {
            s: (round(sum(v) / len(v), 4) if v else None) for s, v in per_slice.items()}

    # Baseline: the CE order, unreordered -- also the layer's shipped fallback.
    for depth in depths:
        base: dict[str, list] = {s: [] for s in SLICES}
        for entry in entries:
            query, targets = entry["query"], set(entry["targets"])
            rank = first_rank(shortlists[depth][query], targets, RETURN_K)
            base["all"].append(rank)
            base[band_of(query)].append(rank)
        report["arms"]["CE only (baseline)@%d" % depth] = {
            s: metrics(v) for s, v in base.items()}

    cached = load_responses()
    if cached:
        print("\nresponse cache: %d entries in %s" % (len(cached), RESPONSES.name),
              flush=True)

    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arm = ARMS[name]
        call = DISPATCH[arm["provider"]]
        for encoding_name in encoding_names:
            encoding = ENCODINGS[encoding_name]
            for depth in depths:
                ranks: dict[str, list] = {s: [] for s in SLICES}
                latencies: list[float] = []       # LIVE calls only -- see below
                totals = {"input": 0, "output": 0, "cache_read": 0}
                # Contract failures are the MODEL breaking the output contract;
                # call failures are the API or the network. The probe used to
                # sum them into one integer, which makes `report.md`'s claim
                # that the index contract is stronger than the permutation
                # contract untestable -- you cannot tell a hallucinated id from
                # a rate limit. Counted apart, by kind.
                contract_failures: dict[str, int] = {}
                call_failures: dict[str, int] = {}
                hits = 0
                samples: list[dict] = []
                print("\n[%s / %s @ top-%d] %s via %s"
                      % (name, encoding_name, depth, arm["model"], arm["provider"]),
                      flush=True)

                for position, entry in enumerate(entries, 1):
                    query, targets = entry["query"], set(entry["targets"])
                    head = shortlists[depth][query]
                    order = head
                    candidates = [(a, text.get(a, "")) for a in head]
                    prompt = encoding["build_prompt"](query, candidates)
                    # The offline fakes are free and instant, so they are never
                    # cached -- caching them would only bloat the file that
                    # exists to protect paid calls.
                    cacheable = arm["provider"] != "local"
                    key = cache_key(arm["model"], encoding_name,
                                    encoding["instruction"], prompt)
                    row = cached.get(key) if cacheable else None
                    try:
                        if row is not None:
                            payload, secs, usage = row["payload"], None, row["usage"]
                            hits += 1
                        else:
                            payload, secs, usage = call(arm, encoding, prompt)
                            if cacheable:
                                append_response({
                                    "key": key, "arm": name, "model": arm["model"],
                                    "encoding": encoding_name, "depth": depth,
                                    "query": query, "payload": payload,
                                    "latency_s": secs, "usage": usage,
                                    "ts": time.time()})
                                cached[key] = {"payload": payload, "usage": usage}
                        # A CACHE HIT HAS NO WALL CLOCK. Folding a zero in here
                        # would drag p50 toward 0 on any resumed run and quietly
                        # destroy the one number this layer is judged on.
                        if secs is not None:
                            latencies.append(secs)
                        for k in totals:
                            totals[k] += usage.get(k, 0)

                        ranking = payload.get("ranking")
                        if encoding_name == "indices":
                            reason = check_indices(ranking, len(head))
                            accepted = None if reason else apply_indices(ranking, head)
                        else:
                            reason = check_permutation(ranking, head)
                            accepted = None if reason else list(ranking)
                        if reason:
                            contract_failures[reason] = contract_failures.get(reason, 0) + 1
                        else:
                            order = accepted
                            if len(samples) < 8:
                                samples.append({
                                    "query": query,
                                    "reason": payload.get("reason", ""),
                                    "slice": band_of(query),
                                    "moved_top": head[:1] != order[:1]})
                    except Exception as exc:                   # noqa: BLE001
                        kind = type(exc).__name__
                        call_failures[kind] = call_failures.get(kind, 0) + 1
                        if sum(call_failures.values()) <= 3:
                            print("  ! " + kind + ": " + str(exc), flush=True)
                    rank = first_rank(order, targets, RETURN_K)
                    ranks["all"].append(rank)
                    ranks[band_of(query)].append(rank)
                    if position % 25 == 0:
                        print("  %d/%d" % (position, len(entries)), flush=True)

                ordered = sorted(latencies)
                report["arms"]["%s/%s@%d" % (name, encoding_name, depth)] = {
                    **{s: metrics(v) for s, v in ranks.items()},
                    "latency_s": {
                        "p50": round(ordered[len(ordered) // 2], 3) if ordered else None,
                        "p95": round(ordered[min(int(len(ordered) * 0.95),
                                                 len(ordered) - 1)], 3) if ordered else None,
                        "max": round(max(ordered), 3) if ordered else None,
                        "n_live": len(ordered), "n_cached": hits},
                    "contract_failures": contract_failures,
                    "call_failures": call_failures,
                    "usage": totals,
                    "rationales": samples,
                }

    print("\nband sizes: %s   cuts %s" % (sizes, [round(c, 4) for c in cuts]))
    print("CE cache:   %s (scored depth %d)" % (ce_path.name, ce_depth))
    if ce_depth < max(depths):
        print("  !! ranks %d-%d are RAW BM25, not cross-encoded. A depth-%d gain "
              "here includes work a real CE@%d would have done first --\n"
              "     re-run: .venv/Scripts/python bakeoff/followup_ce_esci.py "
              "--depth %d --out bakeoff/cache/ce-esci-top50.json"
              % (ce_depth + 1, max(depths), max(depths), max(depths), max(depths)))

    print("\nreachable ceiling (target inside the window at all):")
    for depth in depths:
        c = report["reachable_ceiling"][depth]
        print("  top-%-3d  vague %s   mid %s   literal %s"
              % (depth, c["vague"], c["mid"], c["literal"]))

    print("\n{:<34}{:>8}{:>9}{:>9}{:>9}{:>7}{:>7}".format(
        "arm/encoding@depth", "slice", "R@10", "MRR@10", "p50 s", "ctr", "call"))
    for name, row in report["arms"].items():
        for slice_ in SLICES:
            p50 = row.get("latency_s", {}).get("p50")
            print("{:<34}{:>8}{:>9}{:>9}{:>9}{:>7}{:>7}".format(
                name, slice_, row[slice_]["recall@10"], row[slice_]["mrr@10"],
                p50 if p50 is not None else "-",
                sum(row.get("contract_failures", {}).values()) if "contract_failures" in row else "-",
                sum(row.get("call_failures", {}).values()) if "call_failures" in row else "-"))
    print("  ctr = contract failures (the MODEL broke the output contract)")
    print("  call = call failures (the API or the network). Never sum these.")

    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote " + str(out))
    print("VAGUE is the headline band -- it is the one the gate routes.")
    print("Read every gain against the reachable ceiling for that depth and slice.")
    print("p50 covers LIVE calls only; cached replays carry no wall clock.")


if __name__ == "__main__":
    main()
