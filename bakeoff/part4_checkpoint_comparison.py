"""Cross-encoder checkpoint comparison at depth=50.

Tests 5 different cross-encoder models to find the optimal checkpoint for
reranking BM25's top-50 results down to top-10 recommendations.

Addresses docs/todo.md Item 4 axis 3: "Which cross-encoder — never measured."

Models tested (size/speed/quality trade-offs):
1. TinyBERT-L2: Ultra-lightweight (~17 MB, fastest)
2. MiniLM-L6: Current baseline (~91 MB, proven)
3. MiniLM-L12: Enhanced baseline (~135 MB, better semantic understanding)
4. mMiniLM-L12: Multilingual variant (~110-120 MB, handles paraphrasing)
5. DistilRoBERTa: Semantic specialist (~250 MB, best semantic understanding)

Depth=50 only (rerank top-50 to get top-10). This is the most thorough reranking
configuration, examining deeper into BM25's ranking for better results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bakeoff.dense import catalog_documents  # noqa: E402
from bakeoff.simulate import (  # noqa: E402
    MAX_TURNS, bm25_ranker, bootstrap_delta_ci, by_scenario, load_trajectories,
    paired, play, score,
)

CACHE = ROOT / "bakeoff" / "cache"
DEPTH = 50  # Rerank top-50 to get top-10

# Model dictionary: short_name -> HuggingFace model ID
MODELS = {
    "tinybert-l2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # baseline
    "minilm-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "mminilm-l12": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "distilroberta": "cross-encoder/qnli-distilroberta-base",
    "zerank-1-small": "zeroentropy/zerank-1-small",  # 1.7B ZeroEntropy reranker
}


class CrossEncoderReranker:
    """Cross-encoder reranker with per-model caching."""

    def __init__(self, model_name: str, model_path: str, depth: int) -> None:
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.model_path = model_path
        self.depth = depth

        print(f"  Loading model: {model_path}")
        # ZeroEntropy models require trust_remote_code
        trust_code = model_name.startswith("zerank")
        self.model = CrossEncoder(model_path, max_length=256, device="cpu", trust_remote_code=trust_code)

        # Load catalog documents for scoring
        asins, documents = catalog_documents(ROOT / "data" / "catalog.jsonl")
        self.text = dict(zip(asins, documents))

        # Per-model cache (convert string keys back to tuples)
        cache_file = CACHE / f"ce-{model_name}-depth{depth}.json"
        if cache_file.exists():
            print(f"  Loading cache from {cache_file.name}")
            stored_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            # Convert "query|||asin" string keys back to (query, asin) tuples
            self.cache = {}
            for key, value in stored_cache.items():
                if "|||" in key:
                    query, asin = key.split("|||", 1)
                    self.cache[(query, asin)] = value
        else:
            self.cache = {}
        self.cache_file = cache_file

        # Performance metrics
        self.pairs_scored = 0
        self.seconds = 0.0
        self.turns_reranked = 0

    def rank(self, record: dict, turn: int) -> list[str]:
        """Rerank BM25's top-N for a given turn."""
        query = record["queries"][turn - 1]
        shortlist = record["bm25"][turn - 1][:self.depth]
        if not shortlist:
            return []

        self.turns_reranked += 1

        # Check cache and batch score missing pairs
        missing = [a for a in shortlist if (query, a) not in self.cache]
        if missing:
            t0 = time.perf_counter()
            scores = self.model.predict(
                [[query, self.text.get(a, "")] for a in missing],
                batch_size=64,
                show_progress_bar=False
            )
            self.seconds += time.perf_counter() - t0
            self.pairs_scored += len(missing)

            # Update cache
            for asin, value in zip(missing, scores):
                self.cache[(query, asin)] = float(value)

        # Return reranked shortlist
        return sorted(shortlist, key=lambda a: -self.cache.get((query, a), 0))

    def save_cache(self):
        """Persist cache to disk."""
        CACHE.mkdir(parents=True, exist_ok=True)
        # Convert tuple keys (query, asin) to string keys "query|||asin" for JSON serialization
        serializable_cache = {f"{query}|||{asin}": score for (query, asin), score in self.cache.items()}
        self.cache_file.write_text(json.dumps(serializable_cache, indent=2), encoding="utf-8")
        print(f"  Saved cache to {self.cache_file.name}")


def model_size_mb(model_path: str) -> float:
    """Get the largest single weight file size in MB."""
    try:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(model_path))
        weights = [p for p in root.rglob("*")
                   if p.is_file() and p.suffix in {".bin", ".safetensors", ".onnx"}]
        # A submission bundles one weight format, not all of them.
        return max((p.stat().st_size for p in weights), default=0) / 1e6
    except Exception as e:
        print(f"  Warning: Could not determine model size for {model_path}: {e}")
        return 0.0


def evaluate_shipping_criteria(results: dict) -> dict:
    """Evaluate shipping criteria per part0-decision-rule.md §3."""
    criteria = {
        "ci_excludes_zero": results["delta_ci"]["excludes_zero"],
        "delta_gte_0_020": results["delta_ci"]["mean_delta"] >= 0.020,
        "regress_lte_5pct": results["paired"]["regressed_fraction"] <= 0.05,
        "latency_lte_1s": results["performance"]["seconds_per_reranked_turn"] <= 1.0,
    }

    # All four must hold to ship
    criteria["passes_all"] = all(criteria.values())

    # If regress > 5%, need aggregate gain > 3 SD (+0.070)
    if not criteria["regress_lte_5pct"]:
        sd = results["delta_ci"]["sd"]
        criteria["high_regress_override"] = results["delta_ci"]["mean_delta"] >= 3 * sd
        if criteria["high_regress_override"]:
            criteria["passes_all"] = True  # Override

    return criteria


def report(ledger: str, sample_size: int = 50) -> dict:
    """Run checkpoint comparison for a given ledger."""
    all_records = load_trajectories(ledger)
    # Sample first N sessions for faster comparison
    records = all_records[:sample_size]
    print(f"  Using {len(records)} of {len(all_records)} sessions (sample)")

    baseline = play(records, bm25_ranker())
    base = score(baseline)

    print(f"\n{'=' * 96}")
    print(f"CHECKPOINT COMPARISON - DEPTH {DEPTH} (Rerank top-{DEPTH} to get top-10)")
    print(f"Ledger: {ledger} | Models: {len(MODELS)} | Sample: {sample_size}/{len(all_records)} sessions")
    print(f"Baseline TechnicalScore: {base['technical_score']:.6f}")
    print(f"{'=' * 96}")
    print(f"NOTE: Using 50-session sample for faster comparison (~75% time savings)")
    print(f"{'=' * 96}\n")

    # Header
    print(f"{'Model':<32} {'Tech':>9} {'Delta':>10} {'CI Excl.':>9} {'Hit@10':>8} "
          f"{'MRR':>9} {'s/turn':>8} {'Size(MB)':>10}")
    print(f"{'K0  BM25 baseline':<32} {base['technical_score']:>9.6f} {'--':>10} "
          f"{'--':>9} {base['hit_rate_at_10']:>8.3f} {base['mrr']:>9.6f} "
          f"{0.0:>8.2f} {'--':>10}")

    results_by_model = {}
    model_weights = {}

    for idx, (model_name, model_path) in enumerate(MODELS.items(), start=1):
        print(f"\n[{idx}/{len(MODELS)}] Testing {model_name} ({model_path})")

        # Initialize reranker
        reranker = CrossEncoderReranker(model_name, model_path, DEPTH)

        # Replay sessions with reranking
        sessions = play(records, reranker.rank)

        # Compute metrics
        s = score(sessions)
        p = paired(baseline, sessions)
        ci = bootstrap_delta_ci(baseline, sessions)
        by_scen = by_scenario(sessions)

        # Performance metrics
        per_turn = reranker.seconds / max(reranker.turns_reranked, 1)
        size = model_size_mb(model_path)
        model_weights[model_name] = round(size, 1)

        # Save cache
        reranker.save_cache()

        # Store results
        results_by_model[model_name] = {
            "score": s,
            "paired": p,
            "delta_ci": ci,
            "by_scenario": by_scen,
            "performance": {
                "pairs_scored": reranker.pairs_scored,
                "seconds_total": round(reranker.seconds, 2),
                "seconds_per_reranked_turn": round(per_turn, 4),
                "turns_reranked": reranker.turns_reranked,
                "weights_mb": round(size, 1),
            }
        }

        # Shipping criteria
        criteria = evaluate_shipping_criteria(results_by_model[model_name])
        results_by_model[model_name]["shipping_criteria"] = criteria

        # Print row
        delta = s['technical_score'] - base['technical_score']
        ci_excl = "Yes" if ci["excludes_zero"] else "No"
        label = f"K1{chr(96 + idx)}  {model_name}"
        if model_name == "minilm-l6":
            label += " (baseline)"

        print(f"{label:<32} {s['technical_score']:>9.6f} {delta:>+10.6f} "
              f"{ci_excl:>9} {s['hit_rate_at_10']:>8.3f} {s['mrr']:>9.6f} "
              f"{per_turn:>8.3f} {size:>10.1f}")

    # Comparison rankings
    ranked_by_tech = sorted(
        results_by_model.items(),
        key=lambda x: x[1]["score"]["technical_score"],
        reverse=True
    )
    ranked_by_latency = sorted(
        results_by_model.items(),
        key=lambda x: x[1]["performance"]["seconds_per_reranked_turn"]
    )
    shipping_candidates = [
        name for name, r in results_by_model.items()
        if r["shipping_criteria"]["passes_all"]
    ]

    # Print summary
    print(f"\n{'=' * 96}")
    print("SHIPPING CRITERIA (per part0-decision-rule.md §3)")
    print(f"{'=' * 96}")

    for criterion in ["ci_excludes_zero", "delta_gte_0_020", "regress_lte_5pct", "latency_lte_1s"]:
        passing = [name for name, r in results_by_model.items()
                   if r["shipping_criteria"][criterion]]
        label = {
            "ci_excludes_zero": "(a) CI excludes zero",
            "delta_gte_0_020": "(b) Delta >= +0.020",
            "regress_lte_5pct": "(c) Regress <= 5%",
            "latency_lte_1s": "(d) Latency <= 1.0s",
        }[criterion]
        print(f"  {label:<25}: {', '.join(passing) if passing else 'NONE'}")

    print(f"\n  PASSES ALL CRITERIA: {', '.join(shipping_candidates) if shipping_candidates else 'NONE'}")

    print(f"\n{'=' * 96}")
    print("RANKINGS")
    print(f"{'=' * 96}")
    print(f"  By TechnicalScore: {', '.join(name for name, _ in ranked_by_tech)}")
    print(f"  By Latency (fast): {', '.join(name for name, _ in ranked_by_latency)}")

    # Recommendation
    print(f"\n{'=' * 96}")
    print("RECOMMENDATION")
    print(f"{'=' * 96}")
    if shipping_candidates:
        # Pick the best TechnicalScore among shipping candidates
        best = max(
            [(name, results_by_model[name]) for name in shipping_candidates],
            key=lambda x: x[1]["score"]["technical_score"]
        )
        best_name, best_results = best
        print(f"  Model: {best_name}")
        print(f"  Rationale: Passes all shipping criteria with TechnicalScore "
              f"{best_results['score']['technical_score']:.6f} "
              f"(+{best_results['delta_ci']['mean_delta']:.6f} vs baseline)")
        print(f"  Trade-offs: {best_results['performance']['weights_mb']} MB, "
              f"{best_results['performance']['seconds_per_reranked_turn']:.3f}s/turn")
    else:
        print(f"  NONE - No model passes all shipping criteria at depth={DEPTH}")
        print(f"  Consider testing at shallower depths (10, 20) or accepting trade-offs")

    return {
        "ledger": ledger,
        "depth": DEPTH,
        "k0_baseline": base,
        "models": results_by_model,
        "model_weights_mb": model_weights,
        "comparison": {
            "ranked_by_technical_score": [name for name, _ in ranked_by_tech],
            "ranked_by_latency": [name for name, _ in ranked_by_latency],
            "shipping_criteria_met": shipping_candidates,
        }
    }


if __name__ == "__main__":
    ledgers = sys.argv[1:] if sys.argv[1:] else ["current", "legacy"]
    out = {ledger: report(ledger) for ledger in ledgers}

    path = ROOT / "bakeoff" / "results-checkpoint-comparison.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {path}")
