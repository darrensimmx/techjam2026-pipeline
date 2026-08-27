"""Runs the real evaluator against the real catalog and the full 200-session
public set, then appends one row to benchmarks/history.jsonl.

See benchmarks/README.md for the file formats and workflow this feeds.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CATALOG_GZ = DATA_DIR / "catalog.jsonl.gz"
CATALOG = DATA_DIR / "catalog.jsonl"
PUBLIC_SET = DATA_DIR / "public_set.jsonl"
HISTORY = REPO_ROOT / "benchmarks" / "history.jsonl"

sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate  # noqa: E402
from starter.agent import Agent  # noqa: E402


def ensure_catalog() -> None:
    if CATALOG.exists():
        return
    if not CATALOG_GZ.exists():
        raise SystemExit(f"missing {CATALOG_GZ} -- see data/README.md")
    with gzip.open(CATALOG_GZ, "rb") as src, CATALOG.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real evaluator and record the result")
    parser.add_argument("--label", required=True, help="short label, e.g. phase1-baseline")
    parser.add_argument("--catalog", default=str(CATALOG))
    parser.add_argument("--dataset", default=str(PUBLIC_SET))
    args = parser.parse_args()

    ensure_catalog()

    samples = load_jsonl(Path(args.dataset))
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    elapsed = round(time.time() - started, 1)

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "recommended_technical_score": result["recommended_technical_score"],
        "scenario_metrics": result["scenario_metrics"],
        "reported_token_usage": result["reported_token_usage"],
        "elapsed_seconds": elapsed,
    }

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    print(f"label:           {row['label']}")
    print(f"commit:          {row['commit']} ({row['branch']})")
    print(f"sample_count:    {row['sample_count']}")
    print(f"hit_rate_at_10:  {row['hit_rate_at_10']}")
    print(f"mrr:             {row['mrr']}")
    print(f"mttc:            {row['mttc']}")
    print(f"efficiency:      {row['efficiency']}")
    print(f"technical_score: {row['recommended_technical_score']}")
    print("scenario_metrics:")
    for name, metrics in sorted(row["scenario_metrics"].items()):
        print(
            f"  {name:16s} n={metrics['sample_count']:<4d} "
            f"hit_rate={metrics['hit_rate_at_10']:.4f} mrr={metrics['mrr']:.4f} mttc={metrics['mttc']}"
        )
    print(f"elapsed_seconds: {row['elapsed_seconds']}")
    print(f"\nAppended to {HISTORY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
