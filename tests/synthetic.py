"""FROZEN. Shared synthetic catalog + session builder for the src/ test suite.

Why this exists: tests/fixtures/catalog.jsonl has SIX products against top_k=10,
so any query matching one term returns the whole fixture. A test built on it
passes even with a query-blind ranker -- it cannot detect the single most
important class of retrieval bug. Every src/ test that needs to prove ranking
uses this builder instead, into a tempfile.

It is frozen and shared so three parallel workstreams do not invent three
incompatible fixtures.

Everything is deterministic: same seed, same catalog, byte for byte.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

# Drawn from the evaluator's own MATERIAL_RE / COLOR_RE so that intent_card()
# can actually manufacture a material and a color constraint for any target.
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "yellow")
ITEMS = ("boots", "jacket", "gloves", "scarf", "trousers", "shirt", "belt", "hat", "socks", "vest")
STYLES = ("formal", "casual", "slim fit", "relaxed fit", "crew neck", "long sleeve")
USES = ("hiking", "running", "gym", "winter", "outdoor", "work")
STORES = ("TrailForge", "NorthPeak", "Fieldmark", "Harborline", "Vantage Co")

SEED = 20260831


def build_products(n: int = 250, planted: Sequence[dict] = (), seed: int = SEED) -> list[dict]:
    """A deterministic, lexically varied catalog.

    Variety is the point: material x color alone narrows 250 products to a
    handful, so a ranker that ignores the query genuinely fails these tests.
    Planted products are appended verbatim and keep whatever fields they carry.
    """
    rng = random.Random(seed)
    products: list[dict] = []
    for index in range(n):
        material = MATERIALS[index % len(MATERIALS)]
        color = COLORS[(index // len(MATERIALS)) % len(COLORS)]
        item = ITEMS[index % len(ITEMS)]
        style = rng.choice(STYLES)
        use = rng.choice(USES)
        store = rng.choice(STORES)
        products.append({
            "parent_asin": f"S{index:05d}",
            "title": f"{color.title()} {material.title()} {item.title()}",
            "categories": ["Clothing", "Shoes & Jewelry", item.title()],
            "features": [f"{material} upper", f"{style} cut", f"designed for {use}"],
            "details": {"color": color, "size": str(6 + index % 10), "department": style},
            "store": store,
            "description": f"A {style} {color} {material} {item} built for {use}.",
            "price": round(15.0 + (index % 40) * 2.5, 2),
        })
    products.extend(dict(item) for item in planted)
    return products


def rare_product(parent_asin: str = "RARE0001") -> dict:
    """One product carrying a bigram that appears nowhere else in the catalog.

    Used for the discriminating rank test: a query for "chartreuse alpaca" must
    put this at rank 1 out of 250, which a query-blind ranker cannot do.
    """
    return {
        "parent_asin": parent_asin,
        "title": "Chartreuse Alpaca Cloak",
        "categories": ["Clothing", "Outerwear"],
        "features": ["chartreuse alpaca weave", "hand-loomed", "designed for winter"],
        "details": {"color": "green", "size": "M", "department": "formal"},
        "store": "Vantage Co",
        "description": "A hand-loomed chartreuse alpaca cloak.",
        "price": 249.0,
    }


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> Path:
    """Write rows as JSONL and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return target


def build_catalog(path: str | Path, n: int = 250, planted: Sequence[dict] = (),
                  seed: int = SEED) -> list[dict]:
    """Build the catalog AND write it to `path`. Returns the products."""
    products = build_products(n=n, planted=planted, seed=seed)
    write_jsonl(path, products)
    return products


def profile_for(index: int) -> dict:
    """A user_profile with exactly the five fields the contract requires."""
    return {
        "purchase_frequency": f"{1 + index % 5}-{2 + index % 5} prior purchases",
        "average_prior_rating": 4.0 + (index % 10) / 10.0,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort", "durability"],
        "summary": "Prior purchases emphasize fit, comfort, durability.",
    }


def build_samples(products: Sequence[dict],
                  scenarios: Sequence[str] = ("buying", "browsing", "boundary", "intent_override"),
                  per_scenario: int = 1) -> list[dict]:
    """Sessions in the public_set.jsonl schema.

    Deliberately carries NO `intent_card` and NO `behavior`, exactly like the
    real public set -- so the evaluator takes its materialize_hidden_fields()
    fallback and builds the hidden card out of the target's own listing, which
    is the behaviour our agent actually faces.
    """
    samples: list[dict] = []
    index = 0
    for scenario in scenarios:
        for _ in range(per_scenario):
            product = products[(index * 7) % len(products)]
            samples.append({
                "sample_id": f"synthetic_{index:04d}",
                "scenario_type": scenario,
                "category_bucket": "clothing",
                "difficulty_bucket": "easy",
                "ground_truth": {"parent_asin": str(product["parent_asin"])},
                "user_profile": profile_for(index),
            })
            index += 1
    return samples
