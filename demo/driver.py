"""Drive one session exactly the way the organizer's evaluator does.

This is a deliberate, line-for-line copy of
``evaluator/local_evaluator.py:226-276``, with the source line cited on every
branch. It exists only because ``evaluate()`` is all-or-nothing over N samples
and returns aggregate metrics -- there is no per-turn seam to hook, and a demo
needs one.

The risk a copy creates is DRIVE-LOOP DRIFT: a demo that shows a conversation
the scorer would never produce. Two things hold it down. Every simulated-customer
helper is imported from the vendored module and used unmodified -- we reproduce
the loop, never the customer. And ``tests/test_demo_trace.py`` runs this driver
and the real ``evaluate()`` over the same samples and asserts identical
``(hit, first_hit_turn, best_rank)``; both are deterministic, since
``materialize_hidden_fields`` seeds from ``sample_id\\0scenario_type`` and the
session id does not affect behaviour.

``evaluator/`` itself is never edited. The leaky/scrubbed bracket is applied by
``scripts.evaluate_src.bracket()``, which monkeypatches the imported module
object and restores it on exit -- the technique CLAUDE.md blesses for exactly
this.
"""
from __future__ import annotations

import uuid

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)

# The eight f-strings the simulated customer can emit -- its entire vocabulary,
# and exactly what src/frames.py is anchored against. Labelled here so the
# render can name the template that produced each utterance.
TEMPLATES = (
    ("buying_open", "local_evaluator.py:161", "A key requirement is:"),
    ("browsing_open", "local_evaluator.py:163", "but I'm still exploring."),
    ("refusal", "local_evaluator.py:170", "I don't have a preference for"),
    ("null_nudge", "local_evaluator.py:173", "Those options are not quite right yet"),
    ("exhaustion", "local_evaluator.py:184", "I don't have an additional preference for"),
    ("disclosure", "local_evaluator.py:185", "For that, what matters is:"),
    ("override", "local_evaluator.py:85", "ignore my earlier preference"),
)


def classify_template(message: str, turn: int) -> tuple:
    """Name the template that produced ``message``. Presentation only."""
    text = str(message or "")
    for name, where, marker in TEMPLATES:
        if marker.lower() in text.lower():
            return name, where
    if turn == 1 and text.lower().startswith("i'm looking for"):
        # Frame 2: "I'm looking for {category}. {old_value}" -- the override
        # opener has no marker of its own, only the absence of the other two.
        return "override_open", "local_evaluator.py:162"
    return "unknown", ""


def session_plan(sample: dict, categories: dict, products: dict) -> dict:
    """Everything the evaluator decides before turn 1, made visible.

    None of this is ever shown to the agent; it is the demo's ground truth
    panel, and every field carries ``visible_to_agent: False``.
    """
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)          # :204
    effective = {**sample, "intent_card": card, "behavior": behavior}     # :230
    disclosed: set = set()
    category = coarse_category(categories.get(target, []))                # :232
    opening = initial_message(effective, category, disclosed)             # :232
    product = products.get(target, {})
    return {
        "effective_sample": effective,
        "target": target,
        "target_title": str(product.get("title") or ""),
        "target_price": product.get("price"),
        "coarse_category": category,
        "opening_message": opening,
        "opening_disclosed": sorted(disclosed),
        "card": card,
        "override": (behavior.get("override") or None),
    }


def run_session(agent, sample: dict, catalog_ids: set, categories: dict,
                products: dict, on_turn=None, session_id: str = "") -> dict:
    """One session, up to ten turns. Returns the same summary ``evaluate()`` does.

    ``on_turn(payload)`` fires after every turn with the customer utterance, the
    agent's response, the ranked list and the hit state -- the hook the vendored
    loop does not have.
    """
    plan = session_plan(sample, categories, products)
    effective_sample = plan["effective_sample"]
    target = plan["target"]

    session_id = session_id or ("demo_" + uuid.uuid4().hex)               # :227
    agent.reset(session_id, sample["user_profile"])                       # :228

    disclosed: set = set()
    boundary_used = False
    # An intent_override session's hit check is OFF until the override lands
    # (:234). Everything shown before then is not counted and, per
    # src/shown.py, goes back in play.
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective_sample, plan["coarse_category"], disclosed)             # :232

    hit_turn = None
    best_rank = None
    turns_run = 0
    stop_reason = "max_turns"

    for turn in range(1, MAX_TURNS + 1):                                  # :238
        template, where = classify_template(user_message, turn)
        try:                                                              # :239
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception:                                                 # :241
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            response = {"message": "", "ask_attribute": None, "recommendations": []}  # :243

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)  # :250
        turns_run = turn

        target_rank = (ranked.index(target) + 1) if target in ranked else None
        counted = bool(override_applied and target_rank is not None)      # :251
        if counted:
            best_rank = target_rank
            hit_turn = turn

        if on_turn is not None:
            on_turn({
                "turn": turn,
                "user_message": user_message,
                "evaluator_template": template,
                "evaluator_template_line": where,
                "response": response,
                "ranked": ranked,
                "target": target,
                "target_rank": target_rank,
                "hit_counted": counted,
                # The demo's best moment: the target is right there, and the
                # evaluator is not looking yet.
                "hit_suppressed_by_override": bool(
                    target_rank is not None and not override_applied),
                "override_applied": override_applied,
                "disclosed": sorted(disclosed),
            })

        if counted:                                                       # :251-254
            stop_reason = "hit"
            break
        if turn == MAX_TURNS:                                             # :255
            break

        override = effective_sample.get("behavior", {}).get("override") or {}   # :257
        if not override_applied and turn + 1 == int(override.get("turn", 3)):   # :258
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get(
                "message", "Actually, please ignore my earlier preference."))    # :264
        else:
            user_message, boundary_used = customer_reply(                       # :266
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used)

    return {
        "sample_id": sample["sample_id"],                                 # :269
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "turns_run": turns_run,
        "stop_reason": stop_reason,
        "plan": plan,
    }


def summarise(sessions: list) -> dict:
    """The evaluator's own metric maths, over however many sessions we ran.

    ``metric_summary`` and the score weights come from the vendored module, so
    a demo headline cannot drift from a scored one. It is still only ``n``
    sessions, and every caller must print the bracket alongside.
    """
    from evaluator.local_evaluator import metric_summary

    overall = metric_summary(sessions)
    if not sessions:
        return {**overall, "efficiency": 0.0, "recommended_technical_score": 0.0}
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))   # :281
    score = (0.50 * overall["hit_rate_at_10"]
             + 0.30 * overall["mrr"]
             + 0.20 * efficiency)                                             # :282
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }
