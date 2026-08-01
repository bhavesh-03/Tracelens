"""Causal attribution engine — finds the root cause of a failure in a trace.

Diagnostic pipeline:
  1. Build the execution DAG from the trace steps.
  2. Decompose each step's output into atomic factual claims.
  3. Verify each claim against its direct parent's outputs using ensemble NLI.
  4. Compute per-step attribution scores using a corrected formula that
     correctly penalises leaf hallucinations over root-agent noise.
  5. Return a ranked Diagnosis with the highest-scoring step as root cause.

Attribution formula (corrected from v0.1):

  Old (broken):
    score = novel_ratio × (1 + descendants_count / total_steps)
    Problem: This penalised root agents (most descendants) over leaf agents
             (0 descendants), even though leaf agents directly corrupt the
             final answer.

  New (Bayesian causal attribution):
    p_hallucinated = expected_ungrounded / total_claims   (from ensemble NLI)
    p_propagated   = overlap(step_claims, final_answer_claims) / final_answer_claims
    score          = p_hallucinated × (0.5 + 0.5 × p_propagated)

  Intuition:
    - p_propagated is HIGH for leaf agents (their output IS the final answer)
    - p_propagated is LOW for root agents (their output is diluted through many transforms)
    - This correctly directs the score toward leaf hallucinators.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

from tracelens.claims import decompose_into_claims
from tracelens.config import TraceLensConfig
from tracelens.dag import build_dag
from tracelens.schema import Claim, Diagnosis, StepAttribution, Trace
from tracelens.verify import verify_claim_ensemble

logger = logging.getLogger(__name__)


def _compute_p_propagated(
    step_claims: list[Claim],
    final_answer_claims: list[Claim],
) -> float:
    """Estimate how much of the final answer's content originated from this step.

    Uses a simple token-overlap heuristic: for each final-answer claim, check
    if any step claim shares significant content. Returns the fraction of
    final-answer claims that appear to come from this step.

    This is intentionally approximate — it does not require another LLM call.
    The goal is to weight leaf agents higher than root agents without extra cost.
    """
    if not final_answer_claims or not step_claims:
        return 0.0

    step_texts = {c.text.lower() for c in step_claims}
    matched = 0

    for fa_claim in final_answer_claims:
        fa_lower = fa_claim.text.lower()
        # Check substring overlap: if more than 40% of the final claim's words
        # appear in any step claim, count it as propagated.
        fa_words = set(fa_lower.split())
        for step_text in step_texts:
            step_words = set(step_text.split())
            overlap = len(fa_words & step_words) / len(fa_words) if fa_words else 0.0
            if overlap >= 0.40:
                matched += 1
                break

    return matched / len(final_answer_claims)


def compute_p_ungrounded(claim: Claim) -> float:
    """Probability that a claim is ungrounded, derived from the ensemble verdict."""
    if claim.verdict == "ungrounded":
        return claim.confidence
    elif claim.verdict == "grounded":
        return 1.0 - claim.confidence
    else:
        # uncertain — treat as weakly ungrounded
        return 0.5


def diagnose_trace(trace: Trace, config: TraceLensConfig) -> Diagnosis:
    """Run the full diagnostic pipeline on a trace to find the root cause.

    Returns a Diagnosis with all steps ranked by attribution score and the
    highest-scoring step identified as the root cause (if above threshold).
    """
    # 1. Build DAG
    dag = build_dag(trace)
    trace_steps = trace.steps

    # 2. Decompose the final answer into claims — used for p_propagated
    logger.info("Decomposing final answer into claims...")
    final_answer_claims = decompose_into_claims(
        trace.final_answer, step_id="__final_answer__", config=config
    )
    logger.info(f"Final answer decomposed into {len(final_answer_claims)} claims.")

    step_attributions: dict[str, StepAttribution] = {}
    step_claims_map: dict[str, list[Claim]] = {}

    # 3. For each step: decompose output → verify claims → score
    for step in trace_steps:
        logger.info(f"Evaluating step: {step.step_id} ({step.agent_name})")

        output_text = step.io.output_text or ""
        claims = decompose_into_claims(output_text, step.step_id, config)
        step_claims_map[step.step_id] = claims

        expected_ungrounded = 0.0
        novel_claims: list[Claim] = []

        for claim in claims:
            # Use ensemble NLI with focused evidence window
            verified = verify_claim_ensemble(claim, step, trace_steps, config)

            p_ung = compute_p_ungrounded(verified)
            expected_ungrounded += p_ung

            # Flag as novel hallucination only if the ensemble is confident it's ungrounded
            if verified.verdict == "ungrounded" and verified.confidence > 0.5:
                novel_claims.append(verified)

        total_claims = len(claims)
        p_hallucinated = (expected_ungrounded / total_claims) if total_claims > 0 else 0.0

        # 4. Compute p_propagated using token-overlap with final answer claims
        p_propagated = _compute_p_propagated(claims, final_answer_claims)

        # 5. Attribution score — correctly weights leaf agents over root agents
        # The 0.5 baseline ensures even steps with 0 propagation still get some score
        # when they hallucinate (they may contribute through intermediate steps)
        attribution_score = p_hallucinated * (0.5 + 0.5 * p_propagated)

        logger.info(
            f"  {step.agent_name}: p_hallucinated={p_hallucinated:.3f}, "
            f"p_propagated={p_propagated:.3f}, score={attribution_score:.4f}, "
            f"novel_claims={len(novel_claims)}"
        )

        step_attributions[step.step_id] = StepAttribution(
            step_id=step.step_id,
            agent_name=step.agent_name,
            step_type=step.step_type,
            attribution_score=round(attribution_score, 4),
            novel_claim_ratio=round(p_hallucinated, 4),
            downstream_impact=round(p_propagated, 4),
            novel_claims=novel_claims,
            total_claims=total_claims,
        )

    # 6. Find root cause: highest score above threshold
    threshold = config.attribution_threshold
    sorted_steps = sorted(
        step_attributions.values(),
        key=lambda x: x.attribution_score,
        reverse=True,
    )

    root_cause_attr = None
    for attr in sorted_steps:
        if attr.attribution_score >= threshold:
            root_cause_attr = attr
            break

    # 7. Build summary
    if root_cause_attr:
        vote_summary = ""
        if root_cause_attr.novel_claims:
            claim = root_cause_attr.novel_claims[0]
            breakdown = claim.vote_breakdown
            vote_summary = (
                f" (NLI votes: {breakdown}, "
                f"agreement: {claim.agreement_score:.0%})"
            )
        summary = (
            f"Agent '{root_cause_attr.agent_name}' (step: {root_cause_attr.step_id}) "
            f"is the root cause, introducing {len(root_cause_attr.novel_claims)} "
            f"unsupported claims with an attribution score of {root_cause_attr.attribution_score}"
            f"{vote_summary}."
        )
    else:
        summary = "No hallucination detected. All agent claims are grounded in their inputs."

    return Diagnosis(
        trace_id=trace.trace_id,
        root_cause_step=root_cause_attr,
        all_steps=sorted_steps,
        summary=summary,
        diagnosed_at=datetime.now(UTC).isoformat(),
    )
