"""Causal attribution engine.

Uses the Trace DAG, claim decomposition, and step-level verification
to mathematically determine the root cause of a failure.
"""

import logging
from datetime import UTC

from tracelens.claims import decompose_into_claims
from tracelens.config import TraceLensConfig
from tracelens.dag import build_dag, descendants
from tracelens.schema import Claim, Diagnosis, StepAttribution, Trace
from tracelens.verify import verify_claim

logger = logging.getLogger(__name__)


def compute_p_ungrounded(claim: Claim) -> float:
    """Computes the probability that a claim is ungrounded (hallucinated) 
    based on the NLI judge's verdict and confidence.
    """
    if claim.verdict == "ungrounded":
        return claim.confidence
    elif claim.verdict == "grounded":
        # If the judge is 90% sure it is grounded, there is a 10% chance it is ungrounded.
        return 1.0 - claim.confidence
    else:
        # Uncertain
        return 0.5


def diagnose_trace(trace: Trace, config: TraceLensConfig) -> Diagnosis:
    """Runs the full diagnostic pipeline on a trace to find the root cause."""
    
    # 1. Build the DAG
    dag = build_dag(trace)
    
    step_attributions: dict[str, StepAttribution] = {}
    all_evaluated_claims: list[Claim] = []
    
    # We evaluate every step in the DAG.
    for step in trace.steps:
        logger.info(f"Diagnosing step: {step.step_id} ({step.agent_name})")
        
        # Step A: Decompose the step's output into claims
        output_text = step.io.output_text or ""
        claims = decompose_into_claims(output_text, step.step_id, config)
        
        expected_ungrounded = 0.0
        novel_claims = []
        
        # Step B: Verify each claim against the step's inputs
        # To do this correctly, we need the parent step's output (which became this step's input).
        # In TraceLens, the `step.io.input_text` already contains the context it was given.
        parent_step = None
        if step.parent_step_id:
            parent_step = trace.get_step(step.parent_step_id)
            
        for claim in claims:
            verified_claim = verify_claim(claim, parent_step, config)
            all_evaluated_claims.append(verified_claim)
            
            p_ung = compute_p_ungrounded(verified_claim)
            expected_ungrounded += p_ung
            
            # If it's highly likely to be ungrounded, record it as a novel hallucination
            if p_ung > 0.5:
                novel_claims.append(verified_claim)
                
        # Calculate mathematical hallucination ratio for this step
        total_claims = len(claims)
        novel_ratio = (expected_ungrounded / total_claims) if total_claims > 0 else 0.0
        
        # Calculate downstream impact (number of dependent steps)
        # A hallucination early in a 10-step chain is worse than one at a leaf.
        descendants_list = descendants(dag, step.step_id)
        impact = len(descendants_list) / len(trace.steps) if len(trace.steps) > 0 else 0.0
        
        # Attribution score combines the severity of the hallucination with its blast radius.
        # This prevents penalizing harmless off-topic chatter at a leaf node.
        # But if we just want to find who lied, novel_ratio is the primary metric.
        attribution_score = novel_ratio * (1.0 + impact)
        
        attr = StepAttribution(
            step_id=step.step_id,
            agent_name=step.agent_name,
            step_type=step.step_type,
            attribution_score=round(attribution_score, 4),
            novel_claim_ratio=round(novel_ratio, 4),
            downstream_impact=round(impact, 4),
            novel_claims=novel_claims,
            total_claims=total_claims
        )
        step_attributions[step.step_id] = attr

    from datetime import datetime
    
    # Find the StepAttribution object with the highest score
    root_cause_attr = None
    max_score = -1.0
    threshold = config.attribution_threshold
    
    for attr in step_attributions.values():
        if attr.attribution_score > max_score and attr.attribution_score >= threshold:
            max_score = attr.attribution_score
            root_cause_attr = attr
            
    # Sort all steps by score descending
    sorted_steps = sorted(
        step_attributions.values(), 
        key=lambda x: x.attribution_score, 
        reverse=True
    )
    
    summary = "No hallucination detected."
    if root_cause_attr:
        summary = (
            f"Agent '{root_cause_attr.agent_name}' (step: {root_cause_attr.step_id}) "
            f"is the root cause, introducing {len(root_cause_attr.novel_claims)} "
            f"unsupported claims with an attribution score of {root_cause_attr.attribution_score}."
        )
            
    return Diagnosis(
        trace_id=trace.trace_id,
        root_cause_step=root_cause_attr,
        all_steps=sorted_steps,
        summary=summary,
        diagnosed_at=datetime.now(UTC).isoformat()
    )
