"""Step-level Entailment Verification Engine with Ensemble NLI.

Uses an LLM judge to verify claims against evidence. Runs each claim through
the judge N times (configurable via `nli_ensemble_votes`) and takes a majority
vote, which corrects for LLM non-determinism and gives a calibrated confidence
score rather than a single-shot verdict.

Ensemble design:
  - N independent LLM calls per claim (default: 3)
  - Majority verdict wins (e.g. 2/3 "ungrounded" → "ungrounded")
  - Agreement score = fraction of votes matching the majority
  - Final confidence = mean(individual confidences) × agreement_score
  - If agreement < nli_min_agreement (default: 0.67), verdict is "uncertain"
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter

import litellm

from tracelens.config import TraceLensConfig
from tracelens.schema import Claim, TraceStep

logger = logging.getLogger(__name__)

# Rate limiter: tracks the minimum allowed time between LLM calls
_last_call_time = 0.0
_MIN_CALL_INTERVAL = 1.5  # seconds between calls (safe for 5 RPM free tier with some margin)


ENTAILMENT_SYSTEM_PROMPT = """You are a strict logical entailment checker (NLI judge).
Your job is to determine if a specific CLAIM is logically supported by the provided EVIDENCE.

RULES:
1. STRICT ENTAILMENT: The claim must be fully supported by the evidence. Do not use 
   outside knowledge. Even if the claim SEEMS plausible, if it is not in the evidence, 
   verdict is "ungrounded".
2. MINIMAL EVIDENCE SCOPE: Only use the evidence provided. Ignore everything else.
3. VERDICT OPTIONS:
   - "grounded": The evidence explicitly states or directly implies the claim.
   - "ungrounded": The evidence contradicts the claim, OR the evidence does not contain
     enough information to support the claim (hallucination by omission).
   - "uncertain": The claim is inherently subjective, a matter of opinion, or refers to
     future states that cannot be verified from the evidence.
4. OUTPUT FORMAT: Return a JSON object with exactly three keys:
   - "verdict": string ("grounded", "ungrounded", or "uncertain")
   - "evidence": string (short exact quote from the text that proves/disproves the claim,
     or "None" if hallucinated)
   - "confidence": float (your confidence in your verdict, 0.0 to 1.0)

EXAMPLE:
CLAIM: "The time complexity is O(N)."
EVIDENCE: "[Tool Output]: function uses a single for-loop, making it O(N)."

OUTPUT: {"verdict": "grounded", "evidence": "making it O(N)", "confidence": 0.97}
"""


def build_evidence_window(step: TraceStep, trace_steps: list[TraceStep]) -> str:
    """Build minimal, focused evidence for a step.

    Only includes the step's direct parents' outputs (not the full conversation
    history), which prevents the judge from finding superficial matches deep in
    a long context window and reporting false negatives.
    """
    # Build a lookup for fast parent resolution
    step_map = {s.step_id: s for s in trace_steps}
    evidence_parts = []

    # Check for single-parent (old schema) or multi-parent (new schema)
    parent_ids = getattr(step, "parent_span_ids", None)
    if not parent_ids:
        pid = getattr(step, "parent_step_id", None)
        parent_ids = [pid] if pid else []

    for pid in parent_ids:
        parent = step_map.get(pid)
        if not parent:
            continue
        if parent.step_type == "tool":
            evidence_parts.append(
                f"[Tool Output from {parent.agent_name}]:\n{parent.io.output_text}"
            )
        else:
            # Truncate long agent outputs to avoid polluting the evidence window
            truncated = (parent.io.output_text or "")[:3000]
            if len(parent.io.output_text or "") > 3000:
                truncated += "\n... [truncated]"
            evidence_parts.append(
                f"[Response from {parent.agent_name}]:\n{truncated}"
            )

        # Include parent's tool output if it is different from its text output
        if parent.io.tool_output:
            evidence_parts.append(
                f"[Tool Result used by {parent.agent_name}]:\n{parent.io.tool_output}"
            )

    if not evidence_parts:
        return "No parent context available (this is the root step)."

    return "\n\n---\n\n".join(evidence_parts)


def _single_verify(
    claim: Claim,
    evidence_text: str,
    config: TraceLensConfig,
    max_retries: int = 5,
) -> tuple[str, str, float]:
    """One NLI judge call with rate-limit-aware retry. Returns (verdict, evidence_quote, confidence)."""
    global _last_call_time
    user_prompt = f'CLAIM: "{claim.text}"\n\nEVIDENCE:\n{evidence_text}'

    for attempt in range(max_retries):
        # Throttle: ensure minimum interval between calls
        now = time.time()
        wait = _MIN_CALL_INTERVAL - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()

        try:
            response = litellm.completion(
                model=config.judge_model,
                messages=[
                    {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.temperature,
                response_format={"type": "json_object"},
                num_retries=0,  # we handle retries ourselves
            )
            raw = response.choices[0].message.content or ""
            clean = raw.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.endswith("```"):
                clean = clean[:-3]

            data = json.loads(clean.strip())
            verdict = data.get("verdict", "uncertain")
            if verdict not in ("grounded", "ungrounded", "uncertain"):
                verdict = "uncertain"
            evidence_quote = str(data.get("evidence", ""))
            confidence = float(data.get("confidence", 0.0))
            return verdict, evidence_quote, confidence

        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error and retry with backoff
            if "429" in error_str or "RateLimitError" in type(e).__name__:
                # Extract retry delay from error if available
                retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
                if retry_match:
                    wait_time = float(retry_match.group(1)) + 2  # add buffer
                else:
                    wait_time = 15 * (attempt + 1)  # escalating backoff
                logger.info(
                    f"Rate limited on claim '{claim.claim_id}' "
                    f"(attempt {attempt+1}/{max_retries}), waiting {wait_time:.0f}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"NLI judge call failed for claim '{claim.claim_id}': {e}")
                return "uncertain", f"Error: {e}", 0.0

    logger.warning(f"NLI judge exhausted retries for claim '{claim.claim_id}'")
    return "uncertain", "Error: rate limit exceeded after retries", 0.0


def verify_claim_ensemble(
    claim: Claim,
    step: TraceStep,
    trace_steps: list[TraceStep],
    config: TraceLensConfig,
) -> Claim:
    """Verify a claim using an ensemble of N independent NLI judge calls.

    This corrects for LLM non-determinism. The majority verdict across N runs
    is taken as the final answer. If fewer than `nli_min_agreement` of the
    judges agree, the verdict falls back to "uncertain".

    The final confidence score is:
        mean(individual confidences) × agreement_fraction

    This penalises confident but split verdicts — if 2/3 judges say "ungrounded"
    at 0.9 confidence but 1 says "grounded", the effective confidence drops to
    reflect the genuine ambiguity.

    Mutates and returns the passed Claim object.
    """
    n = config.nli_ensemble_votes
    evidence_text = build_evidence_window(step, trace_steps)
    votes: list[str] = []
    confidences: list[float] = []
    evidence_quotes: list[str] = []

    for i in range(n):
        verdict, ev_quote, conf = _single_verify(claim, evidence_text, config)
        votes.append(verdict)
        confidences.append(conf)
        evidence_quotes.append(ev_quote)
        logger.debug(
            f"  [vote {i+1}/{n}] claim={claim.claim_id} verdict={verdict} conf={conf:.2f}"
        )

    counts = Counter(votes)
    majority_verdict, majority_count = counts.most_common(1)[0]
    agreement = majority_count / n

    # If agreement is below threshold, we cannot be confident → uncertain
    if agreement < config.nli_min_agreement:
        final_verdict = "uncertain"
        logger.info(
            f"Claim '{claim.claim_id}' below agreement threshold "
            f"({agreement:.2f} < {config.nli_min_agreement}) → uncertain. "
            f"Votes: {dict(counts)}"
        )
    else:
        final_verdict = majority_verdict

    # Mean confidence × agreement fraction = calibrated confidence
    mean_conf = sum(confidences) / n
    calibrated_conf = round(mean_conf * agreement, 4)

    # Use the evidence quote from the majority-verdict runs
    majority_quotes = [
        ev for v, ev in zip(votes, evidence_quotes) if v == majority_verdict and ev and ev != "None"
    ]
    best_evidence = majority_quotes[0] if majority_quotes else "None"

    # Mutate the claim with ensemble results
    claim.verdict = final_verdict
    if final_verdict == "grounded":
        claim.grounded = True
    elif final_verdict == "ungrounded":
        claim.grounded = False
    else:
        claim.grounded = None
    claim.evidence = best_evidence
    claim.confidence = calibrated_conf
    claim.agreement_score = round(agreement, 4)
    claim.vote_breakdown = dict(counts)

    logger.info(
        f"Claim '{claim.claim_id}' | ensemble verdict: {final_verdict} "
        f"| conf: {calibrated_conf:.3f} | agreement: {agreement:.2f} | votes: {dict(counts)}"
    )
    return claim


# ---------------------------------------------------------------------------
# Legacy single-call interface (kept for backwards-compat in tests)
# ---------------------------------------------------------------------------

def verify_claim(
    claim: Claim,
    parent_step: TraceStep | None,
    config: TraceLensConfig,
) -> Claim:
    """Single-call NLI verify. Prefer verify_claim_ensemble for production use."""
    trace_steps = [parent_step] if parent_step else []
    # Build a fake step that points to the parent
    from dataclasses import dataclass
    from tracelens.schema import StepIO

    class _FakeStep:
        step_id = "__current__"
        step_type = "agent"
        agent_name = "unknown"
        parent_step_id = parent_step.step_id if parent_step else None
        parent_span_ids: list[str] = [parent_step.step_id] if parent_step else []
        io = StepIO()

    return verify_claim_ensemble(claim, _FakeStep(), trace_steps, config)
