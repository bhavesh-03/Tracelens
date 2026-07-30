"""Step-level Entailment Verification Engine.

Uses an LLM to judge if a given claim is supported by the context (evidence) available to the agent.
"""

import json
import logging

import litellm

from tracelens.config import TraceLensConfig
from tracelens.schema import Claim, TraceStep

logger = logging.getLogger(__name__)


ENTAILMENT_SYSTEM_PROMPT = """You are a strict logical entailment checker (NLI).
Your job is to determine if a specific CLAIM is logically supported by the provided EVIDENCE.

RULES:
1. STRICT ENTAILMENT: The claim must be fully supported by the evidence. Do not use 
   outside knowledge.
2. VERDICT OPTIONS:
   - "grounded": The evidence explicitly supports the claim, or the claim is a direct 
     logical consequence of the evidence.
   - "ungrounded": The evidence contradicts the claim, OR the evidence does not contain 
     enough information to support the claim (hallucination).
   - "uncertain": The claim is inherently unverifiable or subjective.
3. OUTPUT FORMAT: You must return a JSON object with exactly three keys:
   - "verdict": string ("grounded", "ungrounded", or "uncertain")
   - "evidence": string (a short, exact quote from the text that proves or disproves 
     the claim, or "None" if hallucinated)
   - "confidence": float (between 0.0 and 1.0)

EXAMPLE INPUT:
CLAIM: "The time complexity is O(N)."
EVIDENCE: "Code analysis complete. The function uses a single for-loop iterating over 
the array, making it O(N)."

EXAMPLE OUTPUT:
{
  "verdict": "grounded",
  "evidence": "uses a single for-loop iterating over the array, making it O(N)",
  "confidence": 1.0
}
"""

def verify_claim(claim: Claim, parent_step: TraceStep | None, config: TraceLensConfig) -> Claim:
    """Verifies a claim against the inputs and tool outputs of its parent step.
    
    Mutates and returns the passed Claim object with verdict, evidence, and confidence.
    """
    evidence_text = ""
    
    if parent_step:
        evidence_text += f"INPUT TO AGENT:\n{parent_step.io.input_text}\n\n"
        if parent_step.io.tool_output:
            evidence_text += f"TOOL OUTPUT RECEIVED:\n{parent_step.io.tool_output}\n\n"
    else:
        # If there's no parent step (e.g., this is the root node parsing user input),
        # the user's initial query is the only evidence.
        evidence_text = "No parent context available (Root level)."

    user_prompt = f"CLAIM: \"{claim.text}\"\n\nEVIDENCE:\n{evidence_text}"

    try:
        response = litellm.completion(
            model=config.judge_model,
            messages=[
                {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.temperature,
            response_format={"type": "json_object"},
        )
        
        raw_output = response.choices[0].message.content
        if not raw_output:
            raise ValueError("Empty response from LLM")
            
        clean_json = raw_output.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
            
        data = json.loads(clean_json.strip())
        
        claim.verdict = data.get("verdict", "uncertain")
        if claim.verdict not in ("grounded", "ungrounded", "uncertain"):
            claim.verdict = "uncertain"
            
        claim.grounded = (claim.verdict == "grounded")
        claim.evidence = data.get("evidence", "")
        claim.confidence = float(data.get("confidence", 0.0))
        
    except Exception as e:
        logger.error(f"Failed to verify claim '{claim.claim_id}': {e}")
        claim.verdict = "uncertain"
        claim.grounded = None
        claim.evidence = f"Error during verification: {e}"
        claim.confidence = 0.0

    return claim
