"""Claim decomposition engine.

Uses an LLM to break down a block of text into a list of atomic factual claims.
"""

import json
import logging

import litellm

from tracelens.config import TraceLensConfig
from tracelens.schema import Claim

logger = logging.getLogger(__name__)


# The prompt used to instruct the LLM to extract claims.
CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a rigorous analytical engine. 
Your task is to take a block of text and decompose it into a list of atomic factual claims.

RULES:
1. ATOMICITY: Each claim must contain exactly ONE fact. If a sentence contains two facts, 
   split them into two claims.
2. COMPLETENESS: Capture all factual statements made in the text, including implicit assumptions.
3. CONTEXT: Each claim must make sense on its own. Resolve pronouns 
   (e.g. change "it ran fast" to "The algorithm ran fast").
4. IGNORE FILLER: Ignore greetings, opinions, polite filler, or structural text 
   (e.g. "Here is the report:").
5. OUTPUT FORMAT: You must output a raw JSON object with exactly one key "claims" 
   which maps to a list of strings. Do not use markdown blocks.

EXAMPLE INPUT:
"The security scanner ran successfully. It found 2 hardcoded secrets on lines 4 and 10, 
but the overall time complexity is O(N)."

EXAMPLE OUTPUT:
{
  "claims": [
    "The security scanner ran successfully.",
    "The code contains a hardcoded secret on line 4.",
    "The code contains a hardcoded secret on line 10.",
    "The time complexity of the code is O(N)."
  ]
}
"""


def decompose_into_claims(text: str, step_id: str, config: TraceLensConfig) -> list[Claim]:
    """Decompose a text block into a list of Claim objects using the configured judge model."""
    if not text.strip():
        return []

    try:
        response = litellm.completion(
            model=config.judge_model,
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Decompose this text into claims:\n\n{text}"}
            ],
            temperature=config.temperature,
            response_format={"type": "json_object"},
            num_retries=3,
        )
        
        raw_output = response.choices[0].message.content
        if not raw_output:
            return []
            
        # Clean up in case the LLM ignored the "no markdown" rule
        clean_json = raw_output.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
            
        data = json.loads(clean_json.strip())
        claim_texts = data.get("claims", [])
        
        # Enforce max claims to avoid exploding DAG complexity
        if len(claim_texts) > config.max_claims_per_step:
            logger.warning(
                f"Truncating claims from {len(claim_texts)} to {config.max_claims_per_step} "
                f"for step {step_id}."
            )
            claim_texts = claim_texts[:config.max_claims_per_step]

        claims = []
        for i, claim_str in enumerate(claim_texts):
            c_id = f"{step_id}_claim_{i:03d}"
            claims.append(
                Claim(
                    claim_id=c_id,
                    text=claim_str,
                    source_step_id=step_id,
                )
            )
        return claims

    except Exception as e:
        logger.error(f"Failed to decompose claims for step {step_id}: {e}")
        # Fallback: Treat the whole text as one big claim if the LLM fails.
        return [
            Claim(
                claim_id=f"{step_id}_claim_fallback",
                text=text.strip(),
                source_step_id=step_id
            )
        ]
