"""End-to-end TraceLens example for a checkout incident.

Run without an LLM to validate secure capture, redaction, and persistence:

    uv run python examples/checkout_incident_demo.py

To also run claim verification and graph-aware ranking, configure a judge key
and add --diagnose:

    export GOOGLE_API_KEY="..."
    uv run python examples/checkout_incident_demo.py --diagnose
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from tracelens.attribute import diagnose_trace
from tracelens.capture import TraceLensCapture
from tracelens.config import load_config
from tracelens.schema import Trace
from tracelens.store import connect, load_trace, save_diagnosis, save_trace


def build_checkout_trace() -> Trace:
    """Create a realistic trace with one deliberately unsupported diagnosis."""
    trace_id = f"checkout-demo-{uuid.uuid4().hex[:8]}"
    query = "Why did checkout fail for alice@example.com? Order ID: ORD-1042."
    tracer = TraceLensCapture(trace_id=trace_id, project_name="checkout-demo")

    router_id = tracer.add_step(
        agent_name="Router",
        step_type="router",
        input_text=query,
        output_text="Route incident to the payments specialist.",
    )
    tool_id = tracer.add_step(
        agent_name="GatewayStatusTool",
        step_type="tool",
        input_text="Get gateway status for ORD-1042",
        output_text="Gateway timeout after 30 seconds; no charge was created.",
        parent_step_id=router_id,
        tool_name="gateway_status",
        tool_args={"order_id": "ORD-1042"},
        tool_output="Gateway timeout after 30 seconds; no charge was created.",
    )
    payments_id = tracer.add_step(
        agent_name="PaymentsAgent",
        step_type="agent",
        input_text="Investigate gateway status for ORD-1042",
        output_text=(
            "The gateway timed out after 30 seconds and no charge was created. "
            "The checkout database is corrupted."
        ),
        parent_step_id=tool_id,
        model="gemini-2.5-flash",
    )
    tracer.add_step(
        agent_name="ResponseAgent",
        step_type="synthesizer",
        input_text="Prepare the customer-facing incident update.",
        output_text=(
            "Checkout failed because the gateway timed out. No charge was created. "
            "The checkout database is corrupted."
        ),
        parent_step_id=payments_id,
    )

    return tracer.finalize(
        query=query,
        final_answer=(
            "Checkout failed because the gateway timed out. No charge was created. "
            "The checkout database is corrupted."
        ),
        tags=["demo", "checkout", "incident"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a secure TraceLens checkout demo.")
    parser.add_argument("--db", type=Path, default=Path("checkout-demo.db"))
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run LLM claim verification and save the ranked review candidate.",
    )
    args = parser.parse_args()

    config = load_config()
    trace = build_checkout_trace()
    conn = connect(args.db)
    save_trace(conn, trace, config)
    stored = load_trace(conn, trace.trace_id)

    print(f"Saved trace: {trace.trace_id}")
    print(f"Database: {args.db}")
    print(f"Steps: {len(stored['steps'])}")
    print(f"Stored query: {stored['query']}")
    print("Redaction check: email addresses are replaced before persistence.")

    if not args.diagnose:
        print("\nCapture-only run complete. Re-run with --diagnose to call the configured judge.")
        return

    diagnosis = diagnose_trace(trace, config)
    save_diagnosis(conn, diagnosis)
    print(f"\nDiagnosis: {diagnosis.summary}")
    if diagnosis.root_cause_step:
        print("Ranked candidate:", diagnosis.root_cause_step.agent_name)
        for claim in diagnosis.root_cause_step.novel_claims:
            print(f"  - {claim.text}")


if __name__ == "__main__":
    main()
