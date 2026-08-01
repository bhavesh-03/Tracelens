"""
TraceLens Sample Agent — End-to-End Integration Test

A simple 3-agent travel chatbot that demonstrates the full TraceLens stack:
  RouterAgent → FlightAgent + HotelAgent → SynthesizerAgent

This example uses the HTTP ingest API (fire-and-forget spans) rather than
importing the SDK directly, which is the recommended production pattern.

Prerequisites:
  1. Start the ingest server:
       tracelens serve

  2. In a separate terminal, run this script:
       uv run examples/travel_chatbot.py

  3. Open the dashboard to see traces appear live:
       tracelens dashboard
       → http://localhost:8501

The script runs 3 scenarios:
  - Scenario 1: Healthy — all agents respond truthfully
  - Scenario 2: Defective — HotelAgent hallucinates a cancellation policy
  - Scenario 3: Healthy — all agents respond truthfully
"""

from __future__ import annotations

import time
import uuid

from tracelens.integrations.http_client import TraceLensHTTPClient

TRACELENS_ENDPOINT = "http://localhost:4318"
PROJECT = "travel_chatbot"

# ---------------------------------------------------------------------------
# Simulated LLM responses
# ---------------------------------------------------------------------------

def router_agent(query: str) -> str:
    """Route the query to the correct specialist agents."""
    return (
        f"Query classified as TRAVEL_BOOKING. "
        f"Dispatching to FlightAgent and HotelAgent in parallel."
    )


def flight_agent(query: str) -> str:
    """Find flight options."""
    time.sleep(0.1)  # simulate LLM call
    return (
        "Found 3 flights from Mumbai to London:\n"
        "  - Air India AI101: Departs 08:00, arrives 14:30 (6h30m), ₹52,000\n"
        "  - British Airways BA117: Departs 10:15, arrives 16:45 (6h30m), ₹67,000\n"
        "  - Emirates EK500 via Dubai: Departs 07:00, arrives 20:00 (13h), ₹41,000\n"
        "Best value: Emirates EK500."
    )


def hotel_agent(query: str, hallucinate: bool = False) -> str:
    """Find hotel options — optionally hallucinates."""
    time.sleep(0.1)
    base = (
        "Found hotels in London:\n"
        "  - Premier Inn London City: £89/night, 4 stars, free WiFi, breakfast included\n"
        "  - Travelodge Kings Cross: £65/night, 3 stars, central location\n"
        "  - Hilton London Bankside: £210/night, 5 stars, Thames view\n"
    )
    if hallucinate:
        # HALLUCINATION: The cancellation policy was never in any tool output
        base += (
            "Note: All hotels listed have a strict 48-hour cancellation policy "
            "with a £200 penalty for late cancellations. "
            "The Hilton requires a 7-day advance notice for refunds."
        )
    else:
        base += "All hotels offer free cancellation up to 24 hours before check-in."
    return base


def synthesizer_agent(flight_result: str, hotel_result: str, query: str) -> str:
    """Merge flight and hotel results into a final answer."""
    time.sleep(0.1)
    return (
        f"Here's your travel plan for London:\n\n"
        f"✈️ FLIGHTS:\n{flight_result}\n\n"
        f"🏨 HOTELS:\n{hotel_result}\n\n"
        f"💡 Recommendation: Book Emirates EK500 + Premier Inn for the best value at ~₹52,000 + £89/night."
    )


# ---------------------------------------------------------------------------
# One trace execution
# ---------------------------------------------------------------------------

def run_trace(query: str, hallucinate: bool, client: TraceLensHTTPClient) -> str:
    """Run the travel chatbot and push spans to TraceLens."""
    trace_id = f"travel_{uuid.uuid4().hex[:10]}"
    print(f"\n{'='*60}")
    print(f"Running trace: {trace_id}")
    print(f"Query: {query}")
    print(f"Hallucinate: {hallucinate}")
    print(f"{'='*60}")

    t0 = time.time() * 1000

    # --- Step 1: Router ---
    router_start = time.time() * 1000
    router_out = router_agent(query)
    router_end = time.time() * 1000

    router_span_id = f"span_{uuid.uuid4().hex[:8]}"
    client.push_span(
        trace_id=trace_id,
        span_id=router_span_id,
        agent_name="RouterAgent",
        span_type="router",
        parent_span_ids=[],           # root step
        input_text=query,
        output_text=router_out,
        start_time_ms=router_start,
        end_time_ms=router_end,
        metadata={"model": "rule-based"},
    )
    print(f"  RouterAgent: OK")

    # --- Step 2a: FlightAgent (parallel child of router) ---
    flight_start = time.time() * 1000
    flight_out = flight_agent(query)
    flight_end = time.time() * 1000

    flight_span_id = f"span_{uuid.uuid4().hex[:8]}"
    client.push_span(
        trace_id=trace_id,
        span_id=flight_span_id,
        agent_name="FlightAgent",
        span_type="agent",
        parent_span_ids=[router_span_id],   # child of router
        input_text=query,
        output_text=flight_out,
        model="gemini-2.5-flash",
        start_time_ms=flight_start,
        end_time_ms=flight_end,
    )
    print(f"  FlightAgent: OK")

    # --- Step 2b: HotelAgent (parallel child of router) ---
    hotel_start = time.time() * 1000
    hotel_out = hotel_agent(query, hallucinate=hallucinate)
    hotel_end = time.time() * 1000

    hotel_span_id = f"span_{uuid.uuid4().hex[:8]}"
    client.push_span(
        trace_id=trace_id,
        span_id=hotel_span_id,
        agent_name="HotelAgent",
        span_type="agent",
        parent_span_ids=[router_span_id],   # child of router (parallel to FlightAgent)
        input_text=query,
        output_text=hotel_out,
        model="gemini-2.5-flash",
        start_time_ms=hotel_start,
        end_time_ms=hotel_end,
        metadata={"hallucination_injected": hallucinate},
    )
    print(f"  HotelAgent: OK {'(HALLUCINATION INJECTED)' if hallucinate else ''}")

    # --- Step 3: Synthesizer (fan-in: child of BOTH FlightAgent and HotelAgent) ---
    synth_start = time.time() * 1000
    synth_out = synthesizer_agent(flight_out, hotel_out, query)
    synth_end = time.time() * 1000

    synth_span_id = f"span_{uuid.uuid4().hex[:8]}"
    client.push_span(
        trace_id=trace_id,
        span_id=synth_span_id,
        agent_name="SynthesizerAgent",
        span_type="synthesizer",
        parent_span_ids=[flight_span_id, hotel_span_id],  # ← fan-in! Two parents.
        input_text=f"Flight results:\n{flight_out}\n\nHotel results:\n{hotel_out}",
        output_text=synth_out,
        model="gemini-2.5-flash",
        start_time_ms=synth_start,
        end_time_ms=synth_end,
    )
    print(f"  SynthesizerAgent: OK")

    # --- Finalize: tell TraceLens this trace is complete ---
    result = client.finalize(
        trace_id=trace_id,
        query=query,
        final_answer=synth_out,
        tags=[PROJECT, "defective" if hallucinate else "healthy"],
        run_diagnosis=True,
    )
    print(f"\n  ✅ Trace finalized: {result}")
    print(f"  ℹ️  Diagnosis running in background — check dashboard in ~30s")
    return trace_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("🧳 TraceLens Travel Chatbot — Sample Integration Test")
    print("Connecting to TraceLens ingest API at", TRACELENS_ENDPOINT)

    try:
        import httpx
        r = httpx.get(f"{TRACELENS_ENDPOINT}/v1/health", timeout=3)
        r.raise_for_status()
        print("  ✅ API is running\n")
    except Exception as e:
        print(f"\n  ❌ Could not connect to TraceLens API: {e}")
        print("  → Start the server first: tracelens serve")
        return

    with TraceLensHTTPClient(
        endpoint=TRACELENS_ENDPOINT,
        project_name=PROJECT,
    ) as client:
        # Scenario 1: Healthy
        run_trace(
            query="I need to fly from Mumbai to London next month and need a hotel for 5 nights.",
            hallucinate=False,
            client=client,
        )

        time.sleep(1)

        # Scenario 2: Defective — HotelAgent hallucinates cancellation policy
        run_trace(
            query="Book me a business trip to London, 3 nights, flexible cancellation please.",
            hallucinate=True,
            client=client,
        )

        time.sleep(1)

        # Scenario 3: Healthy again
        run_trace(
            query="What's the cheapest way to get to London and stay for a week?",
            hallucinate=False,
            client=client,
        )

    print("\n\n🎉 All traces submitted!")
    print("Open the dashboard to see them: http://localhost:8501")
    print("Select project 'travel_chatbot' from the sidebar.")
    print("Diagnosis runs in the background — allow 1-2 minutes per trace.")


if __name__ == "__main__":
    main()
