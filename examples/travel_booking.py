"""
Interactive Multi-Agent Travel Booking — with Hallucination Detection.

This example demonstrates TraceLens catching a hallucinating agent in a
realistic multi-agent travel booking pipeline.

Architecture:
    User Input
        └──> OrchestratorAgent (root)
                ├──> ResearchAgent    — researches the destination (HONEST)
                ├──> HotelAgent       — recommends hotels (HALLUCINATES!)
                ├──> FlightAgent      — finds flights (HONEST)
                └──> SynthesizerAgent — builds final itinerary

The HotelAgent deliberately injects false claims that were NOT in its
parent's (ResearchAgent's) output. TraceLens's NLI engine will catch this
and flag the HotelAgent as the root cause.
"""

from tracelens.capture import TraceLensCapture
from tracelens.store import connect, save_trace

tracer = TraceLensCapture(project_name="travel_booking_v2")

# ---------------------------------------------------------------------------
# Knowledge bases — simulated "tool" outputs grounding each agent
# ---------------------------------------------------------------------------

DESTINATION_DATA = {
    "beach": {
        "name": "Goa, India",
        "description": (
            "Goa is a coastal state in western India known for its pristine beaches, "
            "Portuguese-era architecture, and vibrant nightlife. The average temperature "
            "is 28°C year-round. Popular beaches include Calangute, Baga, and Anjuna. "
            "The local cuisine features fresh seafood and Goan fish curry."
        ),
    },
    "mountain": {
        "name": "Manali, Himachal Pradesh",
        "description": (
            "Manali is a hill station nestled in the Himalayas at an altitude of 2,050m. "
            "Known for adventure sports like paragliding, trekking, and river rafting. "
            "The Solang Valley and Rohtang Pass are major attractions. Winter temperatures "
            "drop to -5°C with heavy snowfall. The town has a mix of Tibetan and Indian culture."
        ),
    },
    "city": {
        "name": "Jaipur, Rajasthan",
        "description": (
            "Jaipur, the Pink City, is the capital of Rajasthan. Famous for Hawa Mahal, "
            "Amber Fort, and the City Palace. Known for traditional Rajasthani cuisine "
            "including dal baati churma and ghevar. The city has a thriving handicraft "
            "market at Johari Bazaar. Summer temperatures reach 45°C."
        ),
    },
}

HOTEL_DATA = {
    "goa, india": {
        "name": "Taj Fort Aguada Resort & Spa",
        "facts": (
            "Located on Sinquerim Beach. 145 rooms with sea-facing balconies. "
            "Features an outdoor infinity pool, Jiva Spa, and three restaurants. "
            "Room rates start at ₹12,000 per night. Free airport shuttle available."
        ),
    },
    "manali, himachal pradesh": {
        "name": "The Himalayan Resort",
        "facts": (
            "Located on the banks of the Beas River. 80 rooms with mountain views. "
            "Features a heated indoor pool, bonfire area, and one multi-cuisine restaurant. "
            "Room rates start at ₹8,500 per night. Offers guided trekking packages."
        ),
    },
    "jaipur, rajasthan": {
        "name": "Rambagh Palace",
        "facts": (
            "A former royal residence converted into a heritage hotel. 79 rooms with "
            "palatial decor. Features manicured Mughal gardens, a croquet lawn, and "
            "fine dining at Suvarna Mahal. Room rates start at ₹25,000 per night."
        ),
    },
}

# These are FAKE claims the HotelAgent will inject — NOT in the source data
HALLUCINATED_HOTEL_CLAIMS = {
    "goa, india": (
        "The resort also offers complimentary private yacht tours along the "
        "Goan coastline and has an exclusive underwater glass-floor restaurant "
        "with a Michelin-star chef."
    ),
    "manali, himachal pradesh": (
        "The resort also features a private cable car to a mountaintop "
        "observatory and an in-house ice skating rink open year-round."
    ),
    "jaipur, rajasthan": (
        "The palace also includes a private airstrip for guest arrivals "
        "and an underground museum with original crown jewels on display."
    ),
}

FLIGHT_DATA = {
    "goa, india": "Direct flight from Mumbai (BOM) to Goa (GOI). Duration: 1h 15m. Fare: ₹4,500. Airlines: IndiGo, Air India.",
    "manali, himachal pradesh": "Flight from Delhi (DEL) to Kullu-Manali (KUU). Duration: 1h 30m. Fare: ₹6,200. Airlines: Alliance Air. Then 1h taxi to Manali.",
    "jaipur, rajasthan": "Direct flight from Mumbai (BOM) to Jaipur (JAI). Duration: 1h 50m. Fare: ₹5,100. Airlines: IndiGo, SpiceJet.",
}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def research_agent(destination_type: str) -> str:
    """Researches the destination. HONEST — uses only grounded data."""
    with tracer.step("ResearchAgent", step_type="agent", input_text=destination_type) as io:
        data = DESTINATION_DATA.get(destination_type.lower(), DESTINATION_DATA["beach"])
        output = (
            f"Destination: {data['name']}. {data['description']}"
        )
        io.output_text = output
        print(f"\n[ResearchAgent] Researched: {data['name']}")
        return output


def hotel_agent(research_output: str) -> str:
    """Recommends hotels. HALLUCINATES — injects false claims!"""
    with tracer.step("HotelAgent", step_type="agent", input_text=research_output) as io:
        # Extract destination name from research output
        dest_key = None
        for key in HOTEL_DATA:
            if key.lower() in research_output.lower():
                dest_key = key
                break
        dest_key = dest_key or "goa, india"

        hotel = HOTEL_DATA[dest_key]
        hallucination = HALLUCINATED_HOTEL_CLAIMS[dest_key]

        # Mix real facts WITH hallucinated claims — this is what a faulty LLM would do
        output = (
            f"Recommended Hotel: {hotel['name']}. {hotel['facts']} "
            f"{hallucination}"
        )
        io.output_text = output
        print(f"[HotelAgent] Recommended: {hotel['name']} (⚠️  contains hallucinated claims)")
        return output


def flight_agent(research_output: str) -> str:
    """Finds flights. HONEST — uses only grounded data."""
    with tracer.step("FlightAgent", step_type="agent", input_text=research_output) as io:
        dest_key = None
        for key in FLIGHT_DATA:
            if key.lower() in research_output.lower():
                dest_key = key
                break
        dest_key = dest_key or "goa, india"

        output = f"Flight: {FLIGHT_DATA[dest_key]}"
        io.output_text = output
        print(f"[FlightAgent] Found flights to {dest_key.title()}")
        return output


def synthesizer_agent(research: str, hotel: str, flight: str) -> str:
    """Combines all agent outputs into a final itinerary."""
    combined_input = f"Research: {research}\nHotel: {hotel}\nFlight: {flight}"
    with tracer.step("SynthesizerAgent", step_type="synthesizer", input_text=combined_input) as io:
        output = (
            f"Here is your complete travel itinerary:\n\n"
            f"📍 {research}\n\n"
            f"🏨 {hotel}\n\n"
            f"✈️ {flight}\n\n"
            f"Have a wonderful trip!"
        )
        io.output_text = output
        return output


def orchestrator():
    """Main orchestrator — routes user input through the agent pipeline."""
    with tracer.step("OrchestratorAgent", step_type="router", input_text="Start travel booking") as io:
        print("\n" + "=" * 60)
        print("   Welcome to the Multi-Agent Travel Booker v2")
        print("   (with hallucination detection)")
        print("=" * 60)

        print("\nWhere would you like to go?")
        print("  1. Beach")
        print("  2. Mountain")
        print("  3. City")
        choice = input("\nYour choice (1/2/3): ").strip()
        dest_map = {"1": "beach", "2": "mountain", "3": "city"}
        destination_type = dest_map.get(choice, "beach")

        print(f"\nGreat! Planning a {destination_type} trip for you...\n")
        print("-" * 40)

        # Run the agent pipeline
        research = research_agent(destination_type)
        hotel = hotel_agent(research)
        flight = flight_agent(research)

        print("-" * 40)
        final = synthesizer_agent(research, hotel, flight)

        print(f"\n{final}")
        io.output_text = final
        return final


if __name__ == "__main__":
    final_plan = orchestrator()

    trace = tracer.finalize(query="I want to book a trip", final_answer=final_plan)

    db = connect("tracelens.db")
    save_trace(db, trace)

    print("\n" + "=" * 60)
    print("✅ Trace saved!")
    print(f"   Trace ID: {trace.trace_id}")
    print(f"\n   Now run: tracelens diagnose {trace.trace_id}")
    print("   Then check the dashboard: tracelens dashboard")
    print("=" * 60)
