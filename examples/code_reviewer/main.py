"""CLI entrypoint for the Subject Application."""

import argparse

from tracelens.capture import TraceLensCapture
from tracelens.config import load_config
from tracelens.store import connect, save_trace

from .agents import CoordinatorAgent

SAMPLE_CODE = """
def process_user_data(user_id, api_key="sk-1234567890abcdef"):
    # Connect to DB
    query = f"SELECT * FROM users WHERE id = {user_id}"
    print("Running query:", query)
    
    # Process items
    items = [1, 2, 3, 4, 5]
    for i in items:
        for j in items:
            print(i * j)
            
    return "done"
"""

def main():
    parser = argparse.ArgumentParser(description="Run the Code Reviewer Subject App")
    parser.add_argument("--project", type=str, default="code_reviewer", help="Project name for the trace")
    args = parser.parse_args()

    print(f"Initializing TraceLens Capture for project '{args.project}'...")
    tracer = TraceLensCapture(project_name=args.project)
    
    coordinator = CoordinatorAgent(tracer)
    
    print("\n--- Submitting code for review ---")
    print(SAMPLE_CODE)
    print("----------------------------------\n")
    
    # Run the multi-agent pipeline
    print("Agents are reviewing the code (this may take a few seconds)...")
    final_report = coordinator.run_review(SAMPLE_CODE)
    
    print("\n=== FINAL REVIEW REPORT ===")
    print(final_report)
    print("===========================\n")
    
    # Finalize the trace
    print("Finalizing trace...")
    trace = tracer.finalize(
        query=SAMPLE_CODE,
        final_answer=final_report,
        tags=["demo", "subject_app"]
    )
    
    print(f"Trace constructed with {len(trace.steps)} steps.")
    
    # Save to SQLite store
    config = load_config()
    conn = connect(config.db_path)
    save_trace(conn, trace)
    
    print(f"\n✅ Trace successfully saved to database '{config.db_path}'!")
    print(f"Trace ID: {trace.trace_id}")
    print("\nYou can now view this trace using:")
    print("  uv run tracelens report")
    print(f"  uv run tracelens diagnose {trace.trace_id}")

if __name__ == "__main__":
    main()
