"""DAG builder — converts a flat list of TraceSteps into a networkx DiGraph.

The DAG represents information flow: edges go from parent steps (information
source) to child steps (information consumers). This structure allows backward
traversal from the final answer to the root cause.

Key operations:
- ancestors(step_id): All upstream steps that influenced this step.
- descendants(step_id): All downstream steps that this step influenced.
- leaves: Terminal steps whose outputs feed into the final answer.
- path_to_root(step_id): The chain of context from root to this step.
"""

from __future__ import annotations

import networkx as nx

from tracelens.schema import Trace


def build_dag(trace: Trace) -> nx.DiGraph:
    """Build a directed acyclic graph from trace steps.

    Nodes carry the full TraceStep data as attributes.
    Edges go from parent → child (direction of information flow).

    Raises ValueError if the resulting graph contains a cycle.
    """
    g = nx.DiGraph()

    for step in trace.steps:
        g.add_node(
            step.step_id,
            agent_name=step.agent_name,
            step_type=step.step_type,
            input_text=step.io.input_text,
            output_text=step.io.output_text,
            model=step.io.model,
            tool_name=step.io.tool_name,
            tool_args=step.io.tool_args,
            tool_output=step.io.tool_output,
            timestamp_ms=step.timestamp_ms,
            duration_ms=step.duration_ms,
            metadata=step.metadata,
        )

    for step in trace.steps:
        for parent_id in step.parent_span_ids:
            if parent_id in g:
                g.add_edge(parent_id, step.step_id)

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Trace contains a cycle — cannot build a DAG")

    return g


def get_root(g: nx.DiGraph) -> str:
    """Return the primary root node (in-degree 0).

    In fan-out pipelines there may technically be a single root that fans out.
    Raises if there are no roots at all.
    """
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    if not roots:
        raise ValueError("Graph has no root node (no node with in-degree 0)")
    return roots[0]  # primary root


def get_leaves(g: nx.DiGraph) -> list[str]:
    """Return all leaf nodes (out-degree 0) — the terminal outputs."""
    return [n for n in g.nodes if g.out_degree(n) == 0]


def ancestors(g: nx.DiGraph, step_id: str) -> list[str]:
    """All upstream steps that influenced this step, in topological order."""
    anc = nx.ancestors(g, step_id)
    # Return in topological order (root first) for readable output.
    topo = list(nx.topological_sort(g))
    return [n for n in topo if n in anc]


def descendants(g: nx.DiGraph, step_id: str) -> list[str]:
    """All downstream steps influenced by this step, in topological order."""
    desc = nx.descendants(g, step_id)
    topo = list(nx.topological_sort(g))
    return [n for n in topo if n in desc]


def path_to_root(g: nx.DiGraph, step_id: str) -> list[str]:
    """The chain of steps from root to this step (inclusive), ordered root-first.

    This is the "information path" — the sequence of context transformations
    that led to this step's input.
    """
    root = get_root(g)
    try:
        path = nx.shortest_path(g, root, step_id)
        return path
    except nx.NetworkXNoPath:
        return [step_id]


def depth(g: nx.DiGraph, step_id: str) -> int:
    """Distance from the root to this step (root has depth 0)."""
    root = get_root(g)
    try:
        return nx.shortest_path_length(g, root, step_id)
    except nx.NetworkXNoPath:
        return 0


def children(g: nx.DiGraph, step_id: str) -> list[str]:
    """Direct children of a step (immediate consumers of its output)."""
    return list(g.successors(step_id))


def parent(g: nx.DiGraph, step_id: str) -> str | None:
    """Direct parent of a step (the step that invoked it)."""
    preds = list(g.predecessors(step_id))
    return preds[0] if preds else None
