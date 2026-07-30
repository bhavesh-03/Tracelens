"""Tests for tracelens.dag — DAG construction and traversal operations."""

from __future__ import annotations

import pytest

from tracelens.dag import (
    ancestors,
    build_dag,
    children,
    depth,
    descendants,
    get_leaves,
    get_root,
    parent,
    path_to_root,
)
from tracelens.schema import StepIO, Trace, TraceStep


def _make_step(step_id: str, parent_id: str | None = None, agent: str = "a") -> TraceStep:
    return TraceStep(
        step_id=step_id,
        agent_name=agent,
        step_type="agent",
        parent_step_id=parent_id,
        io=StepIO(input_text="in", output_text="out"),
    )


def _make_trace(*steps: TraceStep) -> Trace:
    return Trace(
        trace_id="t1",
        query="q",
        final_answer="a",
        steps=list(steps),
    )


class TestBuildDAG:
    """Tests for DAG construction from traces."""

    def test_single_step(self) -> None:
        trace = _make_trace(_make_step("root"))
        g = build_dag(trace)
        assert len(g.nodes) == 1
        assert len(g.edges) == 0

    def test_linear_chain(self) -> None:
        """root → child → grandchild"""
        trace = _make_trace(
            _make_step("root"),
            _make_step("child", "root"),
            _make_step("grandchild", "child"),
        )
        g = build_dag(trace)
        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        assert ("root", "child") in g.edges
        assert ("child", "grandchild") in g.edges

    def test_branching(self) -> None:
        """root → {child_a, child_b}"""
        trace = _make_trace(
            _make_step("root"),
            _make_step("child_a", "root"),
            _make_step("child_b", "root"),
        )
        g = build_dag(trace)
        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_node_attributes(self) -> None:
        trace = _make_trace(_make_step("root", agent="coordinator"))
        g = build_dag(trace)
        assert g.nodes["root"]["agent_name"] == "coordinator"


class TestDAGTraversal:
    """Tests for graph traversal operations."""

    @pytest.fixture()
    def diamond_dag(self):
        """
        Diamond DAG:
            root
           /    \\
         mid_a  mid_b
           \\    /
            leaf
        """
        trace = _make_trace(
            _make_step("root"),
            _make_step("mid_a", "root"),
            _make_step("mid_b", "root"),
            _make_step("leaf", "mid_a"),
        )
        # Note: leaf has only one parent (mid_a) since Trace schema requires
        # a single parent_step_id. For a true diamond we'd need a different
        # model. This tests the traversal with a branching DAG.
        return build_dag(trace)

    def test_get_root(self, diamond_dag) -> None:
        assert get_root(diamond_dag) == "root"

    def test_get_leaves(self, diamond_dag) -> None:
        leaves = get_leaves(diamond_dag)
        # mid_b has no children, leaf has no children
        assert set(leaves) == {"mid_b", "leaf"}

    def test_ancestors(self, diamond_dag) -> None:
        anc = ancestors(diamond_dag, "leaf")
        assert "root" in anc
        assert "mid_a" in anc
        assert "mid_b" not in anc  # mid_b is not an ancestor of leaf

    def test_descendants(self, diamond_dag) -> None:
        desc = descendants(diamond_dag, "root")
        assert set(desc) == {"mid_a", "mid_b", "leaf"}

    def test_path_to_root(self, diamond_dag) -> None:
        path = path_to_root(diamond_dag, "leaf")
        assert path == ["root", "mid_a", "leaf"]

    def test_depth(self, diamond_dag) -> None:
        assert depth(diamond_dag, "root") == 0
        assert depth(diamond_dag, "mid_a") == 1
        assert depth(diamond_dag, "leaf") == 2

    def test_children(self, diamond_dag) -> None:
        kids = children(diamond_dag, "root")
        assert set(kids) == {"mid_a", "mid_b"}

    def test_parent(self, diamond_dag) -> None:
        assert parent(diamond_dag, "mid_a") == "root"
        assert parent(diamond_dag, "root") is None


class TestRealisticCodeReviewerDAG:
    """Tests with a DAG shaped like the code reviewer subject agent."""

    @pytest.fixture()
    def reviewer_dag(self):
        trace = _make_trace(
            _make_step("coordinator", agent="coordinator"),
            _make_step("linter", "coordinator", agent="syntax_linter"),
            _make_step("security", "coordinator", agent="security_auditor"),
            _make_step("perf", "coordinator", agent="performance_reviewer"),
            _make_step("synthesizer", "coordinator", agent="report_synthesizer"),
        )
        return build_dag(trace)

    def test_coordinator_has_four_children(self, reviewer_dag) -> None:
        kids = children(reviewer_dag, "coordinator")
        assert len(kids) == 4

    def test_all_sub_agents_are_leaves(self, reviewer_dag) -> None:
        leaves = get_leaves(reviewer_dag)
        assert set(leaves) == {"linter", "security", "perf", "synthesizer"}

    def test_depth_of_sub_agents(self, reviewer_dag) -> None:
        for agent in ["linter", "security", "perf", "synthesizer"]:
            assert depth(reviewer_dag, agent) == 1

    def test_ancestors_of_linter(self, reviewer_dag) -> None:
        anc = ancestors(reviewer_dag, "linter")
        assert anc == ["coordinator"]
