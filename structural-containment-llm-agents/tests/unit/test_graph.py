import pytest

from structural_containment.graph.builder import build_directed_graph


@pytest.mark.parametrize("nodes", [["a", "a"], ["a", " "]])
def test_invalid_node_identifiers_are_rejected(nodes):
    with pytest.raises(ValueError):
        build_directed_graph(nodes, [])


def test_non_string_node_identifier_is_rejected():
    with pytest.raises(TypeError):
        build_directed_graph(["a", 1], [])


def test_non_string_edge_endpoint_is_rejected():
    with pytest.raises(TypeError):
        build_directed_graph(["a", "b"], [("a", 1)])


def test_unknown_edge_endpoint_is_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        build_directed_graph(["a"], [("a", "missing")])


def test_duplicate_edges_are_collapsed():
    graph = build_directed_graph(
        ["a", "b"],
        [("a", "b"), ("a", "b")],
    )
    assert graph.number_of_edges() == 1


def test_edge_with_wrong_number_of_endpoints_is_rejected():
    with pytest.raises(ValueError, match="exactly two"):
        build_directed_graph(["a", "b"], [("a", "b", "c")])


def test_self_loop_option_must_be_boolean():
    with pytest.raises(TypeError):
        build_directed_graph(["a"], [], allow_self_loops=1)
