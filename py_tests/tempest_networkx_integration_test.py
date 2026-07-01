import networkx as nx
from tempest import Tempest
import pytest

GPU_USAGE_MODE = False

def test_networkx_integration():
    nx_graph = nx.DiGraph()

    edges_with_timestamps = [
        (0, 1, {'timestamp': 100}),
        (1, 2, {'timestamp': 200}),
        (2, 3, {'timestamp': 300}),
        (3, 0, {'timestamp': 400}),
        (1, 3, {'timestamp': 500}),
    ]
    nx_graph.add_edges_from(edges_with_timestamps)

    tw = Tempest(True, GPU_USAGE_MODE)

    tw.add_edges_from_networkx(nx_graph)

    assert tw.get_node_count() == 4
    assert tw.get_edge_count() == 5

    nx_graph_2 = tw.to_networkx()

    assert nx_graph_2.number_of_nodes() == nx_graph.number_of_nodes()
    assert nx_graph_2.number_of_edges() == nx_graph.number_of_edges()

    for u, v, data in nx_graph.edges(data=True):
        assert nx_graph_2.has_edge(u, v)
        assert nx_graph_2[u][v]['timestamp'] == data['timestamp']

def test_networkx_integration_empty_graph():
    G = nx.DiGraph()
    tw = Tempest(True, GPU_USAGE_MODE)

    tw.add_edges_from_networkx(G)
    assert tw.get_node_count() == 0
    assert tw.get_edge_count() == 0

def test_networkx_integration_invalid_timestamp():
    G = nx.DiGraph()
    G.add_edge(0, 1)

    tw = Tempest(True, GPU_USAGE_MODE)

    with pytest.raises(KeyError):
        tw.add_edges_from_networkx(G)

def test_networkx_integration_with_existing_edges():
    tw = Tempest(True, GPU_USAGE_MODE)

    sources = [0, 1, 4]
    targets = [1, 2, 5]
    timestamps = [50, 150, 600]
    tw.add_multiple_edges(sources, targets, timestamps)

    assert tw.get_node_count() == 5
    assert tw.get_edge_count() == 3

    nx_graph = nx.DiGraph()
    edges_with_timestamps = [
        (0, 1, {'timestamp': 100}),
        (1, 2, {'timestamp': 200}),
        (2, 3, {'timestamp': 300}),
        (3, 0, {'timestamp': 400}),
    ]
    nx_graph.add_edges_from(edges_with_timestamps)

    tw.add_edges_from_networkx(nx_graph)

    assert tw.get_node_count() == 6
    assert tw.get_edge_count() == 7

    nx_graph_2 = tw.to_networkx()

    expected_edges = {
        (0, 1): [50, 100],
        (1, 2): [150, 200],
        (2, 3): [300],
        (3, 0): [400],
        (4, 5): [600]
    }

    for (u, v), expected_timestamps in expected_edges.items():
        edges_found = []
        if nx_graph_2.has_edge(u, v):
            edges_found.append(nx_graph_2[u][v]['timestamp'])

        edges_found.sort()
        expected_timestamps.sort()
        # DiGraph collapses parallel edges to the latest timestamp
        assert edges_found[0] == max(expected_timestamps), f"Edge ({u},{v}) has incorrect timestamp: {edges_found[0]} vs expected {max(expected_timestamps)}"

    for u, v in nx_graph_2.edges():
        assert (u, v) in expected_edges, f"Unexpected edge ({u},{v}) found"

def test_networkx_integration_directed_undirected():
    tw_directed = Tempest(True, GPU_USAGE_MODE)
    tw_directed.add_multiple_edges(
        [0, 1, 2],
        [1, 2, 0],
        [100, 200, 300]
    )

    nx_directed = tw_directed.to_networkx()
    assert isinstance(nx_directed, nx.DiGraph)
    assert nx_directed.has_edge(0, 1) and not nx_directed.has_edge(1, 0)
    assert nx_directed.has_edge(1, 2) and not nx_directed.has_edge(2, 1)
    assert nx_directed.has_edge(2, 0) and not nx_directed.has_edge(0, 2)

    tw_undirected = Tempest(False, GPU_USAGE_MODE)
    tw_undirected.add_multiple_edges(
        [0, 1, 2],
        [1, 2, 0],
        [100, 200, 300]
    )

    nx_undirected = tw_undirected.to_networkx()
    assert isinstance(nx_undirected, nx.Graph)
    assert nx_undirected.has_edge(0, 1) and nx_undirected.has_edge(1, 0)
    assert nx_undirected.has_edge(1, 2) and nx_undirected.has_edge(2, 1)
    assert nx_undirected.has_edge(2, 0) and nx_undirected.has_edge(0, 2)

    assert nx_directed[0][1]['timestamp'] == 100
    assert nx_directed[1][2]['timestamp'] == 200
    assert nx_directed[2][0]['timestamp'] == 300

    assert nx_undirected[0][1]['timestamp'] == 100
    assert nx_undirected[1][2]['timestamp'] == 200
    assert nx_undirected[2][0]['timestamp'] == 300


if __name__ == "__main__":
    test_networkx_integration()
    test_networkx_integration_empty_graph()
    test_networkx_integration_invalid_timestamp()
    test_networkx_integration_with_existing_edges()
    test_networkx_integration_directed_undirected()
    print("All tests passed!")
