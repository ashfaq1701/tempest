import numpy as np
import pytest

from tempest import Tempest


GPU_USAGE_MODE = False


def _base_graph_with_edge_features():
    tw = Tempest(is_directed=True, use_gpu=GPU_USAGE_MODE)

    sources = np.array([0, 1, 2], dtype=np.int32)
    targets = np.array([1, 2, 3], dtype=np.int32)
    timestamps = np.array([10, 20, 30], dtype=np.int64)
    edge_features = np.array(
        [
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
        ],
        dtype=np.float32,
    )

    tw.add_multiple_edges(sources, targets, timestamps, edge_features)

    expected_feature_by_edge = {
        (0, 1, 10): np.array([1.0, 1.5], dtype=np.float32),
        (1, 2, 20): np.array([2.0, 2.5], dtype=np.float32),
        (2, 3, 30): np.array([3.0, 3.5], dtype=np.float32),
    }

    return tw, expected_feature_by_edge


def test_walks_include_edge_features_and_zero_unused_slots():
    tw, expected_feature_by_edge = _base_graph_with_edge_features()

    max_walk_len = 4
    nodes, timestamps, lens, walk_edge_features = tw.get_random_walks_and_times_for_all_nodes(
        max_walk_len=max_walk_len,
        walk_bias="Uniform",
        num_walks_per_node=1,
        walk_direction="Forward_In_Time",
    )

    assert walk_edge_features is not None
    assert walk_edge_features.shape == (len(lens), max_walk_len - 1, 2)

    for walk_idx, walk_len in enumerate(lens):
        walk_len = int(walk_len)

        for step_idx in range(max(0, walk_len - 1)):
            edge = (
                int(nodes[walk_idx, step_idx]),
                int(nodes[walk_idx, step_idx + 1]),
                int(timestamps[walk_idx, step_idx + 1]),
            )
            np.testing.assert_allclose(
                walk_edge_features[walk_idx, step_idx],
                expected_feature_by_edge[edge],
            )

        for step_idx in range(max(0, walk_len - 1), max_walk_len - 1):
            np.testing.assert_allclose(
                walk_edge_features[walk_idx, step_idx],
                np.zeros(2, dtype=np.float32),
            )


def test_walks_return_none_when_edge_features_are_not_provided():
    tw = Tempest(is_directed=True, use_gpu=GPU_USAGE_MODE)
    tw.add_multiple_edges(
        np.array([0, 1], dtype=np.int32),
        np.array([1, 2], dtype=np.int32),
        np.array([10, 20], dtype=np.int64),
    )

    _, _, _, walk_edge_features = tw.get_random_walks_and_times(
        max_walk_len=3,
        walk_bias="Uniform",
        num_walks_total=2,
        walk_direction="Forward_In_Time",
    )

    assert walk_edge_features is None


def test_rejects_mismatched_edge_feature_dimensions_on_subsequent_ingestion():
    tw, _ = _base_graph_with_edge_features()

    with pytest.raises(RuntimeError):
        tw.add_multiple_edges(
            np.array([4], dtype=np.int32),
            np.array([5], dtype=np.int32),
            np.array([40], dtype=np.int64),
            np.array([[9.0, 9.1, 9.2]], dtype=np.float32),
        )


def test_rejects_missing_features_after_feature_mode_is_enabled():
    tw, _ = _base_graph_with_edge_features()

    with pytest.raises(RuntimeError):
        tw.add_multiple_edges(
            np.array([4], dtype=np.int32),
            np.array([5], dtype=np.int32),
            np.array([40], dtype=np.int64),
        )

if __name__ == "__main__":
    test_walks_include_edge_features_and_zero_unused_slots()
    test_walks_return_none_when_edge_features_are_not_provided()
    test_rejects_mismatched_edge_feature_dimensions_on_subsequent_ingestion()
    test_rejects_missing_features_after_feature_mode_is_enabled()
    print("All tests passed!")
