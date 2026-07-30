from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from motiverse.accumulation import (
    accumulate_around_query_motif,
    accumulate_motif_cooccurrences,
    accumulate_sequences_around_hits,
    accumulate_signal_around_hits,
    apply_accumulation_strategy,
    initialize_analysis_tensor,
)


@pytest.fixture
def toy_scores():
    return torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 2.0, 0.0],
                [2.0, 0.0, 3.0, 0.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def thresholds():
    return {"p0.0001": torch.tensor([0.5, 1.5], dtype=torch.float32)}


def test_accumulate_motif_cooccurrences_exact(toy_scores, thresholds):
    result = torch.zeros((2, 2, 3), dtype=torch.float32)

    out = accumulate_motif_cooccurrences(toy_scores, result, thresholds, window_size=1)

    expected = torch.tensor(
        [
            [[0.0, 3.0, 0.0], [5.0, 0.0, 7.0]],
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_accumulate_around_query_motif_exact(toy_scores, thresholds):
    result = torch.zeros((2, 3), dtype=torch.float32)

    out = accumulate_around_query_motif(
        toy_scores,
        result,
        thresholds,
        query_motif_index=0,
        window_size=1,
        device="cpu",
    )

    expected = torch.tensor([[0.0, 3.0, 0.0], [5.0, 0.0, 7.0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_accumulate_signal_around_hits_1d_exact(toy_scores, thresholds):
    signal = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=torch.float32)
    result = torch.zeros((2, 3), dtype=torch.float32)

    out = accumulate_signal_around_hits(signal, toy_scores, result, thresholds, window_size=1)

    expected = torch.tensor([[40.0, 60.0, 80.0], [20.0, 30.0, 40.0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_accumulate_signal_around_hits_2d_shape_and_order(toy_scores, thresholds):
    signal = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]],
        dtype=torch.float32,
    )
    result = torch.zeros((2, 3, 2), dtype=torch.float32)

    out = accumulate_signal_around_hits(signal, toy_scores, result, thresholds, window_size=1)

    expected = torch.tensor(
        [
            [[4.0, 40.0], [6.0, 60.0], [8.0, 80.0]],
            [[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_accumulate_sequences_around_hits_exact(toy_scores, thresholds):
    # A, C, G, T, A one-hot.
    dna = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ]
        ],
        dtype=torch.float32,
    )
    result = torch.zeros((2, 3, 4), dtype=torch.int32)

    out = accumulate_sequences_around_hits(dna, toy_scores, result, thresholds, window_size=1)

    expected = torch.tensor(
        [
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_threshold_is_strict_and_edges_are_ignored(thresholds):
    scores = torch.tensor([[[0.5, 1.0, 0.0, 0.5, 1.0]]], dtype=torch.float32)
    one_threshold = {"p0.0001": torch.tensor([0.5], dtype=torch.float32)}
    result = torch.zeros((1, 1, 3), dtype=torch.float32)

    out = accumulate_motif_cooccurrences(scores, result, one_threshold, window_size=1)

    # Only position 1 is both strictly above threshold and non-edge.
    expected = torch.tensor([[[0.5, 1.0, 0.0]]], dtype=torch.float32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_too_short_sequence_returns_unchanged(toy_scores, thresholds):
    result = torch.ones((2, 2, 11), dtype=torch.float32)

    out = accumulate_motif_cooccurrences(toy_scores, result.clone(), thresholds, window_size=5)

    torch.testing.assert_close(out, result, rtol=0, atol=0)


def test_invalid_signal_rank_raises(toy_scores, thresholds):
    with pytest.raises(ValueError):
        accumulate_signal_around_hits(
            torch.zeros((1, 5, 2, 1), dtype=torch.float32),
            toy_scores,
            torch.zeros((2, 3), dtype=torch.float32),
            thresholds,
            window_size=1,
        )


def test_initialize_shapes_and_reverse_offset_strategy(toy_scores, thresholds):
    motif_names = ["A", "B"]
    seq_tensor = initialize_analysis_tensor(
        True,
        True,
        None,
        2,
        1,
        "cpu",
        torch.float32,
        motif_names,
        strand_specific=False,
    )
    assert seq_tensor.shape == (4, 3, 4)
    assert seq_tensor.dtype == torch.int32

    result = torch.zeros((4, 3), dtype=torch.float32)
    signal = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=torch.float32)
    out = apply_accumulation_strategy(
        toy_scores,
        result,
        thresholds,
        1,
        "cpu",
        sequence_accumulation=False,
        query_motif_index=None,
        strand_sequences=toy_scores.permute(0, 2, 1),
        motif_index_offset=2,
        n_motifs=2,
        signal_accumulation=True,
        signal_data=signal,
    )

    assert torch.count_nonzero(out[:2]) == 0
    assert torch.count_nonzero(out[2:]) > 0
