from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motiverse.processing import (
    _scan_motifs_conv1d,
    process_chromosome_sequences,
)


def test_scan_motifs_conv1d_valid_relu():
    # A C G T one-hot, channel-first for conv1d.
    seq = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).permute(0, 2, 1)
    ac_kernel = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    ).permute(0, 2, 1)

    scores = _scan_motifs_conv1d(seq, ac_kernel)

    torch.testing.assert_close(scores, torch.tensor([[[2.0, 0.0, 0.0]]]), rtol=0, atol=0)

    negative_kernel = -ac_kernel
    negative_scores = _scan_motifs_conv1d(seq, negative_kernel)
    torch.testing.assert_close(negative_scores, torch.zeros_like(negative_scores), rtol=0, atol=0)


def test_process_chromosome_sequences_toy_counts_and_accumulation():
    # A C A T A one-hot.
    chromosome = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    motif_kernel = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    motif_parameters = (
        ["A"],
        motif_kernel,
        None,
        {"p0.0001": torch.tensor([0.5], dtype=torch.float32)},
        None,
        None,
    )

    hit_stats, _scan_time, result = process_chromosome_sequences(
        "chrToy",
        chromosome,
        motif_parameters,
        batch_size=1,
        sequence_length=5,
        device="cpu",
        dtype=torch.float32,
        enable_accumulation=True,
        analysis_window_size=1,
        strand_specific=True,
        p_value_threshold="p0.0001",
    )

    assert hit_stats["p0.0001"] == 3
    torch.testing.assert_close(
        result,
        torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_process_chromosome_sequences_short_chromosome_returns_empty():
    hit_stats, scan_time, result = process_chromosome_sequences(
        "chrTiny",
        np.zeros((3, 4), dtype=np.float32),
        (
            ["A"],
            torch.ones((1, 1, 4), dtype=torch.float32),
            None,
            {"p0.0001": torch.tensor([0.5], dtype=torch.float32)},
            None,
            None,
        ),
        sequence_length=5,
        device="cpu",
        dtype=torch.float32,
    )

    assert hit_stats is None
    assert scan_time == 0
    assert result is None
