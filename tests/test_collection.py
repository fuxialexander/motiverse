from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from motiverse.collection import (
    StreamingHitCollector,
    collect_max_hits_per_peak,
    collect_sum_hits_per_peak,
)
from motiverse.motif_hits import MotifHitZarrIO


@pytest.fixture
def motif_scores():
    return torch.tensor(
        [
            [[0.0, 2.0, 2.0], [1.0, 0.0, 4.0]],
            [[0.5, 0.5, 0.5], [3.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def thresholds():
    return {"p0.0001": torch.tensor([1.0, 2.0], dtype=torch.float32)}


def test_collect_max_hits_per_peak_exact(motif_scores, thresholds):
    hits = collect_max_hits_per_peak(
        motif_scores,
        thresholds,
        peak_start_indices=[10, 20],
        n_motifs=2,
    )

    expected = torch.tensor(
        [
            [10.0, 0.0, 1.0, 2.0],  # first max wins tie at positions 1 and 2
            [10.0, 1.0, 2.0, 4.0],
            [20.0, 1.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(hits, expected, rtol=0, atol=0)


def test_collect_max_threshold_equality_is_not_hit(motif_scores, thresholds):
    strict_thresholds = {"p0.0001": torch.tensor([2.0, 2.0], dtype=torch.float32)}

    hits = collect_max_hits_per_peak(
        motif_scores,
        strict_thresholds,
        peak_start_indices=[10, 20],
        n_motifs=2,
    )

    expected = torch.tensor([[10.0, 1.0, 2.0, 4.0], [20.0, 1.0, 0.0, 3.0]], dtype=torch.float32)
    assert hits.shape == (2, 4)
    torch.testing.assert_close(hits, expected, rtol=0, atol=0)


def test_collect_empty_and_reuses_preallocated_buffer(thresholds):
    scores = torch.zeros((1, 2, 3), dtype=torch.float32)
    empty = collect_max_hits_per_peak(scores, thresholds, [0], 2)
    assert empty.shape == (0, 4)

    scores[:, 0, 1] = 3.0
    buffer = torch.zeros((10, 4), dtype=torch.float32)
    hits = collect_max_hits_per_peak(scores, thresholds, [7], 2, preallocated_buffer=buffer)
    assert hits.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(
        hits,
        torch.tensor([[7.0, 0.0, 1.0, 3.0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_collect_sum_hits_per_peak_exact(motif_scores, thresholds):
    hits = collect_sum_hits_per_peak(
        motif_scores,
        thresholds,
        peak_start_indices=[10, 20],
        n_motifs=2,
    )

    expected = torch.tensor(
        [
            [10.0, 0.0, -1.0, 4.0],
            [10.0, 1.0, -1.0, 4.0],
            [20.0, 1.0, -1.0, 3.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(hits, expected, rtol=0, atol=0)


def test_streaming_hit_collector_buffers_numpy_chunks_without_zarr():
    collector = StreamingHitCollector(
        device="cpu",
        batch_buffer_size=4,
        save_every_n_peaks=10,
        output_path=None,
        peaks_df=None,
        motif_names=["A", "B"],
    )
    hits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)

    collector.add_batch_hits(hits, peaks_in_batch=2)

    assert collector.peaks_processed == 2
    assert len(collector.chunk_hits) == 1
    assert isinstance(collector.chunk_hits[0], np.ndarray)
    assert collector.chunk_hits[0].dtype == np.float32
    assert collector.get_memory_usage_mb() > 0

    collector.chunk_hits.clear()
    collector._save_current_chunk()
    assert collector.peaks_processed == 0
    assert collector.finalize() == 0


def test_motif_hit_zarr_append_groups_numpy_hits_by_chromosome(tmp_path):
    peaks_df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr1"],
            "start": [0, 10, 20],
            "end": [5, 15, 25],
        }
    )
    io = MotifHitZarrIO(tmp_path / "motif_hits.zarr", mode="w")
    io.initialize_for_streaming(
        peaks_df=peaks_df,
        motif_names=["M0", "M1"],
        p_value_threshold="p0.0001",
    )

    io.append_hits(
        np.array(
            [
                [0, 1, 2, 3],
                [1, 0, 0, 4],
                [2, 1, 5, 6],
                [99, 1, 0, 7],
            ],
            dtype=np.float32,
        )
    )

    chr1_hits = io.dataset["motif_hit_data/regions/chr1"][:]
    chr2_hits = io.dataset["motif_hit_data/regions/chr2"][:]

    np.testing.assert_allclose(
        chr1_hits,
        np.array([[0, 1, 2, 3], [1, 1, 5, 6]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        chr2_hits,
        np.array([[0, 0, 0, 4]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    assert io.dataset["metadata"].attrs["n_hits"] == 3
