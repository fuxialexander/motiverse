from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from motiverse import processing


class FakeSequenceDenseZarrIO:
    chroms = ["chr1"]
    chrom_sizes = {"chr1": 100}

    def __init__(self, *_args, **_kwargs):
        pass


def _motif_parameters():
    return (
        ["A"],
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        None,
        {"p0.0001": torch.tensor([0.5], dtype=torch.float32)},
        None,
        None,
    )


@pytest.mark.integration
def test_process_narrowpeak_regions_filters_unavailable_chroms_and_pads(monkeypatch):
    monkeypatch.setattr(processing, "SequenceDenseZarrIO", FakeSequenceDenseZarrIO)
    seen_regions = {}

    def fake_extract(_seq_db, regions_df, _extend_bp, extend_right_only=False):
        seen_regions["chroms"] = regions_df["chrom"].tolist()
        assert not extend_right_only
        seq1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
        seq2 = np.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
            dtype=np.float32,
        )
        metadata = [
            {"chrom": "chr1", "start": 0, "end": 3, "length": 3},
            {"chrom": "chr1", "start": 10, "end": 14, "length": 4},
        ]
        return [seq1, seq2], metadata

    monkeypatch.setattr(processing, "extract_sequences_from_regions", fake_extract)
    tiled_regions = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chrMissing"],
            "start": [0, 10, 0],
            "end": [3, 14, 3],
        }
    )

    (
        hit_stats,
        _scan_time,
        result,
        collected_hits,
        region_stats,
    ) = processing.process_narrowpeak_regions(
        "unused.bed",
        "unused.zarr",
        _motif_parameters(),
        batch_size=2,
        device="cpu",
        dtype=torch.float32,
        enable_accumulation=True,
        analysis_window_size=1,
        strand_specific=True,
        p_value_threshold="p0.0001",
        tiled_regions_df=tiled_regions,
        return_region_stats=True,
    )

    assert seen_regions["chroms"] == ["chr1", "chr1"]
    assert hit_stats["p0.0001"] == 4
    assert result.shape == (1, 1, 3)
    assert collected_hits is None
    assert region_stats["n_input_rows"] == 3
    assert region_stats["n_chrom_filtered_regions"] == 2
    assert region_stats["n_skipped_unavailable_chrom_regions"] == 1
    assert region_stats["n_valid_regions"] == 2
    assert region_stats["n_dropped_invalid_regions"] == 0
    assert region_stats["bp_scanned"] == 7


@pytest.mark.integration
@pytest.mark.parametrize("aggregation_mode", ["max", "sum"])
def test_process_narrowpeak_regions_routes_hit_aggregation(monkeypatch, aggregation_mode):
    monkeypatch.setattr(processing, "SequenceDenseZarrIO", FakeSequenceDenseZarrIO)
    monkeypatch.setattr(
        processing,
        "extract_sequences_from_regions",
        lambda *_args, **_kwargs: (
            [np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)],
            [{"chrom": "chr1", "start": 0, "end": 2, "length": 2}],
        ),
    )

    calls = {"max": 0, "sum": 0, "finalized": 0}

    class FakeCollector:
        def __init__(self, *args, **kwargs):
            self.buffer = torch.zeros((4, 4), dtype=torch.float32)

        def get_batch_buffer(self):
            return self.buffer

        def add_batch_hits(self, batch_hits, peaks_in_batch):
            assert batch_hits.shape == (0, 4)
            assert peaks_in_batch == 1

        def finalize(self):
            calls["finalized"] += 1
            return 0

        def get_memory_usage_mb(self):
            return 0.0

    def fake_max(*_args, **_kwargs):
        calls["max"] += 1
        return torch.zeros((0, 4), dtype=torch.float32)

    def fake_sum(*_args, **_kwargs):
        calls["sum"] += 1
        return torch.zeros((0, 4), dtype=torch.float32)

    import motiverse.collection as collection

    monkeypatch.setattr(collection, "StreamingHitCollector", FakeCollector)
    monkeypatch.setattr(collection, "collect_max_hits_per_peak", fake_max)
    monkeypatch.setattr(collection, "collect_sum_hits_per_peak", fake_sum)

    processing.process_narrowpeak_regions(
        "unused.bed",
        "unused.zarr",
        _motif_parameters(),
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        enable_accumulation=False,
        strand_specific=True,
        p_value_threshold="p0.0001",
        collect_hits=True,
        output_directory="/tmp/not_used",
        tiled_regions_df=pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [2]}),
        hit_aggregation_mode=aggregation_mode,
    )

    assert calls[aggregation_mode] == 1
    assert calls["finalized"] == 1


@pytest.mark.integration
def test_collect_hits_uses_filtered_reset_peak_indices(monkeypatch):
    monkeypatch.setattr(processing, "SequenceDenseZarrIO", FakeSequenceDenseZarrIO)
    monkeypatch.setattr(
        processing,
        "extract_sequences_from_regions",
        lambda *_args, **_kwargs: (
            [np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)],
            [{"chrom": "chr1", "start": 0, "end": 2, "length": 2}],
        ),
    )

    seen = {"peak_indices": None, "writer_chroms": None}

    class FakeCollector:
        def __init__(self, *args, **kwargs):
            self.buffer = torch.zeros((4, 4), dtype=torch.float32)
            seen["writer_chroms"] = kwargs["peaks_df"]["chrom"].tolist()

        def get_batch_buffer(self):
            return self.buffer

        def add_batch_hits(self, batch_hits, peaks_in_batch):
            assert peaks_in_batch == 1

        def finalize(self):
            return 0

        def get_memory_usage_mb(self):
            return 0.0

    def fake_max(_scores, _thresholds, peak_indices, *_args, **_kwargs):
        seen["peak_indices"] = list(peak_indices)
        return torch.zeros((0, 4), dtype=torch.float32)

    import motiverse.collection as collection

    monkeypatch.setattr(collection, "StreamingHitCollector", FakeCollector)
    monkeypatch.setattr(collection, "collect_max_hits_per_peak", fake_max)

    processing.process_narrowpeak_regions(
        "unused.bed",
        "unused.zarr",
        _motif_parameters(),
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        enable_accumulation=False,
        strand_specific=True,
        p_value_threshold="p0.0001",
        collect_hits=True,
        output_directory="/tmp/not_used",
        tiled_regions_df=pd.DataFrame(
            {"chrom": ["chrMissing", "chr1"], "start": [0, 0], "end": [2, 2]}
        ),
    )

    assert seen["writer_chroms"] == ["chr1"]
    assert seen["peak_indices"] == [0]
