from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr

torch = pytest.importorskip("torch")

from motiverse.accumulation import accumulate_around_query_motif
from motiverse.positional_cache import (
    CACHE_ACCUMULATION_SEMANTICS,
    CACHE_COLUMNS,
    CACHE_COORDINATE_FRAME,
    CACHE_ROW_DTYPE,
    HIT_CHUNK_INDEX_COLUMNS,
    HIT_CHUNK_INDEX_SCHEMA_VERSION,
    HIT_STORAGE_LAYOUT,
    PositionalMotifHitCache,
    accumulate_hit_rows,
    build_positional_motif_hit_cache,
    collect_positional_hits,
    motif_center_offsets,
    threshold_vector_checksum,
)
from motiverse.processing import _scan_motifs_conv1d


def _reference_accumulate_hit_rows(region_hits, *, window_size, n_motifs, query_motif_index=None):
    width = 2 * window_size + 1
    if query_motif_index is None:
        out = np.zeros((n_motifs, n_motifs, width), dtype=np.float32)
    else:
        out = np.zeros((n_motifs, width), dtype=np.float32)
    if region_hits.size == 0:
        return out
    centers = region_hits[:, 1].astype(np.int64)
    motifs = region_hits[:, 2].astype(np.int64)
    scores = region_hits[:, 4].astype(np.float32)
    if query_motif_index is None:
        anchor_indices = range(len(region_hits))
    else:
        anchor_indices = np.where(motifs == query_motif_index)[0]
    for anchor_i in anchor_indices:
        deltas = centers - centers[anchor_i]
        valid = (deltas >= -window_size) & (deltas <= window_size)
        rel = (deltas[valid] + window_size).astype(np.int64)
        if query_motif_index is None:
            np.add.at(out, (motifs[anchor_i], motifs[valid], rel), scores[valid])
        else:
            np.add.at(out, (motifs[valid], rel), scores[valid])
    return out


def test_accumulate_hit_rows_matches_reference_with_chunking_and_large_coordinates():
    rows = np.array(
        [
            [100_000_005, 100_000_005, 0, 0, 5],
            [100_000_001, 100_000_001, 1, 0, 2],
            [100_000_002, 100_000_002, 2, 0, 3],
            [100_000_004, 100_000_004, 1, 0, 4],
            [100_000_000, 100_000_000, 0, 0, 1],
        ],
        dtype=CACHE_ROW_DTYPE,
    )

    all_to_all = np.zeros((3, 3, 5), dtype=np.float32)
    accumulate_hit_rows(
        all_to_all,
        rows,
        window_size=2,
        query_motif_index=None,
        pair_chunk_size=3,
    )
    one_to_all = np.zeros((3, 5), dtype=np.float32)
    accumulate_hit_rows(
        one_to_all,
        rows,
        window_size=2,
        query_motif_index=1,
        pair_chunk_size=2,
    )

    np.testing.assert_allclose(
        all_to_all,
        _reference_accumulate_hit_rows(rows, window_size=2, n_motifs=3),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        one_to_all,
        _reference_accumulate_hit_rows(
            rows,
            window_size=2,
            n_motifs=3,
            query_motif_index=1,
        ),
        rtol=0,
        atol=0,
    )


def test_motif_center_offsets_use_active_core_center():
    kernels = np.zeros((2, 7, 4), dtype=np.float32)
    kernels[0, 2:5, :] = 1.0
    kernels[1, :, :] = 0.0

    offsets = motif_center_offsets(kernels)

    np.testing.assert_array_equal(offsets, np.array([3, 3], dtype=np.int32))


def test_collect_positional_hits_reports_start_and_center_positions():
    scores = torch.tensor([[[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]]], dtype=torch.float32)
    thresholds = torch.tensor([0.5, 1.5], dtype=torch.float32)
    starts = torch.tensor([100], dtype=torch.int64)
    center_offsets = torch.tensor([15, 16], dtype=torch.int64)

    rows = collect_positional_hits(
        scores,
        thresholds,
        starts,
        center_offsets,
        strand_id=0,
    )

    expected = torch.tensor(
        [
            [101.0, 116.0, 0.0, 0.0, 1.0],
            [100.0, 116.0, 1.0, 0.0, 2.0],
            [102.0, 118.0, 1.0, 0.0, 3.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    assert rows.dtype == torch.float64


def test_collect_positional_hits_preserves_large_genome_coordinates_exactly():
    start = 100_000_003
    scores = torch.tensor([[[9.0, 8.0, 7.0]]], dtype=torch.float32)
    thresholds = torch.tensor([0.0], dtype=torch.float32)

    rows = collect_positional_hits(
        scores,
        thresholds,
        torch.tensor([start], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        strand_id=0,
    )

    assert rows.dtype == torch.float64
    np.testing.assert_array_equal(
        rows[:, 0].cpu().numpy().astype(np.int64),
        np.array([100_000_003, 100_000_004, 100_000_005], dtype=np.int64),
    )


def test_collect_positional_hits_maps_reverse_positions_to_genome_coordinates():
    scores = torch.tensor([[[0.0, 0.0, 3.0, 0.0]]], dtype=torch.float32)
    thresholds = torch.tensor([2.0], dtype=torch.float32)
    starts = torch.tensor([100], dtype=torch.int64)
    lengths = torch.tensor([8], dtype=torch.int64)
    center_offsets = torch.tensor([1], dtype=torch.int64)

    rows = collect_positional_hits(
        scores,
        thresholds,
        starts,
        center_offsets,
        strand_id=1,
        motif_length=3,
        sequence_lengths=lengths,
        padded_sequence_length=10,
        reverse_strand=True,
    )

    expected = torch.tensor([[105.0, 106.0, 0.0, 1.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)


def test_cache_query_and_sparse_accumulation(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(cache_path), mode="w", zarr_format=2)
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0", "M1"], dtype="<U10"))
    metadata.attrs.update({"motif_count": 2, "schema_version": "positional_hit_cache_v1"})
    hits_group = root.require_group("hits/regions")
    rows = np.array(
        [
            [10, 10, 0, 0, 1],
            [11, 11, 1, 0, 2],
            [12, 12, 0, 0, 3],
            [20, 20, 1, 0, 4],
        ],
        dtype=np.float32,
    )
    arr = hits_group.create_array("chr1", data=rows, chunks=(10, len(CACHE_COLUMNS)))
    arr.attrs["columns"] = CACHE_COLUMNS

    cache = PositionalMotifHitCache(cache_path)
    queried = cache.query("chr1", 9, 13)
    assert queried.shape == (3, len(CACHE_COLUMNS))

    regions = pd.DataFrame({"chrom": ["chr1"], "start": [9], "end": [13]})
    one = cache.accumulate_regions(regions, window_size=1, n_motifs=2, query_motif_index=0)
    np.testing.assert_allclose(
        one,
        np.array([[0, 4, 0], [2, 0, 2]], dtype=np.float32),
        rtol=0,
        atol=0,
    )

    all_to_all = cache.accumulate_regions(regions, window_size=1, n_motifs=2)
    expected = np.array(
        [
            [[0, 4, 0], [2, 0, 2]],
            [[1, 0, 3], [0, 2, 0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(all_to_all, expected, rtol=0, atol=0)


def test_sparse_hit_cache_contract_drops_below_threshold_partner_scores():
    motif_scores = torch.tensor(
        [[[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    thresholds = {"p0.0001": torch.tensor([0.5, 10.0], dtype=torch.float32)}
    dense = torch.zeros((2, 3), dtype=torch.float32)

    accumulate_around_query_motif(
        motif_scores,
        dense,
        thresholds,
        query_motif_index=0,
        window_size=1,
        device="cpu",
    )
    rows = collect_positional_hits(
        motif_scores,
        thresholds["p0.0001"],
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int64),
        strand_id=0,
    )
    sparse = np.zeros((2, 3), dtype=np.float32)
    accumulate_hit_rows(
        sparse,
        rows.cpu().numpy(),
        window_size=1,
        query_motif_index=0,
    )

    np.testing.assert_allclose(
        dense.numpy(),
        np.array([[0, 1, 0], [2, 0, 0]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        sparse,
        np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    assert CACHE_ACCUMULATION_SEMANTICS == "significant_hit_score_pairs"
    assert CACHE_COORDINATE_FRAME == "active_core_center"


def test_cache_query_uses_chunk_index_without_materializing_chromosome(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(cache_path), mode="w")
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0", "M1"], dtype="<U10"))
    metadata.attrs.update(
        {
            "motif_count": 2,
            "schema_version": "positional_hit_cache_v1",
            "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
        }
    )
    hits_group = root.require_group("hits/regions")
    rows = np.array(
        [[100_000_000 + pos, 100_000_000 + pos, pos % 2, 0, pos / 10] for pos in range(10)],
        dtype=CACHE_ROW_DTYPE,
    )
    arr = hits_group.create_array("chr1", data=rows, chunks=(2, len(CACHE_COLUMNS)))
    arr.attrs["columns"] = CACHE_COLUMNS
    index_group = root.require_group("hits/chunk_index")
    index_rows = np.array(
        [
            [0, 2, 100_000_000, 100_000_001],
            [2, 4, 100_000_002, 100_000_003],
            [4, 6, 100_000_004, 100_000_005],
            [6, 8, 100_000_006, 100_000_007],
            [8, 10, 100_000_008, 100_000_009],
        ],
        dtype=np.float64,
    )
    index = index_group.create_array(
        "chr1", data=index_rows, chunks=(2, len(HIT_CHUNK_INDEX_COLUMNS))
    )
    index.attrs["columns"] = HIT_CHUNK_INDEX_COLUMNS
    index.attrs["schema_version"] = HIT_CHUNK_INDEX_SCHEMA_VERSION

    cache = PositionalMotifHitCache(cache_path)
    queried = cache.query("chr1", 100_000_004, 100_000_008, motif_idx=1, min_score=0.5)

    np.testing.assert_allclose(
        queried,
        np.array(
            [
                [100_000_005, 100_000_005, 1, 0, 0.5],
                [100_000_007, 100_000_007, 1, 0, 0.7],
            ],
            dtype=CACHE_ROW_DTYPE,
        ),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        queried[:, 1].astype(np.int64),
        np.array([100_000_005, 100_000_007], dtype=np.int64),
    )
    assert cache._chrom_row_cache == {}
    assert "chr1" in cache._chrom_chunk_index_cache

    batch_rows, batch_stats = cache.query_many_with_stats(
        [
            ("chr1", 100_000_004, 100_000_006),
            ("chr1", 100_000_005, 100_000_006),
        ]
    )
    assert batch_stats["n_intervals"] == 2
    assert batch_stats["indexed_chroms"] == 1
    assert batch_stats["fallback_chroms"] == 0
    assert batch_stats["chrom_chunk_reads"] == 1
    assert cache._chrom_row_cache == {}
    np.testing.assert_allclose(
        batch_rows[("chr1", 100_000_004, 100_000_006)],
        rows[4:6],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        batch_rows[("chr1", 100_000_005, 100_000_006)],
        rows[5:6],
        rtol=0,
        atol=0,
    )


def _write_query_cache(path, *, include_index: bool):
    root = zarr.open_group(str(path), mode="w")
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0", "M1"], dtype="<U10"))
    metadata.attrs.update(
        {
            "motif_count": 2,
            "schema_version": "positional_hit_cache_v1",
            "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
        }
    )
    rows = np.array(
        [[100_000_000 + pos, 100_000_000 + pos, pos % 2, 0, pos / 10] for pos in range(8)],
        dtype=CACHE_ROW_DTYPE,
    )
    hits_group = root.require_group("hits/regions")
    arr = hits_group.create_array("chr1", data=rows, chunks=(2, len(CACHE_COLUMNS)))
    arr.attrs["columns"] = CACHE_COLUMNS
    if not include_index:
        return
    index_group = root.require_group("hits/chunk_index")
    index_rows = np.array(
        [
            [0, 2, 100_000_000, 100_000_001],
            [2, 4, 100_000_002, 100_000_003],
            [4, 6, 100_000_004, 100_000_005],
            [6, 8, 100_000_006, 100_000_007],
        ],
        dtype=np.float64,
    )
    index = index_group.create_array(
        "chr1", data=index_rows, chunks=(2, len(HIT_CHUNK_INDEX_COLUMNS))
    )
    index.attrs["columns"] = HIT_CHUNK_INDEX_COLUMNS
    index.attrs["schema_version"] = HIT_CHUNK_INDEX_SCHEMA_VERSION


@pytest.mark.parametrize(
    "start,end,motif_idx,min_score",
    [
        (100_000_000, 100_000_002, None, None),
        (100_000_001, 100_000_006, 1, None),
        (100_000_002, 100_000_005, None, 0.3),
        (100_000_007, 100_000_008, 1, 0.7),
        (100_000_008, 100_000_009, None, None),
        (100_000_004, 100_000_004, None, None),
    ],
)
def test_chunk_index_query_matches_fallback_query(tmp_path, start, end, motif_idx, min_score):
    indexed_path = tmp_path / "indexed.zarr"
    fallback_path = tmp_path / "fallback.zarr"
    _write_query_cache(indexed_path, include_index=True)
    _write_query_cache(fallback_path, include_index=False)

    indexed = PositionalMotifHitCache(indexed_path)
    fallback = PositionalMotifHitCache(fallback_path)

    indexed_rows = indexed.query("chr1", start, end, motif_idx=motif_idx, min_score=min_score)
    fallback_rows = fallback.query("chr1", start, end, motif_idx=motif_idx, min_score=min_score)

    np.testing.assert_allclose(indexed_rows, fallback_rows, rtol=0, atol=0)


def test_query_many_matches_single_query_and_fallback_query(tmp_path):
    indexed_path = tmp_path / "indexed.zarr"
    fallback_path = tmp_path / "fallback.zarr"
    _write_query_cache(indexed_path, include_index=True)
    _write_query_cache(fallback_path, include_index=False)

    indexed = PositionalMotifHitCache(indexed_path)
    fallback = PositionalMotifHitCache(fallback_path)
    intervals = [
        ("chr1", 100_000_000, 100_000_003),
        ("chr1", 100_000_002, 100_000_007),
        ("chr1", 100_000_006, 100_000_008),
        ("chr1", 100_000_008, 100_000_009),
    ]

    indexed_rows, stats = indexed.query_many_with_stats(
        intervals,
        motif_idx=1,
        min_score=0.3,
    )
    fallback_rows = fallback.query_many(
        intervals,
        motif_idx=1,
        min_score=0.3,
    )

    assert stats["indexed_chroms"] == 1
    assert stats["fallback_chroms"] == 0
    assert indexed._chrom_row_cache == {}
    for key in intervals:
        single_rows = indexed.query(
            key[0],
            key[1],
            key[2],
            motif_idx=1,
            min_score=0.3,
        )
        np.testing.assert_allclose(indexed_rows[key], single_rows, rtol=0, atol=0)
        np.testing.assert_allclose(indexed_rows[key], fallback_rows[key], rtol=0, atol=0)


def _one_hot(sequence: str) -> np.ndarray:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for pos, base in enumerate(sequence):
        encoded[pos, base_to_idx[base]] = 1.0
    return encoded


def _write_sequence_zarr(path, sequence: str):
    root = zarr.open_group(str(path), mode="w")
    root.attrs["assembly"] = "hg38"
    root.attrs["chunk_size"] = 100
    root.attrs["zarr_type"] = "dense"
    chrs = root.create_group("chrs")
    chrs.create_array("chr1", data=_one_hot(sequence), chunks=(100, 4))


def _atg_kernel() -> torch.Tensor:
    kernel = torch.zeros((1, 3, 4), dtype=torch.float32)
    kernel[0, 0, 0] = 1.0  # A
    kernel[0, 1, 3] = 1.0  # T
    kernel[0, 2, 2] = 1.0  # G
    return kernel


def test_build_positional_cache_maps_reverse_only_hit_to_forward_coordinates(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    cache_path = tmp_path / "cache.zarr"
    _write_sequence_zarr(seq_path, "AAAAACATCCCC")
    motif_parameters = (
        ["ATG"],
        _atg_kernel(),
        None,
        {"p0.0001": torch.tensor([2.5], dtype=torch.float32)},
        None,
        None,
    )

    build_positional_motif_hit_cache(
        genome_zarr_path=str(seq_path),
        output_path=cache_path,
        motif_parameters=motif_parameters,
        motif_metadata={
            "motif_source": "test",
            "aligned_motif_path": "test.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "abc",
            "motif_names_checksum": "def",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 2.5,
        },
        target_chromosomes=["chr1"],
        tile_size=20,
        tile_extension_bp=0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        strand_specific=False,
    )

    cache = PositionalMotifHitCache(cache_path)
    rows = cache.query("chr1", 0, 12)

    assert cache.metadata["hit_storage_layout"] == HIT_STORAGE_LAYOUT
    assert cache.metadata["coordinate_frame"] == CACHE_COORDINATE_FRAME
    assert cache.metadata["score_threshold_vector_checksum"] == threshold_vector_checksum(
        torch.tensor([2.5], dtype=torch.float32)
    )
    assert cache.metadata["score_threshold_vector_shape"] == [1]
    assert cache.dataset["hits/regions/chr1"].shape == (1, len(CACHE_COLUMNS))
    assert cache.dataset["hits/regions/chr1"].attrs["storage_layout"] == HIT_STORAGE_LAYOUT
    assert cache.dataset["hits/regions/chr1"].attrs["n_build_chunks"] == 1
    assert "_tmp_hits" not in cache.dataset
    assert rows.shape == (1, len(CACHE_COLUMNS))
    np.testing.assert_allclose(
        rows[0],
        np.array([5.0, 6.0, 0.0, 1.0, 3.0], dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_build_positional_cache_keeps_hit_with_start_before_tile_boundary(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    cache_path = tmp_path / "cache.zarr"
    _write_sequence_zarr(seq_path, "AATGCC")
    motif_parameters = (
        ["ATG"],
        _atg_kernel(),
        None,
        {"p0.0001": torch.tensor([2.5], dtype=torch.float32)},
        None,
        None,
    )

    build_positional_motif_hit_cache(
        genome_zarr_path=str(seq_path),
        output_path=cache_path,
        motif_parameters=motif_parameters,
        motif_metadata={
            "motif_source": "test",
            "aligned_motif_path": "test.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "abc",
            "motif_names_checksum": "def",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 2.5,
        },
        target_chromosomes=["chr1"],
        tile_size=2,
        tile_extension_bp=0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
    )

    cache = PositionalMotifHitCache(cache_path)
    rows = cache.query("chr1", 0, 6)

    assert rows.shape == (1, len(CACHE_COLUMNS))
    np.testing.assert_allclose(
        rows[0],
        np.array([1.0, 2.0, 0.0, 0.0, 3.0], dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_build_positional_cache_with_region_intervals_merges_overlaps(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    cache_path = tmp_path / "cache.zarr"
    _write_sequence_zarr(seq_path, "ATGCCATGCCATG")
    motif_parameters = (
        ["ATG"],
        _atg_kernel(),
        None,
        {"p0.0001": torch.tensor([2.5], dtype=torch.float32)},
        None,
        None,
    )
    region_intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 5],
            "end": [7, 9],
        }
    )

    build_positional_motif_hit_cache(
        genome_zarr_path=str(seq_path),
        output_path=cache_path,
        motif_parameters=motif_parameters,
        motif_metadata={
            "motif_source": "test",
            "aligned_motif_path": "test.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "abc",
            "motif_names_checksum": "def",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 2.5,
        },
        target_chromosomes=["chr1"],
        region_intervals=region_intervals,
        tile_size=5,
        tile_extension_bp=0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
    )

    cache = PositionalMotifHitCache(cache_path)
    rows = cache.query("chr1", 0, 13)

    assert cache.metadata["coverage_scope"] == "region_intervals"
    assert cache.metadata["coverage_regions"] == 1
    assert cache.metadata["coverage_bp"] == 9
    assert cache.metadata["n_regions"] == 1
    assert cache.metadata["bp_scanned"] == 12
    assert cache.dataset["hits/regions/chr1"].attrs["n_build_chunks"] == 1
    np.testing.assert_allclose(
        rows[:, 1],
        np.array([1.0, 6.0], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    assert rows.shape == (2, len(CACHE_COLUMNS))


def test_cache_metadata_validation_reports_motif_checksum_mismatch(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(cache_path), mode="w")
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0"], dtype="<U10"))
    metadata.attrs.update(
        {
            "schema_version": "positional_hit_cache_v1",
            "motif_source": "aligned_pt",
            "aligned_motif_path": "motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "old",
            "motif_names_checksum": "names",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 1.0,
            "cache_accumulation_semantics": CACHE_ACCUMULATION_SEMANTICS,
            "coordinate_frame": CACHE_COORDINATE_FRAME,
            "hit_storage_layout": HIT_STORAGE_LAYOUT,
            "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
        }
    )

    cache = PositionalMotifHitCache(cache_path)
    mismatches = cache.validate_metadata(
        {
            "motif_source": "aligned_pt",
            "aligned_motif_path": "motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": (1, 3, 4),
            "motif_kernel_checksum": "new",
            "motif_names_checksum": "names",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 1.0,
        }
    )

    assert any("motif_kernel_checksum" in mismatch for mismatch in mismatches)


def test_cache_metadata_validation_reports_threshold_vector_checksum_mismatch(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(cache_path), mode="w")
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0"], dtype="<U10"))
    metadata.attrs.update(
        {
            "schema_version": "positional_hit_cache_v1",
            "motif_source": "aligned_pt",
            "aligned_motif_path": "motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "kernel",
            "motif_names_checksum": "names",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 1.0,
            "score_threshold_vector_checksum": threshold_vector_checksum(
                torch.tensor([1.0], dtype=torch.float32)
            ),
            "score_threshold_vector_shape": [1],
            "cache_accumulation_semantics": CACHE_ACCUMULATION_SEMANTICS,
            "coordinate_frame": CACHE_COORDINATE_FRAME,
            "hit_storage_layout": HIT_STORAGE_LAYOUT,
            "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
        }
    )

    cache = PositionalMotifHitCache(cache_path)
    mismatches = cache.validate_metadata(
        {
            "motif_source": "aligned_pt",
            "aligned_motif_path": "motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": (1, 3, 4),
            "motif_kernel_checksum": "kernel",
            "motif_names_checksum": "names",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 1.0,
            "score_threshold_vector_checksum": threshold_vector_checksum(
                torch.tensor([2.0], dtype=torch.float32)
            ),
            "score_threshold_vector_shape": [1],
        }
    )

    assert any("score_threshold_vector_checksum" in mismatch for mismatch in mismatches)


def test_cache_metadata_validation_rejects_wrong_coordinate_semantics(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    root = zarr.open_group(str(cache_path), mode="w")
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0"], dtype="<U10"))
    metadata.attrs.update(
        {
            "schema_version": "positional_hit_cache_v1",
            "motif_source": "aligned_pt",
            "aligned_motif_path": "motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "kernel",
            "motif_names_checksum": "names",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 1.0,
            "cache_accumulation_semantics": "legacy_peak_aggregate",
            "coordinate_frame": "motif_start",
            "hit_storage_layout": HIT_STORAGE_LAYOUT,
            "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
        }
    )

    cache = PositionalMotifHitCache(cache_path)
    mismatches = cache.validate_metadata(
        {
            "motif_source": "aligned_pt",
            "aligned_motif_path": "motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": (1, 3, 4),
            "motif_kernel_checksum": "kernel",
            "motif_names_checksum": "names",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 1.0,
        }
    )

    assert any("cache_accumulation_semantics" in mismatch for mismatch in mismatches)
    assert any("coordinate_frame" in mismatch for mismatch in mismatches)


def test_build_positional_cache_streams_internally_then_finalizes_flat_array(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    cache_path = tmp_path / "cache.zarr"
    _write_sequence_zarr(seq_path, "ATGCCATGCCATG")
    motif_parameters = (
        ["ATG"],
        _atg_kernel(),
        None,
        {"p0.0001": torch.tensor([2.5], dtype=torch.float32)},
        None,
        None,
    )

    build_positional_motif_hit_cache(
        genome_zarr_path=str(seq_path),
        output_path=cache_path,
        motif_parameters=motif_parameters,
        motif_metadata={
            "motif_source": "test",
            "aligned_motif_path": "test.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "abc",
            "motif_names_checksum": "def",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 2.5,
        },
        target_chromosomes=["chr1"],
        tile_size=5,
        tile_extension_bp=0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
    )

    cache = PositionalMotifHitCache(cache_path)
    chrom_array = cache.dataset["hits/regions/chr1"]
    chunk_index = cache.dataset["hits/chunk_index/chr1"]

    assert "_tmp_hits" not in cache.dataset
    assert chrom_array.shape == (3, len(CACHE_COLUMNS))
    assert chrom_array.dtype == CACHE_ROW_DTYPE
    assert chrom_array.attrs["storage_layout"] == HIT_STORAGE_LAYOUT
    assert chrom_array.attrs["n_build_chunks"] == 3
    assert cache.metadata["hit_chunk_index_schema_version"] == HIT_CHUNK_INDEX_SCHEMA_VERSION
    assert chunk_index.attrs["columns"] == HIT_CHUNK_INDEX_COLUMNS
    assert chunk_index.attrs["schema_version"] == HIT_CHUNK_INDEX_SCHEMA_VERSION
    assert chunk_index.shape == (1, len(HIT_CHUNK_INDEX_COLUMNS))

    rows = cache.query("chr1", 0, 13)
    assert cache._chrom_row_cache == {}
    np.testing.assert_allclose(
        rows[:, 0],
        np.array([0.0, 5.0, 10.0], dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_cache_rows_and_accumulation_match_direct_dense_scan_on_toy_genome(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    cache_path = tmp_path / "cache.zarr"
    sequence = "ATGCCATGCCATG"
    _write_sequence_zarr(seq_path, sequence)
    kernel = _atg_kernel()
    thresholds = {"p0.0001": torch.tensor([2.5], dtype=torch.float32)}
    motif_parameters = (["ATG"], kernel, None, thresholds, None, None)

    build_positional_motif_hit_cache(
        genome_zarr_path=str(seq_path),
        output_path=cache_path,
        motif_parameters=motif_parameters,
        motif_metadata={
            "motif_source": "test",
            "aligned_motif_path": "test.pt",
            "loaded_with_aligned": True,
            "motif_count": 1,
            "motif_kernel_shape": [1, 3, 4],
            "motif_kernel_checksum": "abc",
            "motif_names_checksum": "def",
            "threshold_mode": "explicit_score",
            "p_value_threshold": "p0.0001",
            "score_threshold": 2.5,
        },
        target_chromosomes=["chr1"],
        tile_size=5,
        tile_extension_bp=0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
    )

    dense_scores = _scan_motifs_conv1d(
        torch.as_tensor(_one_hot(sequence)).unsqueeze(0).permute(0, 2, 1),
        kernel.permute(0, 2, 1),
    )
    direct_rows = collect_positional_hits(
        dense_scores,
        thresholds["p0.0001"],
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        strand_id=0,
        motif_length=3,
        sequence_lengths=torch.tensor([len(sequence)], dtype=torch.int64),
    ).numpy()

    cache = PositionalMotifHitCache(cache_path)
    cached_rows = cache.query("chr1", 0, len(sequence))
    np.testing.assert_allclose(cached_rows, direct_rows, rtol=0, atol=0)

    regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [len(sequence)]})
    one_to_all = cache.accumulate_regions(regions, window_size=5, n_motifs=1, query_motif_index=0)
    all_to_all = cache.accumulate_regions(regions, window_size=5, n_motifs=1)
    expected_profile = np.zeros((1, 11), dtype=np.float32)
    expected_profile[0, [0, 5, 10]] = [6, 9, 6]

    np.testing.assert_allclose(one_to_all, expected_profile, rtol=0, atol=0)
    np.testing.assert_allclose(all_to_all, expected_profile.reshape(1, 1, 11), rtol=0, atol=0)
