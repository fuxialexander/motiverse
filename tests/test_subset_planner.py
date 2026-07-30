from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr

pytest.importorskip("torch")

from motiverse.positional_cache import (
    CACHE_COLUMNS,
    PositionalMotifHitCache,
)
from motiverse.subset_planner import (
    _add_contribution_to_subset_values,
    accumulate_region_subsets,
    accumulate_region_subsets_from_interval_contribution_cache,
    accumulate_region_subsets_from_interval_contribution_cache_to_zarr,
    accumulate_region_subsets_to_zarr,
    build_interval_contribution_cache,
    plan_region_subset_aggregation,
)


def _write_cache(path):
    root = zarr.open_group(str(path), mode="w")
    metadata = root.require_group("metadata")
    metadata.create_array("motif_names", data=np.asarray(["M0", "M1"], dtype="<U10"))
    metadata.attrs.update(
        {
            "motif_count": 2,
            "schema_version": "positional_hit_cache_v1",
            "motif_source": "aligned_pt",
            "aligned_motif_path": "/tmp/motifs_with_rc_aligned.pt",
            "loaded_with_aligned": True,
            "motif_names_checksum": "names123",
            "motif_kernel_checksum": "kernel456",
            "motif_kernel_shape": [2, 31, 4],
            "threshold_mode": "score_threshold",
            "coordinate_frame": "active_core_center",
        }
    )
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
    arr.attrs["n_hits"] = len(rows)


def _memberships():
    return pd.DataFrame(
        {
            "subset": ["S1", "S2", "S2", "S3"],
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [9, 9, 19, 9],
            "end": [13, 13, 21, 13],
        }
    )


def test_add_contribution_to_subset_values_updates_numpy_in_place():
    values = np.zeros((4, 2, 3), dtype=np.float32)
    contribution = np.arange(6, dtype=np.float32).reshape(2, 3)

    _add_contribution_to_subset_values(
        values,
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([1, 2, 1], dtype=np.int64),
        contribution,
        dtype=np.dtype("float32"),
        chunk_subsets=2,
    )

    expected = np.zeros_like(values)
    expected[0] += contribution
    expected[1] += contribution * np.float32(2)
    expected[3] += contribution
    np.testing.assert_array_equal(values, expected)


def test_add_contribution_to_subset_values_writes_back_array_like(tmp_path):
    class ArrayLike:
        def __init__(self):
            self.data = np.zeros((3, 2, 3), dtype=np.float32)

        def __getitem__(self, key):
            return self.data[key].copy()

        def __setitem__(self, key, value):
            self.data[key] = value

    values = ArrayLike()
    contribution = np.arange(6, dtype=np.float32).reshape(2, 3)

    _add_contribution_to_subset_values(
        values,
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        contribution,
        dtype=np.dtype("float32"),
        chunk_subsets=1,
    )

    expected = np.zeros((3, 2, 3), dtype=np.float32)
    expected[0] += contribution
    expected[2] += contribution * np.float32(3)
    np.testing.assert_array_equal(values.data, expected)


def test_plan_region_subset_aggregation_reports_reuse_and_output_bytes():
    plan = plan_region_subset_aggregation(
        _memberships(),
        subset_column="subset",
        n_motifs=2,
        window_size=1,
    )

    assert plan.n_subsets == 3
    assert plan.n_input_rows == 4
    assert plan.n_dropped_invalid_regions == 0
    assert plan.n_region_memberships == 4
    assert plan.n_unique_regions == 2
    assert plan.reused_region_memberships == 2
    assert plan.reuse_factor == 2.0
    assert plan.output_shape == (3, 2, 2, 3)
    assert plan.output_bytes == 3 * 2 * 2 * 3 * 4


def test_accumulate_region_subsets_matches_repeated_cache_accumulation(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)
    memberships = _memberships()

    result = accumulate_region_subsets(
        cache,
        memberships,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )

    expected = []
    for subset_id in result.subset_ids:
        subset_regions = memberships[memberships["subset"] == subset_id]
        expected.append(cache.accumulate_regions(subset_regions, window_size=1, n_motifs=2))

    assert result.subset_ids == ["S1", "S2", "S3"]
    np.testing.assert_allclose(result.values, np.stack(expected, axis=0), rtol=0, atol=0)


def test_accumulate_region_subsets_one_to_all_matches_repeated_cache_accumulation(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)
    memberships = _memberships()

    result = accumulate_region_subsets(
        cache,
        memberships,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
    )

    expected = []
    for subset_id in result.subset_ids:
        subset_regions = memberships[memberships["subset"] == subset_id]
        expected.append(
            cache.accumulate_regions(
                subset_regions,
                window_size=1,
                n_motifs=2,
                query_motif_index=0,
            )
        )

    assert result.values.shape == (3, 2, 3)
    np.testing.assert_allclose(result.values, np.stack(expected, axis=0), rtol=0, atol=0)


def test_accumulate_region_subsets_duplicate_membership_counts_twice(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S1"],
            "Chromosome": ["chr1", "chr1"],
            "Start": [9, 9],
            "End": [13, 13],
        }
    )

    result = accumulate_region_subsets(
        cache,
        memberships,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    single = cache.accumulate_regions(
        pd.DataFrame({"chrom": ["chr1"], "start": [9], "end": [13]}),
        window_size=1,
        n_motifs=2,
    )

    assert result.plan.n_region_memberships == 2
    assert result.plan.n_unique_regions == 1
    np.testing.assert_allclose(result.values[0], single * 2, rtol=0, atol=0)


def test_accumulate_region_subsets_output_guard_raises(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    _write_cache(cache_path)

    with pytest.raises(MemoryError, match="Planned dense subset output"):
        accumulate_region_subsets(
            PositionalMotifHitCache(cache_path),
            _memberships(),
            subset_column="subset",
            window_size=500,
            n_motifs=10,
            max_output_bytes=100,
        )


def test_accumulate_region_subsets_to_zarr_matches_dense_result(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_path = tmp_path / "subsets.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)

    dense = accumulate_region_subsets(
        cache,
        _memberships(),
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    zarr_result = accumulate_region_subsets_to_zarr(
        cache,
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        chunk_subsets=2,
    )

    root = zarr.open_group(str(output_path), mode="r")

    assert zarr_result.subset_ids == dense.subset_ids
    assert root["metadata/subset_ids"][:].tolist() == dense.subset_ids
    assert root["metadata"].attrs["schema_version"] == "subset_motif_aggregation_v1"
    assert root["metadata"].attrs["complete"] is True
    assert root["metadata"].attrs["n_unique_regions"] == dense.plan.n_unique_regions
    assert root["metadata"].attrs["values_checksum"] == zarr_result.checksum
    assert root["metadata"].attrs["values_checksum_mode"] == "full"
    assert root["metadata"].attrs["values_checksum_status"] == "complete"
    assert zarr_result.timings["unique_regions_processed"] == 2
    assert zarr_result.timings["nonzero_contribution_regions"] == 2
    assert root["metadata"].attrs["query_wall_s"] >= 0
    assert root["metadata"].attrs["accumulate_wall_s"] >= 0
    assert root["metadata"].attrs["subset_update_wall_s"] >= 0
    assert root["metadata"].attrs["checksum_wall_s"] >= 0
    cache_metadata = root["metadata"].attrs["cache_metadata"]
    assert cache_metadata["motif_source"] == "aligned_pt"
    assert cache_metadata["aligned_motif_path"].endswith("motifs_with_rc_aligned.pt")
    assert cache_metadata["motif_names_checksum"] == "names123"
    assert cache_metadata["motif_kernel_checksum"] == "kernel456"
    assert cache_metadata["motif_kernel_shape"] == [2, 31, 4]
    assert cache_metadata["coordinate_frame"] == "active_core_center"
    assert root["values"].chunks == (2, 2, 2, 3)
    np.testing.assert_allclose(root["values"][:], dense.values, rtol=0, atol=0)


def test_accumulate_region_subsets_to_zarr_sample_checksum_mode(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_path = tmp_path / "subsets.zarr"
    _write_cache(cache_path)

    result = accumulate_region_subsets_to_zarr(
        PositionalMotifHitCache(cache_path),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        checksum_mode="sample",
        checksum_sample_subsets=2,
    )
    root = zarr.open_group(str(output_path), mode="r")
    attrs = root["metadata"].attrs

    assert result.checksum == attrs["values_checksum"]
    assert attrs["values_checksum_mode"] == "sample"
    assert attrs["values_checksum_status"] == "sampled"
    assert attrs["values_checksum_sample_subsets"] == 2
    assert attrs["values_checksum_sample_indices"] == [0, 2]


def test_accumulate_region_subsets_to_zarr_none_checksum_mode(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_path = tmp_path / "subsets.zarr"
    _write_cache(cache_path)

    result = accumulate_region_subsets_to_zarr(
        PositionalMotifHitCache(cache_path),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        checksum_mode="none",
    )
    root = zarr.open_group(str(output_path), mode="r")
    attrs = root["metadata"].attrs

    assert result.checksum is None
    assert attrs["values_checksum"] is None
    assert attrs["values_checksum_mode"] == "none"
    assert attrs["values_checksum_status"] == "skipped"
    assert attrs["values_checksum_sample_subsets"] == 0


def test_accumulate_region_subsets_to_zarr_rejects_bad_checksum_mode(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    _write_cache(cache_path)

    with pytest.raises(ValueError, match="checksum_mode"):
        accumulate_region_subsets_to_zarr(
            PositionalMotifHitCache(cache_path),
            _memberships(),
            output_path=tmp_path / "subsets.zarr",
            subset_column="subset",
            window_size=1,
            n_motifs=2,
            checksum_mode="fastish",
        )


def test_accumulate_region_subsets_to_zarr_duplicate_membership_counts_twice(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_path = tmp_path / "subsets.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S1"],
            "chrom": ["chr1", "chr1"],
            "start": [9, 9],
            "end": [13, 13],
        }
    )

    zarr_result = accumulate_region_subsets_to_zarr(
        cache,
        memberships,
        output_path=output_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    root = zarr.open_group(str(output_path), mode="r")
    single = cache.accumulate_regions(
        pd.DataFrame({"chrom": ["chr1"], "start": [9], "end": [13]}),
        window_size=1,
        n_motifs=2,
    )

    assert zarr_result.plan.n_region_memberships == 2
    assert zarr_result.plan.n_unique_regions == 1
    np.testing.assert_allclose(root["values"][0], single * 2, rtol=0, atol=0)


def test_accumulate_region_subsets_to_zarr_one_to_all_matches_dense_result(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_path = tmp_path / "subsets.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)

    dense = accumulate_region_subsets(
        cache,
        _memberships(),
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
    )
    zarr_result = accumulate_region_subsets_to_zarr(
        cache,
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
    )

    root = zarr.open_group(str(output_path), mode="r")
    assert zarr_result.plan.output_shape == (3, 2, 3)
    np.testing.assert_allclose(root["values"][:], dense.values, rtol=0, atol=0)


def test_accumulate_region_subsets_to_zarr_empty_and_invalid_rows_recorded(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_path = tmp_path / "subsets.zarr"
    _write_cache(cache_path)
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2", "S3", "S4"],
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [10, 20, -1, "bad"],
            "end": [10, 19, 5, 30],
        }
    )

    result = accumulate_region_subsets_to_zarr(
        PositionalMotifHitCache(cache_path),
        memberships,
        output_path=output_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    root = zarr.open_group(str(output_path), mode="r")

    assert result.plan.n_input_rows == 4
    assert result.plan.n_dropped_invalid_regions == 4
    assert result.plan.output_shape == (0, 2, 2, 3)
    assert root["metadata"].attrs["complete"] is True
    assert root["metadata"].attrs["n_dropped_invalid_regions"] == 4
    assert root["values"].shape == (0, 2, 2, 3)


def test_accumulate_region_subsets_to_zarr_chunk_size_does_not_change_values(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    output_a = tmp_path / "subsets_a.zarr"
    output_b = tmp_path / "subsets_b.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)

    a = accumulate_region_subsets_to_zarr(
        cache,
        _memberships(),
        output_path=output_a,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        chunk_subsets=1,
    )
    b = accumulate_region_subsets_to_zarr(
        cache,
        _memberships(),
        output_path=output_b,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
        chunk_subsets=3,
    )

    assert a.checksum == b.checksum


def test_accumulate_region_subsets_to_zarr_output_guard_raises(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    _write_cache(cache_path)

    with pytest.raises(MemoryError, match="Planned zarr subset output"):
        accumulate_region_subsets_to_zarr(
            PositionalMotifHitCache(cache_path),
            _memberships(),
            output_path=tmp_path / "subsets.zarr",
            subset_column="subset",
            window_size=500,
            n_motifs=10,
            max_output_bytes=100,
        )


def test_interval_contribution_cache_matches_direct_subset_aggregation(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    contribution_path = tmp_path / "contributions.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)

    build = build_interval_contribution_cache(
        cache,
        _memberships(),
        output_path=contribution_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    direct = accumulate_region_subsets(
        cache,
        _memberships(),
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    from_contributions = accumulate_region_subsets_from_interval_contribution_cache(
        contribution_path,
        _memberships(),
        subset_column="subset",
    )

    root = zarr.open_group(str(contribution_path), mode="r")
    attrs = root["metadata"].attrs

    assert build.n_unique_regions == 2
    assert build.contribution_shape == (2, 2, 3)
    assert attrs["schema_version"] == "interval_contribution_cache_v1"
    assert attrs["semantics"] == "sparse_hit_contribution_per_exact_interval"
    assert attrs["complete"] is True
    assert attrs["n_unique_regions"] == direct.plan.n_unique_regions
    assert attrs["cache_metadata"]["motif_source"] == "aligned_pt"
    assert from_contributions.subset_ids == direct.subset_ids
    assert from_contributions.timings["region_query_strategy"] == ("interval_contribution_cache")
    assert from_contributions.timings["contribution_cache_hits"] == 2
    np.testing.assert_allclose(
        from_contributions.values,
        direct.values,
        rtol=0,
        atol=0,
    )


def test_interval_contribution_cache_zarr_matches_direct_subset_aggregation(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    contribution_path = tmp_path / "contributions.zarr"
    output_path = tmp_path / "subset_from_contributions.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)

    build_interval_contribution_cache(
        cache,
        _memberships(),
        output_path=contribution_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    direct = accumulate_region_subsets(
        cache,
        _memberships(),
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    zarr_result = accumulate_region_subsets_from_interval_contribution_cache_to_zarr(
        contribution_path,
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        chunk_subsets=2,
    )
    root = zarr.open_group(str(output_path), mode="r")
    attrs = root["metadata"].attrs

    assert zarr_result.subset_ids == direct.subset_ids
    assert attrs["schema_version"] == "subset_motif_aggregation_v1"
    assert attrs["complete"] is True
    assert attrs["region_query_strategy"] == "interval_contribution_cache"
    assert attrs["contribution_cache_metadata"]["schema_version"] == (
        "interval_contribution_cache_v1"
    )
    assert attrs["cache_metadata"]["motif_source"] == "aligned_pt"
    assert attrs["contribution_cache_hits"] == 2
    assert root["values"].chunks == (2, 2, 2, 3)
    np.testing.assert_allclose(root["values"][:], direct.values, rtol=0, atol=0)


def test_interval_contribution_cache_buffered_zarr_matches_direct_subset_aggregation(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    contribution_path = tmp_path / "contributions.zarr"
    output_path = tmp_path / "subset_from_contributions.zarr"
    _write_cache(cache_path)
    cache = PositionalMotifHitCache(cache_path)

    build_interval_contribution_cache(
        cache,
        _memberships(),
        output_path=contribution_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    direct = accumulate_region_subsets(
        cache,
        _memberships(),
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    accumulate_region_subsets_from_interval_contribution_cache_to_zarr(
        contribution_path,
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        chunk_subsets=2,
        buffered_subset_updates=True,
        max_buffered_output_bytes=10_000,
    )
    root = zarr.open_group(str(output_path), mode="r")
    attrs = root["metadata"].attrs

    assert attrs["subset_update_mode"] == "buffered_full_output"
    assert attrs["buffered_output_bytes"] == direct.values.nbytes
    np.testing.assert_allclose(root["values"][:], direct.values, rtol=0, atol=0)


def test_interval_contribution_cache_requires_matching_intervals(tmp_path):
    cache_path = tmp_path / "cache.zarr"
    contribution_path = tmp_path / "contributions.zarr"
    _write_cache(cache_path)

    build_interval_contribution_cache(
        PositionalMotifHitCache(cache_path),
        _memberships(),
        output_path=contribution_path,
        subset_column="subset",
        window_size=1,
        n_motifs=2,
    )
    extra_interval = pd.DataFrame(
        {
            "subset": ["S1"],
            "chrom": ["chr1"],
            "start": [30],
            "end": [35],
        }
    )

    with pytest.raises(ValueError, match="missing required interval"):
        accumulate_region_subsets_from_interval_contribution_cache(
            contribution_path,
            extra_interval,
            subset_column="subset",
        )
