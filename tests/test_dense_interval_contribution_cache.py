from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.core.dtype import VariableLengthUTF8

torch = pytest.importorskip("torch")

from motiverse.dense_subset import (
    DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION,
    DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS,
    DENSE_SUBSET_COORDINATE_FRAME,
    DENSE_SUBSET_SEMANTICS,
    _read_dense_query_motif_hit_cache_directory,
    accumulate_dense_score_region_subsets,
    accumulate_dense_score_region_subsets_from_contribution_cache,
    accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr,
    accumulate_dense_score_region_subsets_from_query_motif_hit_cache,
    accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter,
    build_dense_interval_contribution_cache,
    build_dense_query_motif_hit_cache,
    build_dense_query_motif_hit_cache_from_tiles,
    dense_query_motif_hits_from_scores,
    dense_region_contribution_from_scores,
    dense_region_contribution_from_scores_and_query_motif_hits,
    plan_dense_interval_contribution_cache_resources,
)


def _scores_a():
    return torch.tensor(
        [[[0.0, 1.0, 0.0, 2.0, 0.0], [2.0, 0.0, 3.0, 0.0, 4.0]]],
        dtype=torch.float32,
    )


def _scores_b():
    return torch.tensor(
        [[[0.0, 0.0, 1.5, 0.0, 0.0], [0.5, 2.5, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )


def _thresholds():
    return {"p0.0001": torch.tensor([0.5, 1.5], dtype=torch.float32)}


def _target_thresholds():
    return {"p0.0001": torch.tensor([0.5], dtype=torch.float32)}


def _scores_no_target():
    return torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 2.0, 1.0, 3.0]]],
        dtype=torch.float32,
    )


def _memberships():
    return pd.DataFrame(
        {
            "subset": ["S1", "S2", "S2", "S3"],
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [0, 0, 10, 0],
            "end": [5, 5, 15, 5],
        }
    )


def _memberships_with_no_target_interval():
    return pd.DataFrame(
        {
            "subset": ["S1", "S2", "S2", "S3", "S4"],
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
            "start": [0, 0, 10, 0, 20],
            "end": [5, 5, 15, 5, 25],
        }
    )


def _provider_factory(calls):
    score_by_region = {
        ("chr1", 0, 5): _scores_a(),
        ("chr1", 10, 15): _scores_b(),
        ("chr1", 20, 25): _scores_no_target(),
    }

    def provider(chrom, start, end):
        key = (chrom, start, end)
        calls[key] = calls.get(key, 0) + 1
        return score_by_region[key]

    return provider


def _target_provider_factory(calls):
    full_provider = _provider_factory(calls)

    def provider(chrom, start, end):
        return [full_provider(chrom, start, end)[:, 0:1, :]]

    return provider


def _tile_target_provider_factory(calls):
    target_track = np.zeros((31,), dtype=np.float32)
    target_track[[1, 3, 12]] = [1.0, 2.0, 1.5]

    def provider(chrom, start, end):
        key = (chrom, start, end)
        calls[key] = calls.get(key, 0) + 1
        return [torch.from_numpy(target_track[int(start) : int(end)][None, None, :])]

    return provider


def _motif_length_aware_target_provider_factory(calls):
    motif_length = 31
    hits = {1469: 2.0, 1470: 2.0}

    def provider(chrom, start, end):
        key = (chrom, start, end)
        calls[key] = calls.get(key, 0) + 1
        score_length = max(0, int(end) - int(start) - motif_length + 1)
        values = np.zeros((1, 1, score_length), dtype=np.float32)
        for genomic_position, score in hits.items():
            local_position = int(genomic_position) - int(start)
            if 0 <= local_position < score_length:
                values[0, 0, local_position] = score
        return [torch.from_numpy(values)]

    provider.motif_length = motif_length
    return provider


def _checksum(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def _write_single_interval_sparse_cache(full_cache_path, sparse_cache_path):
    source = zarr.open_group(str(full_cache_path), mode="r")
    sparse = zarr.open_group(str(sparse_cache_path), mode="w")
    metadata = sparse.require_group("metadata", overwrite=True)
    metadata.attrs.update(dict(source["metadata"].attrs))
    metadata.attrs["sparse_nonzero_contributions"] = True
    metadata.attrs["n_unique_regions"] = 1
    metadata.attrs["contribution_cache_bytes"] = int(source["values"][0].nbytes)
    intervals = sparse.require_group("intervals", overwrite=True)
    intervals.create_array(
        "chrom",
        shape=(1,),
        dtype=VariableLengthUTF8(),
        overwrite=True,
    )[:] = [source["intervals/chrom"][0]]
    intervals.create_array(
        "start",
        data=np.asarray([source["intervals/start"][0]], dtype=np.int64),
        overwrite=True,
    )
    intervals.create_array(
        "end",
        data=np.asarray([source["intervals/end"][0]], dtype=np.int64),
        overwrite=True,
    )
    values = sparse.create_array(
        "values",
        data=np.asarray(source["values"][0:1]),
        overwrite=True,
    )
    return np.asarray(values[0])


def test_dense_interval_contribution_cache_plan_reports_full_curve_storage():
    plan = plan_dense_interval_contribution_cache_resources(
        _memberships(),
        subset_column="subset",
        n_motifs=2,
        window_size=1,
        query_motif_index=0,
        dtype=np.float32,
        max_cache_bytes=40,
    )
    summary = plan.to_dict()

    assert plan.subset_plan.n_region_memberships == 4
    assert plan.subset_plan.n_unique_regions == 2
    assert plan.contribution_shape == (2, 3)
    assert plan.cache_bytes_per_unique_region == 2 * 3 * 4
    assert plan.contribution_cache_bytes == 48
    assert plan.cache_feasible_under_max is False
    assert plan.recommended_strategy == (
        "skip_exact_interval_cache_use_union_direct_or_tiled_cache"
    )
    assert summary["schema_version"] == "dense_interval_contribution_cache_plan_v1"
    assert summary["contribution_shape"] == [2, 3]
    assert summary["subset_plan"]["reuse_factor"] == 2.0
    first_sparse = summary["sparse_nonzero_scenarios"][0]
    assert first_sparse["nonzero_fraction"] == 0.01
    assert first_sparse["nonzero_region_count"] == 1
    assert first_sparse["sparse_contribution_cache_bytes"] == 24
    assert first_sparse["sparse_cache_to_dense_cache_ratio"] == 0.5
    assert first_sparse["sparse_cache_to_output_bytes_ratio"] == pytest.approx(24 / 72)
    assert first_sparse["sparse_cache_feasible_under_max"] is True


def test_query_motif_hit_contribution_matches_direct_dense_region_contribution():
    scores = _scores_a()
    direct = dense_region_contribution_from_scores(
        scores,
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    anchors = dense_query_motif_hits_from_scores(
        scores[:, 0:1, :],
        score_thresholds=_target_thresholds(),
        query_motif_index=0,
        window_size=1,
        device="cpu",
    )
    anchored = dense_region_contribution_from_scores_and_query_motif_hits(
        scores,
        anchors,
        window_size=1,
        n_motifs=2,
        device="cpu",
    )

    assert anchors.n_hits == 2
    np.testing.assert_allclose(anchored, direct, rtol=0, atol=0)


def test_query_motif_hit_prefilter_matches_direct_dense_subset_and_skips_nohit():
    memberships = _memberships_with_no_target_interval()
    direct_calls = {}
    direct = accumulate_dense_score_region_subsets(
        _provider_factory(direct_calls),
        memberships,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    target_calls = {}
    full_calls = {}
    prefiltered = accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter(
        _target_provider_factory(target_calls),
        _provider_factory(full_calls),
        memberships,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )

    assert prefiltered.subset_ids == direct.subset_ids
    assert prefiltered.timings["query_motif_hit_hit_regions"] == 2
    assert prefiltered.timings["full_score_regions_scanned"] == 2
    assert prefiltered.timings["full_score_regions_skipped"] == 1
    assert ("chr1", 20, 25) not in full_calls
    np.testing.assert_allclose(prefiltered.values, direct.values, rtol=0, atol=0)


def test_dense_query_motif_hit_cache_replay_matches_direct_and_skips_nohit(tmp_path):
    memberships = _memberships_with_no_target_interval()
    direct = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        memberships,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    target_calls = {}
    cache_path = tmp_path / "query_motif_hits.zarr"
    cache = build_dense_query_motif_hit_cache(
        _target_provider_factory(target_calls),
        memberships,
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )
    full_calls = {}
    replay = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        cache_path,
        _provider_factory(full_calls),
        memberships,
        subset_column="subset",
        device="cpu",
    )
    coalesced_full_calls = {}
    coalesced_replay = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        cache_path,
        _provider_factory(coalesced_full_calls),
        memberships,
        subset_column="subset",
        device="cpu",
        coalesce_membership_patterns=True,
    )
    attrs, *_ = _read_dense_query_motif_hit_cache_directory(cache_path)

    assert cache.n_unique_regions == 3
    assert cache.n_anchor_hits == 3
    assert attrs["schema_version"] == "dense_query_motif_hit_cache_v1"
    assert attrs["semantics"] == "query_motif_hit_positions_per_exact_interval"
    assert attrs["motif_source"] == "aligned_pt"
    assert attrs["loaded_with_aligned"] is True
    assert attrs["query_motif_hit_hits"] == 3
    assert attrs["query_motif_hit_hit_regions"] == 2
    assert target_calls == {
        ("chr1", 0, 5): 1,
        ("chr1", 10, 15): 1,
        ("chr1", 20, 25): 1,
    }
    assert full_calls == {
        ("chr1", 0, 5): 1,
        ("chr1", 10, 15): 1,
    }
    assert coalesced_full_calls == {
        ("chr1", 0, 5): 1,
        ("chr1", 10, 15): 1,
    }
    assert replay.timings["region_query_strategy"] == "dense_query_motif_hit_cache_replay"
    assert replay.timings["query_motif_hit_cache_hits"] == 3
    assert replay.timings["full_score_regions_scanned"] == 2
    assert replay.timings["full_score_regions_skipped"] == 1
    assert coalesced_replay.timings["membership_pattern_coalescing"] is True
    assert coalesced_replay.timings["membership_pattern_count"] == 2
    assert coalesced_replay.timings["membership_pattern_interval_assignments"] == 2
    assert coalesced_replay.timings["membership_pattern_accumulator_bytes"] == 48
    np.testing.assert_allclose(replay.values, direct.values, rtol=0, atol=0)
    np.testing.assert_allclose(coalesced_replay.values, direct.values, rtol=0, atol=0)

    with pytest.raises(MemoryError, match="membership-pattern accumulator"):
        accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
            cache_path,
            _provider_factory({}),
            memberships,
            subset_column="subset",
            device="cpu",
            coalesce_membership_patterns=True,
            max_pattern_accumulator_bytes=47,
        )


def test_tile_assisted_dense_query_motif_hit_cache_matches_direct_cache_and_replay(
    tmp_path,
):
    memberships = _memberships_with_no_target_interval()
    direct = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        memberships,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    direct_cache_path = tmp_path / "direct_query_motif_hits.zarr"
    direct_cache = build_dense_query_motif_hit_cache(
        _target_provider_factory({}),
        memberships,
        output_path=direct_cache_path,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )
    tile_calls = {}
    tile_cache_path = tmp_path / "tile_query_motif_hits.zarr"
    tile_cache = build_dense_query_motif_hit_cache_from_tiles(
        _tile_target_provider_factory(tile_calls),
        memberships,
        output_path=tile_cache_path,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        tile_size=10,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )
    replay = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        tile_cache_path,
        _provider_factory({}),
        memberships,
        subset_column="subset",
        device="cpu",
    )
    (
        _direct_attrs,
        _direct_key_to_index,
        _direct_anchor_starts,
        direct_anchor_counts,
        _direct_batch_indices,
        direct_positions,
    ) = _read_dense_query_motif_hit_cache_directory(direct_cache_path)
    (
        tile_attrs,
        _tile_key_to_index,
        _tile_anchor_starts,
        tile_anchor_counts,
        _tile_batch_indices,
        tile_positions,
    ) = _read_dense_query_motif_hit_cache_directory(tile_cache_path)

    assert tile_calls == {
        ("chr1", 0, 11): 1,
        ("chr1", 9, 21): 1,
        ("chr1", 19, 31): 1,
    }
    assert tile_cache.n_unique_regions == direct_cache.n_unique_regions == 3
    assert tile_cache.n_anchor_hits == direct_cache.n_anchor_hits == 3
    assert tile_cache.checksum == direct_cache.checksum
    assert tile_attrs["query_motif_hit_cache_build_mode"] == (
        "tile_assisted_exact_interval_reconstruction"
    )
    assert tile_attrs["tile_size_bp"] == 10
    assert tile_attrs["tile_extension_bp"] == 1
    assert tile_attrs["unique_tiles_processed"] == 3
    assert tile_attrs["tile_anchor_hits"] == 3
    np.testing.assert_array_equal(tile_anchor_counts, direct_anchor_counts)
    np.testing.assert_array_equal(tile_positions, direct_positions)
    np.testing.assert_allclose(replay.values, direct.values, rtol=0, atol=0)


def test_tile_assisted_query_motif_hit_cache_filters_right_edge_by_motif_length(
    tmp_path,
):
    memberships = pd.DataFrame(
        {
            "subset": ["S1"],
            "chrom": ["chr1"],
            "start": [0],
            "end": [2000],
        }
    )
    direct_cache_path = tmp_path / "direct_query_motif_hits.zarr"
    build_dense_query_motif_hit_cache(
        _motif_length_aware_target_provider_factory({}),
        memberships,
        output_path=direct_cache_path,
        subset_column="subset",
        score_thresholds={"p0.0001": torch.tensor([0.5], dtype=torch.float32)},
        window_size=500,
        n_motifs=1,
        query_motif_index=0,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )
    tile_cache_path = tmp_path / "tile_query_motif_hits.zarr"
    tile_cache = build_dense_query_motif_hit_cache_from_tiles(
        _motif_length_aware_target_provider_factory({}),
        memberships,
        output_path=tile_cache_path,
        subset_column="subset",
        score_thresholds={"p0.0001": torch.tensor([0.5], dtype=torch.float32)},
        window_size=500,
        n_motifs=1,
        query_motif_index=0,
        tile_size=1000,
        tile_extension_bp=500,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )

    (
        _direct_attrs,
        _direct_key_to_index,
        _direct_anchor_starts,
        direct_anchor_counts,
        _direct_batch_indices,
        direct_positions,
    ) = _read_dense_query_motif_hit_cache_directory(direct_cache_path)
    (
        tile_attrs,
        _tile_key_to_index,
        _tile_anchor_starts,
        tile_anchor_counts,
        _tile_batch_indices,
        tile_positions,
    ) = _read_dense_query_motif_hit_cache_directory(tile_cache_path)

    assert tile_cache.n_anchor_hits == 1
    assert tile_attrs["motif_length"] == 31
    assert tile_attrs["requested_tile_extension_bp"] == 500
    assert tile_attrs["minimum_tile_extension_bp"] == 530
    assert tile_attrs["tile_extension_bp"] == 530
    assert tile_attrs["tile_anchor_boundary_filtered"] >= 1
    np.testing.assert_array_equal(tile_anchor_counts, direct_anchor_counts)
    np.testing.assert_array_equal(tile_positions, direct_positions)
    np.testing.assert_array_equal(tile_positions, np.asarray([1469], dtype=np.int64))


def test_dense_query_motif_hit_cache_rejects_missing_interval(tmp_path):
    cache_path = tmp_path / "query_motif_hits.zarr"
    build_dense_query_motif_hit_cache(
        _target_provider_factory({}),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )

    with pytest.raises(ValueError, match="missing required interval"):
        accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
            cache_path,
            _provider_factory({}),
            _memberships_with_no_target_interval(),
            subset_column="subset",
            device="cpu",
        )


def test_dense_query_motif_hit_cache_rejects_unaligned_provenance(tmp_path):
    cache_path = tmp_path / "query_motif_hits.zarr"
    build_dense_query_motif_hit_cache(
        _target_provider_factory({}),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        metadata_extra={
            "motif_source": "pwm",
            "loaded_with_aligned": False,
        },
    )

    with pytest.raises(ValueError, match="loaded_with_aligned=True"):
        accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
            cache_path,
            _provider_factory({}),
            _memberships(),
            subset_column="subset",
            device="cpu",
        )


def test_dense_query_motif_hit_cache_rejects_metadata_mismatch(tmp_path):
    cache_path = tmp_path / "query_motif_hits.zarr"
    expected_metadata = {
        "genome_zarr_path": "/data/hg38.zarr",
        "motif_count": 2,
        "motif_names_checksum": "names-a",
        "motif_kernel_checksum": "kernels-a",
        "motif_kernel_shape": [2, 1, 4],
        "aligned_motif_path": "/motifs_with_rc_aligned.pt",
        "motif_source": "aligned_pt",
        "loaded_with_aligned": True,
        "threshold_mode": "pvalue_mapping",
        "p_value_threshold": "p0.0001",
        "score_threshold": None,
        "score_threshold_vector_checksum": "threshold-a",
        "score_threshold_vector_shape": [2],
        "strand_specific": False,
        "strands": 2,
    }
    build_dense_query_motif_hit_cache(
        _target_provider_factory({}),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_target_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        metadata_extra=expected_metadata,
    )

    replay = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        cache_path,
        _provider_factory({}),
        _memberships(),
        subset_column="subset",
        device="cpu",
        expected_metadata={
            **expected_metadata,
            "window_size": 1,
            "query_motif_index": 0,
            "dtype": "float32",
        },
    )
    assert replay.values.shape == (3, 2, 3)

    with pytest.raises(ValueError, match="motif_names_checksum"):
        accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
            cache_path,
            _provider_factory({}),
            _memberships(),
            subset_column="subset",
            device="cpu",
            expected_metadata={
                **expected_metadata,
                "motif_names_checksum": "names-b",
                "window_size": 1,
                "query_motif_index": 0,
                "dtype": "float32",
            },
        )


def test_dense_interval_contribution_cache_replays_one_to_all_full_curves(tmp_path):
    direct_calls = {}
    direct = accumulate_dense_score_region_subsets(
        _provider_factory(direct_calls),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    cache_calls = {}
    cache_path = tmp_path / "dense_interval_contrib.zarr"
    cache = build_dense_interval_contribution_cache(
        _provider_factory(cache_calls),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
            "genome_zarr_path": "/home/xf2217/get_data/hg38.zarr",
        },
    )
    replay = accumulate_dense_score_region_subsets_from_contribution_cache(
        cache.path,
        _memberships(),
        subset_column="subset",
        chunk_subsets=2,
    )
    coalesced_replay = accumulate_dense_score_region_subsets_from_contribution_cache(
        cache.path,
        _memberships(),
        subset_column="subset",
        chunk_subsets=2,
        coalesce_membership_patterns=True,
    )
    metadata = dict(zarr.open_group(str(cache_path), mode="r")["metadata"].attrs)

    assert cache.n_unique_regions == 2
    assert cache.contribution_shape == (2, 3)
    assert cache.semantics == DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS
    assert metadata["schema_version"] == DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION
    assert metadata["complete"] is True
    assert metadata["semantics"] == DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS
    assert metadata["coordinate_frame"] == DENSE_SUBSET_COORDINATE_FRAME
    assert metadata["motif_source"] == "aligned_pt"
    assert metadata["loaded_with_aligned"] is True
    assert metadata["genome_zarr_path"] == "/home/xf2217/get_data/hg38.zarr"
    assert cache_calls == {("chr1", 0, 5): 1, ("chr1", 10, 15): 1}
    assert replay.subset_ids == direct.subset_ids
    assert replay.timings["region_query_strategy"] == "dense_interval_contribution_cache"
    assert replay.timings["contribution_cache_hits"] == 2
    assert coalesced_replay.timings["membership_pattern_coalescing"] is True
    assert coalesced_replay.timings["membership_pattern_count"] == 2
    assert coalesced_replay.timings["membership_pattern_interval_assignments"] == 2
    assert coalesced_replay.timings["membership_pattern_accumulator_bytes"] == 48
    assert coalesced_replay.timings["contribution_cache_read_batches"] == 1
    assert coalesced_replay.timings["contribution_cache_intervals_read"] == 2
    assert coalesced_replay.timings["contribution_cache_interval_overread"] == 0
    np.testing.assert_allclose(replay.values, direct.values, rtol=0, atol=0)
    np.testing.assert_allclose(coalesced_replay.values, direct.values, rtol=0, atol=0)


def test_dense_interval_contribution_cache_replays_to_zarr_and_new_subsets(tmp_path):
    cache_path = tmp_path / "dense_interval_contrib.zarr"
    build_dense_interval_contribution_cache(
        _provider_factory({}),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        checksum_mode="sample",
        checksum_sample_subsets=1,
    )
    alternate = pd.DataFrame(
        {
            "subset": ["A", "A", "B"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 10, 0],
            "end": [5, 15, 5],
        }
    )
    direct = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        alternate,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    output_path = tmp_path / "subset_from_contrib.zarr"
    result = accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr(
        cache_path,
        alternate,
        output_path=output_path,
        subset_column="subset",
        chunk_subsets=1,
        chunk_buffered_subset_updates=True,
        max_active_subset_chunks=1,
        metadata_extra={"motif_source": "aligned_pt"},
    )
    root = zarr.open_group(str(output_path), mode="r")
    values = np.asarray(root["values"][:])
    metadata = dict(root["metadata"].attrs)

    assert result.subset_ids == ["A", "B"]
    assert result.checksum == _checksum(direct.values)
    assert metadata["complete"] is True
    assert metadata["semantics"] == DENSE_SUBSET_SEMANTICS
    assert metadata["region_query_strategy"] == "dense_interval_contribution_cache"
    assert metadata["source_contribution_cache_path"] == str(cache_path)
    assert metadata["subset_update_mode"] == "chunk_buffered_output"
    assert metadata["contribution_cache_hits"] == 2
    assert metadata["motif_source"] == "aligned_pt"
    np.testing.assert_allclose(values, direct.values, rtol=0, atol=0)

    coalesced_output_path = tmp_path / "subset_from_contrib_coalesced.zarr"
    coalesced = accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr(
        cache_path,
        alternate,
        output_path=coalesced_output_path,
        subset_column="subset",
        chunk_subsets=1,
        coalesce_membership_patterns=True,
        max_pattern_accumulator_bytes=2 * 2 * 3 * 4,
        metadata_extra={"motif_source": "aligned_pt"},
    )
    coalesced_root = zarr.open_group(str(coalesced_output_path), mode="r")
    coalesced_values = np.asarray(coalesced_root["values"][:])
    coalesced_metadata = dict(coalesced_root["metadata"].attrs)

    assert coalesced.checksum == _checksum(direct.values)
    assert coalesced_metadata["subset_update_mode"] == ("membership_pattern_coalesced_output")
    assert coalesced_metadata["membership_pattern_coalescing"] is True
    assert coalesced_metadata["membership_pattern_count"] == 2
    assert coalesced_metadata["membership_pattern_accumulator_bytes"] == 48
    assert coalesced_metadata["contribution_cache_read_batches"] == 1
    assert coalesced_metadata["contribution_cache_intervals_read"] == 2
    assert coalesced_metadata["contribution_cache_interval_overread"] == 0
    np.testing.assert_allclose(coalesced_values, direct.values, rtol=0, atol=0)


def test_dense_interval_contribution_cache_coalesced_replay_obeys_memory_guard(
    tmp_path,
):
    cache_path = tmp_path / "dense_interval_contrib.zarr"
    build_dense_interval_contribution_cache(
        _provider_factory({}),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )

    with pytest.raises(MemoryError, match="membership-pattern accumulator"):
        accumulate_dense_score_region_subsets_from_contribution_cache(
            cache_path,
            _memberships(),
            subset_column="subset",
            coalesce_membership_patterns=True,
            max_pattern_accumulator_bytes=47,
        )


def test_dense_interval_contribution_cache_rejects_missing_interval(tmp_path):
    cache_path = tmp_path / "dense_interval_contrib.zarr"
    build_dense_interval_contribution_cache(
        _provider_factory({}),
        _memberships(),
        output_path=cache_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    missing = pd.DataFrame(
        {
            "subset": ["S1"],
            "chrom": ["chr1"],
            "start": [20],
            "end": [25],
        }
    )

    with pytest.raises(ValueError, match="missing required interval chr1:20-25"):
        accumulate_dense_score_region_subsets_from_contribution_cache(
            cache_path,
            missing,
            subset_column="subset",
        )


def test_sparse_dense_interval_contribution_cache_treats_missing_as_zero(tmp_path):
    full_cache_path = tmp_path / "dense_interval_contrib.zarr"
    build_dense_interval_contribution_cache(
        _provider_factory({}),
        _memberships(),
        output_path=full_cache_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    sparse_cache_path = tmp_path / "sparse_dense_interval_contrib.zarr"
    kept_contribution = _write_single_interval_sparse_cache(full_cache_path, sparse_cache_path)

    replay = accumulate_dense_score_region_subsets_from_contribution_cache(
        sparse_cache_path,
        _memberships(),
        subset_column="subset",
    )
    coalesced = accumulate_dense_score_region_subsets_from_contribution_cache(
        sparse_cache_path,
        _memberships(),
        subset_column="subset",
        coalesce_membership_patterns=True,
    )
    expected = np.zeros_like(replay.values)
    expected[0] = kept_contribution
    expected[1] = kept_contribution
    expected[2] = kept_contribution

    assert replay.timings["missing_intervals_are_zero"] is True
    assert replay.timings["contribution_cache_missing_zero_intervals"] == 1
    assert coalesced.timings["missing_intervals_are_zero"] is True
    assert coalesced.timings["contribution_cache_missing_zero_intervals"] == 1
    np.testing.assert_allclose(replay.values, expected, rtol=0, atol=0)
    np.testing.assert_allclose(coalesced.values, expected, rtol=0, atol=0)
