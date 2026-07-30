from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr

torch = pytest.importorskip("torch")

from motiverse import dense_subset as dense_subset_module
from motiverse import dense_subset_workflow as dense_cli
from motiverse.accumulation import (
    accumulate_around_query_motif,
    accumulate_motif_cooccurrences,
)
from motiverse.dense_subset import (
    DENSE_SUBSET_COORDINATE_FRAME,
    DENSE_SUBSET_SEMANTICS,
    DenseGenomeBlockScoreProvider,
    DenseGenomeScoreProvider,
    DenseQueryMotifHitResult,
    accumulate_dense_score_region_subsets,
    accumulate_dense_score_region_subsets_from_query_motif_hit_cache,
    accumulate_dense_score_region_subsets_to_zarr,
    dense_region_contribution_from_scores,
    dense_region_contribution_from_scores_and_query_motif_hits,
    dense_region_one_to_all_sum_contribution_from_scores,
    plan_dense_score_region_subset_resources,
)
from motiverse.positional_cache import (
    threshold_vector_checksum,
)
from motiverse.processing import _scan_motifs_conv1d
from motiverse.sequence_io import reverse_complement_batch

_thresholds_from_args = dense_cli._thresholds_from_args


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


class _FakeHocomoco:
    def __init__(self):
        self.seen_pvalue = None
        self.seen_annotation_pvalue = None

    def get_score_threshold_for_pvalue(self, p_value):
        self.seen_pvalue = p_value
        return np.array([1.25, 2.5], dtype=np.float32)

    def prepare_for_gpu(self, *, device, dtype, p_value):
        self.seen_annotation_pvalue = p_value
        return None, torch.tensor([1.5, 2.75], dtype=dtype, device=device)


def test_dense_subset_cli_thresholds_default_to_pvalue_mapping():
    fake = _FakeHocomoco()

    thresholds, metadata = _thresholds_from_args(
        fake,
        score_threshold=None,
        p_value_threshold="p0.0001",
        use_pvalue_mapping=True,
        n_motifs=2,
        dtype=torch.float32,
        device="cpu",
    )

    assert fake.seen_pvalue == 0.0001
    assert fake.seen_annotation_pvalue is None
    assert torch.allclose(thresholds["p0.0001"], torch.tensor([1.25, 2.5]))
    assert metadata["threshold_mode"] == "pvalue_mapping"
    assert metadata["p_value_threshold"] == "p0.0001"
    assert metadata["score_threshold"] is None
    assert metadata["score_threshold_vector_shape"] == [2]
    assert metadata["score_threshold_vector_checksum"] == threshold_vector_checksum(
        thresholds["p0.0001"]
    )


def test_dense_subset_cli_score_threshold_overrides_pvalue_mapping():
    fake = _FakeHocomoco()

    thresholds, metadata = _thresholds_from_args(
        fake,
        score_threshold=3.0,
        p_value_threshold="p0.0001",
        use_pvalue_mapping=True,
        n_motifs=2,
        dtype=torch.float32,
        device="cpu",
    )

    assert fake.seen_pvalue is None
    assert fake.seen_annotation_pvalue is None
    assert torch.equal(thresholds["p0.0001"], torch.tensor([3.0, 3.0]))
    assert metadata["threshold_mode"] == "score_threshold"
    assert metadata["score_threshold"] == 3.0


def test_dense_subset_cli_no_pvalue_mapping_uses_annotation_thresholds():
    fake = _FakeHocomoco()

    thresholds, metadata = _thresholds_from_args(
        fake,
        score_threshold=None,
        p_value_threshold="p0.0001",
        use_pvalue_mapping=False,
        n_motifs=2,
        dtype=torch.float32,
        device="cpu",
    )

    assert fake.seen_pvalue is None
    assert fake.seen_annotation_pvalue == "p0.0001"
    assert torch.allclose(thresholds["p0.0001"], torch.tensor([1.5, 2.75]))
    assert metadata["threshold_mode"] == "annotation"
    assert metadata["score_threshold"] is None


def test_dense_subset_script_full_curve_one_to_all_records_aligned_source(
    tmp_path,
    monkeypatch,
):
    seq_path = tmp_path / "seq.zarr"
    regions_path = tmp_path / "regions.tsv"
    output_json = tmp_path / "dense_full_curve.json"
    aligned_path = "/home/xf2217/.gcell_data/annotations/hocomoco/motifs_with_rc_aligned.pt"
    _write_sequence_zarr(seq_path, "AAAA")
    pd.DataFrame(
        {
            "subset": ["S1"],
            "chrom": ["chr1"],
            "start": [0],
            "end": [4],
        }
    ).to_csv(regions_path, sep="\t", index=False)

    class FakeAlignedHocomoco:
        motif_names = ["A_ONEBASE", "C_ONEBASE"]
        motif_kernels = _one_base_kernels().numpy()

        def get_score_threshold_for_pvalue(self, p_value):
            assert p_value == 0.0001
            return np.array([0.5, 0.5], dtype=np.float32)

    def fake_load_hocomoco_motifs(**kwargs):
        assert kwargs["use_aligned_motifs"] is True
        assert kwargs["require_aligned_motifs"] is True
        assert kwargs["motif_selection"] is None
        return FakeAlignedHocomoco(), SimpleNamespace(
            motif_names_checksum="toy-motif-names",
            motif_kernel_checksum="toy-kernels",
            motif_kernel_shape=[2, 1, 4],
            motif_source="aligned_pt",
            aligned_motif_path=aligned_path,
            loaded_with_aligned=True,
        )

    monkeypatch.setattr(dense_cli, "load_hocomoco_motifs", fake_load_hocomoco_motifs)
    base_argv = [
        "dense_subset_aggregation.py",
        "--genome-zarr",
        str(seq_path),
        "--regions-tsv",
        str(regions_path),
        "--subset-column",
        "subset",
        "--window",
        "1",
        "--query-motif-index",
        "0",
        "--strand-specific",
        "--dtype",
        "float32",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [*base_argv, "--output-json", str(output_json)],
    )

    dense_cli.main()

    summary = json.loads(output_json.read_text())
    assert summary["schema_version"] == "dense_subset_genome_scan_v1"
    assert summary["mode"] == "dense-exact-subset-genome-scan"
    assert summary["semantics"] == DENSE_SUBSET_SEMANTICS
    assert summary["coordinate_frame"] == DENSE_SUBSET_COORDINATE_FRAME
    assert summary["query_motif_index"] == 0
    assert summary["motif_count"] == 2
    assert summary["motif_source"] == "aligned_pt"
    assert summary["aligned_motif_path"] == aligned_path
    assert summary["loaded_with_aligned"] is True
    assert summary["motif_names_checksum"] == "toy-motif-names"
    assert summary["motif_kernel_shape"] == [2, 1, 4]
    assert summary["threshold_mode"] == "pvalue_mapping"
    assert summary["bp_scanned"] == 4
    assert summary["values_shape"] == [1, 2, 3]
    assert summary["values_checksum_mode"] == "full"
    assert summary["subset_update_mode"] == "in_memory_dense_output"

    plan_json = tmp_path / "plan_only.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--plan-only",
            "--provider",
            "block",
            "--provider-max-gap-bp",
            "10",
            "--provider-max-block-span-bp",
            "100",
            "--provider-max-block-score-gb",
            str(80 / (1024**3)),
            "--output-json",
            str(plan_json),
        ],
    )
    dense_cli.main()
    plan_summary = json.loads(plan_json.read_text())
    assert plan_summary["schema_version"] == "dense_subset_resource_plan_v1"
    assert plan_summary["motif_source"] == "aligned_pt"
    assert (
        plan_summary["contribution_cache_plan"]["schema_version"]
        == "dense_interval_contribution_cache_plan_v1"
    )
    assert plan_summary["contribution_cache_bytes"] == 2 * 3 * 4
    assert plan_summary["contribution_cache_feasible_under_max"] is True
    assert plan_summary["contribution_cache_recommended_strategy"] == (
        "build_only_if_reused_by_multiple_downstream_subset_families"
    )
    assert plan_summary["loaded_with_aligned"] is True
    assert plan_summary["provider_max_block_score_bytes"] == 80
    assert plan_summary["max_block_score_bytes_estimate"] == 32


def test_dense_subset_query_motif_hit_modes_reject_pwm_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dense_subset_aggregation.py",
            "--genome-zarr",
            str(tmp_path / "seq.zarr"),
            "--regions-tsv",
            str(tmp_path / "regions.tsv"),
            "--subset-column",
            "subset",
            "--window",
            "1",
            "--query-motif-index",
            "0",
            "--allow-pwm-fallback",
            "--build-query-motif-hit-cache",
            str(tmp_path / "query_motif_hit_cache.zarr"),
            "--output-json",
            str(tmp_path / "summary.json"),
        ],
    )

    with pytest.raises(SystemExit, match="require aligned HOCOMOCO PT"):
        dense_cli.main()


def _one_hot(sequence: str) -> np.ndarray:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for pos, base in enumerate(sequence):
        encoded[pos, base_to_idx[base]] = 1.0
    return encoded


def _write_sequence_zarr(path, sequence: str):
    root = zarr.open_group(str(path), mode="w")
    root.attrs["assembly"] = "test"
    root.attrs["chunk_size"] = 100
    root.attrs["zarr_type"] = "dense"
    chrs = root.create_group("chrs")
    chrs.create_array("chr1", data=_one_hot(sequence), chunks=(100, 4))


def _one_base_kernels() -> torch.Tensor:
    kernels = torch.zeros((2, 1, 4), dtype=torch.float32)
    kernels[0, 0, 0] = 1.0
    kernels[1, 0, 1] = 1.0
    return kernels


def _memberships():
    return pd.DataFrame(
        {
            "subset": ["S1", "S2", "S2", "S3"],
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [0, 0, 10, 0],
            "end": [5, 5, 15, 5],
        }
    )


def _provider_factory(calls):
    score_by_region = {
        ("chr1", 0, 5): _scores_a(),
        ("chr1", 10, 15): _scores_b(),
    }

    def provider(chrom, start, end):
        key = (chrom, start, end)
        calls[key] = calls.get(key, 0) + 1
        return score_by_region[key]

    return provider


def _checksum(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def test_dense_subset_script_query_motif_hit_prefilter_preserves_full_curves(
    tmp_path,
    monkeypatch,
):
    seq_path = tmp_path / "seq.zarr"
    regions_path = tmp_path / "regions.tsv"
    direct_json = tmp_path / "direct.json"
    prefilter_json = tmp_path / "prefilter.json"
    direct_zarr = tmp_path / "direct.zarr"
    prefilter_zarr = tmp_path / "prefilter.zarr"
    _write_sequence_zarr(seq_path, "AAAACCCC")
    pd.DataFrame(
        {
            "subset": ["S1", "S1"],
            "chrom": ["chr1", "chr1"],
            "start": [0, 4],
            "end": [4, 8],
        }
    ).to_csv(regions_path, sep="\t", index=False)

    class FakeAlignedHocomoco:
        motif_names = ["A_ONEBASE", "C_ONEBASE"]
        motif_kernels = _one_base_kernels().numpy()

        def get_score_threshold_for_pvalue(self, p_value):
            assert p_value == 0.0001
            return np.array([0.5, 0.5], dtype=np.float32)

    def fake_load_hocomoco_motifs(**kwargs):
        assert kwargs["use_aligned_motifs"] is True
        assert kwargs["require_aligned_motifs"] is True
        return FakeAlignedHocomoco(), SimpleNamespace(
            motif_names_checksum="toy-motif-names",
            motif_kernel_checksum="toy-kernels",
            motif_kernel_shape=[2, 1, 4],
            motif_source="aligned_pt",
            aligned_motif_path="/aligned.pt",
            loaded_with_aligned=True,
        )

    monkeypatch.setattr(dense_cli, "load_hocomoco_motifs", fake_load_hocomoco_motifs)
    base_argv = [
        "dense_subset_aggregation.py",
        "--genome-zarr",
        str(seq_path),
        "--regions-tsv",
        str(regions_path),
        "--subset-column",
        "subset",
        "--window",
        "1",
        "--query-motif-index",
        "0",
        "--strand-specific",
        "--dtype",
        "float32",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--output-zarr",
            str(direct_zarr),
            "--output-json",
            str(direct_json),
        ],
    )
    dense_cli.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--query-motif-hit-prefilter",
            "--output-zarr",
            str(prefilter_zarr),
            "--output-json",
            str(prefilter_json),
        ],
    )
    dense_cli.main()

    direct = np.asarray(zarr.open_group(str(direct_zarr), mode="r")["values"][:])
    prefilter = np.asarray(zarr.open_group(str(prefilter_zarr), mode="r")["values"][:])
    prefilter_summary = json.loads(prefilter_json.read_text())

    np.testing.assert_allclose(prefilter, direct, rtol=0, atol=0)
    assert prefilter_summary["mode"] == "dense-exact-subset-query-motif-hit-prefilter"
    assert prefilter_summary["query_motif_hit_prefilter"] is True
    assert prefilter_summary["full_score_regions_scanned"] == 1
    assert prefilter_summary["full_score_regions_skipped"] == 1
    assert prefilter_summary["query_motif_hit_provider_provider_intervals"] == 2
    assert prefilter_summary["provider_intervals"] == 1
    assert prefilter_summary["values_checksum"] == _checksum(prefilter)
    assert prefilter_summary["subset_update_mode"] == ("query_motif_hit_prefilter_in_memory_output")


def test_dense_subset_script_query_motif_hit_cache_replay_preserves_full_curves(
    tmp_path,
    monkeypatch,
):
    seq_path = tmp_path / "seq.zarr"
    regions_path = tmp_path / "regions.tsv"
    direct_json = tmp_path / "direct.json"
    cache_replay_json = tmp_path / "cache_replay.json"
    coalesced_replay_json = tmp_path / "cache_replay_coalesced.json"
    direct_zarr = tmp_path / "direct.zarr"
    cache_replay_zarr = tmp_path / "cache_replay.zarr"
    coalesced_replay_zarr = tmp_path / "cache_replay_coalesced.zarr"
    target_cache = tmp_path / "query_motif_hit_cache.zarr"
    _write_sequence_zarr(seq_path, "AAAACCCC")
    pd.DataFrame(
        {
            "subset": ["S1", "S1"],
            "chrom": ["chr1", "chr1"],
            "start": [0, 4],
            "end": [4, 8],
        }
    ).to_csv(regions_path, sep="\t", index=False)

    class FakeAlignedHocomoco:
        motif_names = ["A_ONEBASE", "C_ONEBASE"]
        motif_kernels = _one_base_kernels().numpy()

        def get_score_threshold_for_pvalue(self, p_value):
            assert p_value == 0.0001
            return np.array([0.5, 0.5], dtype=np.float32)

    def fake_load_hocomoco_motifs(**kwargs):
        assert kwargs["use_aligned_motifs"] is True
        assert kwargs["require_aligned_motifs"] is True
        return FakeAlignedHocomoco(), SimpleNamespace(
            motif_names_checksum="toy-motif-names",
            motif_kernel_checksum="toy-kernels",
            motif_kernel_shape=[2, 1, 4],
            motif_source="aligned_pt",
            aligned_motif_path="/aligned.pt",
            loaded_with_aligned=True,
        )

    monkeypatch.setattr(dense_cli, "load_hocomoco_motifs", fake_load_hocomoco_motifs)
    base_argv = [
        "dense_subset_aggregation.py",
        "--genome-zarr",
        str(seq_path),
        "--regions-tsv",
        str(regions_path),
        "--subset-column",
        "subset",
        "--window",
        "1",
        "--query-motif-index",
        "0",
        "--strand-specific",
        "--dtype",
        "float32",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--output-zarr",
            str(direct_zarr),
            "--output-json",
            str(direct_json),
        ],
    )
    dense_cli.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--build-query-motif-hit-cache",
            str(target_cache),
            "--output-zarr",
            str(cache_replay_zarr),
            "--output-json",
            str(cache_replay_json),
        ],
    )
    dense_cli.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base_argv,
            "--use-query-motif-hit-cache",
            str(target_cache),
            "--coalesce-membership-patterns",
            "--output-zarr",
            str(coalesced_replay_zarr),
            "--output-json",
            str(coalesced_replay_json),
        ],
    )
    dense_cli.main()

    direct = np.asarray(zarr.open_group(str(direct_zarr), mode="r")["values"][:])
    replay = np.asarray(zarr.open_group(str(cache_replay_zarr), mode="r")["values"][:])
    coalesced_replay = np.asarray(
        zarr.open_group(str(coalesced_replay_zarr), mode="r")["values"][:]
    )
    replay_summary = json.loads(cache_replay_json.read_text())
    coalesced_replay_summary = json.loads(coalesced_replay_json.read_text())

    np.testing.assert_allclose(replay, direct, rtol=0, atol=0)
    np.testing.assert_allclose(coalesced_replay, direct, rtol=0, atol=0)
    assert replay_summary["mode"] == "dense-exact-subset-query-motif-hit-cache-replay"
    assert replay_summary["query_motif_hit_cache_path"] == str(target_cache)
    assert replay_summary["query_motif_hit_cache_anchor_hits"] == 2
    assert replay_summary["query_motif_hit_arrays_preloaded"] is True
    assert replay_summary["query_motif_hit_array_bytes"] == 32
    assert replay_summary["query_motif_hit_array_preload_wall_s"] >= 0
    assert replay_summary["query_motif_hit_array_read_wall_s"] >= 0
    assert replay_summary["full_score_regions_scanned"] == 1
    assert replay_summary["full_score_regions_skipped"] == 1
    assert replay_summary["values_checksum"] == _checksum(replay)
    assert replay_summary["subset_update_mode"] == ("query_motif_hit_cache_replay_in_memory_output")
    assert coalesced_replay_summary["subset_update_mode"] == (
        "query_motif_hit_cache_replay_membership_pattern_coalesced"
    )
    assert coalesced_replay_summary["membership_pattern_coalescing"] is True
    assert coalesced_replay_summary["membership_pattern_count"] == 1
    assert coalesced_replay_summary["membership_pattern_interval_assignments"] == 1


def test_query_motif_hit_cache_replay_preloads_anchor_arrays_without_changing_values(
    monkeypatch,
):
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.nbytes = self.values.nbytes

        def __getitem__(self, key):
            return self.values[key]

    class FakeNode:
        def __init__(self, attrs=None):
            self.attrs = attrs or {}

    class FakeRoot:
        def __init__(self):
            self.nodes = {
                "metadata": FakeNode(
                    {
                        "schema_version": "dense_query_motif_hit_cache_v1",
                        "complete": True,
                        "semantics": "query_motif_hit_positions_per_exact_interval",
                        "motif_source": "aligned_pt",
                        "loaded_with_aligned": True,
                        "motif_count": 2,
                        "window_size": 1,
                        "query_motif_index": 0,
                        "dtype": "float32",
                    }
                ),
                "intervals/chrom": FakeArray(["chr1"]),
                "intervals/start": FakeArray([0]),
                "intervals/end": FakeArray([5]),
                "anchors/interval_anchor_start": FakeArray([0]),
                "anchors/interval_anchor_count": FakeArray([1]),
                "anchors/batch_index": FakeArray([0]),
                "anchors/position": FakeArray([2]),
            }

        def __getitem__(self, key):
            return self.nodes[key]

    regions = pd.DataFrame(
        {
            "subset": ["S1", "S2"],
            "chrom": ["chr1", "chr1"],
            "start": [0, 0],
            "end": [5, 5],
        }
    )
    full_scores = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0, 0.0], [0.0, 4.0, 5.0, 6.0, 0.0]]],
        dtype=torch.float32,
    )

    def full_provider(chrom, start, end):
        assert (chrom, start, end) == ("chr1", 0, 5)
        return full_scores

    monkeypatch.setattr(
        dense_subset_module.zarr,
        "open_group",
        lambda path, mode="r": FakeRoot(),
    )

    preload = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        "fake_query_motif_hit_cache.zarr",
        full_provider,
        regions,
        subset_column="subset",
        torch_dtype=torch.float32,
        device="cpu",
        preload_query_motif_hit_arrays=True,
    )
    lazy = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        "fake_query_motif_hit_cache.zarr",
        full_provider,
        regions,
        subset_column="subset",
        torch_dtype=torch.float32,
        device="cpu",
        preload_query_motif_hit_arrays=False,
    )

    expected = np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]] * 2, dtype=np.float32)
    np.testing.assert_allclose(preload.values, expected, rtol=0, atol=0)
    np.testing.assert_allclose(lazy.values, expected, rtol=0, atol=0)
    assert preload.timings["query_motif_hit_arrays_preloaded"] is True
    assert preload.timings["query_motif_hit_array_bytes"] == 16
    assert preload.timings["query_motif_hit_array_preload_wall_s"] >= 0
    assert preload.timings["query_motif_hit_array_read_wall_s"] >= 0
    assert preload.timings["query_motif_hit_cache_total_intervals"] == 1.0
    assert preload.timings["query_motif_hit_cache_required_intervals"] == 1.0
    assert preload.timings["query_motif_hit_cache_hits"] == 1.0
    assert preload.timings["query_motif_hit_cache_missing_intervals"] == 0.0
    assert preload.timings["query_motif_hit_cache_coverage_fraction"] == 1.0
    assert lazy.timings["query_motif_hit_arrays_preloaded"] is False
    assert lazy.timings["query_motif_hit_array_bytes"] >= 0
    assert lazy.timings["query_motif_hit_cache_coverage_fraction"] == 1.0


def test_query_motif_hit_cache_replay_torch_output_accumulator_matches_numpy(
    monkeypatch,
):
    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.nbytes = self.values.nbytes

        def __getitem__(self, key):
            return self.values[key]

    class FakeNode:
        def __init__(self, attrs=None):
            self.attrs = attrs or {}

    class FakeRoot:
        def __init__(self):
            self.nodes = {
                "metadata": FakeNode(
                    {
                        "schema_version": "dense_query_motif_hit_cache_v1",
                        "complete": True,
                        "semantics": "query_motif_hit_positions_per_exact_interval",
                        "motif_source": "aligned_pt",
                        "loaded_with_aligned": True,
                        "motif_count": 2,
                        "window_size": 1,
                        "query_motif_index": 0,
                        "dtype": "float32",
                    }
                ),
                "intervals/chrom": FakeArray(["chr1", "chr1"]),
                "intervals/start": FakeArray([0, 10]),
                "intervals/end": FakeArray([5, 15]),
                "anchors/interval_anchor_start": FakeArray([0, 2]),
                "anchors/interval_anchor_count": FakeArray([2, 1]),
                "anchors/batch_index": FakeArray([0, 1_000_000, 0]),
                "anchors/position": FakeArray([2, 2, 2]),
            }

        def __getitem__(self, key):
            return self.nodes[key]

    regions = pd.DataFrame(
        {
            "subset": ["S1", "S2", "S2"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 0, 10],
            "end": [5, 5, 15],
        }
    )
    scores_a = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0, 0.0], [0.0, 4.0, 5.0, 6.0, 0.0]]],
        dtype=torch.float32,
    )
    scores_b = torch.tensor(
        [[[0.0, 10.0, 20.0, 30.0, 0.0], [0.0, 40.0, 50.0, 60.0, 0.0]]],
        dtype=torch.float32,
    )

    def full_provider(chrom, start, end):
        assert chrom == "chr1"
        return [scores_a, scores_b] if start == 0 else [scores_a]

    monkeypatch.setattr(
        dense_subset_module.zarr,
        "open_group",
        lambda path, mode="r": FakeRoot(),
    )

    numpy_result = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        "fake_query_motif_hit_cache.zarr",
        full_provider,
        regions,
        subset_column="subset",
        torch_dtype=torch.float32,
        device="cpu",
        output_accumulator="numpy",
    )
    torch_result = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        "fake_query_motif_hit_cache.zarr",
        full_provider,
        regions,
        subset_column="subset",
        torch_dtype=torch.float32,
        device="cpu",
        output_accumulator="torch",
    )

    np.testing.assert_array_equal(torch_result.values, numpy_result.values)
    assert torch_result.timings["output_accumulator"] == "torch"
    assert torch_result.timings["output_accumulator_device"] == "cpu"
    assert torch_result.timings["output_accumulator_to_numpy_wall_s"] >= 0


def test_query_motif_hit_cache_replay_rejects_torch_accumulator_with_coalescing(
    monkeypatch,
):
    monkeypatch.setattr(
        dense_subset_module.zarr,
        "open_group",
        lambda path, mode="r": None,
    )
    regions = pd.DataFrame({"subset": ["S1"], "chrom": ["chr1"], "start": [0], "end": [5]})

    with pytest.raises(ValueError, match="cannot be combined"):
        accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
            "fake_query_motif_hit_cache.zarr",
            lambda chrom, start, end: [],
            regions,
            subset_column="subset",
            torch_dtype=torch.float32,
            device="cpu",
            coalesce_membership_patterns=True,
            output_accumulator="torch",
        )


def _direct_all_to_all(scores):
    out = torch.zeros((2, 2, 3), dtype=torch.float32)
    accumulate_motif_cooccurrences(scores, out, _thresholds(), window_size=1)
    return out.numpy()


def _direct_one_to_all(scores):
    out = torch.zeros((2, 3), dtype=torch.float32)
    accumulate_around_query_motif(
        scores,
        out,
        _thresholds(),
        query_motif_index=0,
        window_size=1,
        device="cpu",
    )
    return out.numpy()


def test_dense_region_contribution_preserves_below_threshold_partner_scores():
    contribution = dense_region_contribution_from_scores(
        _scores_a(),
        score_thresholds={"p0.0001": torch.tensor([0.5, 10.0], dtype=torch.float32)},
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )

    np.testing.assert_allclose(
        contribution,
        np.array([[0, 3, 0], [5, 0, 7]], dtype=np.float32),
        rtol=0,
        atol=0,
    )


def test_dense_region_one_to_all_sum_fast_path_matches_dense_window_sum():
    score_batches = [
        torch.cat([_scores_a(), _scores_b()], dim=0),
        _scores_a(),
    ]
    thresholds = {
        "p0.0001": torch.tensor([0.5, 10.0], dtype=torch.float32),
    }

    dense = dense_region_contribution_from_scores(
        score_batches,
        score_thresholds=thresholds,
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )
    reduced = dense_region_one_to_all_sum_contribution_from_scores(
        score_batches,
        score_thresholds=thresholds,
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )

    np.testing.assert_allclose(reduced, dense.sum(axis=-1), rtol=0, atol=0)


def test_dense_region_one_to_all_sum_fast_path_handles_no_valid_target_hits():
    score_batches = [
        torch.zeros((1, 2, 5), dtype=torch.float32),
        torch.ones((1, 2, 2), dtype=torch.float32),
    ]

    reduced = dense_region_one_to_all_sum_contribution_from_scores(
        score_batches,
        score_thresholds=_thresholds(),
        window_size=2,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )

    np.testing.assert_allclose(reduced, np.zeros(2, dtype=np.float32), rtol=0, atol=0)


def test_dense_region_query_motif_hit_direct_slice_fast_path_matches_expected_windows():
    score_batches = [_scores_a(), _scores_b()]

    single = dense_region_contribution_from_scores_and_query_motif_hits(
        score_batches,
        DenseQueryMotifHitResult(
            batch_indices=np.asarray([0], dtype=np.int64),
            positions=np.asarray([1], dtype=np.int64),
            n_hits=1,
        ),
        window_size=1,
        n_motifs=2,
        device="cpu",
    )
    pair = dense_region_contribution_from_scores_and_query_motif_hits(
        score_batches,
        DenseQueryMotifHitResult(
            batch_indices=np.asarray([0, 1_000_000], dtype=np.int64),
            positions=np.asarray([1, 2], dtype=np.int64),
            n_hits=2,
        ),
        window_size=1,
        n_motifs=2,
        device="cpu",
    )

    np.testing.assert_array_equal(single, _scores_a()[0, :, 0:3].numpy())
    expected_pair = _scores_a()[0, :, 0:3] + _scores_b()[0, :, 1:4]
    np.testing.assert_array_equal(pair, expected_pair.numpy())


def test_dense_region_query_motif_hit_stack_all_mode_uses_explicit_stack_sum():
    score_batches = [_scores_a(), _scores_b()]

    contribution = dense_region_contribution_from_scores_and_query_motif_hits(
        score_batches,
        DenseQueryMotifHitResult(
            batch_indices=np.asarray([0, 1_000_000, 0], dtype=np.int64),
            positions=np.asarray([1, 2, 3], dtype=np.int64),
            n_hits=3,
        ),
        window_size=1,
        n_motifs=2,
        device="cpu",
        anchor_contribution_mode="stack_all",
    )

    expected = torch.stack(
        [
            _scores_a()[0, :, 0:3],
            _scores_b()[0, :, 1:4],
            _scores_a()[0, :, 2:5],
        ],
        dim=0,
    ).sum(dim=0)
    np.testing.assert_array_equal(contribution, expected.numpy())


def test_dense_genome_score_provider_matches_direct_interval_scan(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    _write_sequence_zarr(seq_path, "AACGTA")
    kernels = _one_base_kernels()
    provider = DenseGenomeScoreProvider(
        str(seq_path),
        kernels,
        device="cpu",
        dtype=torch.float32,
        strand_specific=False,
    )

    forward_scores, reverse_scores = provider("chr1", 1, 5)
    interval = torch.tensor(_one_hot("ACGT")[None, :, :], dtype=torch.float32)
    expected_forward = _scan_motifs_conv1d(
        interval.permute(0, 2, 1),
        kernels.permute(0, 2, 1),
    )
    expected_reverse = _scan_motifs_conv1d(
        reverse_complement_batch(interval).permute(0, 2, 1),
        kernels.permute(0, 2, 1),
    )

    torch.testing.assert_close(forward_scores, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(reverse_scores, expected_reverse, rtol=0, atol=0)
    assert provider.stats["provider_intervals"] == 1
    assert provider.stats["provider_bp_loaded"] == 4
    assert provider.stats["provider_score_batches"] == 2


def test_dense_genome_block_score_provider_slices_overlapping_intervals(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    _write_sequence_zarr(seq_path, "AACGTACCAA")
    kernels = _one_base_kernels()
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2"],
            "chrom": ["chr1", "chr1"],
            "start": [1, 3],
            "end": [7, 9],
        }
    )
    provider = DenseGenomeBlockScoreProvider(
        str(seq_path),
        kernels,
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=False,
        max_gap_bp=0,
        max_cached_blocks=1,
    )

    for sequence, start, end in [("ACGTAC", 1, 7), ("GTACCA", 3, 9)]:
        forward_scores, reverse_scores = provider("chr1", start, end)
        interval = torch.tensor(_one_hot(sequence)[None, :, :], dtype=torch.float32)
        expected_forward = _scan_motifs_conv1d(
            interval.permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        expected_reverse = _scan_motifs_conv1d(
            reverse_complement_batch(interval).permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        torch.testing.assert_close(forward_scores, expected_forward, rtol=0, atol=0)
        torch.testing.assert_close(reverse_scores, expected_reverse, rtol=0, atol=0)

    assert provider.stats["provider_blocks_planned"] == 1
    assert provider.stats["provider_blocks_scanned"] == 1
    assert provider.stats["provider_bp_loaded"] == 8
    assert provider.stats["provider_intervals"] == 2
    assert provider.stats["provider_interval_slices"] == 2
    assert provider.stats["provider_block_cache_hits"] == 1
    assert provider.stats["provider_block_cache_misses"] == 1
    assert provider.stats["provider_block_cache_evictions"] == 0
    assert provider.stats["provider_block_cache_peak_blocks"] == 1
    for key in (
        "provider_sequence_fetch_wall_s",
        "provider_tensor_prepare_wall_s",
        "provider_forward_scan_wall_s",
        "provider_reverse_complement_wall_s",
        "provider_reverse_scan_wall_s",
        "provider_interval_slice_wall_s",
    ):
        assert key in provider.stats
        assert provider.stats[key] >= 0.0


def test_dense_genome_block_score_provider_records_timing_breakdown(monkeypatch):
    sequence = _one_hot("AACGTACCAA")

    class FakeSequenceDenseZarrIO:
        def __init__(self, path, mode="r"):
            assert path == "fake.zarr"
            assert mode == "r"

        def get_track(self, chrom, start, end, output_format="raw_array"):
            assert chrom == "chr1"
            assert output_format == "raw_array"
            return sequence[int(start) : int(end)]

    monkeypatch.setattr(
        dense_subset_module,
        "SequenceDenseZarrIO",
        FakeSequenceDenseZarrIO,
    )
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2"],
            "chrom": ["chr1", "chr1"],
            "start": [1, 3],
            "end": [7, 9],
        }
    )
    provider = DenseGenomeBlockScoreProvider(
        "fake.zarr",
        _one_base_kernels(),
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=False,
        max_gap_bp=0,
        max_cached_blocks=1,
    )

    provider("chr1", 1, 7)
    provider("chr1", 3, 9)

    assert provider.stats["provider_blocks_scanned"] == 1
    assert provider.stats["provider_block_cache_hits"] == 1
    assert provider.stats["provider_interval_slices"] == 2
    assert provider.stats["provider_sequence_fetch_mode"] == "get_track_fallback"
    assert provider.stats["provider_sequence_fetch_fallbacks"] == 0
    for key in (
        "provider_sequence_fetch_wall_s",
        "provider_tensor_prepare_wall_s",
        "provider_forward_scan_wall_s",
        "provider_reverse_complement_wall_s",
        "provider_reverse_scan_wall_s",
        "provider_interval_slice_wall_s",
    ):
        assert key in provider.stats
        assert provider.stats[key] >= 0.0


def test_dense_genome_block_score_provider_direct_sequence_fetch(monkeypatch):
    sequence = _one_hot("AACGTACCAA").astype(np.uint8)

    class FakeChromArray:
        shape = sequence.shape

        def __getitem__(self, key):
            return sequence[key]

    class FakeSequenceDenseZarrIO:
        dir_chunked = False
        dataset = {"chrs": {"chr1": FakeChromArray()}}

        def __init__(self, path, mode="r"):
            assert path == "fake.zarr"
            assert mode == "r"

        def get_track(self, chrom, start, end, output_format="raw_array"):
            raise AssertionError("direct fetch should not call get_track")

    monkeypatch.setattr(
        dense_subset_module,
        "SequenceDenseZarrIO",
        FakeSequenceDenseZarrIO,
    )
    memberships = pd.DataFrame(
        {
            "subset": ["S1"],
            "chrom": ["chr1"],
            "start": [1],
            "end": [7],
        }
    )
    provider = DenseGenomeBlockScoreProvider(
        "fake.zarr",
        _one_base_kernels(),
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
        max_gap_bp=0,
        max_cached_blocks=1,
    )

    (forward_scores,) = provider("chr1", 1, 7)
    interval = torch.tensor(_one_hot("ACGTAC")[None, :, :], dtype=torch.float32)
    expected_forward = _scan_motifs_conv1d(
        interval.permute(0, 2, 1),
        _one_base_kernels().permute(0, 2, 1),
    )

    torch.testing.assert_close(forward_scores, expected_forward, rtol=0, atol=0)
    assert provider.stats["provider_sequence_fetch_mode"] == "direct_chrom_array"
    assert provider.stats["provider_sequence_fetch_fallbacks"] == 0


def test_dense_genome_block_score_provider_direct_sequence_chunk_cache(monkeypatch):
    sequence = _one_hot("AACGTACCAA").astype(np.uint8)
    reads: list[object] = []

    class FakeChromArray:
        shape = sequence.shape
        chunks = (8, 4)

        def __getitem__(self, key):
            reads.append(key)
            return sequence[key]

    class FakeSequenceDenseZarrIO:
        dir_chunked = False
        dataset = {"chrs": {"chr1": FakeChromArray()}}

        def __init__(self, path, mode="r"):
            assert path == "fake.zarr"
            assert mode == "r"

        def get_track(self, chrom, start, end, output_format="raw_array"):
            raise AssertionError("direct chunk cache should not call get_track")

    monkeypatch.setattr(
        dense_subset_module,
        "SequenceDenseZarrIO",
        FakeSequenceDenseZarrIO,
    )
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2"],
            "chrom": ["chr1", "chr1"],
            "start": [1, 5],
            "end": [4, 7],
        }
    )
    provider = DenseGenomeBlockScoreProvider(
        "fake.zarr",
        _one_base_kernels(),
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
        max_gap_bp=0,
        max_cached_blocks=0,
        max_sequence_cache_bytes=1024,
    )

    (first_scores,) = provider("chr1", 1, 4)
    (second_scores,) = provider("chr1", 5, 7)

    first_interval = torch.tensor(_one_hot("ACG")[None, :, :], dtype=torch.float32)
    second_interval = torch.tensor(_one_hot("AC")[None, :, :], dtype=torch.float32)
    expected_first = _scan_motifs_conv1d(
        first_interval.permute(0, 2, 1),
        _one_base_kernels().permute(0, 2, 1),
    )
    expected_second = _scan_motifs_conv1d(
        second_interval.permute(0, 2, 1),
        _one_base_kernels().permute(0, 2, 1),
    )

    torch.testing.assert_close(first_scores, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second_scores, expected_second, rtol=0, atol=0)
    assert provider.stats["provider_sequence_fetch_mode"] == "direct_chrom_chunk_cache"
    assert provider.stats["provider_sequence_chunk_cache_misses"] == 1
    assert provider.stats["provider_sequence_chunk_cache_hits"] == 1
    assert provider.stats["provider_sequence_chunk_bp_loaded"] == 8
    assert provider.stats["provider_sequence_chunk_cache_peak_bytes"] == 32
    assert len(reads) == 1


def test_dense_genome_block_score_provider_slices_gapped_reverse_intervals(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    sequence = "AACGTACCAATGCA"
    _write_sequence_zarr(seq_path, sequence)
    kernels = _one_base_kernels()
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2", "S3"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [1, 7, 11],
            "end": [5, 10, 14],
        }
    )
    provider = DenseGenomeBlockScoreProvider(
        str(seq_path),
        kernels,
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=False,
        max_gap_bp=3,
        max_block_span_bp=20,
        max_cached_blocks=1,
    )

    for start, end in [(1, 5), (7, 10), (11, 14)]:
        forward_scores, reverse_scores = provider("chr1", start, end)
        interval = torch.tensor(
            _one_hot(sequence[start:end])[None, :, :],
            dtype=torch.float32,
        )
        expected_forward = _scan_motifs_conv1d(
            interval.permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        expected_reverse = _scan_motifs_conv1d(
            reverse_complement_batch(interval).permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        torch.testing.assert_close(forward_scores, expected_forward, rtol=0, atol=0)
        torch.testing.assert_close(reverse_scores, expected_reverse, rtol=0, atol=0)

    assert provider.stats["provider_unique_intervals"] == 3
    assert provider.stats["provider_blocks_planned"] == 1
    assert provider.stats["provider_coalesced_block_bp"] == 13
    assert provider.stats["provider_unique_region_bp"] == 10
    assert provider.stats["provider_block_bp_vs_unique_region_bp"] == 1.3
    assert provider.stats["provider_block_span_min"] == 13
    assert provider.stats["provider_block_span_median"] == 13.0
    assert provider.stats["provider_block_span_max"] == 13
    assert provider.stats["provider_block_score_bytes_sum"] == 208
    assert provider.stats["provider_block_score_bytes_median"] == 208.0
    assert provider.stats["provider_block_score_bytes_max"] == 208
    assert len(provider.stats["provider_block_plan_checksum"]) == 64
    assert provider.stats["provider_blocks_scanned"] == 1
    assert provider.stats["provider_block_cache_hits"] == 2


def test_dense_genome_block_score_provider_caps_coalesced_score_bytes(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    _write_sequence_zarr(seq_path, "AACGTACCAATGCA")
    kernels = _one_base_kernels()
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2", "S3"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [1, 7, 11],
            "end": [5, 10, 14],
        }
    )

    provider = DenseGenomeBlockScoreProvider(
        str(seq_path),
        kernels,
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=False,
        max_gap_bp=3,
        max_block_span_bp=20,
        max_block_score_bytes=80,
        max_cached_blocks=1,
    )

    assert provider.stats["provider_blocks_planned"] == 3
    assert provider.stats["provider_coalesced_block_bp"] == 10
    assert provider.stats["provider_block_bp_vs_unique_region_bp"] == 1.0
    assert provider.stats["provider_block_score_bytes_max"] == 64
    assert provider.stats["provider_max_block_score_bytes"] == 80

    for start, end in [(1, 5), (7, 10), (11, 14)]:
        forward_scores, reverse_scores = provider("chr1", start, end)
        interval = torch.tensor(
            _one_hot("AACGTACCAATGCA"[start:end])[None, :, :],
            dtype=torch.float32,
        )
        expected_forward = _scan_motifs_conv1d(
            interval.permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        expected_reverse = _scan_motifs_conv1d(
            reverse_complement_batch(interval).permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        torch.testing.assert_close(forward_scores, expected_forward, rtol=0, atol=0)
        torch.testing.assert_close(reverse_scores, expected_reverse, rtol=0, atol=0)


def test_dense_genome_block_score_provider_rejects_unavoidable_score_bytes(
    tmp_path,
):
    seq_path = tmp_path / "seq.zarr"
    _write_sequence_zarr(seq_path, "AACGTA")
    kernels = _one_base_kernels()
    memberships = pd.DataFrame(
        {
            "subset": ["S1"],
            "chrom": ["chr1"],
            "start": [1],
            "end": [5],
        }
    )

    with pytest.raises(MemoryError, match="estimated score size"):
        DenseGenomeBlockScoreProvider(
            str(seq_path),
            kernels,
            memberships,
            subset_column="subset",
            device="cpu",
            dtype=torch.float32,
            strand_specific=False,
            max_block_score_bytes=63,
        )


def test_dense_genome_block_score_provider_evicts_bounded_block_cache(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    sequence = "AACGTACCAA"
    _write_sequence_zarr(seq_path, sequence)
    kernels = _one_base_kernels()
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S2"],
            "chrom": ["chr1", "chr1"],
            "start": [1, 6],
            "end": [4, 9],
        }
    )
    provider = DenseGenomeBlockScoreProvider(
        str(seq_path),
        kernels,
        memberships,
        subset_column="subset",
        device="cpu",
        dtype=torch.float32,
        strand_specific=True,
        max_gap_bp=0,
        max_cached_blocks=1,
    )

    for start, end in [(1, 4), (6, 9), (1, 4)]:
        (forward_scores,) = provider("chr1", start, end)
        interval = torch.tensor(
            _one_hot(sequence[start:end])[None, :, :],
            dtype=torch.float32,
        )
        expected_forward = _scan_motifs_conv1d(
            interval.permute(0, 2, 1),
            kernels.permute(0, 2, 1),
        )
        torch.testing.assert_close(forward_scores, expected_forward, rtol=0, atol=0)

    assert provider.stats["provider_blocks_scanned"] == 3
    assert provider.stats["provider_block_cache_hits"] == 0
    assert provider.stats["provider_block_cache_misses"] == 3
    assert provider.stats["provider_block_cache_evictions"] == 2
    assert provider.stats["provider_block_cache_peak_blocks"] == 1


def test_dense_subset_all_to_all_matches_repeated_direct_accumulation():
    calls = {}
    result = accumulate_dense_score_region_subsets(
        _provider_factory(calls),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
    )

    expected = np.stack(
        [
            _direct_all_to_all(_scores_a()),
            _direct_all_to_all(_scores_a()) + _direct_all_to_all(_scores_b()),
            _direct_all_to_all(_scores_a()),
        ],
        axis=0,
    )
    assert result.subset_ids == ["S1", "S2", "S3"]
    assert result.semantics == DENSE_SUBSET_SEMANTICS
    assert calls == {("chr1", 0, 5): 1, ("chr1", 10, 15): 1}
    assert result.timings["unique_regions_processed"] == 2
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=0)


def test_dense_subset_to_zarr_matches_in_memory_exact_result(tmp_path):
    memory_calls = {}
    memory = accumulate_dense_score_region_subsets(
        _provider_factory(memory_calls),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        chunk_subsets=2,
    )

    zarr_calls = {}
    output_path = tmp_path / "dense_subset.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        _provider_factory(zarr_calls),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        chunk_subsets=2,
        metadata_extra={
            "motif_source": "aligned_pt",
            "loaded_with_aligned": True,
        },
    )
    root = zarr.open_group(str(output_path), mode="r")
    values = np.asarray(root["values"][:])
    metadata = dict(root["metadata"].attrs)

    assert result.subset_ids == memory.subset_ids
    assert result.semantics == DENSE_SUBSET_SEMANTICS
    assert result.checksum == _checksum(memory.values)
    assert metadata["complete"] is True
    assert metadata["semantics"] == DENSE_SUBSET_SEMANTICS
    assert metadata["coordinate_frame"] == DENSE_SUBSET_COORDINATE_FRAME
    assert metadata["values_checksum_mode"] == "full"
    assert metadata["values_checksum_status"] == "complete"
    assert metadata["subset_update_mode"] == "zarr_interval_incremental"
    assert metadata["buffered_output_bytes"] == 0
    assert metadata["motif_source"] == "aligned_pt"
    assert metadata["loaded_with_aligned"] is True
    assert tuple(metadata["plan"]["output_shape"]) == memory.values.shape
    assert zarr_calls == memory_calls
    np.testing.assert_allclose(values, memory.values, rtol=0, atol=0)


def test_dense_subset_to_zarr_sample_checksum_mode(tmp_path):
    output_path = tmp_path / "dense_subset_sample.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        _provider_factory({}),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        checksum_mode="sample",
        checksum_sample_subsets=2,
    )
    root = zarr.open_group(str(output_path), mode="r")
    metadata = dict(root["metadata"].attrs)

    assert result.checksum == metadata["values_checksum"]
    assert metadata["values_checksum_mode"] == "sample"
    assert metadata["values_checksum_status"] == "sampled"
    assert metadata["values_checksum_sample_indices"] == [0, 2]


def test_dense_subset_to_zarr_buffered_updates_match_in_memory_exact_result(tmp_path):
    memory = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        chunk_subsets=2,
    )

    output_path = tmp_path / "dense_subset_buffered.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        _provider_factory({}),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        chunk_subsets=2,
        buffered_subset_updates=True,
        max_buffered_output_bytes=memory.values.nbytes,
    )
    root = zarr.open_group(str(output_path), mode="r")
    metadata = dict(root["metadata"].attrs)

    assert result.checksum == _checksum(memory.values)
    assert metadata["complete"] is True
    assert metadata["subset_update_mode"] == "buffered_full_output"
    assert metadata["buffered_output_bytes"] == memory.values.nbytes
    assert metadata["final_zarr_write_wall_s"] >= 0
    np.testing.assert_allclose(root["values"][:], memory.values, rtol=0, atol=0)


def test_dense_subset_to_zarr_buffered_updates_obey_memory_guard(tmp_path):
    with pytest.raises(MemoryError, match="Planned buffered exact dense subset output"):
        accumulate_dense_score_region_subsets_to_zarr(
            _provider_factory({}),
            _memberships(),
            output_path=tmp_path / "dense_subset_buffer_too_large.zarr",
            subset_column="subset",
            score_thresholds=_thresholds(),
            window_size=1,
            n_motifs=2,
            device="cpu",
            buffered_subset_updates=True,
            max_buffered_output_bytes=1,
        )


def test_dense_subset_to_zarr_chunk_buffered_updates_match_in_memory_exact_result(
    tmp_path,
):
    memory = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        chunk_subsets=1,
    )

    output_path = tmp_path / "dense_subset_chunk_buffered.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        _provider_factory({}),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        chunk_subsets=1,
        chunk_buffered_subset_updates=True,
        max_chunk_output_bytes=memory.values[0].nbytes,
        max_active_subset_chunks=1,
    )
    root = zarr.open_group(str(output_path), mode="r")
    metadata = dict(root["metadata"].attrs)

    assert result.checksum == _checksum(memory.values)
    assert metadata["complete"] is True
    assert metadata["subset_update_mode"] == "chunk_buffered_output"
    assert metadata["requested_chunk_subsets"] == 1
    assert metadata["chunk_subsets"] == 1
    assert metadata["chunk_buffer_bytes"] == memory.values[0].nbytes
    assert metadata["max_active_subset_chunks"] == 1
    assert metadata["subset_chunk_flushes"] >= len(memory.subset_ids)
    assert metadata["subset_chunk_reads"] >= 1
    np.testing.assert_allclose(root["values"][:], memory.values, rtol=0, atol=0)


def test_dense_subset_to_zarr_chunk_buffered_updates_obey_chunk_guard(tmp_path):
    with pytest.raises(MemoryError, match="One subset output chunk"):
        accumulate_dense_score_region_subsets_to_zarr(
            _provider_factory({}),
            _memberships(),
            output_path=tmp_path / "dense_subset_chunk_too_large.zarr",
            subset_column="subset",
            score_thresholds=_thresholds(),
            window_size=1,
            n_motifs=2,
            device="cpu",
            chunk_buffered_subset_updates=True,
            max_chunk_output_bytes=1,
        )


def test_dense_subset_to_zarr_none_checksum_mode_preserves_values(tmp_path):
    memory = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
    )

    output_path = tmp_path / "dense_subset_no_checksum.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        _provider_factory({}),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
        checksum_mode="none",
    )
    root = zarr.open_group(str(output_path), mode="r")
    metadata = dict(root["metadata"].attrs)

    assert result.checksum is None
    assert metadata["values_checksum"] is None
    assert metadata["values_checksum_mode"] == "none"
    assert metadata["values_checksum_status"] == "skipped"
    np.testing.assert_allclose(root["values"][:], memory.values, rtol=0, atol=0)


def test_dense_subset_to_zarr_matches_one_to_all_exact_result(tmp_path):
    memory = accumulate_dense_score_region_subsets(
        _provider_factory({}),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        chunk_subsets=2,
    )

    output_path = tmp_path / "dense_subset_one_to_all.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        _provider_factory({}),
        _memberships(),
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
        chunk_subsets=2,
    )
    root = zarr.open_group(str(output_path), mode="r")
    values = np.asarray(root["values"][:])
    metadata = dict(root["metadata"].attrs)

    assert result.subset_ids == memory.subset_ids
    assert result.checksum == _checksum(memory.values)
    assert tuple(metadata["plan"]["output_shape"]) == (3, 2, 3)
    assert metadata["query_motif_index"] == 0
    np.testing.assert_allclose(values, memory.values, rtol=0, atol=0)


def test_dense_subset_to_zarr_handles_empty_memberships(tmp_path):
    memberships = pd.DataFrame(
        {
            "subset": [],
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    output_path = tmp_path / "empty_dense_subset.zarr"
    result = accumulate_dense_score_region_subsets_to_zarr(
        lambda *_args: pytest.fail("empty input should not call provider"),
        memberships,
        output_path=output_path,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
    )
    root = zarr.open_group(str(output_path), mode="r")
    metadata = dict(root["metadata"].attrs)

    assert result.subset_ids == []
    assert result.plan.output_shape == (0, 2, 2, 3)
    assert result.checksum == hashlib.sha256(b"").hexdigest()
    assert root["values"].shape == (0, 2, 2, 3)
    assert metadata["complete"] is True
    assert metadata["unique_regions_processed"] == 0


def test_dense_subset_one_to_all_matches_repeated_direct_accumulation():
    calls = {}
    result = accumulate_dense_score_region_subsets(
        _provider_factory(calls),
        _memberships(),
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        query_motif_index=0,
        device="cpu",
    )

    expected = np.stack(
        [
            _direct_one_to_all(_scores_a()),
            _direct_one_to_all(_scores_a()) + _direct_one_to_all(_scores_b()),
            _direct_one_to_all(_scores_a()),
        ],
        axis=0,
    )
    assert result.plan.output_shape == (3, 2, 3)
    assert calls == {("chr1", 0, 5): 1, ("chr1", 10, 15): 1}
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=0)


def test_dense_subset_duplicate_membership_counts_twice():
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S1"],
            "chrom": ["chr1", "chr1"],
            "start": [0, 0],
            "end": [5, 5],
        }
    )
    calls = {}
    result = accumulate_dense_score_region_subsets(
        _provider_factory(calls),
        memberships,
        subset_column="subset",
        score_thresholds=_thresholds(),
        window_size=1,
        n_motifs=2,
        device="cpu",
    )

    np.testing.assert_allclose(
        result.values[0],
        _direct_all_to_all(_scores_a()) * 2,
        rtol=0,
        atol=0,
    )
    assert calls == {("chr1", 0, 5): 1}


def test_dense_subset_block_provider_matches_interval_provider(tmp_path):
    seq_path = tmp_path / "seq.zarr"
    _write_sequence_zarr(seq_path, "AACGTACCAA")
    kernels = _one_base_kernels()
    memberships = pd.DataFrame(
        {
            "subset": ["S1", "S1", "S2"],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [1, 3, 3],
            "end": [7, 9, 9],
        }
    )
    thresholds = {"p0.0001": torch.zeros((2,), dtype=torch.float32)}
    interval = accumulate_dense_score_region_subsets(
        DenseGenomeScoreProvider(str(seq_path), kernels, device="cpu"),
        memberships,
        subset_column="subset",
        score_thresholds=thresholds,
        window_size=1,
        n_motifs=2,
        device="cpu",
    )
    block_provider = DenseGenomeBlockScoreProvider(
        str(seq_path),
        kernels,
        memberships,
        subset_column="subset",
        device="cpu",
        max_gap_bp=0,
    )
    block = accumulate_dense_score_region_subsets(
        block_provider,
        memberships,
        subset_column="subset",
        score_thresholds=thresholds,
        window_size=1,
        n_motifs=2,
        device="cpu",
    )

    np.testing.assert_allclose(block.values, interval.values, rtol=0, atol=0)
    assert block_provider.stats["provider_blocks_scanned"] == 1


def test_dense_subset_resource_plan_reports_reuse_and_storage_recommendation():
    plan = plan_dense_score_region_subset_resources(
        _memberships(),
        subset_column="subset",
        n_motifs=2,
        window_size=1,
        dtype=np.float32,
        output_path="local_out.zarr",
        storage_threshold_bytes=100,
        mnt_storage_base="/mnt/storage/caesarion_test",
    )
    data = plan.to_dict()

    assert plan.plan.output_shape == (3, 2, 2, 3)
    assert plan.plan.output_bytes == 144
    assert plan.repeated_region_bp == 20
    assert plan.unique_region_bp == 10
    assert plan.coalesced_block_bp == 10
    assert plan.n_coalesced_blocks == 2
    assert plan.bp_reuse_factor == 2
    assert plan.use_mnt_storage is True
    assert plan.recommended_output_path == "/mnt/storage/caesarion_test/local_out.zarr"
    assert data["schema_version"] == "dense_subset_resource_plan_v1"


def test_dense_subset_resource_plan_accounts_for_block_coalescing_gap():
    plan = plan_dense_score_region_subset_resources(
        _memberships(),
        subset_column="subset",
        n_motifs=2,
        window_size=1,
        motif_length=1,
        strands=2,
        dtype=np.float32,
        provider_max_gap_bp=5,
        storage_threshold_bytes=10_000,
    )

    assert plan.n_coalesced_blocks == 1
    assert plan.coalesced_block_bp == 15
    assert plan.block_bp_vs_unique_region_bp == 1.5
    assert plan.coalesced_block_score_bytes_sum == 240
    assert plan.max_block_score_bytes_estimate == 240
    assert plan.median_block_score_bytes_estimate == 240.0
    assert plan.use_mnt_storage is False


def test_dense_subset_resource_plan_caps_block_score_bytes():
    plan = plan_dense_score_region_subset_resources(
        _memberships(),
        subset_column="subset",
        n_motifs=2,
        window_size=1,
        motif_length=1,
        strands=2,
        dtype=np.float32,
        provider_max_gap_bp=5,
        provider_max_block_score_bytes=80,
        storage_threshold_bytes=10_000,
    )
    data = plan.to_dict()

    assert plan.n_coalesced_blocks == 2
    assert plan.coalesced_block_bp == 10
    assert plan.block_bp_vs_unique_region_bp == 1.0
    assert plan.provider_max_block_score_bytes == 80
    assert plan.coalesced_block_score_bytes_sum == 160
    assert plan.max_block_score_bytes_estimate == 80
    assert plan.median_block_score_bytes_estimate == 80.0
    assert data["provider_max_block_score_bytes"] == 80
    assert data["max_block_score_bytes_estimate"] == 80


def test_dense_subset_resource_plan_rejects_unavoidable_score_bytes():
    with pytest.raises(MemoryError, match="estimated score size"):
        plan_dense_score_region_subset_resources(
            _memberships(),
            subset_column="subset",
            n_motifs=2,
            window_size=1,
            motif_length=1,
            strands=2,
            dtype=np.float32,
            provider_max_block_score_bytes=79,
            storage_threshold_bytes=10_000,
        )
