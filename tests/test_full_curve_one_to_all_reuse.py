from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motiverse import full_curve_reuse as reuse_backend

reuse_script = reuse_backend


class _FakeHocomoco:
    motif_names = [
        "P53.H13CORE.0.P.B",
        "ESR1.H13CORE.0.P.B",
        "ESR1.H13CORE.0.P.B_RC",
    ]
    motif_kernels = np.zeros((3, 5, 4), dtype=np.float32)


def _fake_metadata():
    return SimpleNamespace(
        to_dict=lambda: {
            "motif_source": "aligned_pt",
            "aligned_motif_path": "/aligned.pt",
            "loaded_with_aligned": True,
            "use_aligned": True,
            "require_aligned": True,
            "motif_count": 3,
            "motif_kernel_shape": (3, 5, 4),
            "motif_kernel_checksum": "kernel-checksum",
            "motif_names_checksum": "names-checksum",
            "motif_selection": None,
            "cache_schema_version": "positional_hit_cache_v1",
            "threshold_mode": "pvalue_mapping",
        }
    )


def _base_args(tmp_path: Path) -> argparse.Namespace:
    genome = tmp_path / "hg38.zarr"
    motifs = tmp_path / "motifs_with_rc_aligned.pt"
    regions = tmp_path / "regions.tsv"
    dense_script = tmp_path / "dense_subset_aggregation.py"
    genome.mkdir()
    motifs.write_bytes(b"pt")
    regions.write_text("subset\tchrom\tstart\tend\nS1\tchr1\t0\t1000\n")
    dense_script.write_text("# placeholder\n")
    return argparse.Namespace(
        regions_tsv=regions,
        subset_column="subset",
        output_dir=tmp_path / "out",
        genome_zarr=str(genome),
        allow_nondefault_genome=True,
        aligned_motif_path=str(motifs),
        motifs=None,
        query_motif="P53.H13CORE.0.P.B",
        query_motif_index=None,
        p_value_threshold="p0.0001",
        score_threshold=None,
        no_pvalue_mapping=False,
        window=500,
        strand_specific=False,
        provider="interval",
        provider_max_gap_bp=0,
        provider_max_block_span_bp=1_000_000,
        provider_max_block_score_gb=None,
        provider_max_cached_blocks=1,
        provider_max_sequence_cache_gb=0.5,
        anchor_contribution_mode="strict",
        output_accumulator="numpy",
        dtype="float32",
        device="cpu",
        chunk_subsets=64,
        checksum_mode="full",
        max_output_gb=None,
        cache_path=None,
        query_motif_hit_cache_tile_size=None,
        query_motif_hit_cache_tile_extension_bp=None,
        cache_manifest_json=None,
        reuse_cache_manifest=None,
        query_motif_hit_cache_only=False,
        output_zarr=None,
        plan_json=None,
        run_json=None,
        workflow_json=None,
        reuse_existing_cache=False,
        overwrite_cache=False,
        overwrite_output=False,
        plan_only=False,
        coalesce_membership_patterns=False,
        max_membership_pattern_cache_gb=None,
        dense_subset_script=str(dense_script),
        storage_threshold_gb=10.0,
        mnt_storage_base="/mnt/storage/caesarion_benchmarks/genome_motif_analysis",
        subprocess_backend=True,
    )


def test_resolve_query_motif_uses_aligned_pt_and_exact_order(monkeypatch):
    calls = {}

    def fake_load_hocomoco_motifs(**kwargs):
        calls.update(kwargs)
        return _FakeHocomoco(), _fake_metadata()

    monkeypatch.setattr(reuse_backend, "load_hocomoco_motifs", fake_load_hocomoco_motifs)

    index, metadata = reuse_backend.resolve_query_motif_index(
        motif_selection=None,
        query_motif="P53.H13CORE.0.P.B",
        query_motif_index=None,
        aligned_motif_path="/aligned.pt",
    )

    assert index == 0
    assert metadata["resolved_query_motif_name"] == "P53.H13CORE.0.P.B"
    assert metadata["loaded_with_aligned"] is True
    assert calls["use_aligned_motifs"] is True
    assert calls["require_aligned_motifs"] is True
    assert calls["aligned_motif_path"] == "/aligned.pt"


def test_resolve_query_motif_fails_on_ambiguous_partial(monkeypatch):
    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    with pytest.raises(ValueError, match="resolved to 2 motifs"):
        reuse_backend.resolve_query_motif_index(
            motif_selection=None,
            query_motif="ESR1",
            query_motif_index=None,
            aligned_motif_path="/aligned.pt",
        )


def test_full_curve_workflow_builds_query_motif_hit_cache_without_topk(
    tmp_path,
    monkeypatch,
):
    args = _base_args(tmp_path)
    commands = []

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_runner(command, check):
        assert check is True
        commands.append(command)
        output_json = Path(command[command.index("--output-json") + 1])
        if "--plan-only" in command:
            output_json.write_text(
                json.dumps(
                    {
                        "schema_version": "dense_subset_resource_plan_v1",
                        "n_subsets": 2,
                        "n_region_memberships": 4,
                        "n_unique_regions": 1,
                        "reuse_factor": 1.0,
                        "n_motifs": 3,
                        "output_bytes": 24024,
                    }
                )
                + "\n"
            )
        else:
            output_json.write_text(
                json.dumps(
                    {
                        "mode": "dense-exact-subset-query-motif-hit-cache-replay",
                        "loaded_with_aligned": True,
                        "values_checksum_mode": "full",
                        "n_subsets": 2,
                        "n_regions": 4,
                        "n_unique_regions": 1,
                        "reuse_factor": 4.0,
                        "motif_count": 3,
                        "query_motif_hit_hits": 7,
                        "query_motif_hit_hit_regions": 1,
                        "output_bytes": 24024,
                    }
                )
                + "\n"
            )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args, runner=fake_runner)

    assert summary["screening_mode"] == "none"
    assert summary["scalar_summary_policy"] == reuse_backend.SCALAR_SUMMARY_POLICY
    assert summary["scalar_screening_retired"] is True
    assert "top_k_enabled" not in summary
    assert summary["full_curves_required"] is True
    assert summary["output_contract"] == reuse_backend.FULL_CURVE_OUTPUT_CONTRACT
    assert summary["recommended_strategy"] == reuse_backend.FULL_CURVE_REUSE_STRATEGY
    assert summary["precompute_grain"] == "exact_unique_region_query_motif_hit"
    assert summary["combined_reuse_components"] == reuse_backend.FULL_CURVE_REUSE_COMPONENTS
    assert summary["source_precompute_components"] == [
        "exact_unique_region_query_motif_hit",
        "many_subset_dense_full_curve",
    ]
    assert summary["validation_status"] == "PASS"
    assert len(commands) == 2
    joined = "\n".join(" ".join(command) for command in commands).lower()
    assert "topk" not in joined
    assert "top-k" not in joined
    assert "--build-query-motif-hit-cache" in commands[1]
    assert "--query-motif-index" in commands[1]
    assert commands[1][commands[1].index("--query-motif-index") + 1] == "0"
    assert "--coalesce-membership-patterns" not in commands[1]
    assert (args.output_dir / "workflow_summary.json").exists()
    assert (args.output_dir / "eureka_backend_payload.json").exists()
    assert (args.output_dir / "eureka_backend_manifest.json").exists()
    payload = json.loads((args.output_dir / "eureka_backend_payload.json").read_text())
    manifest = json.loads((args.output_dir / "eureka_backend_manifest.json").read_text())
    assert payload["schema_version"] == (reuse_backend.EUREKA_BACKEND_RUN_PAYLOAD_SCHEMA_VERSION)
    assert payload["backend"] == ("motiverse.full_curve_one_to_all_reuse")
    assert payload["value_kind"] == "one-to-all"
    assert payload["screening_mode"] == "none"
    assert payload["scalar_summary_policy"] == reuse_backend.SCALAR_SUMMARY_POLICY
    assert payload["scalar_screening_retired"] is True
    assert payload["output_contract"] == reuse_backend.FULL_CURVE_OUTPUT_CONTRACT
    assert payload["full_curves_required"] is True
    assert payload["motif_names_checksum"] == "names-checksum"
    assert payload["motif_kernel_checksum"] == "kernel-checksum"
    assert payload["p_value_threshold"] == "p0.0001"
    assert payload["n_subsets"] == 2
    assert payload["n_region_memberships"] == 4
    assert payload["n_unique_regions"] == 1
    assert payload["reuse_factor"] == 4.0
    assert payload["reuse_percent"] == 75.0
    assert payload["motif_count"] == 3
    assert payload["query_motif_hit_hits"] == 7
    assert payload["query_motif_hit_cache_anchor_hits"] == 7
    assert payload["query_motif_hit_hit_regions"] == 1
    assert payload["output_bytes"] == 24024
    assert payload["eureka_backend_manifest_json"] == str(
        args.output_dir / "eureka_backend_manifest.json"
    )
    assert manifest["schema_version"] == (reuse_backend.EUREKA_BACKEND_MANIFEST_SCHEMA_VERSION)
    assert manifest["full_curves_required"] is True
    assert manifest["screening_mode"] == "none"
    assert manifest["scalar_screening_retired"] is True
    assert manifest["genome_zarr_path"] == str(args.genome_zarr)
    assert manifest["motif_source"] == "aligned_pt"
    assert manifest["aligned_motif_path"] == str(args.aligned_motif_path)
    assert manifest["loaded_with_aligned"] is True
    assert manifest["motif_count"] == 3
    assert manifest["n_subsets"] == 2
    assert manifest["n_region_memberships"] == 4
    assert manifest["n_unique_regions"] == 1
    assert manifest["reuse_factor"] == 4.0
    assert manifest["reuse_percent"] == 75.0
    assert manifest["output_bytes"] == 24024
    assert manifest["output_contract"] == reuse_backend.FULL_CURVE_OUTPUT_CONTRACT
    assert manifest["input_contract"]["biological_eligibility"] == (
        "complete_full_curve_tensor_required"
    )
    assert manifest["input_contract"]["motif_source_requirement"] == ("aligned_hocomoco_pt")
    assert manifest["subset_aggregation"] == "many_subset_dense_full_curve"
    assert manifest["active_precompute_grain"] == "exact_unique_region_query_motif_hit"
    assert "tile_assisted_exact_interval_query_motif_hit" in manifest["precompute_grains"]
    assert "full-curve-reuse" in manifest["commands"]["run"]
    assert "topk" not in json.dumps(manifest).lower()
    assert "top-k" not in json.dumps(manifest).lower()


def test_full_curve_workflow_routes_tile_assisted_query_motif_hit_cache(
    tmp_path,
    monkeypatch,
):
    args = _base_args(tmp_path)
    args.query_motif_hit_cache_tile_size = 500
    args.query_motif_hit_cache_tile_extension_bp = 501
    commands = []

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_runner(command, check):
        commands.append(command)
        output_json = Path(command[command.index("--output-json") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        if "--plan-only" in command:
            output_json.write_text(json.dumps({"n_unique_regions": 1}) + "\n")
        else:
            output_json.write_text(
                json.dumps(
                    {
                        "mode": "dense-exact-subset-query-motif-hit-cache-replay",
                        "loaded_with_aligned": True,
                        "validation_status": "PASS",
                    }
                )
                + "\n"
            )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args, runner=fake_runner)

    assert summary["screening_mode"] == "none"
    assert summary["full_curves_required"] is True
    assert summary["query_motif_hit_cache_build_mode"] == (
        "tile_assisted_exact_interval_reconstruction"
    )
    assert summary["combined_reuse_components"] == reuse_backend.FULL_CURVE_REUSE_COMPONENTS
    assert summary["source_precompute_components"] == [
        "tile_assisted_exact_interval_query_motif_hit",
        "many_subset_dense_full_curve",
    ]
    assert summary["query_motif_hit_cache_tile_size_bp"] == 500
    assert "--query-motif-hit-cache-tile-size" in commands[1]
    assert commands[1][commands[1].index("--query-motif-hit-cache-tile-size") + 1] == "500"
    assert "--query-motif-hit-cache-tile-extension-bp" in commands[1]
    joined = " ".join(commands[1]).lower()
    assert "topk" not in joined
    assert "top-k" not in joined


def test_full_curve_workflow_builds_query_motif_hit_cache_only_stage(
    tmp_path,
    monkeypatch,
):
    args = _base_args(tmp_path)
    args.subprocess_backend = True
    args.query_motif_hit_cache_only = True
    args.query_motif_hit_cache_tile_size = 500
    args.query_motif_hit_cache_tile_extension_bp = 500
    commands = []

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_runner(command, check):
        assert check is True
        commands.append(command)
        output_json = Path(command[command.index("--output-json") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        if "--plan-only" in command:
            output_json.write_text(
                json.dumps(
                    {
                        "schema_version": "dense_subset_resource_plan_v1",
                        "plan": {
                            "n_subsets": 2,
                            "n_region_memberships": 4,
                            "n_unique_regions": 1,
                            "reuse_factor": 4.0,
                            "n_motifs": 3,
                            "output_bytes": 24024,
                        },
                    }
                )
                + "\n"
            )
        else:
            assert "--query-motif-hit-cache-only" in command
            assert "--build-query-motif-hit-cache" in command
            assert "--output-zarr" not in command
            output_json.write_text(
                json.dumps(
                    {
                        "schema_version": "dense_query_motif_hit_cache_v1",
                        "mode": "dense-exact-query-motif-hit-cache-build",
                        "validation_status": "PASS",
                        "query_motif_hit_cache_path": str(
                            args.output_dir / "query_motif_hit_cache.zarr"
                        ),
                        "query_motif_hit_cache_build_mode": (
                            "tile_assisted_exact_interval_reconstruction"
                        ),
                        "query_motif_hit_cache_tile_size_bp": 500,
                        "query_motif_hit_cache_tile_extension_bp": 530,
                        "query_motif_hit_cache_requested_tile_extension_bp": 500,
                        "query_motif_hit_cache_minimum_tile_extension_bp": 530,
                        "query_motif_hit_cache_anchor_hits": 7,
                        "query_motif_hit_hit_regions": 3,
                        "query_motif_hit_cache_total_intervals": 1.0,
                        "query_motif_hit_cache_required_intervals": 1.0,
                        "query_motif_hit_cache_missing_intervals": 0.0,
                        "query_motif_hit_cache_coverage_fraction": 1.0,
                        "expected_query_motif_hit_cache_metadata": {
                            "window_size": 500,
                            "query_motif_index": 0,
                        },
                        "genome_zarr_path": str(args.genome_zarr),
                        "motif_count": 3,
                        "motif_names_checksum": "names-checksum",
                        "motif_kernel_checksum": "kernel-checksum",
                        "motif_kernel_shape": [3, 5, 4],
                        "motif_source": "aligned_pt",
                        "aligned_motif_path": str(args.aligned_motif_path),
                        "loaded_with_aligned": True,
                        "threshold_mode": "pvalue_mapping",
                        "p_value_threshold": "p0.0001",
                        "window_size": 500,
                        "query_motif_index": 0,
                        "n_subsets": 2,
                        "n_regions": 4,
                        "n_unique_regions": 1,
                        "reuse_factor": 4.0,
                    }
                )
                + "\n"
            )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(
        args,
        runner=fake_runner,
    )
    payload = json.loads((args.output_dir / "eureka_backend_payload.json").read_text())
    manifest = json.loads((args.output_dir / "eureka_backend_manifest.json").read_text())
    cache_manifest = json.loads(
        (args.output_dir / "query_motif_hit_cache_manifest.json").read_text()
    )

    assert summary["validation_status"] == "PASS"
    assert summary["workflow_stage"] == "query_motif_hit_cache_precompute"
    assert summary["query_motif_hit_cache_only"] is True
    assert summary["final_biological_output_ready"] is False
    assert summary["output_contract"] == (reuse_backend.QUERY_MOTIF_HIT_CACHE_ONLY_OUTPUT_CONTRACT)
    assert summary["biological_output_contract"] == ("precompute_only_not_final_biological_output")
    assert summary["subset_aggregation"] == "deferred_many_subset_dense_full_curve"
    assert len(commands) == 2
    assert payload["workflow_stage"] == "query_motif_hit_cache_precompute"
    assert payload["query_motif_hit_cache_only"] is True
    assert payload["final_biological_output_ready"] is False
    assert payload["output_contract"] == (reuse_backend.QUERY_MOTIF_HIT_CACHE_ONLY_OUTPUT_CONTRACT)
    assert payload["output_bytes"] is None
    assert payload["planned_full_curve_output_bytes"] == 24024
    assert payload["query_motif_hit_hits"] == 7
    assert manifest["query_motif_hit_cache_only"] is True
    assert manifest["final_biological_output_ready"] is False
    assert manifest["output_bytes"] is None
    assert manifest["planned_full_curve_output_bytes"] == 24024
    assert manifest["subset_aggregation"] == "deferred_many_subset_dense_full_curve"
    assert manifest["artifacts"]["output_zarr"] is None
    assert "--query-motif-hit-cache-only" in manifest["commands"]["run"]
    assert cache_manifest["cache_path"] == str(args.output_dir / "query_motif_hit_cache.zarr")
    assert cache_manifest["window_size"] == 500
    assert cache_manifest["query_motif_index"] == 0


def test_full_curve_workflow_marks_coalescing_as_pass_warn(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.coalesce_membership_patterns = True

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_runner(command, check):
        output_json = Path(command[command.index("--output-json") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"loaded_with_aligned": True}) + "\n")

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args, runner=fake_runner)

    assert summary["validation_status"] == "PASS_WARN"
    assert summary["strict_shape_default"] is False
    assert summary["membership_pattern_coalescing"] is True
    assert "--coalesce-membership-patterns" in summary["commands"][1]


def test_full_curve_workflow_accepts_caller_supplied_genome(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.allow_nondefault_genome = False

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_runner(command, check):
        output_json = Path(command[command.index("--output-json") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"loaded_with_aligned": True}) + "\n")

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args, runner=fake_runner)

    assert summary["validation_status"] == "PASS"
    assert summary["assembly_policy"] == "caller_supplied"


def test_dense_subset_command_defaults_to_package_backend(tmp_path):
    args = _base_args(tmp_path)
    args.dense_subset_script = None
    args.resolved_query_motif_index = 0

    command = reuse_backend.dense_subset_command(
        args,
        "--plan-only",
        "--output-json",
        str(tmp_path / "plan.json"),
    )

    assert command[:3] == [
        __import__("sys").executable,
        "-m",
        "motiverse.dense_subset_workflow",
    ]
    assert "scripts/dense_subset_aggregation.py" not in " ".join(command)


def test_full_curve_parser_exposes_no_topk_screening_options():
    help_text = reuse_backend.build_parser().format_help().lower()

    assert "topk" not in help_text
    assert "top-k" not in help_text
    assert "screening" not in help_text


def test_output_accumulator_auto_resolves_to_torch_only_on_cuda():
    effective, metadata = reuse_backend.resolve_output_accumulator(
        requested="auto",
        effective_device="cuda",
        coalesce_membership_patterns=False,
    )
    assert effective == "torch"
    assert metadata["requested_output_accumulator"] == "auto"
    assert metadata["output_accumulator_auto_selected"] is True

    effective, metadata = reuse_backend.resolve_output_accumulator(
        requested="auto",
        effective_device="cpu",
        coalesce_membership_patterns=False,
    )
    assert effective == "numpy"
    assert metadata["effective_output_accumulator"] == "numpy"

    effective, metadata = reuse_backend.resolve_output_accumulator(
        requested="auto",
        effective_device="cuda",
        coalesce_membership_patterns=True,
    )
    assert effective == "numpy"
    assert metadata["output_accumulator_auto_selected"] is True

    with pytest.raises(ValueError, match="cannot be combined"):
        reuse_backend.resolve_output_accumulator(
            requested="torch",
            effective_device="cuda",
            coalesce_membership_patterns=True,
        )


def test_script_is_thin_backend_shim():
    assert reuse_script.main is reuse_backend.main


def test_full_curve_workflow_defaults_to_in_process_backend(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.subprocess_backend = False

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_in_process(run_args):
        plan = {"schema_version": "dense_subset_resource_plan_v1"}
        run = {
            "validation_status": "PASS",
            "mode": "dense-exact-subset-query-motif-hit-cache-replay",
        }
        reuse_backend._write_json(run_args.plan_json, plan)
        reuse_backend._write_json(run_args.run_json, run)
        return plan, run

    monkeypatch.setattr(
        reuse_backend,
        "_run_query_motif_hit_reuse_in_process",
        fake_in_process,
    )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args)

    assert summary["execution_backend"] == "in_process"
    assert summary["commands"] == []
    assert summary["run_summary"]["validation_status"] == "PASS"


def test_full_curve_auto_accumulator_records_effective_torch_on_cuda(
    tmp_path,
    monkeypatch,
):
    args = _base_args(tmp_path)
    args.output_accumulator = "auto"
    args.subprocess_backend = False

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )
    monkeypatch.setattr(
        reuse_backend.dense_subset_workflow,
        "resolve_analysis_device",
        lambda requested: (
            "cuda",
            {
                "requested_device": requested,
                "effective_device": "cuda",
                "cuda_available": True,
            },
        ),
    )

    def fake_in_process(run_args):
        assert run_args.requested_output_accumulator == "auto"
        assert run_args.output_accumulator == "torch"
        plan = {"schema_version": "dense_subset_resource_plan_v1"}
        run = {
            "validation_status": "PASS",
            "mode": "dense-exact-subset-query-motif-hit-cache-replay",
            "requested_output_accumulator": run_args.requested_output_accumulator,
            "output_accumulator": run_args.output_accumulator,
            "effective_output_accumulator": run_args.output_accumulator,
        }
        reuse_backend._write_json(run_args.plan_json, plan)
        reuse_backend._write_json(run_args.run_json, run)
        return plan, run

    monkeypatch.setattr(
        reuse_backend,
        "_run_query_motif_hit_reuse_in_process",
        fake_in_process,
    )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args)

    assert summary["requested_output_accumulator"] == "auto"
    assert summary["output_accumulator"] == "torch"
    assert summary["effective_output_accumulator"] == "torch"
    assert summary["output_accumulator_auto_selected"] is True
    assert summary["validation_status"] == "PASS_WARN"


def test_full_curve_workflow_writes_cache_manifest(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.subprocess_backend = False
    cache_path = tmp_path / "query_motif_hit_cache.zarr"
    expected_metadata = {"motif_names_checksum": "names-checksum"}

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_in_process(run_args):
        run = {
            "validation_status": "PASS",
            "mode": "dense-exact-subset-query-motif-hit-cache-replay",
            "query_motif_hit_cache_path": str(cache_path),
            "query_motif_hit_cache_build_mode": ("tile_assisted_exact_interval_reconstruction"),
            "query_motif_hit_cache_tile_size_bp": 500,
            "query_motif_hit_cache_tile_extension_bp": 530,
            "query_motif_hit_cache_requested_tile_extension_bp": 500,
            "query_motif_hit_cache_minimum_tile_extension_bp": 530,
            "query_motif_hit_cache_total_intervals": 7.0,
            "query_motif_hit_cache_required_intervals": 3.0,
            "query_motif_hit_cache_missing_intervals": 0.0,
            "query_motif_hit_cache_coverage_fraction": 1.0,
            "expected_query_motif_hit_cache_metadata": expected_metadata,
            "genome_zarr_path": str(run_args.genome_zarr),
            "motif_count": 3,
            "motif_names_checksum": "names-checksum",
            "motif_kernel_checksum": "kernel-checksum",
            "motif_kernel_shape": [3, 5, 4],
            "motif_source": "aligned_pt",
            "aligned_motif_path": str(run_args.aligned_motif_path),
            "loaded_with_aligned": True,
            "threshold_mode": "pvalue_mapping",
            "window_size": 500,
            "query_motif_index": 0,
            "values_checksum": "values-checksum",
            "values_checksum_status": "complete",
        }
        plan = {"schema_version": "dense_subset_resource_plan_v1"}
        reuse_backend._write_json(run_args.plan_json, plan)
        reuse_backend._write_json(run_args.run_json, run)
        return plan, run

    monkeypatch.setattr(
        reuse_backend,
        "_run_query_motif_hit_reuse_in_process",
        fake_in_process,
    )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args)
    manifest_path = args.output_dir / "query_motif_hit_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert summary["cache_manifest_json"] == str(manifest_path)
    assert manifest["schema_version"] == "query_motif_hit_cache_manifest_v1"
    assert manifest["cache_path"] == str(cache_path)
    assert manifest["expected_metadata"] == expected_metadata
    assert manifest["values_checksum"] == "values-checksum"
    assert (
        manifest["query_motif_hit_cache_build_mode"]
        == "tile_assisted_exact_interval_reconstruction"
    )
    assert manifest["query_motif_hit_cache_tile_extension_bp"] == 530
    assert manifest["query_motif_hit_cache_requested_tile_extension_bp"] == 500
    assert manifest["query_motif_hit_cache_minimum_tile_extension_bp"] == 530
    assert manifest["query_motif_hit_cache_total_intervals"] == 7.0
    assert manifest["query_motif_hit_cache_required_intervals"] == 3.0
    assert manifest["query_motif_hit_cache_missing_intervals"] == 0.0
    assert manifest["query_motif_hit_cache_coverage_fraction"] == 1.0


def test_cache_manifest_writer_uses_expected_metadata_fallback(tmp_path):
    manifest_path = tmp_path / "manifest.json"

    manifest = reuse_backend._write_cache_manifest(
        manifest_path,
        cache_path=tmp_path / "cache.zarr",
        expected_metadata={
            "window_size": 500,
            "strand_specific": False,
            "strands": 2,
            "query_motif_index": 0,
        },
        run_summary={"summary_path": str(tmp_path / "summary.json")},
        workflow_summary={"schema_version": "shared_query_motif_hit_cache_build_v1"},
    )

    assert manifest["window_size"] == 500
    assert manifest["strand_specific"] is False
    assert manifest["strands"] == 2
    assert manifest["query_motif_index"] == 0
    assert manifest["expected_metadata"]["window_size"] == 500


def test_full_curve_workflow_reuses_cache_manifest(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    args.subprocess_backend = False
    cache_path = tmp_path / "existing_cache.zarr"
    cache_path.mkdir()
    manifest_path = tmp_path / "query_motif_hit_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "query_motif_hit_cache_manifest_v1",
                "cache_path": str(cache_path),
                "window_size": None,
                "strand_specific": None,
                "strands": None,
            }
        )
        + "\n"
    )
    args.reuse_cache_manifest = str(manifest_path)

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_in_process(run_args):
        assert run_args.reuse_existing_cache is True
        assert run_args.cache_path == cache_path
        run = {
            "validation_status": "PASS",
            "query_motif_hit_cache_path": str(cache_path),
        }
        plan = {"schema_version": "dense_subset_resource_plan_v1"}
        reuse_backend._write_json(run_args.plan_json, plan)
        reuse_backend._write_json(run_args.run_json, run)
        return plan, run

    monkeypatch.setattr(
        reuse_backend,
        "_run_query_motif_hit_reuse_in_process",
        fake_in_process,
    )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args)

    assert summary["reuse_cache_manifest"] == str(manifest_path)
    assert summary["cache_path"] == str(cache_path)
    assert summary["reuse_cache_manifest_compatibility"]["compatible"] is True
    assert "aligned_motif_path" in summary["reuse_cache_manifest_compatibility"]["missing_fields"]
    assert "window_size" in summary["reuse_cache_manifest_compatibility"]["missing_fields"]


def test_full_curve_workflow_reports_reused_tile_manifest_precompute(
    tmp_path,
    monkeypatch,
):
    args = _base_args(tmp_path)
    args.subprocess_backend = False
    cache_path = tmp_path / "tile_cache.zarr"
    cache_path.mkdir()
    manifest_path = tmp_path / "query_motif_hit_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "query_motif_hit_cache_manifest_v1",
                "cache_path": str(cache_path),
                "genome_zarr_path": str(args.genome_zarr),
                "motif_count": 3,
                "motif_names_checksum": "names-checksum",
                "motif_kernel_checksum": "kernel-checksum",
                "motif_kernel_shape": [3, 5, 4],
                "motif_source": "aligned_pt",
                "aligned_motif_path": str(args.aligned_motif_path),
                "loaded_with_aligned": True,
                "threshold_mode": "pvalue_mapping",
                "p_value_threshold": "p0.0001",
                "score_threshold": None,
                "window_size": 500,
                "query_motif_index": 0,
                "strand_specific": False,
                "strands": 2,
                "query_motif_hit_cache_build_mode": ("tile_assisted_exact_interval_reconstruction"),
                "query_motif_hit_cache_tile_size_bp": 500,
                "query_motif_hit_cache_tile_extension_bp": 530,
                "query_motif_hit_cache_requested_tile_extension_bp": 500,
                "query_motif_hit_cache_minimum_tile_extension_bp": 530,
            }
        )
        + "\n"
    )
    args.reuse_cache_manifest = str(manifest_path)

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    def fake_in_process(run_args):
        run = {
            "validation_status": "PASS",
            "query_motif_hit_cache_path": str(cache_path),
        }
        plan = {"schema_version": "dense_subset_resource_plan_v1"}
        reuse_backend._write_json(run_args.plan_json, plan)
        reuse_backend._write_json(run_args.run_json, run)
        return plan, run

    monkeypatch.setattr(
        reuse_backend,
        "_run_query_motif_hit_reuse_in_process",
        fake_in_process,
    )

    summary = reuse_backend.run_full_curve_one_to_all_reuse(args)

    assert summary["reuse_cache_manifest_compatibility"]["compatible"] is True
    assert summary["reuse_cache_manifest_compatibility"]["missing_fields"] == []
    assert summary["precompute_grain"] == "tile_assisted_exact_interval_query_motif_hit"
    assert summary["source_precompute_components"] == [
        "tile_assisted_exact_interval_query_motif_hit",
        "many_subset_dense_full_curve",
    ]
    assert summary["query_motif_hit_cache_build_mode"] == (
        "tile_assisted_exact_interval_reconstruction"
    )
    assert summary["query_motif_hit_cache_tile_size_bp"] == 500
    assert summary["query_motif_hit_cache_tile_extension_bp"] == 530
    assert summary["query_motif_hit_cache_requested_tile_extension_bp"] == 500
    assert summary["query_motif_hit_cache_minimum_tile_extension_bp"] == 530


def test_full_curve_workflow_rejects_manifest_cache_path_conflict(tmp_path):
    args = _base_args(tmp_path)
    cache_path = tmp_path / "existing_cache.zarr"
    other_cache_path = tmp_path / "other_cache.zarr"
    manifest_path = tmp_path / "query_motif_hit_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "query_motif_hit_cache_manifest_v1",
                "cache_path": str(cache_path),
            }
        )
        + "\n"
    )
    args.reuse_cache_manifest = str(manifest_path)
    args.cache_path = str(other_cache_path)

    with pytest.raises(ValueError, match="does not match"):
        reuse_backend.run_full_curve_one_to_all_reuse(args)


def test_full_curve_workflow_rejects_incompatible_reuse_manifest(
    tmp_path,
    monkeypatch,
):
    args = _base_args(tmp_path)
    args.subprocess_backend = False
    cache_path = tmp_path / "existing_cache.zarr"
    cache_path.mkdir()
    manifest_path = tmp_path / "query_motif_hit_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "query_motif_hit_cache_manifest_v1",
                "cache_path": str(cache_path),
                "genome_zarr_path": str(args.genome_zarr),
                "motif_count": 3,
                "motif_names_checksum": "names-checksum",
                "motif_kernel_checksum": "kernel-checksum",
                "motif_kernel_shape": [3, 5, 4],
                "motif_source": "aligned_pt",
                "aligned_motif_path": "/different_aligned.pt",
                "loaded_with_aligned": True,
                "threshold_mode": "pvalue_mapping",
                "p_value_threshold": "p0.0001",
                "score_threshold": None,
                "window_size": 500,
                "query_motif_index": 0,
                "strand_specific": False,
                "strands": 2,
            }
        )
        + "\n"
    )
    args.reuse_cache_manifest = str(manifest_path)

    monkeypatch.setattr(
        reuse_backend,
        "load_hocomoco_motifs",
        lambda **kwargs: (_FakeHocomoco(), _fake_metadata()),
    )

    with pytest.raises(ValueError, match="aligned_motif_path"):
        reuse_backend.run_full_curve_one_to_all_reuse(args)
