"""Signal-preserving one-to-all reuse backend for many peak-set subsets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import dense_subset_workflow
from .dense_subset import (
    DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION,
    DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS,
    DENSE_SUBSET_COORDINATE_FRAME,
    DENSE_SUBSET_SEMANTICS,
    DENSE_SUBSET_ZARR_SCHEMA_VERSION,
    DenseGenomeBlockScoreProvider,
    DenseGenomeScoreProvider,
    accumulate_dense_score_region_subsets_from_query_motif_hit_cache,
    build_dense_query_motif_hit_cache,
    build_dense_query_motif_hit_cache_from_tiles,
    plan_dense_interval_contribution_cache_resources,
    plan_dense_score_region_subset_resources,
)
from .motif_source import load_hocomoco_motifs
from .peak_set_memberships import build_peak_set_memberships

DEFAULT_GENOME_ZARR = os.environ.get("MOTIVERSE_GENOME_ZARR")
DEFAULT_ALIGNED_MOTIF_PATH = os.environ.get("MOTIVERSE_ALIGNED_MOTIFS")
DEFAULT_QUERY_MOTIF = "P53.H13CORE.0.P.B"
QUERY_MOTIF_HIT_CACHE_MANIFEST_SCHEMA_VERSION = "query_motif_hit_cache_manifest_v1"
DEFAULT_ANCHOR_CONTRIBUTION_MODE = "strict"
DEFAULT_OUTPUT_ACCUMULATOR = "numpy"
DEFAULT_PROVIDER_MAX_SEQUENCE_CACHE_GB = 0.5
FULL_CURVE_OUTPUT_CONTRACT = "complete_subset_motif_position_tensor"
QUERY_MOTIF_HIT_CACHE_ONLY_OUTPUT_CONTRACT = "query_motif_hit_cache_precompute_only"
FULL_CURVE_REUSE_STRATEGY = (
    "combined_unique_or_tile_query_motif_hit_precompute_plus_many_subset_full_curve_replay"
)
SCALAR_SUMMARY_POLICY = "not_valid_for_biological_shape_analysis"
FULL_CURVE_REUSE_COMPONENTS = [
    "unique_region_or_tile_assisted_query_motif_hit_precompute",
    "exact_interval_reconstruction_for_boundary_jitter",
    "many_subset_full_curve_aggregation",
]
FULL_CURVE_PRECOMPUTE_GRAINS = [
    "exact_unique_region_query_motif_hit",
    "tile_assisted_exact_interval_query_motif_hit",
]
EUREKA_BACKEND_RUN_PAYLOAD_SCHEMA_VERSION = "eureka_genome_motif_one_to_all_run_payload_v1"
EUREKA_BACKEND_MANIFEST_SCHEMA_VERSION = "eureka_genome_motif_one_to_all_backend_manifest_v1"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _eureka_input_contract(args: argparse.Namespace) -> dict:
    return {
        "regions_tsv_required_columns": [
            str(args.subset_column),
            "chrom",
            "start",
            "end",
        ],
        "subset_column": str(args.subset_column),
        "coordinate_system": "0_based_half_open",
        "genome_zarr_path": str(args.genome_zarr),
        "genome_requirement": "caller_supplied_sequence_zarr",
        "aligned_motif_path": str(args.aligned_motif_path),
        "motif_source_requirement": "aligned_hocomoco_pt",
        "query_motif": str(args.query_motif),
        "query_motif_index": int(args.resolved_query_motif_index),
        "p_value_threshold": str(args.p_value_threshold),
        "analysis_window_bp": int(args.window),
        "motif_selection": "all_motifs_by_default_or_name_subset_after_aligned_load",
        "output_contract": FULL_CURVE_OUTPUT_CONTRACT,
        "biological_eligibility": "complete_full_curve_tensor_required",
    }


def _eureka_run_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "-m",
        "motiverse",
        "full-curve-reuse",
        "--regions-tsv",
        str(args.regions_tsv),
        "--output-dir",
        str(args.output_dir),
        "--subset-column",
        str(args.subset_column),
        "--genome-zarr",
        str(args.genome_zarr),
        "--aligned-motif-path",
        str(args.aligned_motif_path),
        "--query-motif",
        str(args.query_motif),
        "--p-value-threshold",
        str(args.p_value_threshold),
        "--window",
        str(args.window),
        "--device",
        str(args.device),
        "--provider",
        str(args.provider),
        "--anchor-contribution-mode",
        str(args.anchor_contribution_mode),
        "--output-accumulator",
        str(args.requested_output_accumulator),
    ]
    if args.motifs:
        command.extend(["--motifs", str(args.motifs)])
    if args.query_motif_index is not None:
        command.extend(["--query-motif-index", str(args.query_motif_index)])
    if args.allow_nondefault_genome:
        command.append("--allow-nondefault-genome")
    if args.query_motif_hit_cache_tile_size is not None:
        command.extend(
            [
                "--query-motif-hit-cache-tile-size",
                str(args.query_motif_hit_cache_tile_size),
            ]
        )
    if args.query_motif_hit_cache_tile_extension_bp is not None:
        command.extend(
            [
                "--query-motif-hit-cache-tile-extension-bp",
                str(args.query_motif_hit_cache_tile_extension_bp),
            ]
        )
    if args.reuse_cache_manifest is not None:
        command.extend(["--reuse-cache-manifest", str(args.reuse_cache_manifest)])
    if args.reuse_existing_cache:
        command.append("--reuse-existing-cache")
    if getattr(args, "query_motif_hit_cache_only", False):
        command.append("--query-motif-hit-cache-only")
    if args.plan_only:
        command.append("--plan-only")
    if args.coalesce_membership_patterns:
        command.append("--coalesce-membership-patterns")
    return command


def _eureka_backend_run_payload(
    workflow_summary: dict,
    *,
    payload_json: Path,
    manifest_json: Path,
) -> dict:
    run = workflow_summary.get("run_summary") or {}
    plan = workflow_summary.get("plan_summary") or {}
    plan_body = plan.get("plan", plan)
    motif_count = (
        run.get("motif_count")
        or workflow_summary.get("motif_count")
        or plan.get("motif_count")
        or plan_body.get("n_motifs")
    )
    cache_only = bool(workflow_summary.get("query_motif_hit_cache_only"))
    planned_output_bytes = run.get("output_bytes", plan_body.get("output_bytes"))
    output_bytes = None if cache_only else planned_output_bytes
    n_region_memberships = run.get(
        "n_regions",
        plan_body.get("n_region_memberships"),
    )
    n_unique_regions = run.get(
        "n_unique_regions",
        plan_body.get("n_unique_regions"),
    )
    reuse_factor = run.get("reuse_factor", plan_body.get("reuse_factor"))
    reuse_percent = None
    if n_region_memberships and n_unique_regions:
        reuse_percent = 100.0 * (1.0 - float(n_unique_regions) / float(n_region_memberships))
    query_motif_hit_hits = run.get(
        "query_motif_hit_cache_anchor_hits",
        run.get("query_motif_hit_hits"),
    )
    return {
        "schema_version": EUREKA_BACKEND_RUN_PAYLOAD_SCHEMA_VERSION,
        "backend": "motiverse.full_curve_one_to_all_reuse",
        "analysis_mode": "one-to-all",
        "value_kind": "one-to-all",
        "workflow_stage": workflow_summary.get(
            "workflow_stage",
            "query_motif_hit_cache_precompute" if cache_only else "full_curve_replay",
        ),
        "status": workflow_summary.get("validation_status"),
        "recommended_strategy": workflow_summary.get("recommended_strategy"),
        "reuse_strategy": workflow_summary.get("recommended_strategy"),
        "precompute_grain": workflow_summary.get("precompute_grain"),
        "combined_reuse_components": workflow_summary.get("combined_reuse_components"),
        "subset_aggregation": workflow_summary.get("subset_aggregation"),
        "output_contract": workflow_summary.get("output_contract"),
        "biological_output_contract": workflow_summary.get(
            "biological_output_contract",
            "shape_preserving_full_curve",
        ),
        "final_biological_output_ready": bool(
            workflow_summary.get("final_biological_output_ready", False)
        ),
        "query_motif_hit_cache_only": cache_only,
        "full_curves_required": True,
        "screening_mode": workflow_summary.get("screening_mode", "none"),
        "scalar_summary_policy": workflow_summary.get(
            "scalar_summary_policy",
            SCALAR_SUMMARY_POLICY,
        ),
        "scalar_screening_retired": workflow_summary.get(
            "scalar_screening_retired",
            True,
        ),
        "genome_zarr_path": workflow_summary.get("genome_zarr_path"),
        "motif_source": workflow_summary.get("motif_source"),
        "aligned_motif_path": workflow_summary.get("aligned_motif_path"),
        "loaded_with_aligned": workflow_summary.get("loaded_with_aligned"),
        "motif_count": motif_count,
        "motif_names_checksum": workflow_summary.get("motif_names_checksum"),
        "motif_kernel_checksum": workflow_summary.get("motif_kernel_checksum"),
        "motif_kernel_shape": workflow_summary.get("motif_kernel_shape"),
        "p_value_threshold": workflow_summary.get("p_value_threshold"),
        "query_motif_index": workflow_summary.get("query_motif_index"),
        "query_motif_name": workflow_summary.get("query_motif_name"),
        "window": workflow_summary.get("window_size"),
        "n_subsets": run.get("n_subsets", plan_body.get("n_subsets")),
        "n_region_memberships": n_region_memberships,
        "n_unique_regions": n_unique_regions,
        "reuse_factor": reuse_factor,
        "reuse_percent": reuse_percent,
        "workflow_wall_s": workflow_summary.get("workflow_wall_s"),
        "query_motif_hit_cache_build_wall_s": run.get("query_motif_hit_cache_build_wall_s"),
        "aggregate_wall_s": run.get("aggregate_wall_s"),
        "full_score_provider_wall_s": run.get("full_score_provider_wall_s"),
        "dense_contribution_wall_s": run.get("dense_contribution_wall_s"),
        "subset_update_wall_s": run.get("subset_update_wall_s"),
        "query_motif_hit_cache_build_mode": workflow_summary.get(
            "query_motif_hit_cache_build_mode"
        ),
        "query_motif_hit_cache_tile_size_bp": workflow_summary.get(
            "query_motif_hit_cache_tile_size_bp"
        ),
        "query_motif_hit_cache_tile_extension_bp": workflow_summary.get(
            "query_motif_hit_cache_tile_extension_bp"
        ),
        "query_motif_hit_cache_coverage_fraction": run.get(
            "query_motif_hit_cache_coverage_fraction"
        ),
        "query_motif_hit_cache_missing_intervals": run.get(
            "query_motif_hit_cache_missing_intervals"
        ),
        "query_motif_hit_cache_anchor_hits": query_motif_hit_hits,
        "query_motif_hit_hits": query_motif_hit_hits,
        "query_motif_hit_hit_regions": run.get("query_motif_hit_hit_regions"),
        "output_bytes": output_bytes,
        "planned_full_curve_output_bytes": planned_output_bytes if cache_only else output_bytes,
        "values_checksum": run.get("values_checksum"),
        "values_checksum_mode": run.get("values_checksum_mode"),
        "values_checksum_status": run.get("values_checksum_status"),
        "artifacts": {
            "payload_json": str(payload_json),
            "manifest_json": str(manifest_json),
            "workflow_summary_json": workflow_summary.get("workflow_json"),
            "run_summary_json": workflow_summary.get("run_json"),
            "resource_plan_json": workflow_summary.get("plan_json"),
            "output_zarr": workflow_summary.get("output_zarr"),
            "cache_manifest_json": workflow_summary.get("cache_manifest_json"),
            "cache_path": workflow_summary.get("cache_path"),
        },
    }


def _eureka_backend_manifest(
    workflow_summary: dict,
    args: argparse.Namespace,
    *,
    payload_json: Path,
    manifest_json: Path,
) -> dict:
    run = workflow_summary.get("run_summary") or {}
    plan = workflow_summary.get("plan_summary") or {}
    plan_body = plan.get("plan", plan)
    motif_count = (
        run.get("motif_count")
        or workflow_summary.get("motif_count")
        or plan.get("motif_count")
        or plan_body.get("n_motifs")
    )
    n_region_memberships = run.get(
        "n_regions",
        plan_body.get("n_region_memberships"),
    )
    n_unique_regions = run.get(
        "n_unique_regions",
        plan_body.get("n_unique_regions"),
    )
    reuse_percent = None
    if n_region_memberships and n_unique_regions:
        reuse_percent = 100.0 * (1.0 - float(n_unique_regions) / float(n_region_memberships))
    cache_only = bool(workflow_summary.get("query_motif_hit_cache_only"))
    planned_output_bytes = run.get("output_bytes", plan_body.get("output_bytes"))
    return {
        "schema_version": EUREKA_BACKEND_MANIFEST_SCHEMA_VERSION,
        "backend": "motiverse.full_curve_one_to_all_reuse",
        "analysis_mode": "one-to-all",
        "workflow_stage": workflow_summary.get(
            "workflow_stage",
            "query_motif_hit_cache_precompute" if cache_only else "full_curve_replay",
        ),
        "status": workflow_summary.get("validation_status"),
        "recommended_strategy": FULL_CURVE_REUSE_STRATEGY,
        "combined_reuse_components": FULL_CURVE_REUSE_COMPONENTS,
        "precompute_grains": FULL_CURVE_PRECOMPUTE_GRAINS,
        "active_precompute_grain": workflow_summary.get("precompute_grain"),
        "subset_aggregation": workflow_summary.get(
            "subset_aggregation",
            "many_subset_dense_full_curve",
        ),
        "output_contract": workflow_summary.get(
            "output_contract",
            FULL_CURVE_OUTPUT_CONTRACT,
        ),
        "biological_output_contract": workflow_summary.get(
            "biological_output_contract",
            "shape_preserving_full_curve",
        ),
        "final_biological_output_ready": bool(
            workflow_summary.get("final_biological_output_ready", False)
        ),
        "query_motif_hit_cache_only": cache_only,
        "full_curves_required": True,
        "screening_mode": workflow_summary.get("screening_mode", "none"),
        "scalar_summary_policy": workflow_summary.get(
            "scalar_summary_policy",
            SCALAR_SUMMARY_POLICY,
        ),
        "scalar_screening_retired": workflow_summary.get(
            "scalar_screening_retired",
            True,
        ),
        "genome_zarr_path": workflow_summary.get("genome_zarr_path"),
        "motif_source": workflow_summary.get("motif_source"),
        "aligned_motif_path": workflow_summary.get("aligned_motif_path"),
        "loaded_with_aligned": workflow_summary.get("loaded_with_aligned"),
        "motif_count": motif_count,
        "query_motif_index": workflow_summary.get("query_motif_index"),
        "query_motif_name": workflow_summary.get("query_motif_name"),
        "window": workflow_summary.get("window_size"),
        "n_subsets": run.get("n_subsets", plan_body.get("n_subsets")),
        "n_region_memberships": n_region_memberships,
        "n_unique_regions": n_unique_regions,
        "reuse_factor": run.get("reuse_factor", plan_body.get("reuse_factor")),
        "reuse_percent": reuse_percent,
        "workflow_wall_s": workflow_summary.get("workflow_wall_s"),
        "output_bytes": None if cache_only else planned_output_bytes,
        "planned_full_curve_output_bytes": planned_output_bytes,
        "input_contract": _eureka_input_contract(args),
        "commands": {"run": _eureka_run_command(args)},
        "artifacts": {
            "manifest_json": str(manifest_json),
            "payload_json": str(payload_json),
            "workflow_summary_json": workflow_summary.get("workflow_json"),
            "run_summary_json": workflow_summary.get("run_json"),
            "resource_plan_json": workflow_summary.get("plan_json"),
            "output_zarr": workflow_summary.get("output_zarr"),
            "cache_manifest_json": workflow_summary.get("cache_manifest_json"),
            "cache_path": workflow_summary.get("cache_path"),
        },
    }


def _ensure_runtime_defaults(args: argparse.Namespace) -> None:
    """Fill parser defaults for tests or Python callers that build Namespace directly."""
    if not hasattr(args, "anchor_contribution_mode"):
        args.anchor_contribution_mode = DEFAULT_ANCHOR_CONTRIBUTION_MODE
    if not hasattr(args, "output_accumulator"):
        args.output_accumulator = DEFAULT_OUTPUT_ACCUMULATOR
    if not hasattr(args, "provider_max_sequence_cache_gb"):
        args.provider_max_sequence_cache_gb = DEFAULT_PROVIDER_MAX_SEQUENCE_CACHE_GB
    if not hasattr(args, "query_motif_hit_cache_only"):
        args.query_motif_hit_cache_only = False
    if not hasattr(args, "peak_set"):
        args.peak_set = None
    if not hasattr(args, "pairwise_set_diff"):
        args.pairwise_set_diff = False
    if not hasattr(args, "peak_set_window_bp"):
        args.peak_set_window_bp = 2000
    if not hasattr(args, "min_peaks"):
        args.min_peaks = 0


def resolve_output_accumulator(
    *,
    requested: str,
    effective_device: str,
    coalesce_membership_patterns: bool,
) -> tuple[str, dict]:
    """Resolve high-level accumulator policy to a dense replay implementation."""
    requested = str(requested or DEFAULT_OUTPUT_ACCUMULATOR)
    if requested not in {"numpy", "torch", "auto"}:
        raise ValueError(
            f"output_accumulator must be one of 'numpy', 'torch', or 'auto'; got {requested!r}."
        )
    if requested == "auto":
        effective = (
            "torch"
            if str(effective_device).startswith("cuda") and not bool(coalesce_membership_patterns)
            else "numpy"
        )
    else:
        effective = requested
    if bool(coalesce_membership_patterns) and effective == "torch":
        raise ValueError(
            "output_accumulator='torch' cannot be combined with "
            "coalesce_membership_patterns; use 'numpy' or 'auto'."
        )
    return effective, {
        "requested_output_accumulator": requested,
        "output_accumulator": effective,
        "effective_output_accumulator": effective,
        "output_accumulator_auto_selected": bool(requested == "auto"),
    }


def _load_cache_manifest(path: Path) -> dict:
    manifest = _load_json(path)
    if not manifest:
        raise FileNotFoundError(f"Target-anchor cache manifest does not exist: {path}")
    schema = manifest.get("schema_version")
    if schema != QUERY_MOTIF_HIT_CACHE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Expected query-motif-hit cache manifest schema "
            f"{QUERY_MOTIF_HIT_CACHE_MANIFEST_SCHEMA_VERSION!r}; got {schema!r}."
        )
    cache_path = manifest.get("cache_path")
    if not cache_path:
        raise ValueError(f"Target-anchor cache manifest has no cache_path: {path}")
    return manifest


def _threshold_mode_from_args(args: argparse.Namespace) -> str:
    if args.score_threshold is not None:
        return "score_threshold"
    if args.no_pvalue_mapping:
        return "annotation"
    return "pvalue_mapping"


def _compatible_value(left, right) -> bool:
    if isinstance(left, Path):
        left = str(left)
    if isinstance(right, Path):
        right = str(right)
    if isinstance(left, tuple):
        left = list(left)
    if isinstance(right, tuple):
        right = list(right)
    return left == right


def _validate_reuse_cache_manifest(
    manifest: dict,
    *,
    args: argparse.Namespace,
    motif_metadata: dict,
    query_motif_index: int,
) -> dict:
    """Validate portable query-motif-hit cache provenance before replay."""
    expected = {
        "genome_zarr_path": str(args.genome_zarr),
        "motif_count": motif_metadata.get("motif_count"),
        "motif_names_checksum": motif_metadata.get("motif_names_checksum"),
        "motif_kernel_checksum": motif_metadata.get("motif_kernel_checksum"),
        "motif_kernel_shape": list(motif_metadata.get("motif_kernel_shape") or []),
        "motif_source": "aligned_pt",
        "aligned_motif_path": str(args.aligned_motif_path),
        "loaded_with_aligned": True,
        "threshold_mode": _threshold_mode_from_args(args),
        "p_value_threshold": args.p_value_threshold,
        "score_threshold": args.score_threshold,
        "window_size": int(args.window),
        "query_motif_index": int(query_motif_index),
        "strand_specific": bool(args.strand_specific),
        "strands": 1 if args.strand_specific else 2,
    }
    mismatches: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for key, expected_value in expected.items():
        observed = manifest.get(key)
        if observed is None:
            if expected_value is None and key in manifest:
                continue
            missing.append(key)
            continue
        if not _compatible_value(observed, expected_value):
            mismatches[key] = {
                "manifest": observed,
                "requested": expected_value,
            }
    result = {
        "schema_version": "query_motif_hit_cache_manifest_compatibility_v1",
        "compatible": not mismatches,
        "mismatches": mismatches,
        "missing_fields": missing,
        "missing_field_count": len(missing),
        "checked_fields": sorted(expected),
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{key}: manifest={value['manifest']!r} requested={value['requested']!r}"
            for key, value in sorted(mismatches.items())
        )
        raise ValueError(
            "Target-anchor cache manifest is incompatible with this full-curve "
            f"one-to-all run: {mismatch_text}"
        )
    return result


def _query_motif_hit_cache_source_metadata(
    *,
    args: argparse.Namespace,
    run_summary: dict,
    manifest_payload: dict | None,
) -> dict:
    manifest_payload = manifest_payload or {}
    build_mode = (
        run_summary.get("query_motif_hit_cache_build_mode")
        or manifest_payload.get("query_motif_hit_cache_build_mode")
        or (
            "tile_assisted_exact_interval_reconstruction"
            if getattr(args, "query_motif_hit_cache_tile_size", None) is not None
            else "exact_unique_interval"
        )
    )
    tile_size = (
        run_summary.get("query_motif_hit_cache_tile_size_bp")
        or manifest_payload.get("query_motif_hit_cache_tile_size_bp")
        or getattr(args, "query_motif_hit_cache_tile_size", None)
    )
    tile_extension = (
        run_summary.get("query_motif_hit_cache_tile_extension_bp")
        or manifest_payload.get("query_motif_hit_cache_tile_extension_bp")
        or getattr(args, "query_motif_hit_cache_tile_extension_bp", None)
    )
    requested_tile_extension = run_summary.get(
        "query_motif_hit_cache_requested_tile_extension_bp"
    ) or manifest_payload.get("query_motif_hit_cache_requested_tile_extension_bp")
    minimum_tile_extension = run_summary.get(
        "query_motif_hit_cache_minimum_tile_extension_bp"
    ) or manifest_payload.get("query_motif_hit_cache_minimum_tile_extension_bp")
    precompute_grain = (
        "tile_assisted_exact_interval_query_motif_hit"
        if "tile_assisted" in str(build_mode) or tile_size is not None
        else "exact_unique_region_query_motif_hit"
    )
    return {
        "precompute_grain": precompute_grain,
        "query_motif_hit_cache_build_mode": build_mode,
        "query_motif_hit_cache_tile_size_bp": tile_size,
        "query_motif_hit_cache_tile_extension_bp": tile_extension,
        "query_motif_hit_cache_requested_tile_extension_bp": requested_tile_extension,
        "query_motif_hit_cache_minimum_tile_extension_bp": minimum_tile_extension,
    }


def _write_cache_manifest(
    path: Path,
    *,
    cache_path: str | Path,
    expected_metadata: dict | None,
    run_summary: dict,
    workflow_summary: dict | None = None,
    source_manifest: str | None = None,
) -> dict:
    expected_metadata = expected_metadata or {}

    def _run_or_expected(key: str):
        return run_summary.get(key, expected_metadata.get(key))

    manifest = {
        "schema_version": QUERY_MOTIF_HIT_CACHE_MANIFEST_SCHEMA_VERSION,
        "cache_path": str(cache_path),
        "expected_metadata": expected_metadata,
        "source_manifest": source_manifest,
        "run_summary_path": str(run_summary.get("summary_path", "")),
        "workflow_schema_version": (
            workflow_summary.get("schema_version") if workflow_summary else None
        ),
        "genome_zarr_path": _run_or_expected("genome_zarr_path"),
        "motif_count": _run_or_expected("motif_count"),
        "motif_names_checksum": _run_or_expected("motif_names_checksum"),
        "motif_kernel_checksum": _run_or_expected("motif_kernel_checksum"),
        "motif_kernel_shape": _run_or_expected("motif_kernel_shape"),
        "motif_source": _run_or_expected("motif_source"),
        "aligned_motif_path": _run_or_expected("aligned_motif_path"),
        "loaded_with_aligned": _run_or_expected("loaded_with_aligned"),
        "threshold_mode": _run_or_expected("threshold_mode"),
        "p_value_threshold": _run_or_expected("p_value_threshold"),
        "score_threshold": _run_or_expected("score_threshold"),
        "score_threshold_vector_checksum": run_summary.get("score_threshold_vector_checksum"),
        "score_threshold_vector_shape": run_summary.get("score_threshold_vector_shape"),
        "window_size": _run_or_expected("window_size"),
        "query_motif_index": _run_or_expected("query_motif_index"),
        "strand_specific": _run_or_expected("strand_specific"),
        "strands": _run_or_expected("strands"),
        "values_checksum": run_summary.get("values_checksum"),
        "values_checksum_status": run_summary.get("values_checksum_status"),
        "query_motif_hit_cache_anchor_hits": run_summary.get(
            "query_motif_hit_cache_anchor_hits",
            run_summary.get("query_motif_hit_hits"),
        ),
        "query_motif_hit_hit_regions": run_summary.get("query_motif_hit_hit_regions"),
        "query_motif_hit_cache_checksum": run_summary.get("query_motif_hit_cache_checksum"),
        "query_motif_hit_cache_build_mode": run_summary.get(
            "query_motif_hit_cache_build_mode",
            run_summary.get(
                "query_motif_hit_cache_query_motif_hit_cache_build_mode",
                workflow_summary.get("query_motif_hit_cache_build_mode")
                if workflow_summary
                else None,
            ),
        ),
        "query_motif_hit_cache_tile_size_bp": run_summary.get(
            "query_motif_hit_cache_tile_size_bp",
            workflow_summary.get("query_motif_hit_cache_tile_size_bp")
            if workflow_summary
            else None,
        ),
        "query_motif_hit_cache_tile_extension_bp": run_summary.get(
            "query_motif_hit_cache_tile_extension_bp",
            workflow_summary.get("query_motif_hit_cache_tile_extension_bp")
            if workflow_summary
            else None,
        ),
        "query_motif_hit_cache_requested_tile_extension_bp": run_summary.get(
            "query_motif_hit_cache_requested_tile_extension_bp",
            workflow_summary.get("query_motif_hit_cache_requested_tile_extension_bp")
            if workflow_summary
            else None,
        ),
        "query_motif_hit_cache_minimum_tile_extension_bp": run_summary.get(
            "query_motif_hit_cache_minimum_tile_extension_bp",
            workflow_summary.get("query_motif_hit_cache_minimum_tile_extension_bp")
            if workflow_summary
            else None,
        ),
        "query_motif_hit_cache_total_intervals": run_summary.get(
            "query_motif_hit_cache_total_intervals"
        ),
        "query_motif_hit_cache_required_intervals": run_summary.get(
            "query_motif_hit_cache_required_intervals"
        ),
        "query_motif_hit_cache_missing_intervals": run_summary.get(
            "query_motif_hit_cache_missing_intervals"
        ),
        "query_motif_hit_cache_coverage_fraction": run_summary.get(
            "query_motif_hit_cache_coverage_fraction"
        ),
    }
    _write_json(path, manifest)
    return manifest


def resolve_query_motif_index(
    *,
    motif_selection: str | None,
    query_motif: str | None,
    query_motif_index: int | None,
    aligned_motif_path: str,
) -> tuple[int, dict]:
    """Resolve the one-to-all target against the aligned motif tensor."""
    hocomoco_db, motif_metadata = load_hocomoco_motifs(
        motif_selection=motif_selection,
        aligned_motif_path=aligned_motif_path,
        use_aligned_motifs=True,
        require_aligned_motifs=True,
        threshold_mode="pvalue_mapping",
    )
    motif_names = list(hocomoco_db.motif_names)
    if query_motif_index is not None:
        index = int(query_motif_index)
        if index < 0 or index >= len(motif_names):
            raise ValueError(f"Query motif index {index} is outside [0, {len(motif_names) - 1}].")
        return index, {
            **motif_metadata.to_dict(),
            "resolved_query_motif_index": index,
            "resolved_query_motif_name": motif_names[index],
        }
    if not query_motif:
        raise ValueError("Either --query-motif or --query-motif-index is required.")
    matches = [idx for idx, name in enumerate(motif_names) if name == query_motif]
    if not matches:
        matches = [
            idx
            for idx, name in enumerate(motif_names)
            if name.startswith(query_motif + ".") or name.startswith(query_motif + "_")
        ]
    if not matches:
        matches = [
            idx for idx, name in enumerate(motif_names) if query_motif.upper() in name.upper()
        ]
    if len(matches) != 1:
        preview = [motif_names[idx] for idx in matches[:10]]
        raise ValueError(
            f"Query motif {query_motif!r} resolved to {len(matches)} motifs; "
            f"use --query-motif-index. Matches: {preview}"
        )
    index = int(matches[0])
    return index, {
        **motif_metadata.to_dict(),
        "resolved_query_motif_index": index,
        "resolved_query_motif_name": motif_names[index],
    }


def _append_common_dense_args(command: list[str], args: argparse.Namespace) -> None:
    _ensure_runtime_defaults(args)
    command.extend(
        [
            "--genome-zarr",
            str(args.genome_zarr),
            "--regions-tsv",
            str(args.regions_tsv),
            "--subset-column",
            str(args.subset_column),
            "--window",
            str(args.window),
            "--query-motif-index",
            str(args.resolved_query_motif_index),
            "--aligned-motif-path",
            str(args.aligned_motif_path),
            "--p-value-threshold",
            str(args.p_value_threshold),
            "--provider",
            str(args.provider),
            "--dtype",
            str(args.dtype),
            "--device",
            str(args.device),
            "--chunk-subsets",
            str(args.chunk_subsets),
            "--checksum-mode",
            str(args.checksum_mode),
            "--anchor-contribution-mode",
            str(args.anchor_contribution_mode),
            "--output-accumulator",
            str(args.output_accumulator),
        ]
    )
    if args.motifs:
        command.extend(["--motifs", str(args.motifs)])
    if args.score_threshold is not None:
        command.extend(["--score-threshold", str(args.score_threshold)])
    if args.no_pvalue_mapping:
        command.append("--no-pvalue-mapping")
    if args.strand_specific:
        command.append("--strand-specific")
    if args.provider == "block":
        command.extend(
            [
                "--provider-max-gap-bp",
                str(args.provider_max_gap_bp),
                "--provider-max-block-span-bp",
                str(args.provider_max_block_span_bp),
                "--provider-max-cached-blocks",
                str(args.provider_max_cached_blocks),
                "--provider-max-sequence-cache-gb",
                str(args.provider_max_sequence_cache_gb),
            ]
        )
        if args.provider_max_block_score_gb is not None:
            command.extend(
                [
                    "--provider-max-block-score-gb",
                    str(args.provider_max_block_score_gb),
                ]
            )
    if args.max_output_gb is not None:
        command.extend(["--max-output-gb", str(args.max_output_gb)])
    if args.max_membership_pattern_cache_gb is not None:
        command.extend(
            [
                "--max-membership-pattern-cache-gb",
                str(args.max_membership_pattern_cache_gb),
            ]
        )
    if getattr(args, "query_motif_hit_cache_tile_size", None) is not None:
        command.extend(
            [
                "--query-motif-hit-cache-tile-size",
                str(args.query_motif_hit_cache_tile_size),
            ]
        )
    if getattr(args, "query_motif_hit_cache_tile_extension_bp", None) is not None:
        command.extend(
            [
                "--query-motif-hit-cache-tile-extension-bp",
                str(args.query_motif_hit_cache_tile_extension_bp),
            ]
        )


def dense_subset_command(args: argparse.Namespace, *extra: str) -> list[str]:
    """Build the delegated dense-subset command for auditability."""
    _ensure_runtime_defaults(args)
    if args.dense_subset_script is None:
        command = [
            sys.executable,
            "-m",
            "motiverse.dense_subset_workflow",
        ]
    else:
        command = [sys.executable, str(args.dense_subset_script)]
    _append_common_dense_args(command, args)
    command.extend(extra)
    return command


def _run_command(command: Sequence[str], *, runner) -> None:
    runner(list(command), check=True)


def _guard_paths(args: argparse.Namespace) -> None:
    genome_zarr = Path(args.genome_zarr)
    aligned_motif_path = Path(args.aligned_motif_path)
    if not genome_zarr.exists():
        raise FileNotFoundError(f"Genome zarr does not exist: {genome_zarr}")
    if not aligned_motif_path.exists():
        raise FileNotFoundError(f"Aligned motif PT does not exist: {aligned_motif_path}")
    if args.cache_path.exists() and not (args.reuse_existing_cache or args.overwrite_cache):
        raise FileExistsError(
            f"Target-anchor cache already exists: {args.cache_path}. "
            "Use --reuse-existing-cache or --overwrite-cache."
        )
    if (
        not getattr(args, "query_motif_hit_cache_only", False)
        and args.output_zarr.exists()
        and not args.overwrite_output
    ):
        raise FileExistsError(
            f"Output zarr already exists: {args.output_zarr}. "
            "Use --overwrite-output for an intentional replacement."
        )


def _dtype_pair(dtype_name: str) -> tuple[np.dtype, torch.dtype]:
    dtype = np.dtype(dtype_name)
    torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
    return dtype, torch_dtype


def _provider_from_args(
    args: argparse.Namespace,
    motif_kernels: torch.Tensor,
    regions: pd.DataFrame,
    *,
    torch_dtype: torch.dtype,
):
    _ensure_runtime_defaults(args)
    provider_max_block_score_bytes = None
    if args.provider_max_block_score_gb is not None:
        provider_max_block_score_bytes = int(args.provider_max_block_score_gb * (1024**3))
    provider_max_sequence_cache_bytes = int(args.provider_max_sequence_cache_gb * (1024**3))
    if args.provider == "block":
        max_cached_blocks = (
            None if args.provider_max_cached_blocks < 0 else args.provider_max_cached_blocks
        )
        return DenseGenomeBlockScoreProvider(
            args.genome_zarr,
            motif_kernels,
            regions,
            subset_column=args.subset_column,
            device=args.device,
            dtype=torch_dtype,
            strand_specific=args.strand_specific,
            max_gap_bp=args.provider_max_gap_bp,
            max_block_span_bp=args.provider_max_block_span_bp,
            max_block_score_bytes=provider_max_block_score_bytes,
            max_cached_blocks=max_cached_blocks,
            max_sequence_cache_bytes=provider_max_sequence_cache_bytes,
        )
    return DenseGenomeScoreProvider(
        args.genome_zarr,
        motif_kernels,
        device=args.device,
        dtype=torch_dtype,
        strand_specific=args.strand_specific,
    )


def _write_plan_summary(
    args: argparse.Namespace,
    regions: pd.DataFrame,
    *,
    motif_metadata,
    motif_count: int,
    motif_kernel_shape: list[int],
    threshold_metadata: dict,
    dtype: np.dtype,
) -> dict:
    provider_max_block_score_bytes = None
    if args.provider_max_block_score_gb is not None:
        provider_max_block_score_bytes = int(args.provider_max_block_score_gb * (1024**3))
    storage_threshold_bytes = int(args.storage_threshold_gb * (1024**3))
    resource_plan = plan_dense_score_region_subset_resources(
        regions,
        subset_column=args.subset_column,
        n_motifs=motif_count,
        window_size=args.window,
        query_motif_index=args.resolved_query_motif_index,
        motif_length=int(motif_kernel_shape[1]),
        strands=1 if args.strand_specific else 2,
        dtype=dtype,
        provider_max_gap_bp=args.provider_max_gap_bp,
        provider_max_block_span_bp=args.provider_max_block_span_bp,
        provider_max_block_score_bytes=provider_max_block_score_bytes,
        output_path=args.output_zarr,
        storage_threshold_bytes=storage_threshold_bytes,
        mnt_storage_base=args.mnt_storage_base,
    )
    contribution_cache_plan = plan_dense_interval_contribution_cache_resources(
        regions,
        subset_column=args.subset_column,
        n_motifs=motif_count,
        window_size=args.window,
        query_motif_index=args.resolved_query_motif_index,
        dtype=dtype,
    )
    summary = {
        **resource_plan.to_dict(),
        "mode": "dense-exact-subset-resource-plan",
        "provider": args.provider,
        "genome_zarr_path": str(args.genome_zarr),
        "motif_count": motif_count,
        "motif_names_checksum": motif_metadata.motif_names_checksum,
        "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
        "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
        "motif_source": motif_metadata.motif_source,
        "aligned_motif_path": motif_metadata.aligned_motif_path,
        "loaded_with_aligned": motif_metadata.loaded_with_aligned,
        **getattr(args, "device_metadata", {}),
        **threshold_metadata,
        "strand_specific": bool(args.strand_specific),
        "strands": 1 if args.strand_specific else 2,
        "contribution_cache_plan": contribution_cache_plan.to_dict(),
        "contribution_cache_bytes": contribution_cache_plan.contribution_cache_bytes,
        "contribution_cache_bytes_per_unique_region": (
            contribution_cache_plan.cache_bytes_per_unique_region
        ),
        "contribution_cache_feasible_under_max": (contribution_cache_plan.cache_feasible_under_max),
        "contribution_cache_recommended_strategy": (contribution_cache_plan.recommended_strategy),
    }
    _write_json(args.plan_json, summary)
    return summary


def _run_query_motif_hit_reuse_in_process(args: argparse.Namespace) -> tuple[dict, dict]:
    _ensure_runtime_defaults(args)
    dtype, torch_dtype = _dtype_pair(args.dtype)
    regions = pd.read_csv(args.regions_tsv, sep="\t")
    hocomoco_db, motif_metadata = load_hocomoco_motifs(
        motif_selection=args.motifs,
        aligned_motif_path=args.aligned_motif_path,
        use_aligned_motifs=True,
        require_aligned_motifs=True,
        threshold_mode="score_threshold"
        if args.score_threshold is not None
        else "annotation"
        if args.no_pvalue_mapping
        else "pvalue_mapping",
    )
    motif_names = list(hocomoco_db.motif_names)
    motif_kernels = torch.as_tensor(
        np.asarray(hocomoco_db.motif_kernels, dtype=np.float32),
        dtype=torch_dtype,
        device=args.device,
    )
    thresholds, threshold_metadata = dense_subset_workflow._thresholds_from_args(
        hocomoco_db,
        score_threshold=args.score_threshold,
        p_value_threshold=args.p_value_threshold,
        use_pvalue_mapping=not args.no_pvalue_mapping,
        n_motifs=len(motif_names),
        dtype=torch_dtype,
        device=args.device,
    )
    plan_summary = _write_plan_summary(
        args,
        regions,
        motif_metadata=motif_metadata,
        motif_count=len(motif_names),
        motif_kernel_shape=list(motif_kernels.shape),
        threshold_metadata=threshold_metadata,
        dtype=dtype,
    )
    if args.plan_only:
        return plan_summary, {}

    max_output_bytes = None
    if args.max_output_gb is not None:
        max_output_bytes = int(args.max_output_gb * (1024**3))
    max_pattern_bytes = None
    if args.max_membership_pattern_cache_gb is not None:
        max_pattern_bytes = int(args.max_membership_pattern_cache_gb * (1024**3))

    provider = _provider_from_args(
        args,
        motif_kernels,
        regions,
        torch_dtype=torch_dtype,
    )
    query_idx = int(args.resolved_query_motif_index)
    query_provider = _provider_from_args(
        args,
        motif_kernels[query_idx : query_idx + 1],
        regions,
        torch_dtype=torch_dtype,
    )
    query_thresholds = {key: value[query_idx : query_idx + 1] for key, value in thresholds.items()}
    metadata_extra = {
        "mode": "dense-exact-subset-query-motif-hit-cache-replay",
        "provider": args.provider,
        "genome_zarr_path": str(args.genome_zarr),
        "motif_count": len(motif_names),
        "motif_names_checksum": motif_metadata.motif_names_checksum,
        "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
        "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
        "motif_source": motif_metadata.motif_source,
        "aligned_motif_path": motif_metadata.aligned_motif_path,
        "loaded_with_aligned": motif_metadata.loaded_with_aligned,
        **getattr(args, "device_metadata", {}),
        **threshold_metadata,
        "strand_specific": bool(args.strand_specific),
        "strands": 1 if args.strand_specific else 2,
    }
    expected_cache_metadata = {
        **metadata_extra,
        "window_size": int(args.window),
        "query_motif_index": query_idx,
        "dtype": str(dtype),
    }
    replay_metadata_extra = {
        **metadata_extra,
        "anchor_contribution_mode": args.anchor_contribution_mode,
        "anchor_stack_all_float_order_jitter": bool(args.anchor_contribution_mode == "stack_all"),
        "requested_output_accumulator": getattr(
            args,
            "requested_output_accumulator",
            args.output_accumulator,
        ),
        "output_accumulator": args.output_accumulator,
        "effective_output_accumulator": args.output_accumulator,
        "output_accumulator_auto_selected": bool(
            getattr(args, "requested_output_accumulator", args.output_accumulator) == "auto"
        ),
        "output_accumulator_float_order_jitter": bool(args.output_accumulator == "torch"),
    }
    query_motif_hit_cache_build_summary: dict = {}
    query_motif_hit_cache_path = str(args.cache_path)
    if not args.reuse_existing_cache:
        query_hit_cache_start = time.perf_counter()
        query_hit_cache_metadata = {
            **metadata_extra,
            "mode": "dense-exact-query-motif-hit-cache-build",
        }
        if getattr(args, "query_motif_hit_cache_tile_size", None) is None:
            query_hit_cache_result = build_dense_query_motif_hit_cache(
                query_provider,
                regions,
                output_path=args.cache_path,
                subset_column=args.subset_column,
                score_thresholds=query_thresholds,
                window_size=args.window,
                n_motifs=len(motif_names),
                query_motif_index=query_idx,
                p_value_threshold=args.p_value_threshold,
                dtype=dtype,
                torch_dtype=torch_dtype,
                device=args.device,
                metadata_extra=query_hit_cache_metadata,
            )
        else:
            query_hit_cache_result = build_dense_query_motif_hit_cache_from_tiles(
                query_provider,
                regions,
                output_path=args.cache_path,
                subset_column=args.subset_column,
                score_thresholds=query_thresholds,
                window_size=args.window,
                n_motifs=len(motif_names),
                query_motif_index=query_idx,
                tile_size=int(args.query_motif_hit_cache_tile_size),
                tile_extension_bp=getattr(
                    args,
                    "query_motif_hit_cache_tile_extension_bp",
                    None,
                ),
                p_value_threshold=args.p_value_threshold,
                dtype=dtype,
                torch_dtype=torch_dtype,
                device=args.device,
                metadata_extra=query_hit_cache_metadata,
            )
        query_motif_hit_cache_build_summary = {
            "query_motif_hit_cache_schema_version": (DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION),
            "query_motif_hit_cache_path": query_hit_cache_result.path,
            "query_motif_hit_cache_build_wall_s": (time.perf_counter() - query_hit_cache_start),
            "query_motif_hit_cache_unique_regions": (query_hit_cache_result.n_unique_regions),
            "query_motif_hit_cache_anchor_hits": query_hit_cache_result.n_anchor_hits,
            "query_motif_hit_cache_checksum": query_hit_cache_result.checksum,
            "query_motif_hit_cache_build_mode": (
                "tile_assisted_exact_interval_reconstruction"
                if getattr(args, "query_motif_hit_cache_tile_size", None) is not None
                else "exact_unique_interval"
            ),
            "query_motif_hit_cache_tile_size_bp": getattr(
                args,
                "query_motif_hit_cache_tile_size",
                None,
            ),
            "query_motif_hit_cache_tile_extension_bp": getattr(
                args,
                "query_motif_hit_cache_tile_extension_bp",
                None,
            ),
            **{
                f"query_motif_hit_cache_{key}": value
                for key, value in query_hit_cache_result.timings.items()
            },
        }
        query_motif_hit_cache_path = query_hit_cache_result.path
    if args.query_motif_hit_cache_only:
        provider_stats = getattr(query_provider, "stats", {})
        run_summary = {
            "schema_version": DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION,
            "semantics": DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS,
            "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
            "mode": "dense-exact-query-motif-hit-cache-build",
            "provider": args.provider,
            "execution_backend": "in_process",
            "validation_status": "PASS",
            "genome_zarr_path": str(args.genome_zarr),
            "motif_count": len(motif_names),
            "motif_names_checksum": motif_metadata.motif_names_checksum,
            "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
            "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
            "motif_source": motif_metadata.motif_source,
            "aligned_motif_path": motif_metadata.aligned_motif_path,
            "loaded_with_aligned": motif_metadata.loaded_with_aligned,
            **threshold_metadata,
            "n_subsets": int(plan_summary.get("plan", {}).get("n_subsets", 0)),
            "n_regions": int(plan_summary.get("plan", {}).get("n_region_memberships", 0)),
            "n_unique_regions": int(plan_summary.get("plan", {}).get("n_unique_regions", 0)),
            "reuse_factor": plan_summary.get("plan", {}).get("reuse_factor"),
            "window_size": int(args.window),
            "query_motif_index": query_idx,
            "strand_specific": bool(args.strand_specific),
            "strands": 1 if args.strand_specific else 2,
            "expected_query_motif_hit_cache_metadata": expected_cache_metadata,
            **query_motif_hit_cache_build_summary,
            **query_hit_cache_result.timings,
            "query_motif_hit_cache_path": query_motif_hit_cache_path,
            "query_motif_hit_hits": query_hit_cache_result.n_anchor_hits,
            "query_motif_hit_hit_regions": query_hit_cache_result.timings.get(
                "query_motif_hit_hit_regions"
            ),
            "bp_scanned": int(provider_stats.get("provider_bp_loaded", 0)),
            **{f"query_motif_hit_provider_{key}": value for key, value in provider_stats.items()},
        }
        _write_json(args.run_json, run_summary)
        return plan_summary, run_summary

    aggregate_start = time.perf_counter()
    result = accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
        query_motif_hit_cache_path,
        provider,
        regions,
        subset_column=args.subset_column,
        dtype=dtype,
        torch_dtype=torch_dtype,
        device=args.device,
        max_output_bytes=max_output_bytes,
        chunk_subsets=args.chunk_subsets,
        coalesce_membership_patterns=args.coalesce_membership_patterns,
        max_pattern_accumulator_bytes=max_pattern_bytes,
        expected_metadata=expected_cache_metadata,
        anchor_contribution_mode=args.anchor_contribution_mode,
        output_accumulator=args.output_accumulator,
    )
    aggregate_wall_s = time.perf_counter() - aggregate_start
    values_checksum = dense_subset_workflow._checksum(result.values)
    dense_subset_workflow._write_dense_result_to_zarr(
        result=result,
        output_path=args.output_zarr,
        chunk_subsets=args.chunk_subsets,
        metadata={
            **replay_metadata_extra,
            "schema_version": DENSE_SUBSET_ZARR_SCHEMA_VERSION,
            "complete": True,
            "semantics": DENSE_SUBSET_SEMANTICS,
            "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
            "subset_update_mode": (
                "query_motif_hit_cache_replay_membership_pattern_coalesced"
                if args.coalesce_membership_patterns
                else "query_motif_hit_cache_replay_in_memory_output"
            ),
            "anchor_contribution_mode": args.anchor_contribution_mode,
            "anchor_stack_all_float_order_jitter": bool(
                args.anchor_contribution_mode == "stack_all"
            ),
            "requested_output_accumulator": getattr(
                args,
                "requested_output_accumulator",
                args.output_accumulator,
            ),
            "output_accumulator": args.output_accumulator,
            "effective_output_accumulator": args.output_accumulator,
            "output_accumulator_auto_selected": bool(
                getattr(args, "requested_output_accumulator", args.output_accumulator) == "auto"
            ),
            "output_accumulator_float_order_jitter": bool(args.output_accumulator == "torch"),
            "values_checksum": values_checksum,
            "values_checksum_mode": "full",
            "values_checksum_status": "complete",
            "values_checksum_sample_subsets": 0,
            **query_motif_hit_cache_build_summary,
            **result.timings,
        },
    )
    provider_stats = getattr(provider, "stats", {})
    query_provider_stats = getattr(query_provider, "stats", {})
    run_summary = {
        "schema_version": DENSE_SUBSET_ZARR_SCHEMA_VERSION,
        "semantics": DENSE_SUBSET_SEMANTICS,
        "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
        "mode": "dense-exact-subset-query-motif-hit-cache-replay",
        "provider": args.provider,
        "execution_backend": "in_process",
        "genome_zarr_path": str(args.genome_zarr),
        "motif_count": result.plan.n_motifs,
        "motif_names_checksum": motif_metadata.motif_names_checksum,
        "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
        "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
        "motif_source": motif_metadata.motif_source,
        "aligned_motif_path": motif_metadata.aligned_motif_path,
        "loaded_with_aligned": motif_metadata.loaded_with_aligned,
        **threshold_metadata,
        "n_subsets": result.plan.n_subsets,
        "n_regions": result.plan.n_region_memberships,
        "n_unique_regions": result.plan.n_unique_regions,
        "n_dropped_invalid_regions": result.plan.n_dropped_invalid_regions,
        "reuse_factor": result.plan.reuse_factor,
        "window_size": result.plan.window_size,
        "query_motif_index": result.plan.query_motif_index,
        "strand_specific": bool(args.strand_specific),
        "strands": 1 if args.strand_specific else 2,
        "anchor_contribution_mode": args.anchor_contribution_mode,
        "anchor_stack_all_float_order_jitter": bool(args.anchor_contribution_mode == "stack_all"),
        "requested_output_accumulator": getattr(
            args,
            "requested_output_accumulator",
            args.output_accumulator,
        ),
        "output_accumulator": args.output_accumulator,
        "effective_output_accumulator": args.output_accumulator,
        "output_accumulator_auto_selected": bool(
            getattr(args, "requested_output_accumulator", args.output_accumulator) == "auto"
        ),
        "output_accumulator_float_order_jitter": bool(args.output_accumulator == "torch"),
        "output_bytes": result.plan.output_bytes,
        "values_shape": list(result.values.shape),
        "values_checksum": values_checksum,
        "values_checksum_mode": "full",
        "values_checksum_status": "complete",
        "values_checksum_sample_subsets": 0,
        "expected_query_motif_hit_cache_metadata": expected_cache_metadata,
        "subset_update_mode": (
            "query_motif_hit_cache_replay_membership_pattern_coalesced"
            if args.coalesce_membership_patterns
            else "query_motif_hit_cache_replay_in_memory_output"
        ),
        **query_motif_hit_cache_build_summary,
        "query_motif_hit_cache_path": query_motif_hit_cache_path,
        "aggregate_wall_s": aggregate_wall_s,
        "bp_scanned": int(provider_stats.get("provider_bp_loaded", 0))
        + int(query_provider_stats.get("provider_bp_loaded", 0)),
        "full_provider_bp_loaded": int(provider_stats.get("provider_bp_loaded", 0)),
        "query_motif_hit_provider_bp_loaded": int(
            query_provider_stats.get("provider_bp_loaded", 0)
        ),
        "query_motif_hit_prefilter": False,
        **result.timings,
        **provider_stats,
        **{f"query_motif_hit_provider_{key}": value for key, value in query_provider_stats.items()},
    }
    _write_json(args.run_json, run_summary)
    dense_subset_workflow._update_zarr_metadata(args.output_zarr, run_summary)
    return plan_summary, run_summary


def run_full_curve_one_to_all_reuse(
    args: argparse.Namespace,
    *,
    runner=subprocess.run,
) -> dict:
    """Run the recommended full-curve query-motif-hit reuse workflow.

    The strict default does not coalesce subset membership patterns because that
    can change float32 addition order. Coalescing is still exposed as an opt-in
    speed mode and marked PASS_WARN in the workflow summary.
    """
    _ensure_runtime_defaults(args)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peak_set_inventory = None
    if args.peak_set:
        if args.regions_tsv:
            raise ValueError("Use either --regions-tsv or --peak-set, not both.")
        regions, inventory, union_loci = build_peak_set_memberships(
            list(args.peak_set),
            window_bp=int(args.peak_set_window_bp),
            pairwise_set_diff=bool(args.pairwise_set_diff),
            min_peaks=int(args.min_peaks),
            return_union_loci=True,
        )
        if regions.empty:
            raise ValueError("No peak-set subsets passed --min-peaks.")
        args.regions_tsv = args.output_dir / "peak_set_memberships.tsv"
        regions.to_csv(args.regions_tsv, sep="\t", index=False)
        peak_set_inventory = args.output_dir / "peak_set_inventory.tsv"
        inventory.to_csv(peak_set_inventory, sep="\t", index=False)
        union_loci.to_csv(args.output_dir / "peak_set_union_loci.tsv", sep="\t", index=False)
    elif not args.regions_tsv:
        raise ValueError("Provide --regions-tsv or one or more --peak-set NAME=PATH values.")
    elif args.pairwise_set_diff:
        raise ValueError("--pairwise-set-diff requires repeated --peak-set NAME=PATH inputs.")
    args.reuse_cache_manifest = (
        Path(args.reuse_cache_manifest) if getattr(args, "reuse_cache_manifest", None) else None
    )
    raw_cache_path = args.cache_path
    manifest_payload = None
    if args.reuse_cache_manifest is not None:
        manifest_payload = _load_cache_manifest(args.reuse_cache_manifest)
        manifest_cache_path = str(manifest_payload["cache_path"])
        if raw_cache_path is not None and str(raw_cache_path) != manifest_cache_path:
            raise ValueError(
                "--cache-path does not match --reuse-cache-manifest cache_path: "
                f"{raw_cache_path!r} != {manifest_cache_path!r}"
            )
        raw_cache_path = manifest_cache_path
        args.reuse_existing_cache = True
    args.cache_path = Path(raw_cache_path or args.output_dir / "query_motif_hit_cache.zarr")
    args.output_zarr = Path(args.output_zarr or args.output_dir / "full_curve_one_to_all.zarr")
    args.plan_json = Path(args.plan_json or args.output_dir / "resource_plan.json")
    args.run_json = Path(args.run_json or args.output_dir / "full_curve_one_to_all_summary.json")
    args.workflow_json = Path(args.workflow_json or args.output_dir / "workflow_summary.json")
    args.cache_manifest_json = Path(
        args.cache_manifest_json or args.output_dir / "query_motif_hit_cache_manifest.json"
    )
    args.dense_subset_script = Path(args.dense_subset_script) if args.dense_subset_script else None
    args.subprocess_backend = bool(getattr(args, "subprocess_backend", False))
    if (
        getattr(args, "query_motif_hit_cache_tile_extension_bp", None) is not None
        and getattr(args, "query_motif_hit_cache_tile_size", None) is None
    ):
        raise ValueError(
            "--query-motif-hit-cache-tile-extension-bp requires --query-motif-hit-cache-tile-size."
        )
    if getattr(args, "query_motif_hit_cache_only", False) and getattr(
        args,
        "plan_only",
        False,
    ):
        raise ValueError("--query-motif-hit-cache-only cannot be combined with --plan-only.")
    if getattr(args, "query_motif_hit_cache_only", False) and getattr(
        args,
        "reuse_existing_cache",
        False,
    ):
        raise ValueError(
            "--query-motif-hit-cache-only builds a reusable cache; do not combine "
            "it with --reuse-existing-cache or --reuse-cache-manifest."
        )
    args.device, args.device_metadata = dense_subset_workflow.resolve_analysis_device(
        getattr(args, "device", "auto")
    )
    requested_output_accumulator = getattr(
        args,
        "output_accumulator",
        DEFAULT_OUTPUT_ACCUMULATOR,
    )
    args.output_accumulator, args.output_accumulator_metadata = resolve_output_accumulator(
        requested=requested_output_accumulator,
        effective_device=args.device,
        coalesce_membership_patterns=bool(args.coalesce_membership_patterns),
    )
    args.requested_output_accumulator = requested_output_accumulator
    _guard_paths(args)

    target_index, motif_metadata = resolve_query_motif_index(
        motif_selection=args.motifs,
        query_motif=args.query_motif,
        query_motif_index=args.query_motif_index,
        aligned_motif_path=str(args.aligned_motif_path),
    )
    args.resolved_query_motif_index = target_index
    reuse_manifest_compatibility = None
    if manifest_payload is not None:
        reuse_manifest_compatibility = _validate_reuse_cache_manifest(
            manifest_payload,
            args=args,
            motif_metadata=motif_metadata,
            query_motif_index=target_index,
        )

    workflow_start = time.perf_counter()
    commands: list[list[str]] = []
    if not args.subprocess_backend:
        plan_summary, run_summary = _run_query_motif_hit_reuse_in_process(args)
    else:
        plan_command = dense_subset_command(
            args,
            "--plan-only",
            "--output-json",
            str(args.plan_json),
        )
        commands.append(plan_command)
        _run_command(plan_command, runner=runner)

        if args.plan_only:
            run_summary = {}
        else:
            if args.reuse_existing_cache:
                cache_args = ["--use-query-motif-hit-cache", str(args.cache_path)]
            else:
                cache_args = ["--build-query-motif-hit-cache", str(args.cache_path)]
            if args.query_motif_hit_cache_only:
                cache_args.append("--query-motif-hit-cache-only")
            if args.coalesce_membership_patterns:
                cache_args.append("--coalesce-membership-patterns")
            if args.query_motif_hit_cache_only:
                run_command = dense_subset_command(
                    args,
                    *cache_args,
                    "--output-json",
                    str(args.run_json),
                )
            else:
                run_command = dense_subset_command(
                    args,
                    *cache_args,
                    "--output-zarr",
                    str(args.output_zarr),
                    "--output-json",
                    str(args.run_json),
                )
            commands.append(run_command)
            _run_command(run_command, runner=runner)
            run_summary = _load_json(args.run_json)
        plan_summary = _load_json(args.plan_json)
    cache_source_metadata = _query_motif_hit_cache_source_metadata(
        args=args,
        run_summary=run_summary,
        manifest_payload=manifest_payload,
    )
    workflow_summary = {
        "schema_version": "full_curve_one_to_all_reuse_workflow_v1",
        "mode": (
            "full-curve-one-to-all-query-motif-hit-cache-build"
            if args.query_motif_hit_cache_only
            else "full-curve-one-to-all-query-motif-hit-reuse"
        ),
        "workflow_stage": (
            "query_motif_hit_cache_precompute"
            if args.query_motif_hit_cache_only
            else "full_curve_replay"
        ),
        "recommended_strategy": FULL_CURVE_REUSE_STRATEGY,
        "output_contract": (
            QUERY_MOTIF_HIT_CACHE_ONLY_OUTPUT_CONTRACT
            if args.query_motif_hit_cache_only
            else FULL_CURVE_OUTPUT_CONTRACT
        ),
        "biological_output_contract": (
            "precompute_only_not_final_biological_output"
            if args.query_motif_hit_cache_only
            else "shape_preserving_full_curve"
        ),
        "final_biological_output_ready": not bool(
            args.query_motif_hit_cache_only or args.plan_only
        ),
        "query_motif_hit_cache_only": bool(args.query_motif_hit_cache_only),
        "full_curves_required": True,
        "screening_mode": "none",
        "scalar_summary_policy": SCALAR_SUMMARY_POLICY,
        "scalar_screening_retired": True,
        "precompute_grain": cache_source_metadata["precompute_grain"],
        "subset_aggregation": (
            "deferred_many_subset_dense_full_curve"
            if args.query_motif_hit_cache_only
            else "many_subset_dense_full_curve"
        ),
        "combined_reuse_components": FULL_CURVE_REUSE_COMPONENTS,
        "source_precompute_components": [
            cache_source_metadata["precompute_grain"],
            "deferred_many_subset_dense_full_curve"
            if args.query_motif_hit_cache_only
            else "many_subset_dense_full_curve",
        ],
        "membership_pattern_coalescing": bool(args.coalesce_membership_patterns),
        "membership_pattern_coalescing_default": False,
        "anchor_contribution_mode": args.anchor_contribution_mode,
        "anchor_stack_all_float_order_jitter": bool(args.anchor_contribution_mode == "stack_all"),
        "output_accumulator": args.output_accumulator,
        "effective_output_accumulator": args.output_accumulator,
        "requested_output_accumulator": args.requested_output_accumulator,
        "output_accumulator_auto_selected": bool(args.requested_output_accumulator == "auto"),
        "output_accumulator_float_order_jitter": bool(args.output_accumulator == "torch"),
        "strict_shape_default": not (
            bool(args.coalesce_membership_patterns)
            or args.anchor_contribution_mode == "stack_all"
            or args.output_accumulator == "torch"
        ),
        "genome_zarr_path": str(args.genome_zarr),
        "assembly_policy": "caller_supplied",
        "aligned_motif_path": str(args.aligned_motif_path),
        "query_motif": args.query_motif,
        "query_motif_index": target_index,
        "query_motif_name": motif_metadata["resolved_query_motif_name"],
        "p_value_threshold": str(args.p_value_threshold),
        "score_threshold": args.score_threshold,
        "threshold_mode": (
            "score_threshold" if args.score_threshold is not None else "pvalue_mapping"
        ),
        "motif_source": motif_metadata["motif_source"],
        "loaded_with_aligned": motif_metadata["loaded_with_aligned"],
        **args.device_metadata,
        "motif_names_checksum": motif_metadata["motif_names_checksum"],
        "motif_kernel_checksum": motif_metadata["motif_kernel_checksum"],
        "motif_kernel_shape": list(motif_metadata["motif_kernel_shape"]),
        "regions_tsv": str(args.regions_tsv),
        "peak_set_mode": "pairwise_set_difference"
        if args.pairwise_set_diff
        else ("multiple_peak_sets" if args.peak_set else None),
        "peak_set_window_bp": int(args.peak_set_window_bp) if args.peak_set else None,
        "peak_set_min_peaks": int(args.min_peaks) if args.peak_set else None,
        "peak_set_inventory_tsv": str(peak_set_inventory) if peak_set_inventory else None,
        "subset_column": str(args.subset_column),
        "window_size": int(args.window),
        "cache_path": str(args.cache_path),
        "query_motif_hit_cache_build_mode": cache_source_metadata[
            "query_motif_hit_cache_build_mode"
        ],
        "query_motif_hit_cache_tile_size_bp": cache_source_metadata[
            "query_motif_hit_cache_tile_size_bp"
        ],
        "query_motif_hit_cache_tile_extension_bp": cache_source_metadata[
            "query_motif_hit_cache_tile_extension_bp"
        ],
        "query_motif_hit_cache_requested_tile_extension_bp": cache_source_metadata[
            "query_motif_hit_cache_requested_tile_extension_bp"
        ],
        "query_motif_hit_cache_minimum_tile_extension_bp": cache_source_metadata[
            "query_motif_hit_cache_minimum_tile_extension_bp"
        ],
        "cache_manifest_json": str(args.cache_manifest_json),
        "reuse_existing_cache": bool(args.reuse_existing_cache),
        "reuse_cache_manifest": (
            str(args.reuse_cache_manifest) if args.reuse_cache_manifest is not None else None
        ),
        "reuse_cache_manifest_compatibility": reuse_manifest_compatibility,
        "output_zarr": None if args.query_motif_hit_cache_only else str(args.output_zarr),
        "plan_json": str(args.plan_json),
        "run_json": str(args.run_json),
        "workflow_json": str(args.workflow_json),
        "workflow_wall_s": time.perf_counter() - workflow_start,
        "execution_backend": (
            "subprocess_package_module" if args.subprocess_backend else "in_process"
        ),
        "commands": commands,
        "plan_summary": plan_summary,
        "run_summary": run_summary,
    }
    if (
        args.coalesce_membership_patterns
        or args.anchor_contribution_mode == "stack_all"
        or args.output_accumulator == "torch"
    ):
        workflow_summary["validation_status"] = "PASS_WARN"
        notes = []
        if args.coalesce_membership_patterns:
            notes.append(
                "Membership-pattern coalescing preserves full curves but is not "
                "the strict default because real 16x256 evidence showed float32 "
                "addition-order jitter above 1e-4."
            )
        if args.anchor_contribution_mode == "stack_all":
            notes.append(
                "anchor_contribution_mode=stack_all preserves full curves but may "
                "change float32 addition order for mixed-strand multi-anchor intervals."
            )
        if args.output_accumulator == "torch":
            notes.append(
                "output_accumulator=torch preserves full curves but updates the "
                "output tensor on the analysis device; compare against strict "
                "numpy output before using it as biological evidence."
            )
        workflow_summary["validation_note"] = " ".join(notes)
    else:
        workflow_summary["validation_status"] = run_summary.get(
            "validation_status",
            "PASS" if run_summary else "PLAN_ONLY",
        )
    if run_summary:
        manifest = _write_cache_manifest(
            args.cache_manifest_json,
            cache_path=run_summary.get("query_motif_hit_cache_path", str(args.cache_path)),
            expected_metadata=run_summary.get("expected_query_motif_hit_cache_metadata"),
            run_summary={**run_summary, "summary_path": str(args.run_json)},
            workflow_summary=workflow_summary,
            source_manifest=(
                str(args.reuse_cache_manifest) if args.reuse_cache_manifest is not None else None
            ),
        )
        workflow_summary["cache_manifest"] = manifest
    eureka_payload_json = args.output_dir / "eureka_backend_payload.json"
    eureka_manifest_json = args.output_dir / "eureka_backend_manifest.json"
    eureka_payload = _eureka_backend_run_payload(
        workflow_summary,
        payload_json=eureka_payload_json,
        manifest_json=eureka_manifest_json,
    )
    eureka_manifest = _eureka_backend_manifest(
        workflow_summary,
        args,
        payload_json=eureka_payload_json,
        manifest_json=eureka_manifest_json,
    )
    eureka_payload["eureka_backend_manifest_json"] = str(eureka_manifest_json)
    workflow_summary["eureka_backend_payload"] = eureka_payload
    workflow_summary["eureka_backend_payload_json"] = str(eureka_payload_json)
    workflow_summary["eureka_backend_manifest"] = eureka_manifest
    workflow_summary["eureka_backend_manifest_json"] = str(eureka_manifest_json)
    _write_json(eureka_manifest_json, eureka_manifest)
    _write_json(eureka_payload_json, eureka_payload)
    _write_json(args.workflow_json, workflow_summary)
    return workflow_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-tsv")
    parser.add_argument(
        "--peak-set",
        action="append",
        help="Peak set as NAME=BED_OR_NARROWPEAK; repeat for multiple sets.",
    )
    parser.add_argument(
        "--pairwise-set-diff",
        action="store_true",
        help="Create ordered source-minus-reference peak subsets before full-curve reuse.",
    )
    parser.add_argument(
        "--peak-set-window-bp",
        type=int,
        default=2000,
        help="Center each input peak to this fixed window before membership construction.",
    )
    parser.add_argument(
        "--min-peaks",
        type=int,
        default=0,
        help="Keep constructed peak-set subsets only when their size is greater than this value.",
    )
    parser.add_argument("--subset-column", default="subset_id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--genome-zarr",
        default=DEFAULT_GENOME_ZARR,
        required=DEFAULT_GENOME_ZARR is None,
        help="Genome sequence Zarr (or set MOTIVERSE_GENOME_ZARR).",
    )
    parser.add_argument(
        "--allow-nondefault-genome",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aligned-motif-path",
        default=DEFAULT_ALIGNED_MOTIF_PATH,
        required=DEFAULT_ALIGNED_MOTIF_PATH is None,
        help="Aligned motif tensor (or set MOTIVERSE_ALIGNED_MOTIFS).",
    )
    parser.add_argument("--motifs")
    parser.add_argument("--query-motif", default=DEFAULT_QUERY_MOTIF)
    parser.add_argument("--query-motif-index", type=int)
    parser.add_argument("--p-value-threshold", default="p0.0001")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--no-pvalue-mapping", action="store_true")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--strand-specific", action="store_true")
    parser.add_argument("--provider", choices=["interval", "block"], default="interval")
    parser.add_argument("--provider-max-gap-bp", type=int, default=0)
    parser.add_argument("--provider-max-block-span-bp", type=int, default=1_000_000)
    parser.add_argument("--provider-max-block-score-gb", type=float)
    parser.add_argument("--provider-max-cached-blocks", type=int, default=1)
    parser.add_argument(
        "--provider-max-sequence-cache-gb",
        type=float,
        default=0.5,
        help=(
            "CPU LRU cache size for raw direct chromosome zarr chunks in the "
            "block provider; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--anchor-contribution-mode",
        choices=["strict", "stack_all"],
        default="strict",
        help=(
            "Target-motif hit-cache replay kernel. 'strict' is the "
            "checksum-preserving default; 'stack_all' is faster for multi-anchor "
            "intervals but may introduce small float-order jitter and is labeled "
            "PASS_WARN."
        ),
    )
    parser.add_argument(
        "--output-accumulator",
        choices=["numpy", "torch", "auto"],
        default="numpy",
        help=(
            "Accumulator backend for query-motif hit-cache replay outputs. 'numpy' is "
            "the strict checksum-preserving default. 'torch' keeps full curves "
            "on the analysis device during aggregation and is labeled PASS_WARN "
            "until compared against strict output. 'auto' selects torch only "
            "when the effective analysis device is CUDA and membership-pattern "
            "coalescing is disabled."
        ),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-subsets", type=int, default=64)
    parser.add_argument("--checksum-mode", choices=["full", "sample", "none"], default="full")
    parser.add_argument("--max-output-gb", type=float)
    parser.add_argument("--cache-path")
    parser.add_argument(
        "--query-motif-hit-cache-tile-size",
        dest="query_motif_hit_cache_tile_size",
        type=int,
        help=(
            "Build the compact query-motif hit cache by scanning reusable genome tiles "
            "and reconstructing exact interval hits. Full curves are still replayed "
            "over exact peak intervals."
        ),
    )
    parser.add_argument(
        "--query-motif-hit-cache-tile-extension-bp",
        dest="query_motif_hit_cache_tile_extension_bp",
        type=int,
        help=("Tile scan flank for --query-motif-hit-cache-tile-size. Defaults to --window."),
    )
    parser.add_argument("--cache-manifest-json")
    parser.add_argument(
        "--reuse-cache-manifest",
        help=(
            "Reuse the query-motif hit-cache path recorded in a previous "
            "query_motif_hit_cache_manifest.json. The cache metadata is still "
            "validated against the current run before replay."
        ),
    )
    parser.add_argument("--output-zarr")
    parser.add_argument("--plan-json")
    parser.add_argument("--run-json")
    parser.add_argument("--workflow-json")
    parser.add_argument("--reuse-existing-cache", action="store_true")
    parser.add_argument(
        "--query-motif-hit-cache-only",
        dest="query_motif_hit_cache_only",
        action="store_true",
        help=(
            "Build the exact query-motif hit cache and manifest, then stop before "
            "full-curve subset replay. This is a reusable precompute stage, not "
            "a final biological output."
        ),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--coalesce-membership-patterns",
        action="store_true",
        help=(
            "Optional speed mode. Keeps full curves but may introduce small "
            "float-order jitter, so it is PASS_WARN rather than strict default."
        ),
    )
    parser.add_argument("--max-membership-pattern-cache-gb", type=float)
    parser.add_argument("--dense-subset-script")
    parser.add_argument("--storage-threshold-gb", type=float, default=10.0)
    parser.add_argument(
        "--mnt-storage-base",
        default=str(Path.home() / ".cache" / "motiverse" / "benchmarks"),
    )
    parser.add_argument(
        "--subprocess-backend",
        action="store_true",
        help="Use the package-module subprocess backend instead of in-process calls.",
    )
    return parser


def main() -> None:
    summary = run_full_curve_one_to_all_reuse(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
