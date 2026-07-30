#!/usr/bin/env python
"""Exact dense-window subset aggregation by scanning each unique interval once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from motiverse.dense_subset import (
    DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION,
    DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS,
    DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION,
    DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS,
    DENSE_SUBSET_COORDINATE_FRAME,
    DENSE_SUBSET_SEMANTICS,
    DENSE_SUBSET_ZARR_SCHEMA_VERSION,
    DenseGenomeBlockScoreProvider,
    DenseGenomeScoreProvider,
    _read_zarr_v3_nd_array,
    _write_zarr_v3_1d_array,
    _write_zarr_v3_group,
    _write_zarr_v3_nd_array,
    accumulate_dense_score_region_subsets,
    accumulate_dense_score_region_subsets_from_contribution_cache,
    accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr,
    accumulate_dense_score_region_subsets_from_query_motif_hit_cache,
    accumulate_dense_score_region_subsets_to_zarr,
    accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter,
    build_dense_interval_contribution_cache,
    build_dense_query_motif_hit_cache,
    build_dense_query_motif_hit_cache_from_tiles,
    plan_dense_interval_contribution_cache_resources,
    plan_dense_score_region_subset_resources,
)
from motiverse.motif_source import load_hocomoco_motifs
from motiverse.positional_cache import (
    threshold_vector_checksum,
)


def _checksum(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def resolve_analysis_device(requested_device: str | None) -> tuple[str, dict]:
    """Resolve auto/cuda/cpu into a torch device string plus provenance."""
    requested = str(requested_device or "auto")
    normalized = requested.lower()
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    metadata = {
        "requested_device": requested,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available and cuda_device_count else None
        ),
    }
    if normalized == "auto":
        effective = "cuda" if cuda_available else "cpu"
        metadata["device_resolution"] = (
            "auto_cuda_available" if cuda_available else "auto_cuda_unavailable_cpu"
        )
    elif normalized.startswith("cuda"):
        if not cuda_available:
            raise ValueError(
                "--device cuda was requested, but torch.cuda.is_available() is false. "
                "Use --device auto for CPU fallback or --device cpu for an explicit CPU run."
            )
        if ":" in normalized:
            try:
                index = int(normalized.split(":", 1)[1])
            except ValueError as error:
                raise ValueError(f"Invalid CUDA device string: {requested!r}") from error
            if index < 0 or index >= cuda_device_count:
                raise ValueError(
                    f"CUDA device {requested!r} requested, but only "
                    f"{cuda_device_count} CUDA device(s) are visible."
                )
        effective = requested
        metadata["device_resolution"] = "explicit_cuda"
    else:
        effective = requested
        metadata["device_resolution"] = "explicit_non_cuda"
    metadata["effective_device"] = effective
    return effective, metadata


def _baseline_allclose(
    values: np.ndarray, baseline: np.ndarray
) -> tuple[bool, float, float, float, float]:
    rtol = 2e-5
    atol = 1e-6
    abs_diff = np.abs(values - baseline)
    scale = np.maximum(np.maximum(np.abs(values), np.abs(baseline)), 1.0)
    rel_diff = abs_diff / scale
    return (
        bool(np.all(abs_diff <= (atol + rtol * scale))),
        rtol,
        atol,
        float(np.max(abs_diff)) if abs_diff.size else 0.0,
        float(np.max(rel_diff)) if rel_diff.size else 0.0,
    )


def _zarr_values(path: str | Path) -> np.ndarray:
    direct_values = Path(path) / "values" / "zarr.json"
    if direct_values.exists():
        return _read_zarr_v3_nd_array(Path(path) / "values")
    root = zarr.open_group(str(path), mode="r")
    return np.asarray(root["values"][:])


def _update_zarr_metadata(path: str | Path, metadata: dict) -> None:
    metadata_json = Path(path) / "metadata" / "zarr.json"
    if metadata_json.exists():
        payload = json.loads(metadata_json.read_text())
        attrs = dict(payload.get("attributes", {}))
        attrs.update({str(key): value for key, value in metadata.items()})
        payload["attributes"] = attrs
        metadata_json.write_text(json.dumps(payload, sort_keys=True) + "\n")
        return
    root = zarr.open_group(str(path), mode="a")
    root["metadata"].attrs.update(metadata)


def _write_dense_result_to_zarr(
    *,
    result,
    output_path: str | Path,
    chunk_subsets: int,
    metadata: dict,
) -> None:
    """Persist an in-memory full-curve dense subset result.

    This is used for query-motif-hit prefilter runs, where the current accelerator
    skips full motif scans for no-anchor intervals but still returns the same
    full-curve tensor contract as the direct dense path.
    """
    output_path = Path(output_path)
    if output_path.exists():
        if output_path.is_dir():
            import shutil

            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    values = np.asarray(result.values)
    chunk_shape = (max(1, min(int(chunk_subsets), values.shape[0])),) + values.shape[1:]
    _write_zarr_v3_group(output_path)
    _write_zarr_v3_group(output_path / "metadata", attrs=metadata)
    _write_zarr_v3_1d_array(
        output_path / "metadata" / "subset_ids",
        np.asarray(result.subset_ids, dtype=object),
    )
    _write_zarr_v3_nd_array(
        output_path / "values",
        values.astype(np.float32, copy=False),
        chunk_shape=chunk_shape,
    )


def _thresholds_from_args(
    hocomoco_db,
    *,
    score_threshold: float | None,
    p_value_threshold: str,
    use_pvalue_mapping: bool,
    n_motifs: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    if score_threshold is not None:
        threshold_values = torch.full(
            (n_motifs,),
            float(score_threshold),
            dtype=dtype,
            device=device,
        )
        threshold_mode = "score_threshold"
    elif not use_pvalue_mapping:
        _, threshold_values = hocomoco_db.prepare_for_gpu(
            device=device,
            dtype=dtype,
            p_value=p_value_threshold,
        )
        threshold_mode = "annotation"
    else:
        p_value_numeric = float(str(p_value_threshold).replace("p", ""))
        threshold_values = torch.as_tensor(
            hocomoco_db.get_score_threshold_for_pvalue(p_value_numeric),
            dtype=dtype,
            device=device,
        )
        threshold_mode = "pvalue_mapping"
    metadata = {
        "threshold_mode": threshold_mode,
        "p_value_threshold": p_value_threshold,
        "score_threshold": score_threshold,
        "score_threshold_vector_checksum": threshold_vector_checksum(threshold_values),
        "score_threshold_vector_shape": list(threshold_values.shape),
    }
    return {p_value_threshold: threshold_values}, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-zarr", required=True)
    parser.add_argument("--regions-tsv", required=True)
    parser.add_argument("--subset-column", required=True)
    parser.add_argument("--motifs")
    parser.add_argument("--aligned-motif-path")
    parser.add_argument("--allow-pwm-fallback", action="store_true")
    parser.add_argument(
        "--p-value-threshold",
        default="p0.0001",
        help=(
            "P-value threshold for motif hits when --score-threshold is not "
            "provided. Uses HOCOMOCO p-value mapping."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Explicit score threshold for all motifs; overrides --p-value-threshold.",
    )
    parser.add_argument(
        "--no-pvalue-mapping",
        action="store_true",
        help=(
            "Use annotation file thresholds for --p-value-threshold instead of "
            "computed HOCOMOCO p-value mapping, matching the main CLI flag."
        ),
    )
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--query-motif-index", type=int)
    parser.add_argument(
        "--query-motif-hit-prefilter",
        action="store_true",
        help=(
            "For one-to-all full curves, scan the query motif first and skip "
            "full-motif scoring for exact intervals with no valid query motif hit."
        ),
    )
    parser.add_argument("--strand-specific", action="store_true")
    parser.add_argument("--provider", choices=["interval", "block"], default="interval")
    parser.add_argument("--provider-max-gap-bp", type=int, default=0)
    parser.add_argument("--provider-max-block-span-bp", type=int, default=1_000_000)
    parser.add_argument(
        "--provider-max-block-score-gb",
        type=float,
        help=(
            "Optional estimated per-block score-tensor memory ceiling for the "
            "block provider. Coalescing stops before a merged block would exceed "
            "this limit."
        ),
    )
    parser.add_argument(
        "--provider-max-cached-blocks",
        type=int,
        default=1,
        help="Maximum block score tensors to retain for block provider; use -1 for unbounded.",
    )
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
            "Target-anchor replay contribution kernel. 'strict' is the "
            "checksum-preserving default; 'stack_all' is faster for multi-anchor "
            "intervals but may introduce small float-order jitter."
        ),
    )
    parser.add_argument(
        "--output-accumulator",
        choices=["numpy", "torch"],
        default="numpy",
        help=(
            "Accumulator backend for query-motif-hit cache replay. 'numpy' is the "
            "strict default; 'torch' keeps full-curve output on the analysis "
            "device during aggregation and should be treated as PASS_WARN until "
            "compared against strict output."
        ),
    )
    parser.add_argument("--validate-interval-provider", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-output-gb", type=float)
    parser.add_argument("--output-zarr")
    parser.add_argument(
        "--build-contribution-cache",
        help=(
            "Build a full-curve dense interval contribution cache at this zarr "
            "path before subset aggregation."
        ),
    )
    parser.add_argument(
        "--use-contribution-cache",
        help=(
            "Replay an existing full-curve dense interval contribution cache "
            "instead of rescanning genome intervals."
        ),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="With --build-contribution-cache, stop after writing the cache.",
    )
    parser.add_argument(
        "--max-contribution-cache-gb",
        type=float,
        help="Abort contribution-cache build if planned cache values exceed this size.",
    )
    parser.add_argument("--contribution-cache-chunk-intervals", type=int, default=1)
    parser.add_argument(
        "--coalesce-membership-patterns",
        action="store_true",
        help=(
            "When replaying a contribution cache or query-motif-hit cache, sum "
            "interval contributions by identical subset-membership pattern "
            "before writing full curves."
        ),
    )
    parser.add_argument(
        "--max-membership-pattern-cache-gb",
        type=float,
        help="Abort membership-pattern coalescing if pattern accumulators exceed this size.",
    )
    parser.add_argument(
        "--membership-pattern-read-batch-intervals",
        type=int,
        default=64,
        help="Contribution-cache interval span batch size for membership-pattern replay.",
    )
    parser.add_argument(
        "--missing-contribution-cache-intervals-are-zero",
        action="store_true",
        help=(
            "Treat intervals absent from a contribution cache as zero. Intended "
            "for sparse/nonzero contribution caches."
        ),
    )
    parser.add_argument(
        "--build-query-motif-hit-cache",
        dest="build_query_motif_hit_cache",
        help=(
            "Build a compact exact query-motif-hit cache at this zarr path before "
            "one-to-all full-curve subset aggregation."
        ),
    )
    parser.add_argument(
        "--use-query-motif-hit-cache",
        dest="use_query_motif_hit_cache",
        help=(
            "Replay an existing compact query-motif-hit cache and scan full "
            "partner motifs only for exact intervals with cached query motif hits."
        ),
    )
    parser.add_argument(
        "--query-motif-hit-cache-only",
        dest="query_motif_hit_cache_only",
        action="store_true",
        help="With --build-query-motif-hit-cache, stop after writing the cache.",
    )
    parser.add_argument(
        "--query-motif-hit-cache-tile-size",
        dest="query_motif_hit_cache_tile_size",
        type=int,
        help=(
            "Build --build-query-motif-hit-cache by scanning reusable genome tiles "
            "and reconstructing exact interval query motif hits. Output remains full-curve exact."
        ),
    )
    parser.add_argument(
        "--query-motif-hit-cache-tile-extension-bp",
        dest="query_motif_hit_cache_tile_extension_bp",
        type=int,
        help=(
            "Tile scan flank for --query-motif-hit-cache-tile-size. Defaults to "
            "--window so exact interval anchor filtering has enough context."
        ),
    )
    parser.add_argument("--chunk-subsets", type=int, default=64)
    parser.add_argument(
        "--buffered-subset-updates",
        "--buffered-output",
        dest="buffered_subset_updates",
        action="store_true",
        help=(
            "Accumulate the dense subset output in RAM and write zarr once. "
            "This preserves exact semantics but requires one full output tensor in memory."
        ),
    )
    parser.add_argument(
        "--chunk-buffered-subset-updates",
        "--chunk-buffered-output",
        dest="chunk_buffered_subset_updates",
        action="store_true",
        help=(
            "Keep a bounded LRU of subset-output chunks in RAM and flush chunks "
            "to zarr. This preserves exact semantics without one full output buffer."
        ),
    )
    parser.add_argument(
        "--max-buffered-output-gb",
        type=float,
        help="Abort buffered zarr output if the dense RAM buffer would exceed this size.",
    )
    parser.add_argument(
        "--max-chunk-output-gb",
        type=float,
        help="Limit the RAM footprint of one chunk-buffered subset output chunk.",
    )
    parser.add_argument(
        "--max-active-subset-chunks",
        type=int,
        default=4,
        help="Maximum subset-output chunks kept live by --chunk-buffered-output.",
    )
    parser.add_argument(
        "--checksum-mode",
        choices=["full", "sample", "none"],
        default="full",
        help="Checksum policy for zarr values; full is exact, sample/none are PASS_WARN evidence.",
    )
    parser.add_argument("--checksum-sample-subsets", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--storage-threshold-gb", type=float, default=10.0)
    parser.add_argument(
        "--mnt-storage-base",
        default=str(Path.home() / ".cache" / "motiverse" / "benchmarks"),
    )
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.buffered_subset_updates and args.chunk_buffered_subset_updates:
        raise SystemExit("Use only one of --buffered-output or --chunk-buffered-output.")
    if args.cache_only and not args.build_contribution_cache:
        raise SystemExit("--cache-only requires --build-contribution-cache.")
    if args.query_motif_hit_cache_only and not args.build_query_motif_hit_cache:
        raise SystemExit("--query-motif-hit-cache-only requires --build-query-motif-hit-cache.")
    if args.query_motif_hit_cache_tile_size is not None and not args.build_query_motif_hit_cache:
        raise SystemExit(
            "--query-motif-hit-cache-tile-size requires --build-query-motif-hit-cache."
        )
    if (
        args.query_motif_hit_cache_tile_extension_bp is not None
        and args.query_motif_hit_cache_tile_size is None
    ):
        raise SystemExit(
            "--query-motif-hit-cache-tile-extension-bp requires --query-motif-hit-cache-tile-size."
        )
    if args.build_contribution_cache and args.use_contribution_cache:
        raise SystemExit("Use only one of --build-contribution-cache or --use-contribution-cache.")
    if args.build_query_motif_hit_cache and args.use_query_motif_hit_cache:
        raise SystemExit(
            "Use only one of --build-query-motif-hit-cache or --use-query-motif-hit-cache."
        )
    if (args.build_query_motif_hit_cache or args.use_query_motif_hit_cache) and (
        args.build_contribution_cache or args.use_contribution_cache
    ):
        raise SystemExit(
            "Use query-motif-hit cache modes separately from contribution-cache modes."
        )
    if (
        args.build_query_motif_hit_cache
        or args.use_query_motif_hit_cache
        or args.query_motif_hit_prefilter
    ) and args.query_motif_index is None:
        raise SystemExit("Target-anchor modes require --query-motif-index for one-to-all output.")
    if args.query_motif_hit_prefilter and (
        args.build_query_motif_hit_cache or args.use_query_motif_hit_cache
    ):
        raise SystemExit(
            "Use either direct --query-motif-hit-prefilter or query-motif-hit cache "
            "build/replay, not both."
        )
    if args.allow_pwm_fallback and (
        args.query_motif_hit_prefilter
        or args.build_query_motif_hit_cache
        or args.use_query_motif_hit_cache
    ):
        raise SystemExit(
            "Target-anchor one-to-all modes require aligned HOCOMOCO PT; "
            "remove --allow-pwm-fallback."
        )
    if args.query_motif_hit_prefilter and args.query_motif_index is None:
        raise SystemExit("--query-motif-hit-prefilter requires --query-motif-index.")
    if args.query_motif_hit_prefilter and (
        args.build_contribution_cache or args.use_contribution_cache
    ):
        raise SystemExit(
            "--query-motif-hit-prefilter cannot be combined with contribution-cache "
            "build/replay in this CLI. Use cache replay for precomputed reuse or "
            "query-motif-hit prefilter for direct full-curve scans."
        )
    if args.coalesce_membership_patterns and not (
        args.build_contribution_cache
        or args.use_contribution_cache
        or args.build_query_motif_hit_cache
        or args.use_query_motif_hit_cache
    ):
        raise SystemExit(
            "--coalesce-membership-patterns requires --build-contribution-cache "
            "or --use-contribution-cache or a query-motif-hit cache mode."
        )
    if args.coalesce_membership_patterns and args.chunk_buffered_subset_updates:
        raise SystemExit(
            "--coalesce-membership-patterns currently requires direct zarr or "
            "--buffered-output; omit --chunk-buffered-output."
        )
    if args.coalesce_membership_patterns and args.output_accumulator == "torch":
        raise SystemExit(
            "--output-accumulator=torch cannot be combined with "
            "--coalesce-membership-patterns; benchmark these PASS_WARN speed modes "
            "separately."
        )
    try:
        args.device, device_metadata = resolve_analysis_device(args.device)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    dtype = np.dtype(args.dtype)
    torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
    regions = pd.read_csv(args.regions_tsv, sep="\t")
    hocomoco_db, motif_metadata = load_hocomoco_motifs(
        motif_selection=args.motifs,
        aligned_motif_path=args.aligned_motif_path,
        use_aligned_motifs=True,
        require_aligned_motifs=not args.allow_pwm_fallback,
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
    thresholds, threshold_metadata = _thresholds_from_args(
        hocomoco_db,
        score_threshold=args.score_threshold,
        p_value_threshold=args.p_value_threshold,
        use_pvalue_mapping=not args.no_pvalue_mapping,
        n_motifs=len(motif_names),
        dtype=torch_dtype,
        device=args.device,
    )
    max_output_bytes = None
    if args.max_output_gb is not None:
        max_output_bytes = int(args.max_output_gb * (1024**3))
    max_buffered_output_bytes = None
    if args.max_buffered_output_gb is not None:
        max_buffered_output_bytes = int(args.max_buffered_output_gb * (1024**3))
    max_chunk_output_bytes = None
    if args.max_chunk_output_gb is not None:
        max_chunk_output_bytes = int(args.max_chunk_output_gb * (1024**3))
    max_contribution_cache_bytes = None
    if args.max_contribution_cache_gb is not None:
        max_contribution_cache_bytes = int(args.max_contribution_cache_gb * (1024**3))
    max_membership_pattern_cache_bytes = None
    if args.max_membership_pattern_cache_gb is not None:
        max_membership_pattern_cache_bytes = int(args.max_membership_pattern_cache_gb * (1024**3))
    provider_max_block_score_bytes = None
    if args.provider_max_block_score_gb is not None:
        provider_max_block_score_bytes = int(args.provider_max_block_score_gb * (1024**3))
    storage_threshold_bytes = int(args.storage_threshold_gb * (1024**3))

    metadata_extra = {
        "mode": "dense-exact-subset-query-motif-hit-prefilter"
        if args.query_motif_hit_prefilter
        else "dense-exact-subset-query-motif-hit-cache-replay"
        if args.use_query_motif_hit_cache
        else "dense-exact-subset-genome-scan",
        "provider": args.provider,
        "genome_zarr_path": args.genome_zarr,
        "motif_count": len(motif_names),
        "motif_names_checksum": motif_metadata.motif_names_checksum,
        "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
        "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
        "motif_source": motif_metadata.motif_source,
        "aligned_motif_path": motif_metadata.aligned_motif_path,
        "loaded_with_aligned": motif_metadata.loaded_with_aligned,
        **device_metadata,
        **threshold_metadata,
        "strand_specific": bool(args.strand_specific),
        "strands": 1 if args.strand_specific else 2,
    }
    if args.plan_only:
        resource_plan = plan_dense_score_region_subset_resources(
            regions,
            subset_column=args.subset_column,
            n_motifs=len(motif_names),
            window_size=args.window,
            query_motif_index=args.query_motif_index,
            motif_length=int(motif_kernels.shape[1]),
            strands=1 if args.strand_specific else 2,
            dtype=dtype,
            provider_max_gap_bp=args.provider_max_gap_bp,
            provider_max_block_span_bp=args.provider_max_block_span_bp,
            provider_max_block_score_bytes=provider_max_block_score_bytes,
            output_path=args.output_zarr or args.output_json,
            storage_threshold_bytes=storage_threshold_bytes,
            mnt_storage_base=args.mnt_storage_base,
        )
        contribution_cache_plan = plan_dense_interval_contribution_cache_resources(
            regions,
            subset_column=args.subset_column,
            n_motifs=len(motif_names),
            window_size=args.window,
            query_motif_index=args.query_motif_index,
            dtype=dtype,
            max_cache_bytes=max_contribution_cache_bytes,
        )
        summary = {
            **resource_plan.to_dict(),
            "mode": "dense-exact-subset-resource-plan",
            "provider": args.provider,
            "genome_zarr_path": args.genome_zarr,
            "motif_count": len(motif_names),
            "motif_names_checksum": motif_metadata.motif_names_checksum,
            "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
            "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
            "motif_source": motif_metadata.motif_source,
            "aligned_motif_path": motif_metadata.aligned_motif_path,
            "loaded_with_aligned": motif_metadata.loaded_with_aligned,
            **device_metadata,
            **threshold_metadata,
            "strand_specific": bool(args.strand_specific),
            "strands": 1 if args.strand_specific else 2,
            "contribution_cache_plan": contribution_cache_plan.to_dict(),
            "contribution_cache_bytes": (contribution_cache_plan.contribution_cache_bytes),
            "contribution_cache_bytes_per_unique_region": (
                contribution_cache_plan.cache_bytes_per_unique_region
            ),
            "contribution_cache_feasible_under_max": (
                contribution_cache_plan.cache_feasible_under_max
            ),
            "contribution_cache_recommended_strategy": (
                contribution_cache_plan.recommended_strategy
            ),
        }
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return

    if args.provider == "block":
        max_cached_blocks = (
            None if args.provider_max_cached_blocks < 0 else args.provider_max_cached_blocks
        )
        provider_max_sequence_cache_bytes = int(args.provider_max_sequence_cache_gb * (1024**3))
        provider = DenseGenomeBlockScoreProvider(
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
    else:
        provider = DenseGenomeScoreProvider(
            args.genome_zarr,
            motif_kernels,
            device=args.device,
            dtype=torch_dtype,
            strand_specific=args.strand_specific,
        )
    query_provider = None
    query_thresholds = None
    if args.query_motif_hit_prefilter or args.build_query_motif_hit_cache:
        query_idx = int(args.query_motif_index)
        query_motif_kernels = motif_kernels[query_idx : query_idx + 1]
        query_thresholds = {
            key: value[query_idx : query_idx + 1] for key, value in thresholds.items()
        }
        if args.provider == "block":
            max_cached_blocks = (
                None if args.provider_max_cached_blocks < 0 else args.provider_max_cached_blocks
            )
            provider_max_sequence_cache_bytes = int(args.provider_max_sequence_cache_gb * (1024**3))
            query_provider = DenseGenomeBlockScoreProvider(
                args.genome_zarr,
                query_motif_kernels,
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
        else:
            query_provider = DenseGenomeScoreProvider(
                args.genome_zarr,
                query_motif_kernels,
                device=args.device,
                dtype=torch_dtype,
                strand_specific=args.strand_specific,
            )
    contribution_cache_path = args.use_contribution_cache
    query_motif_hit_cache_path = args.use_query_motif_hit_cache
    cache_build_summary = {}
    query_motif_hit_cache_build_summary = {}
    if args.build_query_motif_hit_cache:
        query_hit_cache_build_start = time.perf_counter()
        query_hit_cache_metadata = {
            **metadata_extra,
            "mode": "dense-exact-query-motif-hit-cache-build",
        }
        if args.query_motif_hit_cache_tile_size is None:
            query_hit_cache_result = build_dense_query_motif_hit_cache(
                query_provider,
                regions,
                output_path=args.build_query_motif_hit_cache,
                subset_column=args.subset_column,
                score_thresholds=query_thresholds,
                window_size=args.window,
                n_motifs=len(motif_names),
                query_motif_index=int(args.query_motif_index),
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
                output_path=args.build_query_motif_hit_cache,
                subset_column=args.subset_column,
                score_thresholds=query_thresholds,
                window_size=args.window,
                n_motifs=len(motif_names),
                query_motif_index=int(args.query_motif_index),
                tile_size=int(args.query_motif_hit_cache_tile_size),
                tile_extension_bp=args.query_motif_hit_cache_tile_extension_bp,
                p_value_threshold=args.p_value_threshold,
                dtype=dtype,
                torch_dtype=torch_dtype,
                device=args.device,
                metadata_extra=query_hit_cache_metadata,
            )
        query_motif_hit_cache_build_summary = {
            "query_motif_hit_cache_schema_version": (DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION),
            "query_motif_hit_cache_path": query_hit_cache_result.path,
            "query_motif_hit_cache_build_wall_s": time.perf_counter() - query_hit_cache_build_start,
            "query_motif_hit_cache_unique_regions": (query_hit_cache_result.n_unique_regions),
            "query_motif_hit_cache_anchor_hits": query_hit_cache_result.n_anchor_hits,
            "query_motif_hit_cache_checksum": query_hit_cache_result.checksum,
            "query_motif_hit_cache_build_mode": (
                "tile_assisted_exact_interval_reconstruction"
                if args.query_motif_hit_cache_tile_size is not None
                else "exact_unique_interval"
            ),
            "query_motif_hit_cache_tile_size_bp": args.query_motif_hit_cache_tile_size,
            "query_motif_hit_cache_tile_extension_bp": (
                args.query_motif_hit_cache_tile_extension_bp
                if args.query_motif_hit_cache_tile_size is not None
                else None
            ),
            **{
                f"query_motif_hit_cache_{key}": value
                for key, value in query_hit_cache_result.timings.items()
            },
        }
        query_motif_hit_cache_path = query_hit_cache_result.path
        if args.query_motif_hit_cache_only:
            provider_stats = getattr(query_provider, "stats", {})
            summary = {
                "schema_version": DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION,
                "mode": "dense-exact-query-motif-hit-cache-build",
                "semantics": DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS,
                "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
                "provider": args.provider,
                "genome_zarr_path": args.genome_zarr,
                "motif_count": len(motif_names),
                "motif_names_checksum": motif_metadata.motif_names_checksum,
                "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
                "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
                "motif_source": motif_metadata.motif_source,
                "aligned_motif_path": motif_metadata.aligned_motif_path,
                "loaded_with_aligned": motif_metadata.loaded_with_aligned,
                **threshold_metadata,
                **query_motif_hit_cache_build_summary,
                "bp_scanned": int(provider_stats.get("provider_bp_loaded", 0)),
                **{
                    f"query_motif_hit_provider_{key}": value
                    for key, value in provider_stats.items()
                },
            }
            output_json = Path(args.output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            return
    if args.build_contribution_cache:
        cache_build_start = time.perf_counter()
        cache_result = build_dense_interval_contribution_cache(
            provider,
            regions,
            output_path=args.build_contribution_cache,
            subset_column=args.subset_column,
            score_thresholds=thresholds,
            window_size=args.window,
            n_motifs=len(motif_names),
            query_motif_index=args.query_motif_index,
            p_value_threshold=args.p_value_threshold,
            dtype=dtype,
            torch_dtype=torch_dtype,
            device=args.device,
            max_cache_bytes=max_contribution_cache_bytes,
            chunk_intervals=args.contribution_cache_chunk_intervals,
            metadata_extra={
                **metadata_extra,
                "mode": "dense-exact-interval-contribution-cache-build",
            },
            checksum_mode=args.checksum_mode,
            checksum_sample_subsets=args.checksum_sample_subsets,
        )
        cache_build_summary = {
            "contribution_cache_schema_version": (DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION),
            "contribution_cache_path": cache_result.path,
            "contribution_cache_build_wall_s": time.perf_counter() - cache_build_start,
            "contribution_cache_unique_regions": cache_result.n_unique_regions,
            "contribution_cache_shape": list(cache_result.contribution_shape),
            "contribution_cache_checksum": cache_result.checksum,
            **{f"contribution_cache_{key}": value for key, value in cache_result.timings.items()},
        }
        contribution_cache_path = cache_result.path
        if args.cache_only:
            summary = {
                "schema_version": DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION,
                "mode": "dense-exact-interval-contribution-cache-build",
                "semantics": DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS,
                "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
                "provider": args.provider,
                "genome_zarr_path": args.genome_zarr,
                "motif_count": len(motif_names),
                "motif_names_checksum": motif_metadata.motif_names_checksum,
                "motif_kernel_checksum": motif_metadata.motif_kernel_checksum,
                "motif_kernel_shape": list(motif_metadata.motif_kernel_shape),
                "motif_source": motif_metadata.motif_source,
                "aligned_motif_path": motif_metadata.aligned_motif_path,
                "loaded_with_aligned": motif_metadata.loaded_with_aligned,
                **threshold_metadata,
                **cache_build_summary,
                "bp_scanned": int(provider.stats["provider_bp_loaded"]),
                **provider.stats,
            }
            output_json = Path(args.output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            return
    aggregate_start = time.perf_counter()
    replay_metadata_extra = metadata_extra
    if contribution_cache_path:
        replay_metadata_extra = {
            **metadata_extra,
            "mode": "dense-exact-subset-contribution-cache-replay",
        }
    if query_motif_hit_cache_path:
        replay_metadata_extra = {
            **metadata_extra,
            "mode": "dense-exact-subset-query-motif-hit-cache-replay",
        }
    query_motif_hit_expected_metadata = {
        **metadata_extra,
        "window_size": int(args.window),
        "query_motif_index": (
            int(args.query_motif_index) if args.query_motif_index is not None else None
        ),
        "dtype": str(dtype),
    }

    if args.output_zarr:
        if contribution_cache_path:
            result = accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr(
                contribution_cache_path,
                regions,
                output_path=args.output_zarr,
                subset_column=args.subset_column,
                dtype=dtype,
                max_output_bytes=max_output_bytes,
                chunk_subsets=args.chunk_subsets,
                metadata_extra=replay_metadata_extra,
                checksum_mode=args.checksum_mode,
                checksum_sample_subsets=args.checksum_sample_subsets,
                buffered_subset_updates=args.buffered_subset_updates,
                max_buffered_output_bytes=max_buffered_output_bytes,
                chunk_buffered_subset_updates=args.chunk_buffered_subset_updates,
                max_chunk_output_bytes=max_chunk_output_bytes,
                max_active_subset_chunks=args.max_active_subset_chunks,
                coalesce_membership_patterns=args.coalesce_membership_patterns,
                max_pattern_accumulator_bytes=max_membership_pattern_cache_bytes,
                membership_pattern_read_batch_intervals=(
                    args.membership_pattern_read_batch_intervals
                ),
                missing_intervals_are_zero=(
                    True if args.missing_contribution_cache_intervals_are_zero else None
                ),
            )
        elif query_motif_hit_cache_path:
            if args.buffered_subset_updates or args.chunk_buffered_subset_updates:
                raise SystemExit(
                    "Target-anchor cache replay currently writes zarr from the "
                    "in-memory full-curve result; omit buffered-output flags "
                    "for this mode."
                )
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
                max_pattern_accumulator_bytes=max_membership_pattern_cache_bytes,
                expected_metadata=query_motif_hit_expected_metadata,
                anchor_contribution_mode=args.anchor_contribution_mode,
                output_accumulator=args.output_accumulator,
            )
            values_checksum = _checksum(result.values)
            _write_dense_result_to_zarr(
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
                    "output_accumulator": args.output_accumulator,
                    "output_accumulator_float_order_jitter": bool(
                        args.output_accumulator == "torch"
                    ),
                    "values_checksum": values_checksum,
                    "values_checksum_mode": "full",
                    "values_checksum_status": "complete",
                    "values_checksum_sample_subsets": 0,
                    **query_motif_hit_cache_build_summary,
                    **result.timings,
                },
            )
        else:
            if args.query_motif_hit_prefilter:
                if args.buffered_subset_updates or args.chunk_buffered_subset_updates:
                    raise SystemExit(
                        "--query-motif-hit-prefilter currently writes zarr from "
                        "the in-memory full-curve result; omit buffered-output "
                        "flags for this mode."
                    )
                result = accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter(
                    query_provider,
                    provider,
                    regions,
                    subset_column=args.subset_column,
                    score_thresholds=query_thresholds,
                    window_size=args.window,
                    n_motifs=len(motif_names),
                    query_motif_index=int(args.query_motif_index),
                    p_value_threshold=args.p_value_threshold,
                    dtype=dtype,
                    torch_dtype=torch_dtype,
                    device=args.device,
                    max_output_bytes=max_output_bytes,
                    chunk_subsets=args.chunk_subsets,
                )
                values_shape = list(result.values.shape)
                values_checksum = _checksum(result.values)
                _write_dense_result_to_zarr(
                    result=result,
                    output_path=args.output_zarr,
                    chunk_subsets=args.chunk_subsets,
                    metadata={
                        **metadata_extra,
                        "schema_version": DENSE_SUBSET_ZARR_SCHEMA_VERSION,
                        "complete": True,
                        "semantics": DENSE_SUBSET_SEMANTICS,
                        "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
                        "subset_update_mode": "query_motif_hit_prefilter_in_memory_output",
                        "values_checksum": values_checksum,
                        "values_checksum_mode": "full",
                        "values_checksum_status": "complete",
                        "values_checksum_sample_subsets": 0,
                        "query_motif_hit_provider_stats": getattr(query_provider, "stats", {}),
                    },
                )
            else:
                result = accumulate_dense_score_region_subsets_to_zarr(
                    provider,
                    regions,
                    output_path=args.output_zarr,
                    subset_column=args.subset_column,
                    score_thresholds=thresholds,
                    window_size=args.window,
                    n_motifs=len(motif_names),
                    query_motif_index=args.query_motif_index,
                    dtype=dtype,
                    torch_dtype=torch_dtype,
                    device=args.device,
                    max_output_bytes=max_output_bytes,
                    chunk_subsets=args.chunk_subsets,
                    metadata_extra=metadata_extra,
                    checksum_mode=args.checksum_mode,
                    checksum_sample_subsets=args.checksum_sample_subsets,
                    buffered_subset_updates=args.buffered_subset_updates,
                    max_buffered_output_bytes=max_buffered_output_bytes,
                    chunk_buffered_subset_updates=args.chunk_buffered_subset_updates,
                    max_chunk_output_bytes=max_chunk_output_bytes,
                    max_active_subset_chunks=args.max_active_subset_chunks,
                )
        if (
            args.query_motif_hit_prefilter or query_motif_hit_cache_path
        ) and not contribution_cache_path:
            values_shape = list(result.values.shape)
            values_checksum = _checksum(result.values)
        else:
            values_shape = list(result.plan.output_shape)
            values_checksum = result.checksum
        values_for_validation = None
        root = zarr.open_group(str(args.output_zarr), mode="r")
        zarr_metadata = dict(root["metadata"].attrs)
        values_checksum_mode = zarr_metadata.get("values_checksum_mode")
        values_checksum_status = zarr_metadata.get("values_checksum_status")
        values_checksum_sample_subsets = zarr_metadata.get("values_checksum_sample_subsets")
        subset_update_mode = zarr_metadata.get("subset_update_mode")
        buffered_output_bytes = zarr_metadata.get("buffered_output_bytes")
        chunk_metadata = {
            key: zarr_metadata.get(key)
            for key in (
                "requested_chunk_subsets",
                "chunk_subsets",
                "chunk_buffer_bytes",
                "max_active_subset_chunks",
                "subset_chunk_reads",
                "subset_chunk_read_wall_s",
                "subset_chunk_flushes",
            )
        }
    else:
        if contribution_cache_path:
            result = accumulate_dense_score_region_subsets_from_contribution_cache(
                contribution_cache_path,
                regions,
                subset_column=args.subset_column,
                dtype=dtype,
                max_output_bytes=max_output_bytes,
                chunk_subsets=args.chunk_subsets,
                coalesce_membership_patterns=args.coalesce_membership_patterns,
                max_pattern_accumulator_bytes=max_membership_pattern_cache_bytes,
                membership_pattern_read_batch_intervals=(
                    args.membership_pattern_read_batch_intervals
                ),
                missing_intervals_are_zero=(
                    True if args.missing_contribution_cache_intervals_are_zero else None
                ),
            )
        elif query_motif_hit_cache_path:
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
                max_pattern_accumulator_bytes=max_membership_pattern_cache_bytes,
                expected_metadata=query_motif_hit_expected_metadata,
                anchor_contribution_mode=args.anchor_contribution_mode,
                output_accumulator=args.output_accumulator,
            )
        else:
            if args.query_motif_hit_prefilter:
                result = accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter(
                    query_provider,
                    provider,
                    regions,
                    subset_column=args.subset_column,
                    score_thresholds=query_thresholds,
                    window_size=args.window,
                    n_motifs=len(motif_names),
                    query_motif_index=int(args.query_motif_index),
                    p_value_threshold=args.p_value_threshold,
                    dtype=dtype,
                    torch_dtype=torch_dtype,
                    device=args.device,
                    max_output_bytes=max_output_bytes,
                    chunk_subsets=args.chunk_subsets,
                )
            else:
                result = accumulate_dense_score_region_subsets(
                    provider,
                    regions,
                    subset_column=args.subset_column,
                    score_thresholds=thresholds,
                    window_size=args.window,
                    n_motifs=len(motif_names),
                    query_motif_index=args.query_motif_index,
                    dtype=dtype,
                    torch_dtype=torch_dtype,
                    device=args.device,
                    max_output_bytes=max_output_bytes,
                    chunk_subsets=args.chunk_subsets,
                )
        values_shape = list(result.values.shape)
        values_checksum = _checksum(result.values)
        values_for_validation = result.values
        values_checksum_mode = "full"
        values_checksum_status = "complete"
        values_checksum_sample_subsets = 0
        subset_update_mode = "in_memory_dense_output"
        buffered_output_bytes = result.plan.output_bytes
        chunk_metadata = {
            "requested_chunk_subsets": args.chunk_subsets,
            "chunk_subsets": args.chunk_subsets,
            "chunk_buffer_bytes": result.plan.output_bytes,
            "max_active_subset_chunks": None,
            "subset_chunk_reads": 0,
            "subset_chunk_read_wall_s": 0.0,
            "subset_chunk_flushes": 0,
        }
    aggregate_wall_s = time.perf_counter() - aggregate_start
    summary = {
        "schema_version": (
            DENSE_SUBSET_ZARR_SCHEMA_VERSION if args.output_zarr else "dense_subset_genome_scan_v1"
        ),
        "semantics": DENSE_SUBSET_SEMANTICS,
        "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
        "mode": "dense-exact-subset-contribution-cache-replay"
        if contribution_cache_path
        else "dense-exact-subset-query-motif-hit-cache-replay"
        if query_motif_hit_cache_path
        else "dense-exact-subset-query-motif-hit-prefilter"
        if args.query_motif_hit_prefilter
        else "dense-exact-subset-genome-scan",
        "provider": args.provider,
        "genome_zarr_path": args.genome_zarr,
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
        "output_accumulator": args.output_accumulator,
        "output_accumulator_float_order_jitter": bool(args.output_accumulator == "torch"),
        "output_bytes": result.plan.output_bytes,
        "values_shape": values_shape,
        "values_checksum": values_checksum,
        "values_checksum_mode": values_checksum_mode,
        "values_checksum_status": values_checksum_status,
        "values_checksum_sample_subsets": values_checksum_sample_subsets,
        "subset_update_mode": subset_update_mode,
        "buffered_output_bytes": buffered_output_bytes,
        **chunk_metadata,
        **cache_build_summary,
        **query_motif_hit_cache_build_summary,
        "contribution_cache_path": contribution_cache_path,
        "query_motif_hit_cache_path": query_motif_hit_cache_path,
        "aggregate_wall_s": aggregate_wall_s,
        "bp_scanned": int(provider.stats["provider_bp_loaded"])
        + int(
            getattr(query_provider, "stats", {}).get("provider_bp_loaded", 0)
            if query_provider is not None
            else 0
        ),
        "full_provider_bp_loaded": int(provider.stats["provider_bp_loaded"]),
        "query_motif_hit_provider_bp_loaded": int(
            getattr(query_provider, "stats", {}).get("provider_bp_loaded", 0)
            if query_provider is not None
            else 0
        ),
        "query_motif_hit_prefilter": bool(args.query_motif_hit_prefilter),
        **result.timings,
        **provider.stats,
    }
    if query_provider is not None:
        summary.update(
            {
                f"query_motif_hit_provider_{key}": value
                for key, value in getattr(query_provider, "stats", {}).items()
            }
        )
    if args.validate_interval_provider:
        baseline_provider = DenseGenomeScoreProvider(
            args.genome_zarr,
            motif_kernels,
            device=args.device,
            dtype=torch_dtype,
            strand_specific=args.strand_specific,
        )
        baseline_start = time.perf_counter()
        baseline = accumulate_dense_score_region_subsets(
            baseline_provider,
            regions,
            subset_column=args.subset_column,
            score_thresholds=thresholds,
            window_size=args.window,
            n_motifs=len(motif_names),
            query_motif_index=args.query_motif_index,
            dtype=dtype,
            torch_dtype=torch_dtype,
            device=args.device,
            max_output_bytes=max_output_bytes,
            chunk_subsets=args.chunk_subsets,
        )
        summary["baseline_wall_s"] = time.perf_counter() - baseline_start
        summary["baseline_provider"] = "interval"
        summary["baseline_values_checksum"] = _checksum(baseline.values)
        summary["baseline_checksum_equal"] = None
        if summary.get("values_checksum_mode", "full") == "full":
            summary["baseline_checksum_equal"] = (
                summary["values_checksum"] == summary["baseline_values_checksum"]
            )
        if values_for_validation is None:
            values_for_validation = _zarr_values(args.output_zarr)
        (
            summary["baseline_allclose"],
            summary["baseline_rtol"],
            summary["baseline_atol"],
            summary["baseline_max_abs_diff"],
            summary["baseline_max_rel_diff"],
        ) = _baseline_allclose(values_for_validation, baseline.values)
        summary["baseline_provider_bp_loaded"] = baseline_provider.stats["provider_bp_loaded"]
        summary["baseline_provider_scan_wall_s"] = baseline_provider.stats["provider_scan_wall_s"]
    if args.output_zarr:
        _update_zarr_metadata(args.output_zarr, summary)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
