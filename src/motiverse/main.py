"""Main orchestrator and CLI for genome motif analysis.

This module contains the main orchestrator function and command-line interface
for the genome motif analysis pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .accumulation import initialize_analysis_tensor
from .config_utils import parse_plot_pairs_string
from .data_utils import (
    download_remap_narrowpeak,
    generate_tiled_regions,
    get_signal_data,
    load_signal_database,
    save_analysis_results,
)
from .motif_source import load_hocomoco_motifs
from .positional_cache import (
    CACHE_ACCUMULATION_SEMANTICS,
    PositionalMotifHitCache,
    build_positional_motif_hit_cache,
    threshold_vector_checksum,
)
from .processing import (
    process_chromosome_sequences,
    process_narrowpeak_regions,
    set_torch_compile,
)
from .sequence_io import (
    CelltypeDenseZarrIO,
    SequenceDenseZarrIO,
    read_narrowpeak,
)

logger = logging.getLogger(__name__)


_FULL_CURVE_REUSE_COMMANDS = {
    "full-curve-reuse",
    "full-curve-one-to-all-reuse",
    "profile-peak-groups",
    "multi-peak-set-reuse",
    "profile-peak-sets",
    "pairwise-peak-set-reuse",
    "compare-peak-sets",
}
_CHIP_ATLAS_COMMANDS = {"chip-atlas-contexts", "chip-atlas-subsets", "prepare-chip-atlas-groups"}
_CONTEXT_RANKING_COMMANDS = {
    "motif-context-ranking",
    "cross-context-motif-ranking",
    "rank-context-motifs",
}


def _run_backend_subcommand_if_requested(argv=None):
    """Route package-level backend subcommands before legacy CLI parsing."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        return False

    command = raw_args[0]
    command_args = raw_args[1:]
    if command in _FULL_CURVE_REUSE_COMMANDS:
        from . import full_curve_reuse

        reuse_args = full_curve_reuse.build_parser().parse_args(command_args)
        if command in {"pairwise-peak-set-reuse", "compare-peak-sets"}:
            reuse_args.pairwise_set_diff = True
        summary = full_curve_reuse.run_full_curve_one_to_all_reuse(reuse_args)
    elif command in _CHIP_ATLAS_COMMANDS:
        from . import chip_atlas

        summary = chip_atlas.run(chip_atlas.build_parser().parse_args(command_args))
    elif command in _CONTEXT_RANKING_COMMANDS:
        from . import context_ranking

        summary = context_ranking.run(context_ranking.build_parser().parse_args(command_args))
    else:
        return False

    print(json.dumps(summary, indent=2, sort_keys=True))
    return True


def _resolve_query_motif_index(hocomoco_db, motif_names, query_motif_name):
    """Resolve a CLI one-to-all target as an index or motif name."""
    target = str(query_motif_name).strip()
    if target.isdigit():
        target_index = int(target)
        if target_index < 0 or target_index >= len(motif_names):
            raise ValueError(f"Motif index {target_index} out of range [0, {len(motif_names) - 1}]")
        return target_index, motif_names[target_index]

    target_index = hocomoco_db.resolve_motif_index(target)
    return target_index, motif_names[target_index]


def motif_info_payload(
    *,
    motif_query=None,
    motif_selection=None,
    aligned_motif_path=None,
    use_aligned_motifs=True,
    require_aligned_motifs=True,
    threshold_mode="pvalue_mapping",
):
    """Load motif source only and return serializable provenance/target metadata."""
    hocomoco_db, motif_source_metadata = load_hocomoco_motifs(
        motif_selection=motif_selection,
        aligned_motif_path=aligned_motif_path,
        use_aligned_motifs=use_aligned_motifs,
        require_aligned_motifs=require_aligned_motifs,
        threshold_mode=threshold_mode,
    )
    motif_names = list(hocomoco_db.motif_names)
    payload = motif_source_metadata.to_dict()
    payload.update(
        {
            "schema_version": "genome_motif_analysis_motif_info_v1",
            "motif_count": len(motif_names),
            "query": str(motif_query) if motif_query is not None else None,
            "resolved_query_motif_index": None,
            "resolved_query_motif_name": None,
        }
    )
    if motif_query is not None and str(motif_query):
        target_index, target_name = _resolve_query_motif_index(
            hocomoco_db,
            motif_names,
            motif_query,
        )
        payload["resolved_query_motif_index"] = int(target_index)
        payload["resolved_query_motif_name"] = target_name
    return payload


def _extend_regions_for_cache(regions_df, genome_database, extend_bp, extend_right_only):
    """Apply scan-time extension to regions before querying positional cache."""
    if extend_bp <= 0:
        return regions_df.copy()
    regions_df = regions_df.copy()
    starts = []
    ends = []
    for row in regions_df.itertuples(index=False):
        chrom = getattr(row, "chrom", getattr(row, "Chromosome", None))
        start = int(getattr(row, "start", getattr(row, "Start", 0)))
        end = int(getattr(row, "end", getattr(row, "End", 0)))
        chrom_size = int(genome_database.chrom_sizes[chrom])
        if extend_right_only:
            starts.append(start)
            ends.append(min(chrom_size, end + extend_bp))
        else:
            starts.append(max(0, start - extend_bp))
            ends.append(min(chrom_size, end + extend_bp))
    regions_df["start"] = starts
    regions_df["end"] = ends
    return regions_df


def analyze_genome_sequences(
    genome_zarr_path,
    target_chromosomes=None,
    output_directory=None,
    batch_size=1000,
    sequence_length=1000,
    use_mixed_precision=True,
    enable_accumulation=True,
    analysis_window_size=500,
    motif_selection=None,
    narrowpeak_file_path=None,
    region_extension_bp=0,
    strand_specific=False,
    sequence_accumulation=False,
    query_motif_name=None,
    plot_config=None,
    use_remap=False,
    remap_gene=None,
    signal_accumulation=False,
    signal_zarr_path=None,
    signal_celltype_id=None,
    p_value_threshold="p0.0001",
    collect_hits=False,
    use_pvalue_mapping=True,
    score_threshold=None,
    use_tiling=False,
    tile_size=500,
    tile_extension_bp=15,
    use_torch_compile=False,
    hit_aggregation_mode="max",
    aligned_motif_path=None,
    use_aligned_motifs=True,
    require_aligned_motifs=True,
    build_hit_cache=False,
    hit_cache_path=None,
    cache_output_path=None,
    command_argv=None,
):
    """
    Orchestrates comprehensive genome-wide motif analysis pipeline.

    This is the main function that coordinates the entire analysis workflow:
    loading motif data, configuring analysis parameters, processing genomic
    sequences, and aggregating results across chromosomes or regions.

    Args:
        genome_zarr_path (str): Path to reference genome in Zarr format.
        target_chromosomes (list, optional): Specific chromosomes to analyze.
        output_directory (str, optional): Directory for saving results.
        batch_size (int): Number of sequences to process simultaneously.
        sequence_length (int): Length of sequence segments for chromosome processing.
        use_mixed_precision (bool): Whether to use reduced precision for speed.
        enable_accumulation (bool): Whether to perform spatial pattern analysis.
        analysis_window_size (int): Window size for spatial accumulation.
        motif_selection (str, optional): Subset of motifs to analyze.
        narrowpeak_file_path (str, optional): Path to narrowPeak file for region analysis.
        region_extension_bp (int): Base pairs to extend narrowPeak regions.
        strand_specific (bool): Whether to analyze only forward strand.
        sequence_accumulation (bool): Whether to accumulate sequence content.
        query_motif_name (str, optional): Name/index of query motif for focused analysis.
        plot_config (dict, optional): Configuration for visualization plots.
        use_remap (bool): Whether to download narrowPeak data from ReMap database for query motif.
        remap_gene (str, optional): Gene name to use for ReMap download (defaults to motif name).
        signal_accumulation (bool): Whether to accumulate signal data around motif hits.
        signal_zarr_path (str, optional): Path to signal zarr for signal accumulation.
        signal_celltype_id (str, optional): For CelltypeDenseZarrIO, select a specific cell type ID.
                                          If provided, only that cell type is loaded/queried (reduces memory usage).
        p_value_threshold (str): P-value threshold for motif hits (e.g., "p0.0001").
        collect_hits (bool): Whether to collect and save sparse motif hits (requires --narrowpeak or --tiling).
        use_pvalue_mapping (bool): Whether to use p-value mapping for score thresholds.
        score_threshold (float, optional): Explicit score threshold for all motifs (overrides p-value).
        use_tiling (bool): Whether to generate tiles across genome instead of narrowPeak file.
        tile_size (int): Size of tiles in base pairs for tiling mode.
        tile_extension_bp (int): Base pairs to extend tiles on right side during scanning.
        use_torch_compile (bool): Whether to use torch.compile for accelerated computations.
                                 Requires PyTorch 2.0+ and may have warmup overhead on first run.
        hit_aggregation_mode (str): How to aggregate scores when collecting hits: "max" (default) for maximum score,
                                    or "sum" for sum of all scores above threshold per peak.
        aligned_motif_path (str, optional): Path to HOCOMOCO aligned motif tensor.
        use_aligned_motifs (bool): Whether to prefer aligned motif tensor loading.
        require_aligned_motifs (bool): If True, fail if motifs fall back to PWM source.
        build_hit_cache (bool): Build a positional genome-tile motif-hit cache and return.
        hit_cache_path (str, optional): Existing positional cache to use for region accumulation.
        cache_output_path (str, optional): Output path for a newly built cache.
        command_argv (list, optional): CLI arguments to record in result/cache metadata.
    """
    # Configure torch.compile if requested
    if use_torch_compile:
        set_torch_compile(True)
        logger.info("torch.compile enabled - will compile core operations for better performance")

    # Configure computational resources
    computational_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select appropriate precision based on hardware capabilities
    if use_mixed_precision:
        if computational_device == "cpu" or torch.cuda.is_bf16_supported():
            numerical_precision = torch.bfloat16
        else:
            numerical_precision = torch.float16
    else:
        numerical_precision = torch.float32

    logger.info(
        f"Computational setup: {computational_device} device, {numerical_precision} precision"
    )

    threshold_mode = (
        "score_threshold"
        if score_threshold is not None
        else "pvalue_mapping"
        if use_pvalue_mapping
        else "annotation"
    )

    # Load motif data using HocomocoIO. Aligned PT is the default and is made
    # explicit here because cache/query correctness depends on a stable motif basis.
    hocomoco_db, motif_source_metadata = load_hocomoco_motifs(
        motif_selection=motif_selection,
        aligned_motif_path=aligned_motif_path,
        use_aligned_motifs=use_aligned_motifs,
        require_aligned_motifs=require_aligned_motifs,
        threshold_mode=threshold_mode,
    )

    # Handle non-aligned PWM-only reverse complements for backward compatibility.
    # Aligned tensors already include RC motifs and HocomocoIO returns self.
    if strand_specific and not getattr(hocomoco_db, "_loaded_with_aligned", False):
        hocomoco_db = hocomoco_db.create_reverse_complements()

    # Get motif data
    motif_names = hocomoco_db.motif_names
    motif_kernels = hocomoco_db.motif_kernels

    logger.info(f"Prepared {len(motif_names)} motifs with length {motif_kernels.shape[1]}")
    logger.info(
        "Motif source: %s (%s)",
        motif_source_metadata.motif_source,
        motif_source_metadata.aligned_motif_path,
    )

    # Transfer motif data to computational device
    motif_kernels_gpu, annotation_thresholds = hocomoco_db.prepare_for_gpu(
        device=computational_device,
        dtype=numerical_precision,
        p_value=p_value_threshold,
    )

    # Get score thresholds - prioritize p-value mapping (default behavior)
    thresholds_gpu = None

    # First try: Use p-value mapping if enabled (default)
    if use_pvalue_mapping and p_value_threshold:
        try:
            # Extract numeric p-value from string like "p0.0001"
            p_value_numeric = float(p_value_threshold.replace("p", ""))
            score_thresholds_from_pvalue = torch.tensor(
                hocomoco_db.get_score_threshold_for_pvalue(p_value_numeric)
            ).to(device=computational_device, dtype=numerical_precision)
            thresholds_gpu = score_thresholds_from_pvalue
            logger.info(
                f"Using computed score thresholds from p-value mapping (p={p_value_numeric})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to compute score thresholds from p-value mapping: {e}. "
                f"Falling back to annotation thresholds."
            )

    # Fallback: Use annotation thresholds if p-value mapping failed or disabled
    if thresholds_gpu is None:
        thresholds_gpu = annotation_thresholds
        if thresholds_gpu is not None:
            logger.info(f"Using annotation file thresholds (p={p_value_threshold})")

    # Override with explicit score threshold if provided (highest priority)
    if score_threshold is not None:
        score_thresholds_tensor = torch.full(
            (len(motif_names),),
            score_threshold,
            dtype=numerical_precision,
            device=computational_device,
        )
        thresholds_gpu = score_thresholds_tensor
        logger.info(f"Using explicit score threshold: {score_threshold}")

    # Prepare complete motif parameter package
    # For compatibility with existing functions, we need to create the expected structure:
    # (motif_names, motif_kernels, _, score_thresholds, _, _)
    gpu_motif_parameters = (
        motif_names,
        motif_kernels_gpu,
        None,  # placeholder for processed_motif_data[2]
        {p_value_threshold: thresholds_gpu} if thresholds_gpu is not None else {},
        None,  # placeholder for processed_motif_data[4]
        None,  # placeholder for processed_motif_data[5]
    )

    run_metadata = motif_source_metadata.to_dict()
    run_metadata.update(
        {
            "p_value_threshold": p_value_threshold,
            "score_threshold": score_threshold,
            "score_threshold_vector_checksum": threshold_vector_checksum(thresholds_gpu)
            if thresholds_gpu is not None
            else None,
            "score_threshold_vector_shape": list(thresholds_gpu.shape)
            if thresholds_gpu is not None
            else None,
            "threshold_mode": threshold_mode,
            "analysis_window_size": analysis_window_size,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "dtype": str(numerical_precision),
            "device": computational_device,
            "torch_compile": bool(use_torch_compile),
            "genome_zarr_path": str(genome_zarr_path),
            "output_directory": str(output_directory) if output_directory else None,
            "target_chromosomes": list(target_chromosomes)
            if target_chromosomes is not None
            else None,
            "narrowpeak_file_path": str(narrowpeak_file_path) if narrowpeak_file_path else None,
            "requested_narrowpeak_file_path": str(narrowpeak_file_path)
            if narrowpeak_file_path
            else None,
            "resolved_narrowpeak_file_path": str(narrowpeak_file_path)
            if narrowpeak_file_path
            else None,
            "region_extension_bp": int(region_extension_bp),
            "use_remap": bool(use_remap),
            "remap_gene": remap_gene,
            "remap_tf_name": None,
            "use_tiling": bool(use_tiling),
            "tile_size": int(tile_size),
            "tile_extension_bp": int(tile_extension_bp),
            "requested_query_motif": query_motif_name,
            "resolved_query_motif_index": None,
            "resolved_query_motif_name": None,
            "command_argv": list(command_argv) if command_argv is not None else None,
        }
    )

    if build_hit_cache:
        cache_path = (
            cache_output_path
            or hit_cache_path
            or str(Path(output_directory or ".") / "positional_motif_hits.zarr")
        )
        cache_region_intervals = None
        if narrowpeak_file_path:
            genome_database_for_cache = SequenceDenseZarrIO(genome_zarr_path, mode="r")
            peak_regions = read_narrowpeak(narrowpeak_file_path)
            peak_regions = peak_regions[
                peak_regions["chrom"].isin(set(genome_database_for_cache.chroms))
            ]
            cache_region_intervals = _extend_regions_for_cache(
                peak_regions,
                genome_database_for_cache,
                region_extension_bp,
                extend_right_only=False,
            )
        build_positional_motif_hit_cache(
            genome_zarr_path=genome_zarr_path,
            output_path=cache_path,
            motif_parameters=gpu_motif_parameters,
            motif_metadata=run_metadata,
            target_chromosomes=target_chromosomes,
            region_intervals=cache_region_intervals,
            tile_size=tile_size,
            tile_extension_bp=tile_extension_bp,
            batch_size=batch_size,
            device=computational_device,
            dtype=numerical_precision,
            strand_specific=strand_specific,
            p_value_threshold=p_value_threshold,
        )
        return

    # Resolve query motif for single-motif analysis mode
    query_motif_index = None
    resolved_motif_name = None
    if query_motif_name:
        if sequence_accumulation:
            raise ValueError("--one-to-all is not compatible with --sequence-accumulation")
        try:
            query_motif_index, resolved_motif_name = _resolve_query_motif_index(
                hocomoco_db,
                motif_names,
                query_motif_name,
            )
        except ValueError as error:
            raise ValueError(f"Could not identify query motif: {error}") from error
        run_metadata["resolved_query_motif_index"] = int(query_motif_index)
        run_metadata["resolved_query_motif_name"] = resolved_motif_name

    # Initialize genome database early if needed for ReMap or tiling
    genome_database = None
    if use_remap or narrowpeak_file_path or use_tiling:
        genome_database = SequenceDenseZarrIO(genome_zarr_path, mode="r")

    # Handle ReMap database integration
    if use_remap:
        if not query_motif_name:
            logger.error("--remap requires --one-to-all to specify query motif")
            return

        try:
            # Get TF name from the resolved motif name (not the user input)
            tf_name = hocomoco_db.get_tf_name_from_motif(resolved_motif_name)
            run_metadata["remap_tf_name"] = tf_name
            logger.info(
                f"Using TF '{tf_name}' for ReMap lookup (from motif '{resolved_motif_name}')"
            )

            # Download ReMap narrowPeak file
            remap_output_dir = output_directory if output_directory else "/tmp/remap_data"
            narrowpeak_file_path = download_remap_narrowpeak(
                tf_name,
                genome_database,
                output_dir=remap_output_dir,
                gene_name=remap_gene,
            )
            run_metadata["resolved_narrowpeak_file_path"] = str(narrowpeak_file_path)

        except (ValueError, Exception) as e:
            logger.error(f"ReMap integration failed: {e}")
            return

    # Generate tiled regions if tiling mode is enabled
    if use_tiling:
        if narrowpeak_file_path:
            logger.error("Cannot use both --narrowpeak and --tiling mode")
            return
        if genome_database is None:
            genome_database = SequenceDenseZarrIO(genome_zarr_path, mode="r")
        narrowpeak_file_path = "tiled_regions"  # Placeholder to trigger narrowPeak processing
        run_metadata["narrowpeak_file_path"] = narrowpeak_file_path
        run_metadata["resolved_narrowpeak_file_path"] = narrowpeak_file_path
        # Generate tiled regions (will be used instead of narrowpeak file)
        tiled_regions_df = generate_tiled_regions(
            genome_database,
            tile_size=tile_size,
            target_chromosomes=target_chromosomes,
        )
        # Override region_extension_bp for tiling mode (right-side only extension)
        region_extension_bp = tile_extension_bp
        logger.info(
            f"Tiling mode: Using {tile_size}bp tiles with {tile_extension_bp}bp right-side extension"
        )
        if enable_accumulation and tile_size < (2 * analysis_window_size + 1):
            logger.warning(
                f"Tile size ({tile_size}bp) is smaller than the spatial analysis window "
                f"({2 * analysis_window_size + 1}bp). Spatial accumulation will be skipped "
                "for these tiles. Use --no-accumulation if you only need hit collection "
                "to improve performance."
            )

    # Process narrowPeak regions if specified (or tiled regions)
    if narrowpeak_file_path:
        analysis_start_time = time.time()
        run_metadata["resolved_narrowpeak_file_path"] = str(narrowpeak_file_path)

        # Pass tiled regions if in tiling mode
        tiled_regions = tiled_regions_df if use_tiling else None
        extend_right_only = use_tiling  # Use right-side-only extension for tiling

        if hit_cache_path:
            if genome_database is None:
                genome_database = SequenceDenseZarrIO(genome_zarr_path, mode="r")
            if tiled_regions is not None:
                peak_regions = tiled_regions.copy()
            else:
                peak_regions = read_narrowpeak(narrowpeak_file_path)
            peak_regions = peak_regions[peak_regions["chrom"].isin(set(genome_database.chroms))]
            query_regions = _extend_regions_for_cache(
                peak_regions,
                genome_database,
                region_extension_bp,
                extend_right_only,
            )
            cache = PositionalMotifHitCache(hit_cache_path, mode="r")
            cache_mismatches = cache.validate_metadata(run_metadata)
            if cache_mismatches:
                mismatch_text = "; ".join(cache_mismatches)
                raise ValueError(
                    "Hit cache metadata does not match current motif analysis "
                    f"settings: {mismatch_text}"
                )
            cache_results = cache.accumulate_regions(
                query_regions,
                window_size=analysis_window_size,
                n_motifs=len(motif_names),
                query_motif_index=query_motif_index,
            )
            analysis_results = torch.as_tensor(
                cache_results, dtype=numerical_precision, device=computational_device
            )
            cache_source_metadata = cache.metadata
            cache_metadata = dict(run_metadata)
            cache_metadata.update(
                {
                    "hit_cache_path": str(hit_cache_path),
                    "cache_accumulation": True,
                    "cache_accumulation_semantics": CACHE_ACCUMULATION_SEMANTICS,
                    "coordinate_frame": cache_source_metadata.get("coordinate_frame"),
                    "cache_schema_version": cache_source_metadata.get("schema_version"),
                    "hit_storage_layout": cache_source_metadata.get("hit_storage_layout"),
                    "hit_chunk_index_schema_version": cache_source_metadata.get(
                        "hit_chunk_index_schema_version"
                    ),
                    "row_dtype": cache_source_metadata.get("row_dtype"),
                    "cache_n_hits": cache_source_metadata.get("n_hits"),
                    "n_regions": int(len(query_regions)),
                }
            )
            save_analysis_results(
                output_directory,
                analysis_results,
                motif_names,
                sequence_accumulation,
                query_motif_index,
                strand_specific,
                filename_prefix="narrowpeak_cache_analysis",
                plot_config=plot_config,
                signal_accumulation=signal_accumulation,
                metadata_extra=cache_metadata,
            )
            total_duration = time.time() - analysis_start_time
            logger.info(
                "Cache-backed narrowPeak analysis completed: %.1fs total",
                total_duration,
            )
            return

        (
            _,
            scan_duration,
            analysis_results,
            collected_hits,
            region_stats,
        ) = process_narrowpeak_regions(
            narrowpeak_file_path,
            genome_zarr_path,
            gpu_motif_parameters,
            region_extension_bp,
            batch_size,
            computational_device,
            numerical_precision,
            enable_accumulation,
            analysis_window_size,
            strand_specific,
            sequence_accumulation,
            query_motif_index,
            signal_accumulation,
            signal_zarr_path,
            signal_celltype_id,
            p_value_threshold,
            collect_hits,
            output_directory,
            tiled_regions_df=tiled_regions,
            extend_right_only=extend_right_only,
            hit_aggregation_mode=hit_aggregation_mode,
            return_region_stats=True,
        )

        narrowpeak_metadata = dict(run_metadata)
        narrowpeak_metadata.update(region_stats)
        narrowpeak_metadata["provider_scan_wall_s"] = scan_duration

        save_analysis_results(
            output_directory,
            analysis_results,
            motif_names,
            sequence_accumulation,
            query_motif_index,
            strand_specific,
            filename_prefix="narrowpeak_analysis",
            plot_config=plot_config,
            signal_accumulation=signal_accumulation,
            metadata_extra=narrowpeak_metadata,
        )

        # Note: Sparse hits are already saved by StreamingHitCollector during processing
        if collect_hits:
            hits_output_path = f"{output_directory}/motif_hits.zarr"
            logger.info(f"Sparse motif hits already saved to {hits_output_path} via streaming")

        total_duration = time.time() - analysis_start_time
        logger.info(
            f"narrowPeak analysis completed: {total_duration:.1f}s total ({scan_duration:.1f}s scanning)"
        )
        return

    # Process whole chromosomes
    if genome_database is None:
        genome_database = SequenceDenseZarrIO(genome_zarr_path, mode="r")
    available_chromosomes = genome_database.chroms
    chromosomes_to_analyze = target_chromosomes if target_chromosomes else available_chromosomes

    # Load signal data if signal accumulation is enabled
    signal_database = None
    if signal_accumulation and signal_zarr_path:
        signal_database = load_signal_database(signal_zarr_path, celltype_id=signal_celltype_id)

    # Determine signal dimensions if signal accumulation is enabled
    n_signal_dimensions = None
    if signal_accumulation and signal_database is not None:
        # Check signal data dimensionality from the first available chromosome
        available_chromosomes_list = list(available_chromosomes)
        if available_chromosomes_list:
            sample_chrom = available_chromosomes_list[-1]
            sample_signal = get_signal_data(
                signal_database,
                sample_chrom,
                0,
                min(1000, genome_database.chrom_sizes[sample_chrom]),
                celltype_id=signal_celltype_id,
            )
            if sample_signal.ndim == 2:
                n_signal_dimensions = sample_signal.shape[1]
            elif sample_signal.ndim == 1:
                n_signal_dimensions = 1
            else:
                raise ValueError(
                    f"Signal data must have 1 or 2 dimensions, got {sample_signal.ndim}"
                )
            logger.info(f"Detected signal data with {n_signal_dimensions} dimensions")
            if isinstance(signal_database, CelltypeDenseZarrIO):
                logger.info(
                    f"Cell types will be processed as independent dimensions: {signal_database.ids}"
                )

    # Initialize global accumulation tensor for cross-chromosome aggregation
    global_results_tensor = initialize_analysis_tensor(
        enable_accumulation,
        sequence_accumulation,
        query_motif_index,
        len(motif_names),
        analysis_window_size,
        computational_device,
        numerical_precision,
        motif_names,
        strand_specific,
        signal_accumulation,
        n_signal_dimensions,
    )

    # Process each chromosome sequentially
    cumulative_time, cumulative_scan_time, cumulative_base_pairs = 0, 0, 0

    for chromosome_index, chromosome_name in enumerate(chromosomes_to_analyze):
        logger.info(
            f"\nAnalyzing {chromosome_name} ({chromosome_index + 1}/{len(chromosomes_to_analyze)})"
        )
        chromosome_start_time = time.time()
        chromosome_scan_time = 0
        chromosome_length = genome_database.chrom_sizes[chromosome_name]

        # Get chunking information
        n_chunks = genome_database.chrom_n_chunks.get(chromosome_name, 1)
        chunk_size = genome_database.chunk_size or chromosome_length

        logger.info(f"Processing {chromosome_name} in {n_chunks} chunks of size {chunk_size}")

        # Initialize chromosome-level results accumulator
        chromosome_results = None

        # Method 2 optimization: Load entire chromosome once, then slice chunks from memory
        # This is much faster than calling get_track() for each chunk
        # Estimate memory: chromosome_length * 4 (one-hot) * 4 bytes (float32) = ~16 bytes per bp
        # For chr1 (~250Mbp): ~4GB, which is reasonable for most systems
        load_full_chromosome = True  # Can be made conditional based on memory if needed

        if load_full_chromosome and n_chunks > 1:
            logger.info(
                f"Loading entire {chromosome_name} ({chromosome_length:,} bp) to memory for efficient chunk slicing"
            )
            full_chromosome_sequence = genome_database.get_track(
                chromosome_name, 0, chromosome_length, output_format="raw_array"
            ).astype(np.float32)
            logger.debug(
                f"Loaded {chromosome_name} ({full_chromosome_sequence.shape[0]:,} bp) to memory"
            )

        # Process chromosome chunk by chunk
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, chromosome_length)

            logger.info(f"  Processing chunk {chunk_idx + 1}/{n_chunks}: {chunk_start}-{chunk_end}")

            # Load chunk sequence data - use Method 2 if chromosome is loaded
            if load_full_chromosome and n_chunks > 1:
                # Slice from in-memory chromosome array (Method 2)
                chunk_sequence = full_chromosome_sequence[chunk_start:chunk_end]
            else:
                # Fallback to per-chunk loading (original method)
                chunk_sequence = genome_database.get_track(
                    chromosome_name, chunk_start, chunk_end, output_format="raw_array"
                ).astype(np.float32)

            # Load chunk signal data if signal accumulation is enabled
            chunk_signal = None
            if signal_accumulation and signal_database:
                chunk_signal = get_signal_data(
                    signal_database,
                    chromosome_name,
                    chunk_start,
                    chunk_end,
                    celltype_id=signal_celltype_id,
                ).astype(np.float32)
                # Signal can be 1D (seq_length,) or 2D (seq_length, n_celltypes)

            # Process chunk through analysis pipeline
            _, chunk_scan_time, chunk_results = process_chromosome_sequences(
                f"{chromosome_name}_chunk_{chunk_idx}",
                chunk_sequence,
                gpu_motif_parameters,
                batch_size,
                sequence_length,
                computational_device,
                numerical_precision,
                enable_accumulation,
                analysis_window_size,
                strand_specific,
                sequence_accumulation,
                query_motif_index,
                signal_accumulation,
                chunk_signal,
                p_value_threshold,
            )

            # Aggregate chunk results into chromosome results
            if enable_accumulation and chunk_results is not None:
                if chromosome_results is None:
                    chromosome_results = chunk_results
                else:
                    chromosome_results += chunk_results

            # Update chunk timing
            chromosome_scan_time += chunk_scan_time

            # Clean up memory after each chunk
            del chunk_sequence, chunk_results
            if chunk_signal is not None:
                del chunk_signal
            gc.collect()
            if computational_device == "cuda":
                torch.cuda.empty_cache()

        # Clean up full chromosome sequence if it was loaded
        if load_full_chromosome and n_chunks > 1:
            del full_chromosome_sequence
            gc.collect()

        # Aggregate chromosome results into global results
        if enable_accumulation and chromosome_results is not None:
            global_results_tensor += chromosome_results

        # Clean up chromosome-level memory
        if chromosome_results is not None:
            del chromosome_results
        gc.collect()
        if computational_device == "cuda":
            torch.cuda.empty_cache()

        # Update timing statistics
        chromosome_duration = time.time() - chromosome_start_time
        cumulative_time += chromosome_duration
        cumulative_scan_time += chromosome_scan_time
        cumulative_base_pairs += chromosome_length

        logger.info(
            f"  Chromosome completed: {chromosome_duration:.1f}s total ({chromosome_scan_time:.1f}s scanning)"
        )

    # Save final aggregated results
    save_analysis_results(
        output_directory,
        global_results_tensor,
        motif_names,
        sequence_accumulation,
        query_motif_index,
        strand_specific,
        plot_config=plot_config,
        signal_accumulation=signal_accumulation,
        metadata_extra=run_metadata,
    )

    # Report final performance statistics
    average_throughput = cumulative_base_pairs / cumulative_scan_time / 1e6
    logger.info("\nGenome analysis completed:")
    logger.info(f"  Total time: {cumulative_time / 60:.1f} minutes")
    logger.info(f"  Average throughput: {average_throughput:.1f} Mbp/s")


def main():
    """
    Command-line interface for genome motif analysis pipeline.

    Parses command-line arguments and configures the analysis pipeline
    with appropriate parameters. Provides flexible options for different
    analysis modes and computational configurations.
    """
    if _run_backend_subcommand_if_requested():
        return

    argument_parser = argparse.ArgumentParser(
        description="Comprehensive genome motif analysis with spatial pattern detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Modes:
  Co-occurrence: Analyze spatial relationships between all motif pairs
  Query-centered: Focus analysis around one specific motif
  Sequence content: Accumulate DNA composition around binding sites

Plotting Options:
  --plot: Enable automatic plot generation after analysis
  --plot-types: cooccurrence,power_spectrum,autocorrelation,quality
  --plot-pairs: Specify motif pairs as "motif1,motif2;motif3,motif4"
               Supports names: "CTCF,TP53;NFKB1,STAT1"
               Or indices: "0,1;2,3"

Examples:
  # Profile full motif-position curves for every peak group
  motiverse profile-peak-groups --regions-tsv subsets.tsv \
      --genome-zarr genome.zarr --aligned-motif-path motifs.pt \
      --query-motif CTCF.H13CORE.0.P.B --output-dir results/

  # Analyze all chromosomes with co-occurrence patterns
  motiverse --genome-zarr genome.zarr --output results/

  # Focus on CTCF binding sites with plots
  motiverse --genome-zarr genome.zarr --one-to-all CTCF.H13CORE.0.P.B \
      --plot --plot-pairs "CTCF,TP53"

  # Aggregate signal around all motifs (all-to-one mode)
  motiverse --all-to-one --signal-zarr signal.zarr --output results/

  # Analyze only CTCF and TP53 family motifs using prefix matching
  motiverse --motifs "CTCF,TP53" --plot

  # Mix prefix matching with indices and ranges
  motiverse --motifs "CTCF,TP53,0-5,10" --output results/

  # Analyze sequence content around peaks from ChIP-seq
  motiverse --narrowpeak peaks.bed --sequence-accumulation

  # Use ReMap database to automatically download FOXA1 ChIP-seq peaks
  motiverse --one-to-all FOXA1 --remap --plot

  # Use ReMap with different gene name for download
  motiverse --one-to-all FOXA1.D.0.02 --remap --remap-gene FOXA1 --plot
        """,
    )

    # Input/Output configuration
    argument_parser.add_argument(
        "--genome-zarr",
        default=os.environ.get("MOTIVERSE_GENOME_ZARR"),
        required=os.environ.get("MOTIVERSE_GENOME_ZARR") is None,
        help="Path to reference genome Zarr (or set MOTIVERSE_GENOME_ZARR)",
    )
    argument_parser.add_argument(
        "--output",
        "-o",
        default="motiverse_results",
        help="Output directory for results (default: %(default)s)",
    )

    # Analysis scope configuration
    argument_parser.add_argument(
        "--chr",
        nargs="+",
        help="Specific chromosomes to analyze (e.g., chr1 chr2 chrX)",
    )
    argument_parser.add_argument(
        "--narrowpeak",
        help="Path to narrowPeak file for region-specific analysis",
    )
    argument_parser.add_argument(
        "--remap",
        action="store_true",
        help="Use ReMap database to download narrowPeak files for query motif",
    )
    argument_parser.add_argument(
        "--remap-gene",
        type=str,
        help="Gene name to use for ReMap download (defaults to motif name)",
    )
    argument_parser.add_argument(
        "--extend",
        type=int,
        default=0,
        help="Base pairs to extend each narrowPeak region (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--tiling",
        action="store_true",
        help="Generate tiles across entire genome instead of using narrowPeak file (500bp tiles by default)",
    )
    argument_parser.add_argument(
        "--tile-size",
        type=int,
        default=500,
        help="Size of tiles in base pairs for --tiling mode (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--tile-extension",
        type=int,
        default=15,
        help="Base pairs to extend tiles on right side during scanning to avoid boundary issues (default: %(default)s). "
        "Note: Original tile boundaries are saved to zarr, not extended boundaries.",
    )

    # Processing parameters
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Sequences processed simultaneously (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--seq-length",
        type=int,
        default=2000,
        help="Length of sequence segments in bp (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--analysis-window",
        type=int,
        default=500,
        help="Half-width of spatial analysis window in bp (default: %(default)s)",
    )

    # Computational configuration
    argument_parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use 32-bit precision instead of mixed precision",
    )
    argument_parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use torch.compile to accelerate core computations (requires PyTorch 2.0+). "
        "May have warmup overhead on first run but should provide speedup on subsequent runs.",
    )
    argument_parser.add_argument(
        "--no-accumulation",
        action="store_true",
        help="Disable spatial pattern accumulation (count hits only)",
    )
    argument_parser.add_argument(
        "--collect-hits",
        action="store_true",
        help="Collect sparse motif hits (requires --narrowpeak or --tiling)",
    )
    argument_parser.add_argument(
        "--hit-aggregation",
        type=str,
        choices=["max", "sum"],
        default="max",
        help="How to aggregate motif scores per peak when collecting hits: "
        "'max' (default) saves maximum score per motif per peak, "
        "'sum' saves sum of all scores above threshold per motif per peak (default: %(default)s)",
    )

    # Analysis mode selection
    argument_parser.add_argument(
        "--sequence-accumulation",
        action="store_true",
        help="Accumulate DNA sequence content instead of motif scores",
    )
    argument_parser.add_argument(
        "--one-to-all",
        help="Query motif name/index for focused analysis mode",
    )
    argument_parser.add_argument(
        "--all-to-one",
        action="store_true",
        help="Aggregate signal from zarr file around each motif binding site",
    )
    argument_parser.add_argument(
        "--signal-zarr",
        help="Path to signal dense zarr file for --all-to-one mode",
    )
    argument_parser.add_argument(
        "--signal-celltype-id",
        help="For CelltypeDenseZarrIO: select a single celltype ID to load/query (reduces memory usage)",
    )
    argument_parser.add_argument(
        "--strand-specific",
        action="store_true",
        help="Analyze only forward strand with reverse complement motifs",
    )

    # Motif/cache provenance
    argument_parser.add_argument(
        "--aligned-motif-path",
        help="Path to the HOCOMOCO motifs_with_rc_aligned.pt tensor. Defaults to gcell's configured aligned motif path.",
    )
    argument_parser.add_argument(
        "--allow-pwm-fallback",
        action="store_true",
        help="Allow fallback to PWM motifs if the aligned motif tensor is unavailable. By default aligned motifs are required.",
    )
    argument_parser.add_argument(
        "--no-aligned-motifs",
        action="store_true",
        help="Disable aligned motif loading and use PWM motifs. Requires --allow-pwm-fallback.",
    )
    argument_parser.add_argument(
        "--motif-info",
        nargs="?",
        const="",
        metavar="MOTIF",
        help="Load motifs, optionally resolve a motif name/index, print JSON provenance, and exit before genome scanning.",
    )
    argument_parser.add_argument(
        "--build-hit-cache",
        action="store_true",
        help="Build an exact positional genome-tile motif-hit cache and exit.",
    )
    argument_parser.add_argument(
        "--hit-cache",
        help="Path to an existing positional motif-hit cache for cache-backed region accumulation, or cache output path with --build-hit-cache.",
    )
    argument_parser.add_argument(
        "--cache-output",
        help="Output path for --build-hit-cache (defaults to <output>/positional_motif_hits.zarr).",
    )

    # Motif selection
    argument_parser.add_argument(
        "--motifs",
        help="Motif selection criteria. Supports: indices (0,1,5), ranges (0-10), exact names, "
        "prefix matching (CTCF matches all CTCF.* motifs), or mixed (CTCF,TP53,0-5)",
    )
    # P-value threshold
    argument_parser.add_argument(
        "--p-value-threshold",
        type=str,
        default="p0.0001",
        help="P-value threshold for motif hits (default: %(default)s). "
        "This can be any numeric p-value (e.g., 'p0.0001', 'p0.001', 'p1e-5'). "
        "The system will compute accurate score thresholds using FIMO p-value mapping.",
    )
    argument_parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Explicit score threshold for all motifs (overrides p-value threshold). "
        "If specified, all motifs will use this score cutoff directly.",
    )
    argument_parser.add_argument(
        "--no-pvalue-mapping",
        action="store_true",
        help="Disable p-value mapping and use annotation file thresholds (limited to p0.001, p0.0005, p0.0001)",
    )
    # Plotting configuration
    argument_parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable automatic plot generation after analysis",
    )
    argument_parser.add_argument(
        "--plot-types",
        default="cooccurrence,power_spectrum,autocorrelation",
        help="Comma-separated plot types: cooccurrence,power_spectrum,autocorrelation (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--plot-pairs",
        help="Motif pairs to plot as 'motif1,motif2;motif3,motif4'. Use names or indices.",
    )
    argument_parser.add_argument(
        "--plot-smooth",
        type=int,
        default=1,
        help="Smoothing window for co-occurrence plots (default: %(default)s)",
    )
    argument_parser.add_argument(
        "--plot-max-delay",
        type=int,
        default=500,
        help="Maximum delay for autocorrelation plots (default: %(default)s)",
    )

    # Convenience options
    argument_parser.add_argument("--test", action="store_true", help="Test mode: analyze only chr2")
    argument_parser.add_argument(
        "--all", action="store_true", help="Analyze all available chromosomes"
    )

    # Parse command line arguments
    analysis_config = argument_parser.parse_args()

    # Validate signal accumulation arguments
    if analysis_config.all_to_one and not analysis_config.signal_zarr:
        argument_parser.error("--all-to-one requires --signal-zarr to specify signal data")
    if analysis_config.signal_zarr and not analysis_config.all_to_one:
        argument_parser.error("--signal-zarr can only be used with --all-to-one")
    if analysis_config.all_to_one and analysis_config.sequence_accumulation:
        argument_parser.error("--all-to-one cannot be used with --sequence-accumulation")
    if analysis_config.all_to_one and analysis_config.one_to_all:
        argument_parser.error("--all-to-one cannot be used with --one-to-all")
    if (
        analysis_config.collect_hits
        and not analysis_config.narrowpeak
        and not analysis_config.tiling
    ):
        argument_parser.error("--collect-hits requires either --narrowpeak or --tiling")
    if analysis_config.tiling and analysis_config.narrowpeak:
        argument_parser.error("Cannot use both --tiling and --narrowpeak")
    if analysis_config.no_aligned_motifs and not analysis_config.allow_pwm_fallback:
        argument_parser.error("--no-aligned-motifs requires --allow-pwm-fallback")
    threshold_mode = (
        "score_threshold"
        if analysis_config.score_threshold is not None
        else "pvalue_mapping"
        if not analysis_config.no_pvalue_mapping
        else "annotation"
    )
    if analysis_config.motif_info is not None:
        motif_query = analysis_config.motif_info or analysis_config.one_to_all
        payload = motif_info_payload(
            motif_query=motif_query,
            motif_selection=analysis_config.motifs,
            aligned_motif_path=analysis_config.aligned_motif_path,
            use_aligned_motifs=not analysis_config.no_aligned_motifs,
            require_aligned_motifs=not analysis_config.allow_pwm_fallback,
            threshold_mode=threshold_mode,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if analysis_config.one_to_all and analysis_config.sequence_accumulation:
        argument_parser.error("--one-to-all cannot be used with --sequence-accumulation")
    if analysis_config.remap and not analysis_config.one_to_all:
        argument_parser.error("--remap requires --one-to-all")
    if analysis_config.build_hit_cache and analysis_config.remap:
        argument_parser.error("--build-hit-cache does not support --remap")
    if analysis_config.hit_cache and analysis_config.signal_zarr:
        argument_parser.error("--hit-cache does not support --all-to-one signal accumulation")
    if analysis_config.hit_cache and not analysis_config.build_hit_cache:
        if analysis_config.sequence_accumulation:
            argument_parser.error("--hit-cache does not support --sequence-accumulation")
        if not (analysis_config.narrowpeak or analysis_config.tiling or analysis_config.remap):
            argument_parser.error(
                "--hit-cache requires --narrowpeak, --tiling, or --remap unless "
                "--build-hit-cache is set"
            )

    # Configure chromosome selection
    if analysis_config.test:
        selected_chromosomes = ["chr2"]
    elif analysis_config.all:
        selected_chromosomes = None  # Process all chromosomes
    elif analysis_config.chr:
        # User specified chromosomes explicitly
        selected_chromosomes = analysis_config.chr
    else:
        # Default to all chromosomes from genome zarr
        # Load genome database to get available chromosomes
        genome_db_temp = SequenceDenseZarrIO(analysis_config.genome_zarr, mode="r")
        selected_chromosomes = list(genome_db_temp.chroms)
        # Note: genome database will be loaded again in analyze_genome_sequences if needed

    # Configure plotting parameters
    plot_config = {
        "enable_plotting": analysis_config.plot,
        "plot_types": analysis_config.plot_types.split(",") if analysis_config.plot_types else [],
        "motif_pairs": parse_plot_pairs_string(analysis_config.plot_pairs),
        "smoothing_window": analysis_config.plot_smooth,
        "max_delay": analysis_config.plot_max_delay,
    }

    # Execute main analysis pipeline
    analyze_genome_sequences(
        genome_zarr_path=analysis_config.genome_zarr,
        target_chromosomes=selected_chromosomes,
        output_directory=analysis_config.output,
        batch_size=analysis_config.batch_size,
        sequence_length=analysis_config.seq_length,
        use_mixed_precision=not analysis_config.fp32,
        enable_accumulation=not analysis_config.no_accumulation,
        analysis_window_size=analysis_config.analysis_window,
        motif_selection=analysis_config.motifs,
        narrowpeak_file_path=analysis_config.narrowpeak,
        region_extension_bp=analysis_config.extend,
        strand_specific=analysis_config.strand_specific,
        sequence_accumulation=analysis_config.sequence_accumulation,
        query_motif_name=analysis_config.one_to_all,
        plot_config=plot_config,
        use_remap=analysis_config.remap,
        remap_gene=analysis_config.remap_gene,
        signal_accumulation=bool(analysis_config.all_to_one),
        signal_zarr_path=analysis_config.signal_zarr,
        signal_celltype_id=analysis_config.signal_celltype_id,
        p_value_threshold=analysis_config.p_value_threshold,
        collect_hits=analysis_config.collect_hits,
        use_pvalue_mapping=not analysis_config.no_pvalue_mapping,
        score_threshold=analysis_config.score_threshold,
        use_tiling=analysis_config.tiling,
        tile_size=analysis_config.tile_size,
        tile_extension_bp=analysis_config.tile_extension,
        use_torch_compile=analysis_config.torch_compile,
        hit_aggregation_mode=analysis_config.hit_aggregation,
        aligned_motif_path=analysis_config.aligned_motif_path,
        use_aligned_motifs=not analysis_config.no_aligned_motifs,
        require_aligned_motifs=not analysis_config.allow_pwm_fallback,
        build_hit_cache=analysis_config.build_hit_cache,
        hit_cache_path=analysis_config.hit_cache,
        cache_output_path=analysis_config.cache_output,
        command_argv=sys.argv[1:],
    )


if __name__ == "__main__":
    main()
