"""Processing functions for chromosome and region-based motif analysis.

This module contains functions for processing genomic data including whole chromosome
analysis and narrowPeak region-specific analysis.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .sequence_io import (
    CelltypeDenseZarrIO,
    SequenceDenseZarrIO,
    extract_sequences_from_regions,
    read_narrowpeak,
    reverse_complement_batch,
)

# Import functions will be resolved at runtime to avoid circular imports

logger = logging.getLogger(__name__)

# Global flag to track if torch.compile is enabled
_USE_TORCH_COMPILE = False
_compiled_conv_fn = None
_compiled_accumulation_fns = {}


def _scan_motifs_conv1d(sequences, kernels):
    """
    Core motif scanning function using 1D convolution.

    This function is designed to be compiled with torch.compile for better performance.
    It performs the core PWM scanning operation via convolution.

    Args:
        sequences: Tensor of shape (batch, 4, length) - one-hot encoded sequences
        kernels: Tensor of shape (n_motifs, 4, motif_length) - motif PWMs

    Returns:
        Tensor of shape (batch, n_motifs, seq_length - motif_length + 1) - motif scores
    """
    raw_scores = F.conv1d(sequences, kernels, padding="valid")
    return F.relu(raw_scores)


def _get_compiled_conv_fn():
    """Get or create compiled version of convolution function."""
    global _compiled_conv_fn
    if _compiled_conv_fn is None and _USE_TORCH_COMPILE:
        # Check if torch.compile is available (requires PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                "torch.compile not available (requires PyTorch 2.0+). "
                "Falling back to uncompiled version."
            )
            return _scan_motifs_conv1d
        try:
            _compiled_conv_fn = torch.compile(_scan_motifs_conv1d, mode="reduce-overhead")
            logger.info("Compiled motif scanning convolution with torch.compile")
        except Exception as e:
            logger.warning(
                f"Failed to compile convolution function: {e}. Using uncompiled version."
            )
            _compiled_conv_fn = _scan_motifs_conv1d
    return _compiled_conv_fn if _USE_TORCH_COMPILE and _compiled_conv_fn else _scan_motifs_conv1d


def set_torch_compile(enabled=True):
    """
    Enable or disable torch.compile optimization.

    Args:
        enabled (bool): If True, enable torch.compile for core operations.
                       Note: This requires PyTorch 2.0+ and may have a warmup overhead.
    """
    global _USE_TORCH_COMPILE, _compiled_conv_fn, _compiled_accumulation_fns
    _USE_TORCH_COMPILE = enabled
    # Clear compiled cache when toggling
    _compiled_conv_fn = None
    _compiled_accumulation_fns = {}
    if enabled:
        logger.info("torch.compile enabled for core computations")
    else:
        logger.info("torch.compile disabled")


def process_chromosome_sequences(
    chromosome_name,
    chromosome_data,
    motif_parameters,
    batch_size=100,
    sequence_length=10000,
    device="cuda",
    dtype=torch.bfloat16,
    enable_accumulation=True,
    analysis_window_size=500,
    strand_specific=False,
    sequence_accumulation=False,
    query_motif_index=None,
    signal_accumulation=False,
    signal_data=None,
    p_value_threshold="p0.0001",
):
    """
    Processes a single chromosome through motif scanning and pattern accumulation.

    This function handles the complete analysis pipeline for one chromosome:
    segmenting into manageable chunks, performing PWM scanning, counting hits,
    and accumulating spatial patterns according to the specified analysis mode.

    Args:
        chromosome_name (str): Name of the chromosome being processed (e.g., 'chr1').
        chromosome_data (np.ndarray): One-hot encoded chromosome sequence with shape (length, 4).
        motif_parameters (tuple): Motif data including kernels, thresholds, and metadata.
        batch_size (int): Number of sequence segments to process simultaneously.
        sequence_length (int): Length of each sequence segment in base pairs.
        device (str): PyTorch computation device ('cuda' or 'cpu').
        dtype (torch.dtype): Precision for computations (bfloat16, float16, or float32).
        enable_accumulation (bool): Whether to perform spatial pattern accumulation.
        analysis_window_size (int): Half-width for spatial analysis windows.
        strand_specific (bool): If True, only analyze forward strand.
        sequence_accumulation (bool): If True, accumulate DNA content instead of scores.
        query_motif_index (int, optional): Index for single-motif analysis mode.
        signal_accumulation (bool): If True, accumulate signal data around motif hits.
        signal_data (np.ndarray, optional): Signal values with same shape as chromosome_data.

    Returns:
        tuple: (hit_counts_dict, processing_time_seconds, accumulation_tensor)
            - hit_counts_dict: Counts of significant hits at different p-value thresholds
            - processing_time_seconds: Total time spent on motif scanning
            - accumulation_tensor: Spatial patterns tensor (None if accumulation disabled)
    """
    (
        motif_names,
        motif_kernels,
        _,
        score_thresholds,
        _,
        _,
    ) = motif_parameters

    chromosome_length = chromosome_data.shape[0]
    n_motifs = len(motif_names)

    logger.info(f"  Processing {chromosome_name}: {chromosome_length:,} bp")

    # Segment chromosome into manageable chunks
    n_sequence_chunks = chromosome_length // sequence_length
    usable_chromosome_length = n_sequence_chunks * sequence_length

    if usable_chromosome_length == 0:
        logger.warning(
            f"Chromosome {chromosome_name} too short for segment length {sequence_length}"
        )
        return None, 0, None

    # Reshape chromosome data for batch processing
    chromosome_segments = chromosome_data[:usable_chromosome_length].reshape(
        n_sequence_chunks, sequence_length, 4
    )

    # Determine signal dimensions if signal accumulation is enabled
    n_signal_dimensions = None
    if signal_accumulation and signal_data is not None:
        if signal_data.ndim == 2:
            # 2D signal: (seq_length, n_celltypes)
            n_signal_dimensions = signal_data.shape[1]
        elif signal_data.ndim == 1:
            # 1D signal: (seq_length,)
            n_signal_dimensions = 1
        else:
            raise ValueError(f"Signal data must have 1 or 2 dimensions, got {signal_data.ndim}")

    # Initialize accumulation tensor based on analysis mode
    from .accumulation import initialize_analysis_tensor

    results_tensor = initialize_analysis_tensor(
        enable_accumulation,
        sequence_accumulation,
        query_motif_index,
        n_motifs,
        analysis_window_size,
        device,
        dtype,
        motif_names,
        strand_specific,
        signal_accumulation,
        n_signal_dimensions,
    )

    # Track hit statistics across different significance thresholds
    # Only include thresholds that are actually available in score_thresholds
    available_thresholds = ["p0.001", "p0.0005", "p0.0001"]
    hit_statistics = {
        threshold: 0 for threshold in available_thresholds if threshold in score_thresholds
    }
    scan_start_time = time.time()
    n_processing_batches = (n_sequence_chunks + batch_size - 1) // batch_size

    # Process chromosome in batches
    for batch_index in tqdm(range(n_processing_batches), desc=f"Scanning {chromosome_name}"):
        batch_start = batch_index * batch_size
        batch_end = min(batch_start + batch_size, n_sequence_chunks)

        # Load current batch to GPU
        current_batch = torch.tensor(
            chromosome_segments[batch_start:batch_end], dtype=dtype, device=device
        )

        # Define strands to analyze (forward only, or forward + reverse)
        analysis_strands = [("forward", current_batch, 0)]
        if not strand_specific:
            reverse_complement_batch_data = reverse_complement_batch(current_batch)
            analysis_strands.append(("reverse", reverse_complement_batch_data, n_motifs))

        # Process each strand
        for strand_name, strand_sequences, motif_index_offset in analysis_strands:
            # Prepare tensors for convolution (PyTorch expects channel-first format)
            sequences_for_convolution = strand_sequences.permute(0, 2, 1)  # (batch, 4, length)
            kernels_for_convolution = motif_kernels.permute(0, 2, 1)  # (n_motifs, 4, motif_length)

            # Perform PWM scanning via convolution (use compiled version if available)
            conv_fn = _get_compiled_conv_fn()
            motif_scores = conv_fn(sequences_for_convolution, kernels_for_convolution)

            # Count significant hits at different thresholds
            for p_value_key in hit_statistics:
                current_threshold = score_thresholds[p_value_key]
                significant_hits = motif_scores > current_threshold.unsqueeze(0).unsqueeze(-1)
                hit_statistics[p_value_key] += significant_hits.sum().item()

            # Perform spatial pattern accumulation if enabled
            if enable_accumulation:
                # Prepare signal data for current batch if signal accumulation is enabled
                current_signal_data = None
                if signal_accumulation and signal_data is not None:
                    batch_start_bp = batch_start * sequence_length
                    batch_end_bp = batch_end * sequence_length

                    if signal_data.ndim == 2:
                        # 2D signal: (seq_length, n_celltypes)
                        batch_signal = signal_data[
                            batch_start_bp:batch_end_bp
                        ]  # (batch_length, n_celltypes)
                        current_signal_data = torch.tensor(
                            batch_signal.reshape(batch_end - batch_start, sequence_length, -1),
                            dtype=dtype,
                            device=device,
                        )
                    else:
                        # 1D signal: (seq_length,)
                        current_signal_data = torch.tensor(
                            signal_data[batch_start_bp:batch_end_bp].reshape(
                                batch_end - batch_start, sequence_length
                            ),
                            dtype=dtype,
                            device=device,
                        )

                from .accumulation import apply_accumulation_strategy

                results_tensor = apply_accumulation_strategy(
                    motif_scores,
                    results_tensor,
                    score_thresholds,
                    analysis_window_size,
                    device,
                    sequence_accumulation,
                    query_motif_index,
                    strand_sequences,
                    motif_index_offset,
                    n_motifs,
                    signal_accumulation,
                    current_signal_data,
                    p_value_threshold,
                )

    total_scan_time = time.time() - scan_start_time
    scan_throughput = chromosome_length / total_scan_time / 1e6  # Mbp/s

    logger.info(f"  Scanning completed in {total_scan_time:.1f}s ({scan_throughput:.1f} Mbp/s)")
    for p_value_key in hit_statistics:
        logger.info(f"  {p_value_key}: {hit_statistics[p_value_key]:,} hits")

    return hit_statistics, total_scan_time, results_tensor


def process_narrowpeak_regions(
    narrowpeak_file_path,
    genome_zarr_path,
    motif_parameters,
    region_extension_bp=0,
    batch_size=1000,
    device="cuda",
    dtype=torch.bfloat16,
    enable_accumulation=True,
    analysis_window_size=500,
    strand_specific=False,
    sequence_accumulation=False,
    query_motif_index=None,
    signal_accumulation=False,
    signal_zarr_path=None,
    signal_celltype_id=None,
    p_value_threshold="p0.0001",
    collect_hits=False,
    output_directory=None,
    tiled_regions_df=None,
    extend_right_only=False,
    hit_aggregation_mode="max",
    return_region_stats=False,
):
    """
    Processes genomic regions defined by a narrowPeak file for motif analysis.

    This function extracts sequences from specified genomic regions (typically
    ChIP-seq peaks) and performs focused motif analysis on these regions of interest.
    This approach is more targeted than whole-genome scanning and reduces computational cost.

    Args:
        narrowpeak_file_path (str): Path to ENCODE narrowPeak format file defining regions.
        genome_zarr_path (str): Path to genome reference in Zarr format.
        motif_parameters (tuple): Motif kernels, thresholds, and associated metadata.
        region_extension_bp (int): Base pairs to extend each region in both directions.
        batch_size (int): Number of regions to process simultaneously.
        device (str): PyTorch computation device.
        dtype (torch.dtype): Numerical precision for calculations.
        enable_accumulation (bool): Whether to accumulate spatial patterns.
        analysis_window_size (int): Window size for pattern accumulation.
        strand_specific (bool): Whether to analyze only forward strand.
        sequence_accumulation (bool): Whether to accumulate sequence content.
        query_motif_index (int, optional): Index for single-motif analysis.
        signal_accumulation (bool): Whether to accumulate signal data around motif hits.
        signal_zarr_path (str, optional): Path to signal zarr for signal accumulation.
        signal_celltype_id (str, optional): For CelltypeDenseZarrIO, select a specific cell type ID.
                                          If provided, only that cell type is loaded/queried.
        collect_hits (bool): Whether to collect and save sparse motif hits.
        hit_aggregation_mode (str): How to aggregate scores per peak: "max" (default) for maximum score,
                                    or "sum" for sum of all scores above threshold.

    Returns:
        tuple: (hit_statistics, scan_time, accumulation_tensor, collected_hits)
            Hit counts, processing time, accumulated spatial patterns, and
            collected sparse hits.
    """
    motif_names, motif_kernels, _, score_thresholds, _, _ = motif_parameters
    n_motifs = len(motif_names)

    # Load sequence database
    # Note: We use per-chromosome loading in extract_sequences_from_regions for efficiency
    sequence_database = SequenceDenseZarrIO(genome_zarr_path, mode="r")

    # Load signal data if signal accumulation is enabled
    signal_database = None
    if signal_accumulation and signal_zarr_path:
        from .data_utils import load_signal_database

        signal_database = load_signal_database(signal_zarr_path, celltype_id=signal_celltype_id)

    # Get available chromosomes from both databases
    available_chroms = set(sequence_database.chroms)
    if signal_database:
        available_chroms = available_chroms.intersection(set(signal_database.chroms))

    logger.info(f"Available chromosomes for analysis: {sorted(available_chroms)}")

    # Load genomic regions and filter by available chromosomes
    # Use tiled regions if provided, otherwise read from narrowPeak file
    if tiled_regions_df is not None:
        all_peak_regions = tiled_regions_df.copy()
        logger.info("Using pre-generated tiled regions")
    else:
        all_peak_regions = read_narrowpeak(narrowpeak_file_path)
    peak_regions = (
        all_peak_regions[all_peak_regions["chrom"].isin(available_chroms)]
        .copy()
        .reset_index(drop=True)
    )

    skipped_regions = len(all_peak_regions) - len(peak_regions)
    if skipped_regions > 0:
        logger.warning(
            f"Skipped {skipped_regions} regions on chromosomes not available in datasets"
        )
    logger.info(f"Processing {len(peak_regions)} regions")

    # Process regions in batches to avoid loading all sequences into memory at once
    # For large peak sets (>10^5), extract sequences lazily during processing
    use_streaming_extraction = len(peak_regions) > 50000

    if use_streaming_extraction:
        logger.info(
            f"Large peak set detected ({len(peak_regions)} regions). "
            "Using streaming sequence extraction to reduce memory usage."
        )
        region_sequences = None  # Will extract on-demand
        region_metadata = None  # Will extract on-demand
        total_sequence_length = None  # Will compute during processing
        total_valid_regions = 0
    else:
        # For smaller peak sets, extract all sequences upfront (original behavior)
        region_sequences, region_metadata = extract_sequences_from_regions(
            sequence_database,
            peak_regions,
            region_extension_bp,
            extend_right_only=extend_right_only,
        )
        total_sequence_length = sum(info["length"] for info in region_metadata)
        total_valid_regions = len(region_metadata)
        logger.info(
            f"Extracted {len(region_sequences)} sequences: {total_sequence_length:,} bp total"
        )

    # Determine signal dimensions if signal accumulation is enabled
    n_signal_dimensions = None
    if signal_accumulation and signal_database is not None:
        # Check signal data dimensionality from the first available chromosome
        sample_chrom = list(available_chroms)[0]
        if isinstance(signal_database, CelltypeDenseZarrIO):
            # For CelltypeDenseZarrIO, get chrom_sizes from the first celltype
            chrom_sizes = getattr(signal_database, "chrom_sizes", {})
            if not chrom_sizes and hasattr(signal_database, "first_zarr"):
                # Fallback to first zarr's chrom_sizes if available
                chrom_sizes = getattr(
                    signal_database.first_zarr, "chrom_sizes", {sample_chrom: 1000}
                )
        else:
            chrom_sizes = signal_database.chrom_sizes

        from .data_utils import get_signal_data

        sample_signal = get_signal_data(
            signal_database,
            sample_chrom,
            0,
            min(1000, chrom_sizes.get(sample_chrom, 1000)),
            celltype_id=signal_celltype_id,
        )
        if sample_signal.ndim == 2:
            n_signal_dimensions = sample_signal.shape[1]
        elif sample_signal.ndim == 1:
            n_signal_dimensions = 1
        else:
            raise ValueError(f"Signal data must have 1 or 2 dimensions, got {sample_signal.ndim}")
        logger.info(f"Detected signal data with {n_signal_dimensions} dimensions")

    # Initialize accumulation tensor for the analysis
    from .accumulation import initialize_analysis_tensor

    results_tensor = initialize_analysis_tensor(
        enable_accumulation,
        sequence_accumulation,
        query_motif_index,
        n_motifs,
        analysis_window_size,
        device,
        dtype,
        motif_names,
        strand_specific,
        signal_accumulation,
        n_signal_dimensions,
    )

    # Track hit statistics across different significance thresholds
    # Only include thresholds that are actually available in score_thresholds
    available_thresholds = ["p0.001", "p0.0005", "p0.0001"]
    hit_statistics = {
        threshold: 0 for threshold in available_thresholds if threshold in score_thresholds
    }

    # Initialize streaming hit collection to avoid OOM
    collected_hits_manager = None
    if collect_hits:
        # Use tiled regions if provided, otherwise read narrowPeak file
        # For saving, use original boundaries (not extended)
        peak_regions_for_streaming = peak_regions.copy()
        # For tiling mode, region_extension_bp is used for scanning but not saved.
        # Save original tile boundaries (no extension in saved coordinates).
        saved_extension = 0 if tiled_regions_df is not None else region_extension_bp

        from .collection import StreamingHitCollector

        collected_hits_manager = StreamingHitCollector(
            device=device,
            batch_buffer_size=batch_size * n_motifs,
            save_every_n_peaks=1000,  # Save every 1000 peaks to prevent OOM
            output_path=f"{output_directory}/motif_hits.zarr",
            peaks_df=peak_regions_for_streaming,
            motif_names=motif_names,
            p_value_threshold=p_value_threshold,
            region_extension_bp=saved_extension,  # Use 0 for tiling (original boundaries saved)
        )

    scan_start_time = time.time()

    # Calculate number of batches based on peak regions (not pre-extracted sequences)
    n_processing_batches = (len(peak_regions) + batch_size - 1) // batch_size

    # Track total sequence length for reporting
    if total_sequence_length is None:
        total_sequence_length = 0

    # Process regions in batches
    for batch_index in tqdm(range(n_processing_batches), desc="Processing narrowPeak regions"):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(peak_regions))
        current_batch_peaks = peak_regions.iloc[batch_start:batch_end]

        # Extract sequences for current batch (on-demand for streaming mode)
        if use_streaming_extraction:
            # Extract sequences on-demand for this batch only
            batch_sequences, batch_metadata = extract_sequences_from_regions(
                sequence_database,
                current_batch_peaks,
                region_extension_bp,
                extend_right_only=extend_right_only,
            )
            current_batch_sequences = batch_sequences
            # Update total sequence length
            total_sequence_length += sum(info["length"] for info in batch_metadata)
            total_valid_regions += len(batch_metadata)
        else:
            # Use pre-extracted sequences (original behavior)
            current_batch_sequences = region_sequences[batch_start:batch_end]
            if region_metadata:
                batch_metadata = region_metadata[batch_start:batch_end]
            else:
                batch_metadata = []

        if not current_batch_sequences:
            continue

        # Pad sequences to uniform length for efficient batch processing
        max_sequence_length = max(seq.shape[0] for seq in current_batch_sequences)
        padded_batch_sequences = []

        for sequence in current_batch_sequences:
            if sequence.shape[0] < max_sequence_length:
                padding_needed = max_sequence_length - sequence.shape[0]
                padded_sequence = np.pad(sequence, ((0, padding_needed), (0, 0)), mode="constant")
                padded_batch_sequences.append(padded_sequence)
            else:
                padded_batch_sequences.append(sequence)

        batch_tensor = torch.tensor(np.array(padded_batch_sequences), dtype=dtype, device=device)

        # Save batch size before cleanup (needed for hit collection)
        peaks_in_batch = len(current_batch_sequences)

        # Clean up batch sequences immediately to free memory (important for streaming mode)
        # Note: Keep batch_metadata as it's needed for signal accumulation later
        if use_streaming_extraction:
            del current_batch_sequences, padded_batch_sequences
            import gc

            gc.collect()

        # Analyze both strands unless strand-specific mode is enabled
        analysis_strands = [("forward", batch_tensor, 0)]
        if not strand_specific:
            reverse_batch = reverse_complement_batch(batch_tensor)
            analysis_strands.append(("reverse", reverse_batch, n_motifs))

        # Process each strand
        for strand_name, strand_data, motif_offset in analysis_strands:
            # Perform motif scanning
            sequences_for_conv = strand_data.permute(0, 2, 1)
            kernels_for_conv = motif_kernels.permute(0, 2, 1)
            # Use compiled version if available
            conv_fn = _get_compiled_conv_fn()
            motif_scores = conv_fn(sequences_for_conv, kernels_for_conv)

            # Update hit statistics
            for p_value_threshold_key in hit_statistics:
                threshold_tensor = score_thresholds[p_value_threshold_key]
                hits_above_threshold = motif_scores > threshold_tensor.unsqueeze(0).unsqueeze(-1)
                hit_statistics[p_value_threshold_key] += hits_above_threshold.sum().item()

            # Collect sparse hits if enabled
            if collect_hits:
                # Use actual peak indices from the DataFrame, not just batch indices
                peak_indices = list(range(batch_start, batch_end))

                # Select aggregation function based on mode
                if hit_aggregation_mode == "sum":
                    from .collection import collect_sum_hits_per_peak

                    batch_hits_tensor = collect_sum_hits_per_peak(
                        motif_scores,
                        score_thresholds,
                        peak_indices,
                        n_motifs,
                        p_value_threshold,
                        collected_hits_manager.get_batch_buffer(),
                    )
                else:  # default to "max"
                    from .collection import collect_max_hits_per_peak

                    batch_hits_tensor = collect_max_hits_per_peak(
                        motif_scores,
                        score_thresholds,
                        peak_indices,
                        n_motifs,
                        p_value_threshold,
                        collected_hits_manager.get_batch_buffer(),
                    )
                # peaks_in_batch was already computed before cleanup
                collected_hits_manager.add_batch_hits(batch_hits_tensor, peaks_in_batch)

            # Accumulate spatial patterns
            if enable_accumulation:
                # Extract signal data for current batch regions if signal accumulation is enabled
                current_signal_data = None
                if signal_accumulation and signal_database:
                    # Extract signal for each region in the batch
                    batch_signal_data = []
                    # Use peaks_in_batch to iterate (sequences may be deleted in streaming mode)
                    for seq_idx in range(peaks_in_batch):
                        # Get region info - use batch_metadata if streaming, otherwise use full region_metadata
                        if use_streaming_extraction:
                            region_info = batch_metadata[seq_idx]
                        else:
                            region_info = region_metadata[batch_start + seq_idx]
                        from .data_utils import get_signal_data

                        signal_track = get_signal_data(
                            signal_database,
                            region_info["chrom"],
                            region_info["start"],
                            region_info["end"],
                            celltype_id=signal_celltype_id,
                        )

                        # Handle padding for both 1D and 2D signal data
                        if signal_track.ndim == 1:
                            # 1D signal: (seq_length,)
                            if signal_track.shape[0] < max_sequence_length:
                                padding_needed = max_sequence_length - signal_track.shape[0]
                                padded_signal = np.pad(
                                    signal_track, (0, padding_needed), mode="constant"
                                )
                            else:
                                padded_signal = signal_track[:max_sequence_length]
                        else:
                            # 2D signal: (seq_length, n_celltypes)
                            if signal_track.shape[0] < max_sequence_length:
                                padding_needed = max_sequence_length - signal_track.shape[0]
                                padded_signal = np.pad(
                                    signal_track,
                                    ((0, padding_needed), (0, 0)),
                                    mode="constant",
                                )
                            else:
                                padded_signal = signal_track[:max_sequence_length]

                        batch_signal_data.append(padded_signal)

                    current_signal_data = torch.tensor(
                        np.array(batch_signal_data), dtype=dtype, device=device
                    )

                from .accumulation import apply_accumulation_strategy

                results_tensor = apply_accumulation_strategy(
                    motif_scores,
                    results_tensor,
                    score_thresholds,
                    analysis_window_size,
                    device,
                    sequence_accumulation,
                    query_motif_index,
                    strand_data,
                    motif_offset,
                    n_motifs,
                    signal_accumulation,
                    current_signal_data,
                    p_value_threshold,
                )

    total_scan_time = time.time() - scan_start_time
    if total_sequence_length > 0:
        throughput = total_sequence_length / total_scan_time / 1e6
        logger.info(
            f"Region scanning completed in {total_scan_time:.1f}s ({throughput:.1f} Mbp/s, {total_sequence_length:,} bp total)"
        )
    else:
        logger.info(f"Region scanning completed in {total_scan_time:.1f}s")
    for p_value_key in hit_statistics:
        logger.info(f"  {p_value_key}: {hit_statistics[p_value_key]:,} hits")

    # Finalize collected hits if enabled
    collected_hits = None
    if collect_hits:
        total_hits = collected_hits_manager.finalize()
        logger.info(f"Streaming collection finalized. Total hits: {total_hits:,}")
        logger.info(
            f"Final hit collector memory usage: {collected_hits_manager.get_memory_usage_mb():.1f} MB"
        )
        # For compatibility with existing save logic, return empty list since hits are already saved
        collected_hits = []

    if return_region_stats:
        region_stats = {
            "n_input_rows": int(len(all_peak_regions)),
            "n_chrom_filtered_regions": int(len(peak_regions)),
            "n_skipped_unavailable_chrom_regions": int(skipped_regions),
            "n_valid_regions": int(total_valid_regions),
            "n_dropped_invalid_regions": int(len(peak_regions) - total_valid_regions),
            "bp_scanned": int(total_sequence_length),
            "region_extension_bp": int(region_extension_bp),
            "extend_right_only": bool(extend_right_only),
        }
        return (
            hit_statistics,
            total_scan_time,
            results_tensor,
            collected_hits,
            region_stats,
        )

    return hit_statistics, total_scan_time, results_tensor, collected_hits
