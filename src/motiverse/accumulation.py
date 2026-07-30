"""Core accumulation algorithms for motif analysis.

This module contains the core functions for accumulating spatial patterns around
transcription factor binding sites, including co-occurrence analysis, query-centered
analysis, signal accumulation, and sequence content analysis.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def accumulate_motif_cooccurrences(
    motif_scores,
    cooccurrence_tensor,
    score_thresholds,
    window_size=500,
    p_value_threshold="p0.0001",
):
    """
    Computes motif co-occurrence patterns using vectorized operations.

    This function identifies significant motif hits and accumulates the scores
    of all motifs within a sliding window around each hit. The result captures
    spatial relationships between different transcription factor binding motifs.

    Args:
        motif_scores (torch.Tensor): Motif scanning scores with shape (batch, n_motifs, seq_length).
            Each value represents the log-likelihood score of a motif at that position.
        cooccurrence_tensor (torch.Tensor): Accumulation tensor with shape (n_motifs, n_motifs, window_width).
            This tensor is updated in-place with co-occurrence patterns.
        score_thresholds (dict): P-value threshold tensors for motif significance detection.
            Keys are p-value strings (e.g., 'p0.0001'), values are threshold tensors.
        window_size (int): Half-width of the analysis window in base pairs.
            Total window width is 2*window_size + 1.
        p_value_threshold (str): P-value threshold for motif hits.
    Returns:
        torch.Tensor: Updated co-occurrence tensor with accumulated spatial relationships.

    Note:
        Uses unfold operation for efficient sliding window computation and einsum
        for vectorized matrix multiplication across all motif pairs.
    """
    batch_size, n_motifs, seq_length = motif_scores.shape
    significance_threshold = score_thresholds[p_value_threshold]

    # Identify significant motif hits across all sequences
    significant_hits_mask = motif_scores > significance_threshold.unsqueeze(0).unsqueeze(-1)

    # Create sliding windows of motif scores for co-occurrence analysis
    full_window_width = 2 * window_size + 1

    # Check if sequences are long enough for the requested window
    if seq_length < full_window_width:
        return cooccurrence_tensor

    score_windows = motif_scores.unfold(2, full_window_width, 1)

    # Extract hit positions that have complete windows (avoid edge effects)
    valid_hit_positions = significant_hits_mask[:, :, window_size : seq_length - window_size].to(
        motif_scores.dtype
    )

    # Compute co-occurrence: for each hit of motif i, sum scores of motif j in the window
    batch_cooccurrences = torch.einsum("bip,bjpk->bijk", valid_hit_positions, score_windows)
    total_cooccurrences = batch_cooccurrences.sum(dim=0)
    cooccurrence_tensor += total_cooccurrences

    return cooccurrence_tensor


def accumulate_around_query_motif(
    motif_scores,
    target_aggregation_tensor,
    score_thresholds,
    query_motif_index,
    window_size=500,
    device="cuda",
    p_value_threshold="p0.0001",
):
    """
    Aggregates motif scores around binding sites of a single query motif.

    This function focuses analysis on one specific motif of interest and examines
    what other motifs co-occur in its vicinity. This is useful for understanding
    the regulatory context around a particular transcription factor.

    Args:
        motif_scores (torch.Tensor): Motif scanning scores with shape (batch, n_motifs, seq_length).
        target_aggregation_tensor (torch.Tensor): Accumulation tensor with shape (n_motifs, window_width).
            Updated in-place with aggregated scores around query motif hits.
        score_thresholds (dict): P-value threshold tensors for significance detection.
        query_motif_index (int): Index of the motif to use as anchor points for aggregation.
        window_size (int): Half-width of the aggregation window in base pairs.
        device (str): PyTorch device for computation ('cuda' or 'cpu').
        p_value_threshold (str): P-value threshold for motif hits.
    Returns:
        torch.Tensor: Updated aggregation tensor with patterns around query motif.

    Note:
        Uses advanced indexing for efficient extraction of windows around
        sparse hit locations, avoiding the need to process all positions.
    """
    batch_size, n_motifs, seq_length = motif_scores.shape
    target_threshold = score_thresholds[p_value_threshold][query_motif_index]

    # Find all significant hits of the query motif
    query_motif_scores = motif_scores[:, query_motif_index, :]
    target_hits_mask = query_motif_scores > target_threshold

    # Get coordinates of all query motif hits
    hit_batch_indices, hit_positions = torch.where(target_hits_mask)
    if len(hit_batch_indices) == 0:
        return target_aggregation_tensor

    # Filter hits that have complete windows (avoid sequence boundaries)
    valid_position_mask = (hit_positions >= window_size) & (
        hit_positions < seq_length - window_size
    )
    valid_batch_indices = hit_batch_indices[valid_position_mask]
    valid_hit_positions = hit_positions[valid_position_mask]
    if len(valid_batch_indices) == 0:
        return target_aggregation_tensor

    # Create indices for extracting windows around each valid hit
    relative_positions = torch.arange(-window_size, window_size + 1, device=device)
    absolute_window_positions = valid_hit_positions.unsqueeze(1) + relative_positions.unsqueeze(0)

    # Use advanced indexing to extract all windows efficiently
    batch_indices_expanded = valid_batch_indices.unsqueeze(1).unsqueeze(2)
    motif_indices_expanded = torch.arange(n_motifs, device=device).unsqueeze(0).unsqueeze(2)
    position_indices_expanded = absolute_window_positions.unsqueeze(1)
    # Extract scores from all windows around target hits
    extracted_windows = motif_scores[
        batch_indices_expanded, motif_indices_expanded, position_indices_expanded
    ]
    aggregated_windows = extracted_windows.sum(dim=0)
    target_aggregation_tensor += aggregated_windows

    return target_aggregation_tensor


def accumulate_signal_around_hits(
    signal_data,
    motif_scores,
    signal_aggregation_tensor,
    score_thresholds,
    window_size=500,
    p_value_threshold="p0.0001",
):
    """
    Accumulates signal data around transcription factor binding sites.

    This function aggregates signal values (e.g., ATAC-seq, ChIP-seq) around motif
    binding sites instead of motif scores or sequence content. Useful for understanding
    the chromatin accessibility or histone modifications around TF binding sites.

    Now supports 2D signal data where the second dimension represents different
    cell types or factors that are processed independently.

    Args:
        signal_data (torch.Tensor): Signal values with shape (batch, seq_length) for 1D signal
            or (batch, seq_length, n_celltypes) for 2D signal.
        motif_scores (torch.Tensor): Motif scanning scores with shape (batch, n_motifs, score_length).
        signal_aggregation_tensor (torch.Tensor): Accumulation tensor with shape (n_motifs, window_width)
            for 1D signal or (n_motifs, window_width, n_celltypes) for 2D signal.
            Updated in-place with signal values around binding sites.
        score_thresholds (dict): P-value threshold tensors for motif significance.
        window_size (int): Half-width of the signal window to extract.
        p_value_threshold (str): P-value threshold for motif hits.
    Returns:
        torch.Tensor: Updated signal aggregation tensor showing signal patterns.

    Note:
        Handles length mismatches between signal and score tensors that can
        occur due to motif length variations during convolution operations.
        For 2D signal, processes each cell type/factor dimension independently.
    """
    batch_size, n_motifs, score_length = motif_scores.shape
    significance_threshold = score_thresholds[p_value_threshold]

    # Identify significant motif binding sites
    significant_hits_mask = motif_scores > significance_threshold.unsqueeze(0).unsqueeze(-1)

    # Handle both 1D and 2D signal data
    if signal_data.dim() == 2:
        # 1D signal: (batch, seq_length)
        _, seq_length = signal_data.shape

        # Create sliding windows of signal data
        full_window_width = 2 * window_size + 1

        # Check if sequence is long enough for the requested window
        if seq_length < full_window_width:
            return signal_aggregation_tensor

        signal_windows = signal_data.unfold(1, full_window_width, 1)
        unfolded_length = signal_windows.shape[1]

        # Handle potential length differences between scores and signal
        length_difference = score_length - unfolded_length
        if length_difference > 0:
            # Scores are longer - trim the hits mask to match signal windows
            trim_start = length_difference // 2
            trim_end = score_length - (length_difference - trim_start)
            valid_hits = significant_hits_mask[:, :, trim_start:trim_end].to(motif_scores.dtype)
        else:
            # Signal is longer - pad the hits mask
            pad_start = -length_difference // 2
            pad_end = -length_difference - pad_start
            valid_hits = F.pad(significant_hits_mask, (pad_start, pad_end), value=0).to(
                motif_scores.dtype
            )

        # Aggregate signal around hits using einsum: hits × signal_windows → motif_profiles
        batch_signal_accumulation = torch.einsum("bip,bpk->bik", valid_hits, signal_windows)
        total_signal_accumulation = batch_signal_accumulation.sum(dim=0)
        signal_aggregation_tensor += total_signal_accumulation.to(signal_aggregation_tensor.dtype)

    elif signal_data.dim() == 3:
        # 2D signal: (batch, seq_length, n_celltypes)
        _, seq_length, n_celltypes = signal_data.shape

        # Create sliding windows of signal data for each cell type
        full_window_width = 2 * window_size + 1

        # Check if sequence is long enough for the requested window
        if seq_length < full_window_width:
            return signal_aggregation_tensor

        # Unfold along sequence dimension: (batch, unfolded_length, n_celltypes, window_width)
        signal_windows = signal_data.unfold(1, full_window_width, 1)
        unfolded_length = signal_windows.shape[1]

        # Handle potential length differences between scores and signal
        length_difference = score_length - unfolded_length
        if length_difference > 0:
            # Scores are longer - trim the hits mask to match signal windows
            trim_start = length_difference // 2
            trim_end = score_length - (length_difference - trim_start)
            valid_hits = significant_hits_mask[:, :, trim_start:trim_end].to(motif_scores.dtype)
        else:
            # Signal is longer - pad the hits mask
            pad_start = -length_difference // 2
            pad_end = -length_difference - pad_start
            valid_hits = F.pad(significant_hits_mask, (pad_start, pad_end), value=0).to(
                motif_scores.dtype
            )

        # Aggregate signal around hits for each cell type independently
        # valid_hits: (batch, n_motifs, positions)
        # signal_windows: (batch, positions, n_celltypes, window_width)
        # Result: (batch, n_motifs, n_celltypes, window_width)
        batch_signal_accumulation = torch.einsum("bip,bpck->bick", valid_hits, signal_windows)
        total_signal_accumulation = batch_signal_accumulation.sum(
            dim=0
        )  # Sum over batches -> (n_motifs, n_celltypes, window_width)

        # Reorder dimensions to match signal_aggregation_tensor: (n_motifs, window_width, n_celltypes)
        total_signal_accumulation = total_signal_accumulation.permute(0, 2, 1)
        signal_aggregation_tensor += total_signal_accumulation.to(signal_aggregation_tensor.dtype)
    else:
        raise ValueError(f"Signal data must have 2 or 3 dimensions, got {signal_data.dim()}")

    return signal_aggregation_tensor


def accumulate_sequences_around_hits(
    dna_sequences,
    motif_scores,
    sequence_aggregation_tensor,
    score_thresholds,
    window_size=500,
    p_value_threshold="p0.0001",
):
    """
    Accumulates one-hot encoded DNA sequences around transcription factor binding sites.

    This function reveals the sequence composition and preferences around motif
    binding sites by aggregating the actual DNA content rather than motif scores.
    Useful for discovering sequence context and identifying potential co-binding motifs.

    Args:
        dna_sequences (torch.Tensor): One-hot encoded DNA with shape (batch, seq_length, 4).
            Last dimension represents [A, C, G, T] nucleotides.
        motif_scores (torch.Tensor): Motif scanning scores with shape (batch, n_motifs, score_length).
        sequence_aggregation_tensor (torch.Tensor): Accumulation tensor with shape (n_motifs, window_width, 4).
            Updated in-place with nucleotide frequencies around binding sites.
        score_thresholds (dict): P-value threshold tensors for motif significance.
        window_size (int): Half-width of the sequence window to extract.
        p_value_threshold (str): P-value threshold for motif hits.
    Returns:
        torch.Tensor: Updated sequence aggregation tensor showing nucleotide patterns.

    Note:
        Handles length mismatches between sequence and score tensors that can
        occur due to motif length variations during convolution operations.
    """
    batch_size, n_motifs, score_length = motif_scores.shape
    _, seq_length, _ = dna_sequences.shape
    significance_threshold = score_thresholds[p_value_threshold]

    # Identify significant motif binding sites
    significant_hits_mask = motif_scores > significance_threshold.unsqueeze(0).unsqueeze(-1)

    # Create sliding windows of DNA sequence
    full_window_width = 2 * window_size + 1

    # Check if sequence is long enough for the requested window
    if seq_length < full_window_width:
        return sequence_aggregation_tensor

    sequence_windows = dna_sequences.unfold(1, full_window_width, 1)
    unfolded_length = sequence_windows.shape[1]

    # Handle potential length differences between scores and sequences
    # This can occur when motifs have different lengths after padding
    length_difference = score_length - unfolded_length
    if length_difference > 0:
        # Scores are longer - trim the hits mask to match sequence windows
        trim_start = length_difference // 2
        trim_end = score_length - (length_difference - trim_start)
        valid_hits = significant_hits_mask[:, :, trim_start:trim_end].to(motif_scores.dtype)
    else:
        # Sequences are longer - pad the hits mask
        pad_start = -length_difference // 2
        pad_end = -length_difference - pad_start
        valid_hits = F.pad(significant_hits_mask, (pad_start, pad_end), value=0).to(
            motif_scores.dtype
        )

    # Reorganize sequence windows for efficient computation: (batch, position, nucleotide, window)
    sequence_windows_reordered = sequence_windows.to(motif_scores.dtype).permute(0, 1, 3, 2)

    # Aggregate sequences around hits using einsum: hits × sequence_windows → motif_profiles
    batch_sequence_accumulation = torch.einsum(
        "bip,bpck->bick", valid_hits, sequence_windows_reordered
    )
    total_sequence_accumulation = batch_sequence_accumulation.sum(dim=0)
    sequence_aggregation_tensor += total_sequence_accumulation.to(sequence_aggregation_tensor.dtype)

    return sequence_aggregation_tensor


def initialize_analysis_tensor(
    enable_accumulation,
    sequence_accumulation,
    query_motif_index,
    n_motifs,
    analysis_window_size,
    device,
    dtype,
    motif_names,
    strand_specific=False,
    signal_accumulation=False,
    n_signal_dimensions=None,
):
    """
    Initializes the appropriate tensor for spatial pattern accumulation.

    Creates different tensor shapes depending on the analysis mode:
    - Co-occurrence: (n_motifs, n_motifs, window_width) for all pairwise interactions
    - Single-motif: (n_motifs, window_width) for patterns around one query motif
    - Sequence: (n_motifs, window_width, 4) for nucleotide content analysis
    - Signal 1D: (n_motifs, window_width) for 1D signal accumulation
    - Signal 2D: (n_motifs, window_width, n_celltypes) for 2D signal accumulation

    Args:
        enable_accumulation (bool): Whether accumulation is enabled.
        sequence_accumulation (bool): Whether to accumulate sequence content.
        query_motif_index (int, optional): Index for single-motif mode.
        n_motifs (int): Number of motifs in the analysis.
        analysis_window_size (int): Half-width of analysis windows.
        device (str): PyTorch device for tensor allocation.
        dtype (torch.dtype): Data type for the tensor.
        motif_names (list): Names of motifs for logging.
        strand_specific (bool): Whether analysis is strand-specific.
        signal_accumulation (bool): Whether to accumulate signal data.
        n_signal_dimensions (int, optional): Number of cell types/factors for 2D signal data.

    Returns:
        torch.Tensor or None: Initialized accumulation tensor, or None if disabled.
    """
    if not enable_accumulation:
        return None

    window_width = 2 * analysis_window_size + 1

    if signal_accumulation:
        # For signal accumulation analysis, account for reverse complement if needed
        total_motif_count = n_motifs * (1 if strand_specific else 2)

        if n_signal_dimensions is not None and n_signal_dimensions > 1:
            # 2D signal: (n_motifs, window_width, n_celltypes)
            tensor = torch.zeros(
                total_motif_count,
                window_width,
                n_signal_dimensions,
                device=device,
                dtype=dtype,
            )
            logger.info(f"Initialized 2D signal accumulation tensor: {tensor.shape}")
            logger.info(f"Processing {n_signal_dimensions} cell types/factors independently")
        else:
            # 1D signal: (n_motifs, window_width)
            tensor = torch.zeros(total_motif_count, window_width, device=device, dtype=dtype)
            logger.info(f"Initialized 1D signal accumulation tensor: {tensor.shape}")

    elif sequence_accumulation:
        # For sequence content analysis, account for reverse complement if needed
        total_motif_count = n_motifs * (1 if strand_specific else 2)
        tensor = torch.zeros(total_motif_count, window_width, 4, device=device, dtype=torch.int32)
        logger.info(f"Initialized sequence content tensor: {tensor.shape}")

    elif query_motif_index is not None:
        # Single-motif analysis: aggregate all motifs around one target
        tensor = torch.zeros(n_motifs, window_width, device=device, dtype=dtype)
        logger.info(f"Initialized query-centered analysis tensor: {tensor.shape}")
        logger.info(f"Query motif: {motif_names[query_motif_index]} (index {query_motif_index})")

    else:
        # Co-occurrence analysis: all motif pairs
        logger.info(
            f"Initializing motif co-occurrence tensor: {n_motifs}x{n_motifs}x{window_width} with dtype {dtype}"
        )
        tensor = torch.zeros(n_motifs, n_motifs, window_width, device=device, dtype=dtype)
        logger.info(f"Initialized motif co-occurrence tensor: {tensor.shape}")

    logger.info(f"Analysis window: ±{analysis_window_size}bp (total {window_width}bp)")
    return tensor


def apply_accumulation_strategy(
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
    signal_accumulation=False,
    signal_data=None,
    p_value_threshold="p0.0001",
):
    """
    Applies the appropriate accumulation method based on analysis configuration.

    Routes to the correct accumulation function depending on whether the analysis
    is focused on sequence content, a single query motif, or motif co-occurrences.
    Handles strand-specific indexing for reverse complement analysis.

    Args:
        motif_scores (torch.Tensor): Motif scanning scores for current batch.
        results_tensor (torch.Tensor): Accumulation tensor to update.
        score_thresholds (dict): Significance thresholds for each p-value.
        analysis_window_size (int): Half-width of analysis windows.
        device (str): PyTorch computation device.
        sequence_accumulation (bool): Whether accumulating sequence content.
        query_motif_index (int, optional): Query motif for single-motif analysis.
        strand_sequences (torch.Tensor): DNA sequences for current strand.
        motif_index_offset (int): Offset for reverse strand motif indexing.
        n_motifs (int): Number of motifs being analyzed.
        signal_accumulation (bool): Whether accumulating signal data.
        signal_data (torch.Tensor, optional): Signal data for current strand.
        p_value_threshold (str): P-value threshold for motif hits (e.g., "p0.0001").

    Returns:
        torch.Tensor: Updated results tensor with new accumulation data.
    """
    if signal_accumulation:
        # Accumulate signal data around binding sites
        tensor_slice = slice(motif_index_offset, motif_index_offset + n_motifs)
        if signal_data is not None:
            results_tensor[tensor_slice] = accumulate_signal_around_hits(
                signal_data,
                motif_scores,
                results_tensor[tensor_slice],
                score_thresholds=score_thresholds,
                window_size=analysis_window_size,
                p_value_threshold=p_value_threshold,
            )

    elif sequence_accumulation:
        # Accumulate DNA sequence content around binding sites
        tensor_slice = slice(motif_index_offset, motif_index_offset + n_motifs)
        results_tensor[tensor_slice] = accumulate_sequences_around_hits(
            strand_sequences,
            motif_scores,
            results_tensor[tensor_slice],
            score_thresholds=score_thresholds,
            window_size=analysis_window_size,
            p_value_threshold=p_value_threshold,
        )

    elif query_motif_index is not None:
        # Single-motif analysis: aggregate around one query motif
        results_tensor = accumulate_around_query_motif(
            motif_scores,
            results_tensor,
            score_thresholds=score_thresholds,
            query_motif_index=query_motif_index,
            window_size=analysis_window_size,
            device=device,
            p_value_threshold=p_value_threshold,
        )

    else:
        # Co-occurrence analysis: all motif pairs
        results_tensor = accumulate_motif_cooccurrences(
            motif_scores,
            results_tensor,
            score_thresholds=score_thresholds,
            window_size=analysis_window_size,
            p_value_threshold=p_value_threshold,
        )

    return results_tensor
