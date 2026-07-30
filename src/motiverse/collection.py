"""Hit collection and streaming functionality for motif analysis.

This module contains classes and functions for collecting and streaming motif hits
efficiently, including the StreamingHitCollector for memory-efficient processing.
"""

import gc
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StreamingHitCollector:
    """
    Streaming collector for sparse motif hits that saves to zarr periodically to avoid memory exhaustion.
    Processes hits in chunks and saves every N peaks to prevent OOM kills.
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_buffer_size: int = 100000,
        save_every_n_peaks: int = 100000,
        output_path: str = None,
        peaks_df=None,
        motif_names: list = None,
        p_value_threshold: str = "p0.0001",
        region_extension_bp: int = 0,
    ):
        """
        Initialize the streaming hit collector.

        Args:
            device: Device to store tensors on
            batch_buffer_size: Size of buffer for batch processing
            save_every_n_peaks: How often to save to disk and clear memory
            output_path: Path to save zarr file
            peaks_df: DataFrame with peak information
            motif_names: List of motif names
            p_value_threshold: P-value threshold used
            region_extension_bp: Extension applied to peaks
        """
        self.device = device
        self.batch_buffer_size = batch_buffer_size
        self.save_every_n_peaks = save_every_n_peaks
        self.output_path = output_path
        self.peaks_df = peaks_df
        self.motif_names = motif_names
        self.p_value_threshold = p_value_threshold
        self.region_extension_bp = region_extension_bp

        # Pre-allocate batch buffer for temporary storage
        self.batch_buffer = torch.zeros((batch_buffer_size, 4), device=device, dtype=torch.float32)

        # Current chunk storage (smaller, gets saved periodically)
        self.chunk_hits = []
        self.peaks_processed = 0
        self.total_hits_saved = 0
        self.zarr_initialized = False
        self.motif_hits_io = None

    def get_batch_buffer(self) -> torch.Tensor:
        """Get the pre-allocated batch buffer for temporary storage."""
        return self.batch_buffer

    def add_batch_hits(self, batch_hits: torch.Tensor, peaks_in_batch: int):
        """
        Add hits from a batch and save to zarr if chunk is full.

        Args:
            batch_hits: Tensor of shape (n_hits, 4) with hit data
            peaks_in_batch: Number of peaks processed in this batch
        """
        if batch_hits.size(0) > 0:
            # Keep chunk data columnar until the zarr writer. This avoids a
            # row-by-row Python tuple conversion on large hit sets.
            hits_cpu = batch_hits.detach().cpu().numpy().astype(np.float32, copy=False)
            self.chunk_hits.append(hits_cpu)

        self.peaks_processed += peaks_in_batch

        # Save chunk if we've processed enough peaks
        if self.peaks_processed >= self.save_every_n_peaks:
            self._save_current_chunk()

    def _save_current_chunk(self):
        """Save current chunk of hits to zarr and clear memory."""
        if not self.chunk_hits:
            self.peaks_processed = 0  # Reset counter even if no hits
            return

        if not self.zarr_initialized:
            self._initialize_zarr()

        # Save hits to zarr
        hits_to_save = np.concatenate(self.chunk_hits, axis=0)
        logger.info(
            f"Saving {hits_to_save.shape[0]:,} hits from {self.peaks_processed:,} peaks to zarr..."
        )

        try:
            self._append_hits_to_zarr(hits_to_save)

            self.total_hits_saved += hits_to_save.shape[0]
            logger.info(f"Saved chunk. Total hits saved so far: {self.total_hits_saved:,}")
        except Exception as e:
            logger.error(f"Error saving hits chunk: {e}")

        # Clear memory
        self.chunk_hits.clear()
        self.peaks_processed = 0

        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _initialize_zarr(self):
        """Initialize zarr file for streaming writes."""
        from .motif_hits import MotifHitZarrIO

        logger.info(f"Initializing streaming zarr file at {self.output_path}")
        self.motif_hits_io = MotifHitZarrIO(self.output_path, mode="w")

        # Initialize for streaming with proper setup
        self.motif_hits_io.initialize_for_streaming(
            peaks_df=self.peaks_df,
            motif_names=self.motif_names,
            p_value_threshold=self.p_value_threshold,
            region_extension_bp=self.region_extension_bp,
        )

        self.zarr_initialized = True

    def _append_hits_to_zarr(self, hits_chunk):
        """Append hits to existing zarr structure using streaming."""
        # Use the new streaming append method
        self.motif_hits_io.append_hits(hits_chunk)

    def finalize(self):
        """Save any remaining hits and finalize the zarr file."""
        if self.chunk_hits:
            self._save_current_chunk()

        # Get final hit count from zarr metadata
        if self.zarr_initialized and self.motif_hits_io:
            final_total = self.motif_hits_io.dataset["metadata"].attrs["n_hits"]
            logger.info(f"Streaming collection complete. Total hits saved: {final_total:,}")
            return final_total
        else:
            logger.info(
                f"Streaming collection complete. Total hits saved: {self.total_hits_saved:,}"
            )
            return self.total_hits_saved

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        batch_buffer_mb = self.batch_buffer.numel() * 4 / (1024 * 1024)
        chunk_hits_mb = sum(arr.nbytes for arr in self.chunk_hits) / (1024 * 1024)
        return batch_buffer_mb + chunk_hits_mb


def collect_max_hits_per_peak(
    motif_scores,
    score_thresholds,
    peak_start_indices,
    n_motifs,
    p_value_threshold="p0.0001",
    preallocated_buffer=None,
):
    """
    Efficiently collect the position of maximum score for each motif within each peak region.
    Uses vectorized GPU operations and pre-allocated memory for optimal performance.

    Args:
        motif_scores (torch.Tensor): Motif scanning scores with shape (batch, n_motifs, seq_length).
        score_thresholds (dict): P-value threshold tensors for motif significance.
        peak_start_indices (list): Start indices for each peak in the batch.
        n_motifs (int): Number of motifs.
        p_value_threshold (str): P-value threshold for significance.
        preallocated_buffer (torch.Tensor, optional): Pre-allocated buffer for results.

    Returns:
        torch.Tensor: Tensor with shape (n_hits, 4) containing [peak_idx, motif_idx, rel_pos, score].
                     Returns only valid hits (above threshold).
    """
    significance_threshold = score_thresholds[p_value_threshold]  # (n_motifs,)
    batch_size, _, seq_length = motif_scores.shape
    device = motif_scores.device

    # Vectorized approach: find max scores and positions for all motifs in all peaks
    max_scores, max_positions = torch.max(motif_scores, dim=2)  # (batch_size, n_motifs)

    # Create significance mask - broadcast threshold across batch
    significance_mask = max_scores > significance_threshold.unsqueeze(0)  # (batch_size, n_motifs)

    # Get indices of significant hits
    peak_indices, motif_indices = torch.where(significance_mask)
    n_hits = len(peak_indices)

    if n_hits == 0:
        # Return empty tensor with correct shape
        return torch.zeros((0, 4), device=device, dtype=torch.float32)

    # Use pre-allocated buffer if provided and large enough
    if preallocated_buffer is not None and preallocated_buffer.size(0) >= n_hits:
        hits_tensor = preallocated_buffer[:n_hits]
    else:
        hits_tensor = torch.zeros((n_hits, 4), device=device, dtype=torch.float32)

    peak_index_lookup = torch.as_tensor(peak_start_indices, device=device, dtype=torch.float32)
    hits_tensor[:, 0] = peak_index_lookup[peak_indices]  # peak_idx
    hits_tensor[:, 1] = motif_indices.float()  # motif_idx
    hits_tensor[:, 2] = max_positions[peak_indices, motif_indices].float()  # rel_pos
    hits_tensor[:, 3] = max_scores[peak_indices, motif_indices]  # score

    return hits_tensor


def collect_sum_hits_per_peak(
    motif_scores,
    score_thresholds,
    peak_start_indices,
    n_motifs,
    p_value_threshold="p0.0001",
    preallocated_buffer=None,
):
    """
    Efficiently collect the sum of all scores above threshold for each motif within each peak region.
    Uses vectorized GPU operations and pre-allocated memory for optimal performance.

    Args:
        motif_scores (torch.Tensor): Motif scanning scores with shape (batch, n_motifs, seq_length).
        score_thresholds (dict): P-value threshold tensors for motif significance.
        peak_start_indices (list): Start indices for each peak in the batch.
        n_motifs (int): Number of motifs.
        p_value_threshold (str): P-value threshold for significance.
        preallocated_buffer (torch.Tensor, optional): Pre-allocated buffer for results.

    Returns:
        torch.Tensor: Tensor with shape (n_hits, 4) containing [peak_idx, motif_idx, rel_pos, score].
                     rel_pos is set to -1 to indicate this is a sum (not a single position).
                     Returns only motifs with at least one score above threshold.
    """
    significance_threshold = score_thresholds[p_value_threshold]  # (n_motifs,)
    batch_size, _, seq_length = motif_scores.shape
    device = motif_scores.device

    # Create significance mask for all positions
    # Broadcast threshold: (n_motifs,) -> (batch_size, n_motifs, seq_length)
    threshold_expanded = significance_threshold.unsqueeze(0).unsqueeze(-1)  # (1, n_motifs, 1)
    significance_mask = motif_scores > threshold_expanded  # (batch_size, n_motifs, seq_length)

    # Sum scores above threshold for each motif in each peak
    # Only sum where mask is True, set to 0 where False
    masked_scores = motif_scores * significance_mask.float()  # (batch_size, n_motifs, seq_length)
    sum_scores = masked_scores.sum(dim=2)  # (batch_size, n_motifs)

    # Find which peaks/motifs have at least one score above threshold
    has_hits = sum_scores > 0  # (batch_size, n_motifs)

    # Get indices of peaks/motifs with hits
    peak_indices, motif_indices = torch.where(has_hits)
    n_hits = len(peak_indices)

    if n_hits == 0:
        # Return empty tensor with correct shape
        return torch.zeros((0, 4), device=device, dtype=torch.float32)

    # Use pre-allocated buffer if provided and large enough
    if preallocated_buffer is not None and preallocated_buffer.size(0) >= n_hits:
        hits_tensor = preallocated_buffer[:n_hits]
    else:
        hits_tensor = torch.zeros((n_hits, 4), device=device, dtype=torch.float32)

    peak_index_lookup = torch.as_tensor(peak_start_indices, device=device, dtype=torch.float32)
    hits_tensor[:, 0] = peak_index_lookup[peak_indices]  # peak_idx
    hits_tensor[:, 1] = motif_indices.float()  # motif_idx
    hits_tensor[:, 2] = -1.0  # rel_pos = -1 indicates this is a sum (not a single position)
    hits_tensor[:, 3] = sum_scores[peak_indices, motif_indices]  # sum of scores

    return hits_tensor
