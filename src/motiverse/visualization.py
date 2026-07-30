"""Visualization functions for motif analysis results.

This module contains functions for generating plots and visualizations including
co-occurrence matrices, heatmaps, power spectra, and other analysis plots.
"""

import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .plot_utils import (
    plot_autocorrelation,
    plot_cooccurrence_matrix,
    plot_power_spectrum,
)

logger = logging.getLogger(__name__)


def generate_analysis_plots(
    results_tensor,
    motif_names,
    output_directory,
    plot_config,
    sequence_accumulation=False,
    query_motif_index=None,
    strand_specific=False,
):
    """
    Generates visualization plots for analysis results.

    Creates various plots based on the analysis type and user specifications:
    - Co-occurrence matrices for motif spatial relationships
    - Power spectrum analysis for periodic patterns
    - Autocorrelation plots for temporal dependencies
    - Quality scatter plots for motif assessment

    Args:
        results_tensor (torch.Tensor): Analysis results tensor from genome scanning.
        motif_names (list): List of motif names for labeling.
        output_directory (str): Directory to save plots.
        plot_config (dict): Configuration dictionary with plotting parameters:
            - enable_plotting (bool): Whether to generate plots
            - plot_types (list): Types of plots to generate
            - motif_pairs (list): Specific motif pairs to plot
            - smoothing_window (int): Smoothing parameter for co-occurrence plots
            - max_delay (int): Maximum delay for autocorrelation analysis
        sequence_accumulation (bool): Whether results contain sequence content.
        query_motif_index (int, optional): Index of query motif for single-motif analysis.
        strand_specific (bool): Whether analysis was strand-specific.
    """
    if not plot_config.get("enable_plotting", False) or results_tensor is None:
        return

    logger.info("Generating analysis plots...")
    plot_start_time = time.time()

    # Create plots directory
    plots_directory = Path(output_directory) / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    # Prepare extended motif names including reverse complements
    extended_motif_names = list(motif_names)
    if not strand_specific and not sequence_accumulation and query_motif_index is None:
        # For co-occurrence analysis with both strands, add reverse complement names
        extended_motif_names.extend([f"{name}_RC" for name in motif_names])

    plot_types = plot_config.get("plot_types", ["cooccurrence"])
    motif_pairs = plot_config.get("motif_pairs", [])
    smoothing_window = plot_config.get("smoothing_window", 1)
    max_delay = plot_config.get("max_delay", 500)

    # Generate co-occurrence matrix plots
    if "cooccurrence" in plot_types and query_motif_index is None and not sequence_accumulation:
        logger.info("  Generating co-occurrence matrix plot...")
        plot_cooccurrence_matrix(
            results_tensor.cpu().float(),
            motif_names_i=extended_motif_names,
            motif_names_j=extended_motif_names,
            smoothing_window=smoothing_window,
        )

    # Generate specific motif pair plots
    if motif_pairs and results_tensor.dim() >= 2:

        def resolve_extended_motif_index(motif_identifier, extended_names):
            """Helper to resolve motif indices for extended (with RC) motif names."""
            if isinstance(motif_identifier, int):
                if 0 <= motif_identifier < len(extended_names):
                    return motif_identifier
                else:
                    raise ValueError(
                        f"Index {motif_identifier} out of range [0, {len(extended_names) - 1}]"
                    )

            # Try exact match first
            try:
                return extended_names.index(motif_identifier)
            except ValueError:
                pass

            # Try prefix matching
            for i, name in enumerate(extended_names):
                if name.startswith(motif_identifier + ".") or name.startswith(
                    motif_identifier + "_"
                ):
                    return i

            raise ValueError(f"Motif '{motif_identifier}' not found in extended motif names")

        for pair_idx, (motif_i, motif_j) in enumerate(motif_pairs):
            # Resolve motif indices
            try:
                if isinstance(motif_i, str):
                    idx_i = resolve_extended_motif_index(motif_i, extended_motif_names)
                else:
                    idx_i = int(motif_i)

                if isinstance(motif_j, str):
                    idx_j = resolve_extended_motif_index(motif_j, extended_motif_names)
                else:
                    idx_j = int(motif_j)

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not resolve motif pair ({motif_i}, {motif_j}): {e}")
                continue

            # Extract curve data based on analysis type
            if query_motif_index is not None:
                # Single-motif analysis: results_tensor is (n_motifs, window_width)
                if idx_i < results_tensor.shape[0]:
                    curve_data = results_tensor[idx_i].cpu().float().numpy()
                    curve_name = f"{extended_motif_names[idx_i]}_around_target"
                else:
                    logger.warning(f"Motif index {idx_i} out of range for target analysis")
                    continue
            else:
                # Co-occurrence analysis: results_tensor is (n_motifs, n_motifs, window_width)
                if idx_i < results_tensor.shape[0] and idx_j < results_tensor.shape[1]:
                    curve_data = results_tensor[idx_i, idx_j].cpu().float().numpy()
                    curve_name = f"{extended_motif_names[idx_i]}_vs_{extended_motif_names[idx_j]}"
                else:
                    logger.warning(f"Motif indices ({idx_i}, {idx_j}) out of range")
                    continue

            # Generate power spectrum plots
            if "power_spectrum" in plot_types:
                logger.info(f"  Generating power spectrum for {curve_name}...")
                plt.figure(figsize=(10, 6))
                plot_power_spectrum(curve_data)
                plt.suptitle(f"Power Spectrum: {curve_name}")
                plt.savefig(
                    str(plots_directory / f"power_spectrum_{curve_name}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # Generate autocorrelation plots
            if "autocorrelation" in plot_types:
                logger.info(f"  Generating autocorrelation for {curve_name}...")
                plt.figure(figsize=(10, 6))
                plot_autocorrelation(curve_data, max_delay=max_delay)
                plt.suptitle(f"Autocorrelation: {curve_name}")
                plt.savefig(
                    str(plots_directory / f"autocorrelation_{curve_name}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # Generate motif quality plots if annotations are available
    if "quality" in plot_types:
        try:
            annotation_file = plot_config.get("annotation_file")
            if annotation_file and Path(annotation_file).exists():
                logger.warning(
                    "Quality plotting is not implemented; annotation file was not used: %s",
                    annotation_file,
                )
        except Exception as e:
            logger.warning(f"Could not generate quality plot: {e}")

    plot_duration = time.time() - plot_start_time
    logger.info(f"Plot generation completed in {plot_duration:.1f}s")
    logger.info(f"Plots saved to: {plots_directory}")


def create_query_motif_heatmap(
    results_tensor,
    motif_names,
    motif_index_file,
    output_directory,
    query_motif_index=None,
    min_sum_threshold=100,
    min_max_std_ratio=5,
    figsize=(10, 20),
    vmax=50,
    vmin=0,
    celltype_suffix="",
):
    """
    Creates a clustered heatmap for query-centered motif analysis with motif names as row labels.

    Args:
        results_tensor (torch.Tensor): Target analysis results with shape (n_motifs, window_width)
        motif_names (list): List of motif names for indexing
        motif_index_file (str): Path to motif index file for motif name mapping
        output_directory (str): Directory to save the heatmap
        query_motif_index (int, optional): Index of query motif for filename
        min_sum_threshold (float): Minimum row sum for filtering (default: 1000)
        min_max_std_ratio (float): Minimum max/std ratio for filtering (default: 5)
        figsize (tuple): Figure size (default: (12, 10))
        vmax (float): Maximum value for colormap (default: 50)
        vmin (float): Minimum value for colormap (default: 0)

    Returns:
        pd.DataFrame: Filtered DataFrame used for plotting
    """
    if results_tensor is None or results_tensor.dim() != 2:
        logger.warning(
            "Results tensor is not suitable for query motif heatmap (expected 2D tensor)"
        )
        return None

    logger.info("Creating query motif heatmap with motif names...")

    # Convert tensor to numpy array
    data = results_tensor.cpu().float().numpy()

    # Read motif names from index file if it exists
    motif_index_path = Path(motif_index_file)
    if motif_index_path.exists():
        indexed_motif_names = []
        try:
            with motif_index_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            indexed_motif_names.append(parts[1])

            # Use indexed motif names if available and matches data shape
            if len(indexed_motif_names) == data.shape[0]:
                motif_labels = indexed_motif_names
                logger.info(f"Using motif names from index file: {len(motif_labels)} motifs")
            else:
                motif_labels = motif_names[: data.shape[0]]
                logger.warning("Motif index file length mismatch, using original names")
        except Exception as e:
            logger.warning(f"Error reading motif index file: {e}, using original names")
            motif_labels = motif_names[: data.shape[0]]
    else:
        motif_labels = motif_names[: data.shape[0]]
        logger.info("Motif index file not found, using original motif names")

    # Create DataFrame with motif names as index
    df = pd.DataFrame(data, index=motif_labels)

    logger.info(f"Data shape: {df.shape}")

    # Apply filtering criteria
    row_sums = df.sum(axis=1)
    row_maxs = df.max(axis=1)
    row_stds = df.std(axis=1)

    # Filter rows: sum > threshold and max > ratio * std
    filter_mask = (row_sums > min_sum_threshold) & (row_maxs > row_stds * min_max_std_ratio)
    filtered_df = df[filter_mask]

    logger.info(
        f"Filtered data shape: {filtered_df.shape} (filtered {df.shape[0] - filtered_df.shape[0]} rows)"
    )

    if filtered_df.empty:
        logger.warning("No motifs passed filtering criteria, skipping heatmap")
        return None

    # Create output directory
    plots_dir = Path(output_directory) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create the clustered heatmap
    plt.figure(figsize=figsize)

    try:
        g = sns.clustermap(
            filtered_df,
            row_cluster=True,
            col_cluster=False,
            z_score=0,
            vmax=vmax,
            vmin=vmin,
            metric="correlation",
            cmap="viridis",
            figsize=figsize,
            yticklabels=True,
            xticklabels=False,
        )

        # Adjust layout for better readability
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)

        # Generate filename
        if query_motif_index is not None and query_motif_index < len(motif_names):
            target_name = motif_names[query_motif_index].replace(".", "_").replace("/", "_")
            filename = f"query_motif_heatmap_{target_name}{celltype_suffix}.png"
        else:
            filename = f"query_motif_heatmap{celltype_suffix}.png"

        output_path = plots_dir / filename

        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved query motif heatmap to: {output_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        plt.close()
        return filtered_df

    return filtered_df
