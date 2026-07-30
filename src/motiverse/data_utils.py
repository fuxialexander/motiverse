"""Data utilities for loading signals and other data sources.

This module contains utilities for loading and processing signal data, ReMap integration,
and saving analysis results.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from .sequence_io import (
    CelltypeDenseZarrIO,
    ConcatAggregator,
    KeyQuery,
    NoNormalizer,
    SequenceDenseZarrIO,
    SignalDenseZarrIO,
    SumAggregator,
)

logger = logging.getLogger(__name__)


def generate_tiled_regions(
    genome_database: SequenceDenseZarrIO,
    tile_size: int = 500,
    target_chromosomes: list = None,
) -> pd.DataFrame:
    """
    Generate non-overlapping tiled regions across the genome.

    Creates a DataFrame with tiles of specified size for whole-genome scanning.
    Each tile has the same size (except possibly the last tile on each chromosome).

    Args:
        genome_database: SequenceDenseZarrIO object with chromosome information
        tile_size: Size of each tile in base pairs (default: 500)
        target_chromosomes: List of chromosomes to tile (None = all chromosomes)

    Returns:
        DataFrame with columns: chrom, start, end
    """
    logger.info(f"Generating {tile_size}bp tiles across genome...")

    chromosomes = target_chromosomes if target_chromosomes else genome_database.chroms

    tiles = []
    for chrom in chromosomes:
        chrom_size = genome_database.chrom_sizes[chrom]

        # Generate non-overlapping tiles
        for start in range(0, chrom_size, tile_size):
            end = min(start + tile_size, chrom_size)
            tiles.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                }
            )

    tiles_df = pd.DataFrame(tiles)
    logger.info(f"Generated {len(tiles_df):,} tiles across {len(chromosomes)} chromosomes")
    chrom_counts = tiles_df["chrom"].value_counts().sort_index()
    for chrom, count in chrom_counts.items():
        logger.info(f"  {chrom}: {count:,} tiles")

    return tiles_df


def get_signal_data(signal_database, chrom, start, end, celltype_id=None):
    """
    Extract signal data from either SignalDenseZarrIO or CelltypeDenseZarrIO.

    For SignalDenseZarrIO: Returns 1D signal (seq_length,)
    For CelltypeDenseZarrIO:
        - If celltype_id is provided: Returns 1D signal (seq_length,) for that cell type only
        - If celltype_id is None: Returns 2D signal (seq_length, n_celltypes) by concatenating
          all cell types using DefaultQuery with raw_array output format.

    Args:
        signal_database: Either SignalDenseZarrIO or CelltypeDenseZarrIO instance
        chrom (str): Chromosome name
        start (int): Start position
        end (int): End position
        celltype_id (str, optional): For CelltypeDenseZarrIO, select a specific cell type ID.
                                    If provided, returns 1D signal for that cell type only.

    Returns:
        np.ndarray: Signal data with shape (seq_length,) for 1D or (seq_length, n_celltypes) for 2D
    """
    if isinstance(signal_database, CelltypeDenseZarrIO):
        # If a specific celltype is requested, fetch only that as a 1D array
        if celltype_id is not None:
            return signal_database.get_track(
                chrom,
                start,
                end,
                query=KeyQuery(celltype_id),
                aggregator=SumAggregator(),
                normalizer=NoNormalizer(),
                output_format="raw_array",
            )

        # Otherwise, concatenate all cell types into a 2D array (seq_len, n_celltypes)
        signal_data = signal_database.get_track(
            chrom,
            start,
            end,
            query=None,  # DefaultQuery will be used
            aggregator=ConcatAggregator(),
            normalizer=NoNormalizer(),
            output_format="raw_array",
        ).T

        return signal_data

    else:
        # Standard SignalDenseZarrIO - return 1D signal
        return signal_database.get_track(chrom, start, end, output_format="raw_array")


def load_signal_database(signal_zarr_path, celltype_id=None):
    """
    Load signal database with automatic type detection.

    Tries to detect the appropriate signal zarr type by attempting to load
    as CelltypeDenseZarrIO first, then falling back to SignalDenseZarrIO.

    For CelltypeDenseZarrIO, if celltype_id is provided, only that cell type
    will be loaded to reduce memory usage.

    Args:
        signal_zarr_path (str): Path to signal zarr file
        celltype_id (str, optional): For CelltypeDenseZarrIO, select a specific cell type ID.
                                    If provided, only that cell type is loaded.

    Returns:
        Union[CelltypeDenseZarrIO, SignalDenseZarrIO]: Loaded signal database

    Raises:
        Exception: If both loading attempts fail
    """
    # Try to detect the type of signal zarr file
    try:
        # First try CelltypeDenseZarrIO
        # Subset to a specific celltype at open-time if requested
        signal_database = CelltypeDenseZarrIO(
            signal_zarr_path, mode="r", celltype_ids=[celltype_id] if celltype_id else None
        )
        signal_database.load_to_memory_dense()
        logger.info(f"Loaded CelltypeDenseZarr signal data from {signal_zarr_path}")
        if celltype_id:
            logger.info(f"Subset to cell type: {celltype_id}")
        else:
            logger.info(f"Found {signal_database.n_celltypes} cell types: {signal_database.ids}")
        return signal_database
    except Exception:
        # Fallback to SignalDenseZarrIO
        try:
            signal_database = SignalDenseZarrIO(signal_zarr_path, mode="r")
            signal_database.load_to_memory_dense()
            logger.info(f"Loaded SignalDenseZarr signal data from {signal_zarr_path}")
            return signal_database
        except Exception as e:
            logger.error(f"Failed to load signal data from {signal_zarr_path}: {e}")
            raise


def download_remap_narrowpeak(
    tf_name, genome_database=None, assembly="hg38", output_dir="/tmp", gene_name=None
):
    """
    Download narrowPeak file from ReMap database for a given transcription factor.

    Args:
        tf_name (str): Transcription factor name (e.g., "FOXA1")
        genome_database: SequenceDenseZarrIO object to get assembly from, if available
        assembly (str): Genome assembly (default: "hg38", used if genome_database is None)
        output_dir (str): Directory to save the downloaded file
        gene_name (str, optional): Gene name to use for ReMap download (defaults to tf_name)

    Returns:
        str: Path to the downloaded narrowPeak file

    Raises:
        Exception: If download fails
    """
    # Use gene_name if provided, otherwise default to tf_name
    download_gene_name = gene_name if gene_name is not None else tf_name
    # Use assembly from genome_database if available
    if genome_database is not None and hasattr(genome_database, "assembly"):
        assembly = genome_database.assembly
        logger.info(f"Using assembly '{assembly}' from genome database")
    else:
        logger.info(f"Using default assembly '{assembly}'")
    remap_url = f"https://remap.univ-amu.fr/storage/remap2022/{assembly}/MACS2/TF/{download_gene_name}/remap2022_{download_gene_name}_nr_macs2_{assembly}_v1_0.bed.gz"

    output_path = (
        Path(output_dir) / f"remap2022_{download_gene_name}_nr_macs2_{assembly}_v1_0.bed.gz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists():
        logger.info(f"ReMap file already exists: {output_path}")
        return str(output_path)

    logger.info(
        f"Downloading ReMap data for gene '{download_gene_name}' (motif: {tf_name}) from: {remap_url}"
    )

    try:
        # Use pandas to download and immediately save the file
        # This handles the download and saves it as a compressed file
        df = pd.read_csv(remap_url, sep="\t", header=None, compression="gzip")

        # Save as compressed bed file
        df.to_csv(output_path, sep="\t", header=False, index=False, compression="gzip")

        logger.info(f"Successfully downloaded ReMap data to: {output_path}")
        return str(output_path)

    except Exception as e:
        raise Exception(f"Failed to download ReMap data for {tf_name}: {e}")


def save_analysis_results(
    output_directory,
    results_tensor,
    motif_names,
    sequence_accumulation,
    query_motif_index,
    strand_specific,
    filename_prefix="genome_analysis",
    plot_config=None,
    signal_accumulation=False,
    metadata_extra=None,
):
    """
    Saves analysis results and metadata to disk with descriptive filenames.

    Creates output files with names that clearly indicate the analysis type
    and parameters used. Also saves motif name mappings for result interpretation.
    Optionally generates visualization plots.

    Args:
        output_directory (str): Directory path for saving results.
        results_tensor (torch.Tensor): Accumulated analysis results.
        motif_names (list): List of motif names for indexing.
        sequence_accumulation (bool): Whether results contain sequence content.
        query_motif_index (int, optional): Index of query motif for single-motif analysis.
        strand_specific (bool): Whether analysis was strand-specific.
        filename_prefix (str): Prefix for output filenames.
        plot_config (dict, optional): Configuration for plot generation.
        signal_accumulation (bool): Whether results contain signal accumulation data.
        metadata_extra (dict, optional): Additional serializable run metadata.
    """
    if not output_directory or results_tensor is None:
        return

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    save_start_time = time.time()

    # Import torch here to avoid circular imports
    import torch

    # Generate descriptive filename based on analysis type
    if signal_accumulation:
        if results_tensor.dim() == 3:
            # 2D signal results: (n_motifs, window_width, n_celltypes)
            results_filename = f"{output_directory}/{filename_prefix}_2d_signal_accumulation.pt"
        else:
            # 1D signal results: (n_motifs, window_width)
            results_filename = f"{output_directory}/{filename_prefix}_signal_accumulation.pt"
    elif sequence_accumulation:
        results_filename = f"{output_directory}/{filename_prefix}_sequence_content.pt"
    elif query_motif_index is not None:
        query_motif_name = motif_names[query_motif_index].replace("/", "_").replace(".", "_")
        results_filename = f"{output_directory}/{filename_prefix}_target_{query_motif_name}.pt"
    else:
        results_filename = f"{output_directory}/{filename_prefix}_motif_cooccurrences.pt"

    # Save the main results tensor
    torch.save(results_tensor.cpu(), results_filename)
    save_duration = time.time() - save_start_time
    logger.info(f"Saved analysis results to {results_filename} in {save_duration:.1f}s")

    # Save run metadata next to the tensor. This is intentionally lightweight so
    # benchmark and cache code can validate motif source and tensor shape without
    # loading the full result.
    metadata = {
        "schema_version": "genome_motif_analysis_result_v1",
        "result_file": str(results_filename),
        "result_shape": list(results_tensor.shape),
        "result_dtype": str(results_tensor.dtype),
        "motif_count": len(motif_names),
        "sequence_accumulation": bool(sequence_accumulation),
        "signal_accumulation": bool(signal_accumulation),
        "query_motif_index": query_motif_index,
        "query_motif_name": motif_names[query_motif_index]
        if query_motif_index is not None
        else None,
        "strand_specific": bool(strand_specific),
        "save_duration_s": save_duration,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    metadata_file = f"{output_directory}/{Path(results_filename).stem}_metadata.json"
    from .motif_source import write_metadata_json

    write_metadata_json(metadata_file, metadata)

    # Save motif name mapping for result interpretation
    motif_mapping_file = f"{output_directory}/{Path(results_filename).stem}_motif_index.txt"
    with Path(motif_mapping_file).open("w") as mapping_file:
        # Add header with analysis information
        if query_motif_index is not None:
            mapping_file.write(
                f"# Query-centered analysis around: {motif_names[query_motif_index]} (index {query_motif_index})\n"
            )
        mapping_file.write("# Motif index mapping:\n")

        # Write forward strand motifs
        for motif_index, motif_name in enumerate(motif_names):
            mapping_file.write(f"{motif_index}\t{motif_name}\n")

        # Add reverse complement motifs if applicable
        if (sequence_accumulation or signal_accumulation) and not strand_specific:
            mapping_file.write("# Reverse complement motifs:\n")
            for motif_index, motif_name in enumerate(motif_names):
                reverse_index = motif_index + len(motif_names)
                mapping_file.write(f"{reverse_index}\t{motif_name}_reverse_complement\n")

    # Generate plots if requested
    if plot_config and plot_config.get("enable_plotting", False):
        # Import here to avoid circular imports
        from .visualization import create_query_motif_heatmap, generate_analysis_plots

        generate_analysis_plots(
            results_tensor,
            motif_names,
            output_directory,
            plot_config,
            sequence_accumulation,
            query_motif_index,
            strand_specific,
        )

    # Generate query motif heatmap for one-to-all analysis (both motif scores and signal)
    if query_motif_index is not None and not sequence_accumulation:
        # Import here to avoid circular imports
        from .visualization import create_query_motif_heatmap

        if signal_accumulation:
            if results_tensor.dim() == 3:
                logger.info("Generating 2D signal aggregation heatmaps...")
                # For 2D signal, create separate heatmaps for each cell type/factor
                n_celltypes = results_tensor.shape[2]
                for celltype_idx in range(n_celltypes):
                    celltype_tensor = results_tensor[:, :, celltype_idx]
                    create_query_motif_heatmap(
                        celltype_tensor,
                        motif_names,
                        motif_mapping_file,
                        output_directory,
                        query_motif_index=query_motif_index,
                        celltype_suffix=f"_celltype_{celltype_idx}",
                    )
            else:
                logger.info("Generating 1D signal aggregation heatmap...")
                create_query_motif_heatmap(
                    results_tensor,
                    motif_names,
                    motif_mapping_file,
                    output_directory,
                    query_motif_index=query_motif_index,
                )
        else:
            logger.info("Generating query motif heatmap...")
            create_query_motif_heatmap(
                results_tensor,
                motif_names,
                motif_mapping_file,
                output_directory,
                query_motif_index=query_motif_index,
            )
