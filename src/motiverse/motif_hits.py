from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec
from zarr.core.dtype import VariableLengthUTF8

# Use constants and utils directly
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")


def _write_motif_names(metadata_group, motif_names: list[str], overwrite: bool):
    motif_names_array = metadata_group.create_array(
        "motif_names",
        shape=(len(motif_names),),
        dtype=VariableLengthUTF8(),
        overwrite=overwrite,
    )
    motif_names_array[:] = list(motif_names)


class MotifHitZarrIO:
    """
    Zarr-based storage for sparse motif hits with max score per motif per peak.

    Storage structure follows the peaks pattern from ZarrIO:
    motif_hits.zarr/
    ├── motif_hit_data/regions/
    │   ├── chr1              # (n_hits_chr1, 4) [peak_idx, motif_idx, rel_pos, score]
    │   ├── chr2              # (n_hits_chr2, 4)
    │   └── ...
    ├── peak_regions/         # Original narrowPeak data per chromosome
    │   ├── chr1              # (n_peaks_chr1, 2) [start, end] (original coordinates)
    │   ├── chr2
    │   └── ...
    └── metadata/
        ├── motif_names       # (n_motifs,) string array
        └── attrs             # p_value_threshold, region_extension_bp, etc.

    Coordinate System:
    - peak_regions stores original narrowPeak coordinates
    - rel_pos is position within extended region: extended_start + rel_pos = absolute_pos
    - Extended region: [peak_start - extension, peak_end + extension]
    - To get absolute coordinates: (peak_start - extension) + rel_pos

    Usage:
        # Create sparse hits from the Motiverse CLI.
        motiverse --narrowpeak peaks.bed --collect-hits --output results/

        # Use with FootprintMixin (drop-in replacement for MotifDenseZarrIO)
        from motiverse.motif_hits import MotifHitZarrIO
        from motiverse.sequence_io import SignalDenseZarrIO

        signal_io = SignalDenseZarrIO("atac_signal.zarr")
        motif_hits = MotifHitZarrIO("results/motif_hits.zarr")

        # Get footprints - same interface as MotifDenseZarrIO
        footprints = signal_io.get_genome_wide_footprint_all_motifs(motif_hits)
    """

    def __init__(self, path: str, mode="r", chroms: list[str] | None = None):
        """
        Initialize MotifHitZarrIO.

        Args:
            path: Path to zarr store
            mode: File mode ('r', 'w', 'a')
            chroms: List of chromosomes to process
        """
        self.path = str(path)
        self.mode = mode

        # Open or create zarr store. Read mode can still open existing zarr v2
        # stores; write mode uses the default zarr format that matches the
        # zarr.codecs compressor used throughout the current codebase.
        if mode == "w":
            self.dataset = zarr.open_group(self.path, mode="w")
        else:
            self.dataset = zarr.open_group(self.path, mode=mode)

        # Set chromosomes
        if chroms is not None:
            self.chroms = chroms
        elif mode == "r":
            try:
                # Get chromosomes from peak_regions
                available_chroms = list(self.dataset["peak_regions/regions"].keys())
                self.chroms = available_chroms
            except KeyError:
                # Fallback to standard chromosomes
                self.chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        else:
            # Default for write mode
            self.chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    def write_motif_hits(
        self,
        peaks_df: pd.DataFrame,
        collected_hits: list[tuple],
        motif_names: list[str],
        p_value_threshold: str = "p0.0001",
        region_extension_bp: int = 0,
        overwrite: bool = False,
    ):
        """
        Write sparse motif hits and peak regions to zarr store.

        Args:
            peaks_df: DataFrame with narrowPeak regions
            collected_hits: List of (peak_idx, motif_idx, rel_pos, score) tuples
            motif_names: List of motif names
            p_value_threshold: P-value threshold used for hit collection
            region_extension_bp: Extension applied to peak regions
            overwrite: Whether to overwrite existing data
        """
        if self.mode == "r":
            logging.error("Cannot write motif hits: Zarr dataset is read-only.")
            return False

        # Standardize column names for on-disk compatibility (capitalized)
        column_mapping = {
            "chrom": "Chromosome",
            "start": "Start",
            "end": "End",
            "name": "Name",
        }

        # Rename columns if they exist in lowercase
        peaks_df = peaks_df.rename(columns=column_mapping)

        # Store metadata
        metadata_group = self.dataset.require_group("metadata", overwrite=overwrite)
        _write_motif_names(metadata_group, motif_names, overwrite=overwrite)
        metadata_group.attrs.update(
            {
                "p_value_threshold": p_value_threshold,
                "region_extension_bp": region_extension_bp,
                "n_motifs": len(motif_names),
                "n_hits": len(collected_hits),
            }
        )

        # Write peak regions
        self._write_peaks_to_zarr(peaks_df, "peak_regions", overwrite)

        # Organize hits by chromosome and convert to local peak indices
        hits_by_chrom = {}

        # Create mapping from global peak_idx to (chrom, local_peak_idx)
        chrom_peak_counts = {}
        peak_to_local = {}

        for global_peak_idx, (_, peak_row) in enumerate(peaks_df.iterrows()):
            chrom = peak_row["Chromosome"]
            if chrom not in chrom_peak_counts:
                chrom_peak_counts[chrom] = 0
            local_peak_idx = chrom_peak_counts[chrom]
            peak_to_local[global_peak_idx] = (chrom, local_peak_idx)
            chrom_peak_counts[chrom] += 1

        for peak_idx, motif_idx, rel_pos, score in collected_hits:
            chrom, local_peak_idx = peak_to_local[peak_idx]

            if chrom not in hits_by_chrom:
                hits_by_chrom[chrom] = []
            hits_by_chrom[chrom].append([local_peak_idx, motif_idx, rel_pos, score])

        # Write hits per chromosome
        hits_group = self.dataset.require_group("motif_hit_data/regions", overwrite=overwrite)

        for chrom in tqdm(self.chroms, desc="Writing motif hits by chromosome", leave=False):
            if chrom in hits_by_chrom:
                hits_data = np.array(hits_by_chrom[chrom], dtype=np.float32)
            else:
                # Empty array for chromosomes with no hits
                hits_data = np.zeros((0, 4), dtype=np.float32)

            ds = hits_group.create_array(
                chrom,
                data=hits_data.astype(np.float32),
                chunks=(1000, 4),
                compressors=COMPRESSOR,
                overwrite=overwrite,
            )
            ds.attrs["n_hits"] = len(hits_data)
            ds.attrs["columns"] = ["peak_idx", "motif_idx", "rel_pos", "score"]

        logging.info(f"Wrote {len(collected_hits)} sparse motif hits to {self.path}")

    def initialize_for_streaming(
        self,
        peaks_df: pd.DataFrame,
        motif_names: list[str],
        p_value_threshold: str = "p0.0001",
        region_extension_bp: int = 0,
    ):
        """
        Initialize zarr store for streaming writes.

        Args:
            peaks_df: DataFrame with narrowPeak regions
            motif_names: List of motif names
            p_value_threshold: P-value threshold used for hit collection
            region_extension_bp: Extension applied to peak regions
        """
        if self.mode == "r":
            logging.error("Cannot initialize streaming: Zarr dataset is read-only.")
            return False

        # Standardize column names for on-disk compatibility (capitalized)
        column_mapping = {
            "chrom": "Chromosome",
            "start": "Start",
            "end": "End",
            "name": "Name",
        }
        peaks_df = peaks_df.rename(columns=column_mapping)

        # Store metadata
        metadata_group = self.dataset.require_group("metadata", overwrite=True)
        _write_motif_names(metadata_group, motif_names, overwrite=True)
        metadata_group.attrs.update(
            {
                "p_value_threshold": p_value_threshold,
                "region_extension_bp": region_extension_bp,
                "n_motifs": len(motif_names),
                "n_hits": 0,  # Will be updated as we stream
            }
        )

        # Write peak regions
        self._write_peaks_to_zarr(peaks_df, "peak_regions", overwrite=True)

        # Create empty hit data structure per chromosome
        hits_group = self.dataset.require_group("motif_hit_data/regions", overwrite=True)

        # Initialize peak index mappings for streaming
        self._initialize_peak_mappings(peaks_df)

        # Initialize empty hit arrays per chromosome
        for chrom in self.chroms:
            hits_group.create_array(
                chrom,
                shape=(0, 4),
                dtype=np.float32,
                chunks=(1000, 4),
                compressors=COMPRESSOR,
            )
            hits_group[chrom].attrs["n_hits"] = 0
            hits_group[chrom].attrs["columns"] = [
                "peak_idx",
                "motif_idx",
                "rel_pos",
                "score",
            ]

        logging.info(f"Initialized streaming zarr store at {self.path}")

    def _initialize_peak_mappings(self, peaks_df: pd.DataFrame):
        """
        Initialize mappings from global to local peak indices using vectorized operations.

        Much more efficient than iterrows() - uses pandas vectorized operations.
        For millions of peaks, this is ~10-50x faster than the original iterrows() approach.
        """
        # Reset index to ensure we have sequential global indices
        peaks_df_indexed = peaks_df.reset_index(drop=True)

        # Compute local peak indices within each chromosome using cumcount (vectorized)
        local_indices = peaks_df_indexed.groupby("Chromosome").cumcount()

        # Create mapping from global index to (chrom, local_idx) using zip (fastest)
        # zip with Series access is much faster than iterrows() or itertuples()
        self._peak_to_local = {
            global_idx: (chrom, int(local_idx))
            for global_idx, chrom, local_idx in zip(
                peaks_df_indexed.index,
                peaks_df_indexed["Chromosome"],
                local_indices,
            )
        }
        self._peak_local_chroms = peaks_df_indexed["Chromosome"].to_numpy()
        self._peak_local_indices = local_indices.to_numpy(dtype=np.int64)

        # Store chromosome peak counts for reference (vectorized)
        self._chrom_peak_counts = peaks_df_indexed.groupby("Chromosome").size().to_dict()

    def append_hits(self, hits_chunk: list[tuple] | np.ndarray):
        """
        Append a chunk of hits to the zarr store.

        Args:
            hits_chunk: List/array of (peak_idx, motif_idx, rel_pos, score) rows
        """
        if self.mode == "r":
            logging.error("Cannot append hits: Zarr dataset is read-only.")
            return

        if not hasattr(self, "_peak_to_local"):
            logging.error(
                "Zarr store not initialized for streaming. Call initialize_for_streaming() first."
            )
            return

        if isinstance(hits_chunk, np.ndarray):
            hit_array = hits_chunk.astype(np.float32, copy=False)
        else:
            hit_array = np.asarray(hits_chunk, dtype=np.float32)

        if hit_array.size == 0:
            return
        hit_array = hit_array.reshape(-1, 4)

        peak_indices = hit_array[:, 0].astype(np.int64, copy=False)
        if hasattr(self, "_peak_local_chroms") and hasattr(self, "_peak_local_indices"):
            valid = (peak_indices >= 0) & (peak_indices < len(self._peak_local_chroms))
            if not np.any(valid):
                return
            valid_hits = hit_array[valid]
            valid_peak_indices = peak_indices[valid]
            local_chroms = self._peak_local_chroms[valid_peak_indices]
            local_peak_indices = self._peak_local_indices[valid_peak_indices]
            hits_by_chrom = {}
            for chrom in np.unique(local_chroms):
                chrom_mask = local_chroms == chrom
                hits_by_chrom[chrom] = np.column_stack(
                    [
                        local_peak_indices[chrom_mask].astype(np.float32),
                        valid_hits[chrom_mask, 1:],
                    ]
                ).astype(np.float32, copy=False)
        else:
            hits_by_chrom = {}
            for row in hit_array:
                peak_idx = int(row[0])
                if peak_idx in self._peak_to_local:
                    chrom, local_peak_idx = self._peak_to_local[peak_idx]

                    if chrom not in hits_by_chrom:
                        hits_by_chrom[chrom] = []
                    hits_by_chrom[chrom].append([local_peak_idx, row[1], row[2], row[3]])

        # Append to zarr arrays per chromosome
        hits_group = self.dataset["motif_hit_data/regions"]
        total_new_hits = 0

        for chrom, chrom_hits in hits_by_chrom.items():
            if len(chrom_hits):
                chrom_array = hits_group[chrom]
                current_size = chrom_array.shape[0]
                new_hits = np.asarray(chrom_hits, dtype=np.float32)

                # Resize array to accommodate new hits
                chrom_array.resize((current_size + len(new_hits), 4))

                # Append new hits
                chrom_array[current_size:] = new_hits

                # Update hit count
                chrom_array.attrs["n_hits"] = current_size + len(new_hits)
                total_new_hits += len(new_hits)

        # Update total hit count in metadata
        current_total = self.dataset["metadata"].attrs["n_hits"]
        self.dataset["metadata"].attrs["n_hits"] = current_total + total_new_hits

        logging.debug(f"Appended {total_new_hits} hits to zarr store")

    def _write_peaks_to_zarr(self, peaks_df: pd.DataFrame, group_name: str, overwrite: bool):
        """Write peaks data to zarr store by chromosome."""
        peaks_df = peaks_df.sort_values(["Chromosome", "Start"])

        # Only write Start and End (coordinates) - skip Name to avoid string handling
        cols_to_write = ["Start", "End"]

        regions_group = self.dataset.require_group(f"{group_name}/regions", overwrite=overwrite)

        for chrom in tqdm(self.chroms, desc=f"Writing {group_name} chromosomes", leave=False):
            chrom_peaks = peaks_df[peaks_df["Chromosome"] == chrom]

            if len(chrom_peaks) > 0:
                chrom_peaks_data = chrom_peaks[cols_to_write].values.astype(np.int32)
            else:
                # Empty array for chromosomes with no peaks
                chrom_peaks_data = np.zeros((0, len(cols_to_write)), dtype=np.int32)

            ds = regions_group.create_array(
                chrom,
                data=chrom_peaks_data.astype(np.int32),
                chunks=(1000, len(cols_to_write)),
                compressors=COMPRESSOR,
                overwrite=overwrite,
            )
            ds.attrs["n_regions"] = len(chrom_peaks_data)
            ds.attrs["columns"] = cols_to_write

    def get_motif_match_indices(
        self,
        chrom: str,
        start: int,
        end: int,
        zarr_group_name: str = "motif_hit_data",
        cutoff: float = 5,
        motif_idx: int | list[int] | None = None,
        regions: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Get motif match indices in format compatible with FootprintMixin.

        Args:
            chrom: Chromosome name
            start: Start position
            end: End position
            zarr_group_name: Group name (for compatibility, uses motif_hit_data)
            cutoff: Score cutoff threshold
            motif_idx: Specific motif index to filter
            regions: Additional regions to filter (not implemented yet)

        Returns:
            np.ndarray: Array with shape (n_hits, 3) containing [motif_idx, position, score]
        """
        try:
            # Use in-memory data if available, otherwise load from zarr
            if hasattr(self, "data") and chrom in self.data:
                hits_data = self.data[chrom]  # (n_hits, 4)
            else:
                hits_data = self.dataset[f"motif_hit_data/regions/{chrom}"][:]  # (n_hits, 4)

            if hits_data.shape[0] == 0:
                return np.array([]).reshape(0, 3)

            # Use in-memory peak data if available, otherwise load from zarr
            if hasattr(self, "peak_data") and chrom in self.peak_data:
                peak_data = self.peak_data[chrom]  # (n_peaks, 2 or 3)
            else:
                peak_data = self.dataset[f"peak_regions/regions/{chrom}"][:]  # (n_peaks, 2 or 3)

            # Convert relative positions to absolute coordinates using vectorized operations
            # hits_data: (n_hits, 4) -> [local_peak_idx, motif_idx_hit, rel_pos, score]

            if len(hits_data) == 0:
                return np.array([]).reshape(0, 3)

            # Extract columns
            local_peak_indices = hits_data[:, 0].astype(int)
            motif_indices = hits_data[:, 1].astype(int)
            rel_positions = hits_data[:, 2].astype(int)
            scores = hits_data[:, 3]

            # Filter valid peak indices
            valid_mask = local_peak_indices < len(peak_data)
            if not np.any(valid_mask):
                return np.array([]).reshape(0, 3)

            # Apply valid mask
            valid_local_indices = local_peak_indices[valid_mask]
            valid_motif_indices = motif_indices[valid_mask]
            valid_rel_positions = rel_positions[valid_mask]
            valid_scores = scores[valid_mask]

            # Convert to absolute positions vectorized
            # Account for region extension: rel_pos is within extended region,
            # but peak_data contains original coordinates
            region_extension_bp = 0
            if hasattr(self, "metadata") and "attrs" in self.metadata:
                region_extension_bp = self.metadata["attrs"].get("region_extension_bp", 0)
            else:
                try:
                    region_extension_bp = self.dataset["metadata"].attrs.get(
                        "region_extension_bp", 0
                    )
                except (KeyError, AttributeError):
                    region_extension_bp = 0

            peak_starts = peak_data[valid_local_indices, 0]  # Vectorized peak start lookup
            # Adjust for extension: extended region starts at (peak_start - extension)
            extended_region_starts = peak_starts - region_extension_bp
            absolute_positions = extended_region_starts + valid_rel_positions

            # Apply genomic region filter
            region_mask = (absolute_positions >= start) & (absolute_positions < end)
            if not np.any(region_mask):
                return np.array([]).reshape(0, 3)

            # Apply score cutoff
            score_mask = valid_scores[region_mask] >= cutoff
            if not np.any(score_mask):
                return np.array([]).reshape(0, 3)

            # Apply motif filter if specified
            filtered_motif_indices = valid_motif_indices[region_mask][score_mask]
            filtered_positions = absolute_positions[region_mask][score_mask]
            filtered_scores = valid_scores[region_mask][score_mask]

            if motif_idx is not None:
                if isinstance(motif_idx, (int, np.integer)):
                    motif_filter_mask = filtered_motif_indices == motif_idx
                else:
                    # Array or list of indices
                    motif_idx_set = set(motif_idx)
                    # Optimization: if requesting all or most motifs, skip filtering or check set size
                    if len(motif_idx_set) < self.n_motifs:
                        motif_filter_mask = np.isin(filtered_motif_indices, list(motif_idx_set))
                    else:
                        motif_filter_mask = None

                if motif_filter_mask is not None:
                    filtered_motif_indices = filtered_motif_indices[motif_filter_mask]
                    filtered_positions = filtered_positions[motif_filter_mask]
                    filtered_scores = filtered_scores[motif_filter_mask]

            # Stack results
            results = np.column_stack([filtered_motif_indices, filtered_positions, filtered_scores])

            return results.astype(int)

        except KeyError:
            logging.warning(f"No motif hit data found for chromosome {chrom}")
            return np.array([]).reshape(0, 3)

    def get_chunk_motif_match_indices(
        self,
        chrom: str,
        chunk_idx: int,
        zarr_group_name: str = "motif_hit_data",
        cutoff: float = 5,
        motif_idx: int | None = None,
    ) -> np.ndarray:
        """
        Get motif match indices for a specific chunk (for compatibility).

        Args:
            chrom: Chromosome name
            chunk_idx: Chunk index
            zarr_group_name: Group name
            cutoff: Score cutoff
            motif_idx: Motif index filter

        Returns:
            np.ndarray: Motif match indices
        """
        # For sparse storage, we don't have chunk-based organization
        # Instead, get all hits for chromosome and filter by position range
        chunk_size = getattr(self, "chunk_size", 1000000)  # Default 1MB chunks
        start = chunk_idx * chunk_size
        end = (chunk_idx + 1) * chunk_size

        return self.get_motif_match_indices(chrom, start, end, zarr_group_name, cutoff, motif_idx)

    @property
    def motif_names(self) -> list[str]:
        """Get list of motif names."""
        # Use in-memory data if available, otherwise load from zarr
        if hasattr(self, "metadata") and "motif_names" in self.metadata:
            return self.metadata["motif_names"].tolist()
        try:
            return self.dataset["metadata/motif_names"][:].tolist()
        except KeyError:
            return []

    @property
    def n_motifs(self) -> int:
        """Get number of motifs."""
        # Use in-memory data if available, otherwise load from zarr
        if (
            hasattr(self, "metadata")
            and "attrs" in self.metadata
            and "n_motifs" in self.metadata["attrs"]
        ):
            return self.metadata["attrs"]["n_motifs"]
        try:
            return self.dataset["metadata"].attrs["n_motifs"]
        except KeyError:
            return len(self.motif_names)

    def load_to_memory_dense(self):
        """Loads the entire dataset into memory as dense numpy arrays (per chromosome)."""
        if self.dataset is None:
            raise OSError("Cannot load to memory: Zarr dataset not loaded.")

        self.data = {}
        self.peak_data = {}
        self.metadata = {}

        # Load motif hit data per chromosome
        for chr_name in self.chroms:
            try:
                # Load motif hit data for this chromosome
                chrom_ref = self.dataset["motif_hit_data/regions"][chr_name]
                self.data[chr_name] = chrom_ref[:].astype(np.float32)
            except (KeyError, IndexError, AttributeError) as e:
                logging.warning(
                    f"Failed to load motif hit data for chromosome {chr_name} to dense memory: {e}"
                )

        # Load peak regions per chromosome
        for chr_name in self.chroms:
            try:
                # Load peak regions for this chromosome
                peak_ref = self.dataset["peak_regions/regions"][chr_name]
                self.peak_data[chr_name] = peak_ref[:].astype(np.int32)
            except (KeyError, IndexError, AttributeError) as e:
                logging.warning(
                    f"Failed to load peak regions for chromosome {chr_name} to dense memory: {e}"
                )

        # Load metadata
        try:
            # Load motif names
            self.metadata["motif_names"] = self.dataset["metadata/motif_names"][:]
            # Load metadata attributes
            self.metadata["attrs"] = dict(self.dataset["metadata"].attrs)
        except (KeyError, IndexError, AttributeError) as e:
            logging.warning(f"Failed to load metadata to dense memory: {e}")
