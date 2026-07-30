"""Exact positional motif-hit cache for repeated genome motif queries."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec
from zarr.core.dtype import VariableLengthUTF8

from .processing import _get_compiled_conv_fn
from .sequence_io import SequenceDenseZarrIO

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "positional_hit_cache_v1"
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
CACHE_COLUMNS = ["start_pos", "center_pos", "motif_idx", "strand", "score"]
CACHE_ROW_DTYPE = np.float64
CACHE_ACCUMULATION_SEMANTICS = "significant_hit_score_pairs"
CACHE_COORDINATE_FRAME = "active_core_center"
HIT_STORAGE_LAYOUT = "chrom_array_v1"
HIT_CHUNK_INDEX_SCHEMA_VERSION = "chrom_chunk_index_v1"
HIT_CHUNK_INDEX_COLUMNS = ["row_start", "row_end", "center_min", "center_max"]
CACHE_INVARIANT_METADATA = {
    "cache_accumulation_semantics": CACHE_ACCUMULATION_SEMANTICS,
    "coordinate_frame": CACHE_COORDINATE_FRAME,
    "hit_storage_layout": HIT_STORAGE_LAYOUT,
    "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
}
METADATA_VALIDATION_KEYS = [
    "motif_source",
    "aligned_motif_path",
    "loaded_with_aligned",
    "motif_count",
    "motif_kernel_shape",
    "motif_kernel_checksum",
    "motif_names_checksum",
    "threshold_mode",
    "p_value_threshold",
    "score_threshold",
    "score_threshold_vector_checksum",
    "score_threshold_vector_shape",
]
QueryInterval = tuple[str, int, int]


def _chrom_size_from_store(sequence_database: SequenceDenseZarrIO, chrom: str) -> int:
    """Return the actual stored chromosome length when the zarr array exposes it."""
    try:
        chrom_obj = sequence_database.dataset["chrs"][chrom]
        if getattr(sequence_database, "dir_chunked", False):
            chunk_names = sorted(chrom_obj.keys())
            if chunk_names:
                return int(sum(chrom_obj[name].shape[0] for name in chunk_names))
        return int(chrom_obj.shape[0])
    except (AttributeError, KeyError, TypeError):
        return int(sequence_database.chrom_sizes[chrom])


def _json_safe_attrs(metadata: dict) -> dict:
    """Convert metadata values to zarr-attribute-friendly JSON scalars/lists."""
    safe: dict = {}
    for key, value in metadata.items():
        if isinstance(value, tuple):
            safe[key] = list(value)
        elif isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _normalized_metadata_value(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return [_normalized_metadata_value(item) for item in value]
    if isinstance(value, list):
        return [_normalized_metadata_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CacheBuildStats:
    path: str
    n_hits: int
    bp_scanned: int
    n_regions: int
    schema_version: str = SCHEMA_VERSION


def _merged_region_intervals(
    regions: pd.DataFrame,
    *,
    chrom_sizes: dict[str, int],
    target_chromosomes: list[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Return sorted, merged cache coverage intervals by chromosome."""
    if regions is None or len(regions) == 0:
        return {}
    allowed_chroms = set(target_chromosomes or chrom_sizes.keys())
    intervals_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for row in regions.itertuples(index=False):
        chrom = str(getattr(row, "chrom", getattr(row, "Chromosome", "")))
        if chrom not in allowed_chroms or chrom not in chrom_sizes:
            continue
        try:
            start = int(getattr(row, "start") if hasattr(row, "start") else getattr(row, "Start"))
            end = int(getattr(row, "end") if hasattr(row, "end") else getattr(row, "End"))
        except (TypeError, ValueError):
            continue
        start = max(0, start)
        end = min(int(chrom_sizes[chrom]), end)
        if end <= start:
            continue
        intervals_by_chrom.setdefault(chrom, []).append((start, end))

    merged_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for chrom, intervals in intervals_by_chrom.items():
        intervals = sorted(intervals)
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
        merged_by_chrom[chrom] = merged
    return merged_by_chrom


def motif_center_offsets(motif_kernels: np.ndarray | torch.Tensor) -> np.ndarray:
    """Compute active-core center offsets for aligned motif kernels.

    The aligned HOCOMOCO tensor has shape ``(motif, length, base)``. Positions
    are reported as center coordinates using the active non-zero kernel span.
    """
    if isinstance(motif_kernels, torch.Tensor):
        kernels_np = motif_kernels.detach().cpu().numpy()
    else:
        kernels_np = np.asarray(motif_kernels)
    offsets = []
    for kernel in kernels_np:
        active = np.where(np.abs(kernel).sum(axis=-1) > 0)[0]
        if len(active) == 0:
            offsets.append(kernel.shape[0] // 2)
        else:
            offsets.append(int(round(float(active[0] + active[-1]) / 2.0)))
    return np.asarray(offsets, dtype=np.int32)


def threshold_vector_checksum(thresholds: np.ndarray | torch.Tensor) -> str:
    """Return a stable checksum for the per-motif score threshold vector."""
    if isinstance(thresholds, torch.Tensor):
        thresholds_np = thresholds.detach().to(torch.float32).cpu().numpy()
    else:
        thresholds_np = np.asarray(thresholds, dtype=np.float32)
    thresholds_np = np.ascontiguousarray(thresholds_np, dtype=np.float32)
    return hashlib.sha256(thresholds_np.tobytes()).hexdigest()


def collect_positional_hits(
    motif_scores: torch.Tensor,
    thresholds: torch.Tensor,
    sequence_starts: torch.Tensor,
    center_offsets: torch.Tensor,
    *,
    strand_id: int,
    motif_length: int | None = None,
    sequence_lengths: torch.Tensor | None = None,
    padded_sequence_length: int | None = None,
    reverse_strand: bool = False,
    p_value_threshold: str = "p0.0001",
    return_batch_indices: bool = False,
) -> torch.Tensor:
    """Collect every significant motif hit as positional rows.

    Returns a float64 tensor with columns:
    ``start_pos, center_pos, motif_idx, strand, score``.
    """
    del p_value_threshold  # Included for call-site symmetry and future schemas.
    hit_mask = motif_scores > thresholds.unsqueeze(0).unsqueeze(-1)
    batch_indices, motif_indices, hit_positions = torch.where(hit_mask)
    if len(batch_indices) == 0:
        empty = torch.zeros(
            (0, len(CACHE_COLUMNS)),
            dtype=torch.float64,
            device=motif_scores.device,
        )
        if return_batch_indices:
            return empty, torch.zeros((0,), dtype=torch.int64, device=motif_scores.device)
        return empty

    if motif_length is None:
        motif_length = 0
    if reverse_strand:
        if padded_sequence_length is None:
            padded_sequence_length = motif_scores.shape[-1] + motif_length - 1
        start_positions = (
            sequence_starts[batch_indices] + padded_sequence_length - hit_positions - motif_length
        )
        center_positions = (
            sequence_starts[batch_indices]
            + padded_sequence_length
            - 1
            - hit_positions
            - center_offsets[motif_indices]
        )
    else:
        start_positions = sequence_starts[batch_indices] + hit_positions
        center_positions = start_positions + center_offsets[motif_indices]

    if sequence_lengths is not None and motif_length:
        sequence_ends = sequence_starts[batch_indices] + sequence_lengths[batch_indices]
        valid = (start_positions >= sequence_starts[batch_indices]) & (
            start_positions + motif_length <= sequence_ends
        )
        if not torch.all(valid):
            batch_indices = batch_indices[valid]
            motif_indices = motif_indices[valid]
            hit_positions = hit_positions[valid]
            start_positions = start_positions[valid]
            center_positions = center_positions[valid]
            if len(batch_indices) == 0:
                empty = torch.zeros(
                    (0, len(CACHE_COLUMNS)),
                    dtype=torch.float64,
                    device=motif_scores.device,
                )
                if return_batch_indices:
                    return empty, torch.zeros((0,), dtype=torch.int64, device=motif_scores.device)
                return empty

    rows = torch.empty(
        (len(batch_indices), len(CACHE_COLUMNS)),
        dtype=torch.float64,
        device=motif_scores.device,
    )
    rows[:, 0] = start_positions.double()
    rows[:, 1] = center_positions.double()
    rows[:, 2] = motif_indices.double()
    rows[:, 3] = float(strand_id)
    rows[:, 4] = motif_scores[batch_indices, motif_indices, hit_positions].double()
    if return_batch_indices:
        return rows, batch_indices
    return rows


def accumulate_hit_rows(
    out: np.ndarray,
    region_hits: np.ndarray,
    *,
    window_size: int,
    query_motif_index: int | None,
    pair_chunk_size: int = 2_000_000,
) -> None:
    """Accumulate sparse co-occurrence profiles from cached positional hits.

    The cache stores only significant motif hits. For each anchor hit, this adds
    every partner hit within ``window_size`` by partner score, preserving the
    original sparse-cache accumulation semantics while batching pair updates.
    """
    if region_hits.size == 0:
        return
    region_hits = region_hits.reshape(-1, len(CACHE_COLUMNS))
    if region_hits.shape[0] == 0:
        return

    centers = region_hits[:, 1].astype(np.int64, copy=False)
    motifs = region_hits[:, 2].astype(np.int64, copy=False)
    scores = region_hits[:, 4].astype(np.float32, copy=False)
    if centers.shape[0] > 1 and np.any(np.diff(centers) < 0):
        order = np.lexsort((motifs, centers))
        centers = centers[order]
        motifs = motifs[order]
        scores = scores[order]

    if query_motif_index is None:
        anchor_indices = np.arange(len(region_hits), dtype=np.int64)
    else:
        anchor_indices = np.flatnonzero(motifs == query_motif_index).astype(np.int64)
    if anchor_indices.size == 0:
        return

    lefts = np.searchsorted(
        centers,
        centers[anchor_indices] - int(window_size),
        side="left",
    )
    rights = np.searchsorted(
        centers,
        centers[anchor_indices] + int(window_size),
        side="right",
    )
    counts = (rights - lefts).astype(np.int64, copy=False)
    if not np.any(counts):
        return

    pair_chunk_size = max(1, int(pair_chunk_size))
    start = 0
    while start < anchor_indices.size:
        total_pairs = 0
        end = start
        while end < anchor_indices.size:
            next_total = total_pairs + int(counts[end])
            if end > start and next_total > pair_chunk_size:
                break
            total_pairs = next_total
            end += 1
            if total_pairs >= pair_chunk_size:
                break
        if total_pairs == 0:
            start = max(end, start + 1)
            continue

        chunk_counts = counts[start:end]
        chunk_lefts = lefts[start:end]
        chunk_rights = rights[start:end]
        chunk_anchors = anchor_indices[start:end]
        anchor_repeated = np.repeat(chunk_anchors, chunk_counts)
        partner_indices = np.empty(total_pairs, dtype=np.int64)
        offset = 0
        for left, right in zip(chunk_lefts, chunk_rights, strict=True):
            width = int(right - left)
            if width:
                partner_indices[offset : offset + width] = np.arange(
                    int(left),
                    int(right),
                    dtype=np.int64,
                )
                offset += width
        if offset != total_pairs:
            partner_indices = partner_indices[:offset]
            anchor_repeated = anchor_repeated[:offset]
        rel = centers[partner_indices] - centers[anchor_repeated] + int(window_size)
        if query_motif_index is None:
            np.add.at(
                out,
                (motifs[anchor_repeated], motifs[partner_indices], rel),
                scores[partner_indices],
            )
        else:
            np.add.at(out, (motifs[partner_indices], rel), scores[partner_indices])
        start = end


class PositionalMotifHitCache:
    """Read/query an exact positional motif-hit cache."""

    def __init__(self, path: str | Path, mode: str = "r"):
        self.path = str(path)
        self.mode = mode
        if mode == "w":
            self.dataset = zarr.open_group(self.path, mode="w")
        else:
            self.dataset = zarr.open_group(self.path, mode=mode)
        self._chrom_row_cache: dict[str, np.ndarray] = {}
        self._chrom_center_cache: dict[str, np.ndarray] = {}
        self._chrom_chunk_index_cache: dict[str, np.ndarray] = {}

    @property
    def metadata(self) -> dict:
        return dict(self.dataset["metadata"].attrs)

    @property
    def motif_names(self) -> list[str]:
        return self.dataset["metadata/motif_names"][:].tolist()

    @property
    def chroms(self) -> list[str]:
        try:
            return list(self.dataset["hits/regions"].keys())
        except KeyError:
            return []

    def _load_chrom_rows(self, chrom: str) -> np.ndarray:
        if chrom in self._chrom_row_cache:
            return self._chrom_row_cache[chrom]
        try:
            rows = self.dataset[f"hits/regions/{chrom}"][:]
        except KeyError:
            rows = np.zeros((0, len(CACHE_COLUMNS)), dtype=CACHE_ROW_DTYPE)
        rows = rows.reshape(-1, len(CACHE_COLUMNS)).astype(CACHE_ROW_DTYPE, copy=False)
        if rows.shape[0] > 1 and np.any(np.diff(rows[:, 1]) < 0):
            rows = rows[np.lexsort((rows[:, 2], rows[:, 1]))]
        self._chrom_row_cache[chrom] = rows
        self._chrom_center_cache[chrom] = rows[:, 1].astype(np.float64, copy=False)
        return rows

    def _load_chrom_chunk_index(self, chrom: str) -> np.ndarray | None:
        if chrom in self._chrom_chunk_index_cache:
            return self._chrom_chunk_index_cache[chrom]
        try:
            index_rows = self.dataset[f"hits/chunk_index/{chrom}"][:]
        except KeyError:
            return None
        index_rows = np.asarray(index_rows, dtype=np.float64).reshape(
            -1, len(HIT_CHUNK_INDEX_COLUMNS)
        )
        self._chrom_chunk_index_cache[chrom] = index_rows
        return index_rows

    @staticmethod
    def _empty_rows() -> np.ndarray:
        return np.zeros((0, len(CACHE_COLUMNS)), dtype=CACHE_ROW_DTYPE)

    @staticmethod
    def _filter_rows(
        rows: np.ndarray,
        motif_idx: int | list[int] | None = None,
        min_score: float | None = None,
    ) -> np.ndarray:
        rows = rows.reshape(-1, len(CACHE_COLUMNS)).astype(CACHE_ROW_DTYPE, copy=False)
        if rows.shape[0] == 0:
            return PositionalMotifHitCache._empty_rows()
        mask = np.ones(rows.shape[0], dtype=bool)
        if motif_idx is not None:
            motif_indices = np.asarray([motif_idx] if isinstance(motif_idx, int) else motif_idx)
            mask &= np.isin(rows[:, 2].astype(np.int32), motif_indices)
        if min_score is not None:
            mask &= rows[:, 4] >= min_score
        return rows[mask].astype(CACHE_ROW_DTYPE, copy=False)

    def _query_chrom_indexed(
        self,
        chrom: str,
        start: int,
        end: int,
        motif_idx: int | list[int] | None = None,
        min_score: float | None = None,
    ) -> np.ndarray | None:
        index_rows = self._load_chrom_chunk_index(chrom)
        if index_rows is None:
            return None
        if index_rows.shape[0] == 0:
            return self._empty_rows()
        try:
            chrom_array = self.dataset[f"hits/regions/{chrom}"]
        except KeyError:
            return self._empty_rows()

        overlaps = (index_rows[:, 3] >= start) & (index_rows[:, 2] < end)
        if not np.any(overlaps):
            return self._empty_rows()

        parts: list[np.ndarray] = []
        for row_start, row_end, _, _ in index_rows[overlaps]:
            row_start_i = int(row_start)
            row_end_i = int(row_end)
            if row_end_i <= row_start_i:
                continue
            chunk_rows = np.asarray(
                chrom_array[row_start_i:row_end_i],
                dtype=CACHE_ROW_DTYPE,
            ).reshape(-1, len(CACHE_COLUMNS))
            centers = chunk_rows[:, 1]
            left = int(np.searchsorted(centers, start, side="left"))
            right = int(np.searchsorted(centers, end, side="left"))
            if right > left:
                parts.append(chunk_rows[left:right])
        if not parts:
            return self._empty_rows()
        return self._filter_rows(
            np.concatenate(parts, axis=0),
            motif_idx=motif_idx,
            min_score=min_score,
        )

    @staticmethod
    def _normalize_query_intervals(
        intervals,
    ) -> list[QueryInterval]:
        keys: list[QueryInterval] = []
        seen: set[QueryInterval] = set()
        for interval in intervals:
            chrom, start, end = interval
            key = (str(chrom), int(start), int(end))
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def _query_many_chrom_indexed(
        self,
        chrom: str,
        intervals: list[QueryInterval],
        *,
        motif_idx: int | list[int] | None = None,
        min_score: float | None = None,
    ) -> tuple[dict[QueryInterval, np.ndarray], dict[str, int]] | None:
        index_rows = self._load_chrom_chunk_index(chrom)
        if index_rows is None:
            return None
        stats = {
            "indexed_chroms": 1,
            "fallback_chroms": 0,
            "chrom_chunk_reads": 0,
            "empty_intervals": 0,
            "rows_returned": 0,
        }
        if index_rows.shape[0] == 0:
            stats["empty_intervals"] = len(intervals)
            return {key: self._empty_rows() for key in intervals}, stats
        try:
            chrom_array = self.dataset[f"hits/regions/{chrom}"]
        except KeyError:
            stats["empty_intervals"] = len(intervals)
            return {key: self._empty_rows() for key in intervals}, stats

        center_mins = index_rows[:, 2]
        center_maxs = index_rows[:, 3]
        chunk_to_intervals: dict[int, list[QueryInterval]] = {}
        results: dict[QueryInterval, np.ndarray] = {}
        parts_by_interval: dict[QueryInterval, list[np.ndarray]] = {key: [] for key in intervals}
        for key in intervals:
            _, start, end = key
            if end <= start:
                results[key] = self._empty_rows()
                stats["empty_intervals"] += 1
                continue
            first_chunk = int(np.searchsorted(center_maxs, start, side="left"))
            last_chunk = int(np.searchsorted(center_mins, end, side="left"))
            if last_chunk <= first_chunk:
                results[key] = self._empty_rows()
                stats["empty_intervals"] += 1
                continue
            for chunk_index in range(first_chunk, last_chunk):
                chunk_to_intervals.setdefault(chunk_index, []).append(key)

        for chunk_index in sorted(chunk_to_intervals):
            row_start, row_end, _, _ = index_rows[chunk_index]
            row_start_i = int(row_start)
            row_end_i = int(row_end)
            if row_end_i <= row_start_i:
                continue
            chunk_rows = np.asarray(
                chrom_array[row_start_i:row_end_i],
                dtype=CACHE_ROW_DTYPE,
            ).reshape(-1, len(CACHE_COLUMNS))
            stats["chrom_chunk_reads"] += 1
            centers = chunk_rows[:, 1]
            for key in chunk_to_intervals[chunk_index]:
                _, start, end = key
                left = int(np.searchsorted(centers, start, side="left"))
                right = int(np.searchsorted(centers, end, side="left"))
                if right > left:
                    parts_by_interval[key].append(chunk_rows[left:right])

        for key in intervals:
            if key in results:
                continue
            parts = parts_by_interval[key]
            if not parts:
                results[key] = self._empty_rows()
                stats["empty_intervals"] += 1
                continue
            rows = self._filter_rows(
                np.concatenate(parts, axis=0),
                motif_idx=motif_idx,
                min_score=min_score,
            )
            results[key] = rows
            stats["rows_returned"] += int(rows.shape[0])
        return results, stats

    def query_many_with_stats(
        self,
        intervals,
        motif_idx: int | list[int] | None = None,
        min_score: float | None = None,
    ) -> tuple[dict[QueryInterval, np.ndarray], dict[str, int]]:
        """Return exact cache rows for many intervals while reusing row chunks."""
        keys = self._normalize_query_intervals(intervals)
        stats = {
            "n_intervals": len(keys),
            "indexed_chroms": 0,
            "fallback_chroms": 0,
            "chrom_chunk_reads": 0,
            "empty_intervals": 0,
            "rows_returned": 0,
        }
        results: dict[QueryInterval, np.ndarray] = {}
        intervals_by_chrom: dict[str, list[QueryInterval]] = {}
        for key in keys:
            intervals_by_chrom.setdefault(key[0], []).append(key)

        for chrom, chrom_intervals in intervals_by_chrom.items():
            indexed = self._query_many_chrom_indexed(
                chrom,
                chrom_intervals,
                motif_idx=motif_idx,
                min_score=min_score,
            )
            if indexed is not None:
                chrom_results, chrom_stats = indexed
                results.update(chrom_results)
                for key, value in chrom_stats.items():
                    stats[key] += int(value)
                continue

            stats["fallback_chroms"] += 1
            rows = self._load_chrom_rows(chrom)
            if rows.shape[0] == 0:
                for key in chrom_intervals:
                    results[key] = self._empty_rows()
                    stats["empty_intervals"] += 1
                continue
            centers = self._chrom_center_cache[chrom]
            for key in chrom_intervals:
                _, start, end = key
                if end <= start:
                    results[key] = self._empty_rows()
                    stats["empty_intervals"] += 1
                    continue
                left = int(np.searchsorted(centers, start, side="left"))
                right = int(np.searchsorted(centers, end, side="left"))
                if right <= left:
                    results[key] = self._empty_rows()
                    stats["empty_intervals"] += 1
                    continue
                interval_rows = self._filter_rows(
                    rows[left:right],
                    motif_idx=motif_idx,
                    min_score=min_score,
                )
                results[key] = interval_rows
                stats["rows_returned"] += int(interval_rows.shape[0])
        return results, stats

    def query_many(
        self,
        intervals,
        motif_idx: int | list[int] | None = None,
        min_score: float | None = None,
    ) -> dict[QueryInterval, np.ndarray]:
        """Return exact cache rows for many intervals."""
        results, _ = self.query_many_with_stats(
            intervals,
            motif_idx=motif_idx,
            min_score=min_score,
        )
        return results

    def validate_metadata(self, expected_metadata: dict) -> list[str]:
        """Return cache metadata mismatches against the current motif/run settings."""
        cache_metadata = self.metadata
        mismatches: list[str] = []
        if cache_metadata.get("schema_version") != SCHEMA_VERSION:
            mismatches.append(
                f"schema_version cache={cache_metadata.get('schema_version')!r} "
                f"expected={SCHEMA_VERSION!r}"
            )
        for key, expected_value in CACHE_INVARIANT_METADATA.items():
            if key not in cache_metadata:
                mismatches.append(f"{key} missing from cache metadata")
                continue
            cache_value = _normalized_metadata_value(cache_metadata[key])
            expected_value = _normalized_metadata_value(expected_value)
            if cache_value != expected_value:
                mismatches.append(f"{key} cache={cache_value!r} expected={expected_value!r}")
        for key in METADATA_VALIDATION_KEYS:
            if key not in expected_metadata:
                continue
            if key not in cache_metadata:
                mismatches.append(f"{key} missing from cache metadata")
                continue
            cache_value = _normalized_metadata_value(cache_metadata[key])
            expected_value = _normalized_metadata_value(expected_metadata[key])
            if cache_value != expected_value:
                mismatches.append(f"{key} cache={cache_value!r} expected={expected_value!r}")
        return mismatches

    def query(
        self,
        chrom: str,
        start: int,
        end: int,
        motif_idx: int | list[int] | None = None,
        min_score: float | None = None,
    ) -> np.ndarray:
        """Return cache rows whose center coordinate falls in ``[start, end)``."""
        indexed_rows = self._query_chrom_indexed(
            chrom,
            start,
            end,
            motif_idx=motif_idx,
            min_score=min_score,
        )
        if indexed_rows is not None:
            return indexed_rows

        rows = self._load_chrom_rows(chrom)

        if rows.shape[0] == 0:
            return rows.reshape(0, len(CACHE_COLUMNS)).astype(CACHE_ROW_DTYPE)

        centers = self._chrom_center_cache[chrom]
        left = int(np.searchsorted(centers, start, side="left"))
        right = int(np.searchsorted(centers, end, side="left"))
        return self._filter_rows(rows[left:right], motif_idx=motif_idx, min_score=min_score)

    def accumulate_regions(
        self,
        regions_df: pd.DataFrame,
        *,
        window_size: int,
        n_motifs: int | None = None,
        query_motif_index: int | None = None,
    ) -> np.ndarray:
        """Accumulate sparse one-to-all or all-to-all profiles from cached hits.

        The sparse cache semantics use significant motif-hit scores, not all
        below-threshold PWM scores. Regions are counted independently, matching
        repeated subset semantics even when genomic intervals overlap.
        """
        if n_motifs is None:
            n_motifs = int(self.metadata.get("motif_count", len(self.motif_names)))
        width = 2 * window_size + 1
        if query_motif_index is None:
            out = np.zeros((n_motifs, n_motifs, width), dtype=np.float32)
        else:
            out = np.zeros((n_motifs, width), dtype=np.float32)

        for row in regions_df.itertuples(index=False):
            chrom = getattr(row, "chrom", getattr(row, "Chromosome", None))
            start = int(getattr(row, "start", getattr(row, "Start", 0)))
            end = int(getattr(row, "end", getattr(row, "End", 0)))
            region_hits = self.query(chrom, start, end)
            if region_hits.size == 0:
                continue

            accumulate_hit_rows(
                out,
                region_hits,
                window_size=window_size,
                query_motif_index=query_motif_index,
            )
        return out


def _write_chrom_chunk_index(index_group, chrom: str, chrom_array) -> None:
    n_rows = int(chrom_array.shape[0])
    if n_rows == 0:
        index_rows = np.zeros((0, len(HIT_CHUNK_INDEX_COLUMNS)), dtype=np.float64)
    else:
        row_chunk_size = int(chrom_array.chunks[0]) if chrom_array.chunks else n_rows
        records = []
        for row_start in range(0, n_rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, n_rows)
            centers = np.asarray(chrom_array[row_start:row_end, 1], dtype=np.float64)
            if centers.size == 0:
                continue
            records.append((row_start, row_end, float(centers[0]), float(centers[-1])))
        index_rows = np.asarray(records, dtype=np.float64).reshape(-1, len(HIT_CHUNK_INDEX_COLUMNS))
    chunks = (
        max(1, min(index_rows.shape[0], 10_000)),
        len(HIT_CHUNK_INDEX_COLUMNS),
    )
    index_array = index_group.create_array(
        chrom,
        data=index_rows,
        chunks=chunks,
        compressors=COMPRESSOR,
        overwrite=True,
    )
    index_array.attrs["columns"] = HIT_CHUNK_INDEX_COLUMNS
    index_array.attrs["schema_version"] = HIT_CHUNK_INDEX_SCHEMA_VERSION


def build_positional_motif_hit_cache(
    *,
    genome_zarr_path: str,
    output_path: str | Path,
    motif_parameters: tuple,
    motif_metadata: dict,
    target_chromosomes: list[str] | None = None,
    region_intervals: pd.DataFrame | None = None,
    tile_size: int = 500,
    tile_extension_bp: int = 15,
    batch_size: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    strand_specific: bool = False,
    p_value_threshold: str = "p0.0001",
) -> CacheBuildStats:
    """Build a genome-tile positional motif-hit cache."""
    motif_names, motif_kernels, _, score_thresholds, _, _ = motif_parameters
    thresholds = score_thresholds[p_value_threshold]
    threshold_shape = (
        list(thresholds.shape)
        if isinstance(thresholds, torch.Tensor)
        else list(np.asarray(thresholds).shape)
    )
    center_offsets_np = motif_center_offsets(motif_kernels)
    center_offsets = torch.as_tensor(center_offsets_np, device=device)
    sequence_database = SequenceDenseZarrIO(genome_zarr_path, mode="r")
    chrom_sizes = {
        str(chrom): _chrom_size_from_store(sequence_database, str(chrom))
        for chrom in sequence_database.chroms
    }
    merged_region_intervals = _merged_region_intervals(
        region_intervals,
        chrom_sizes=chrom_sizes,
        target_chromosomes=target_chromosomes,
    )
    use_region_intervals = region_intervals is not None
    chroms = (
        list(merged_region_intervals)
        if use_region_intervals
        else target_chromosomes or list(sequence_database.chroms)
    )
    coverage_bp = int(
        sum(
            end - start
            for intervals in merged_region_intervals.values()
            for start, end in intervals
        )
    )
    coverage_regions = int(sum(len(intervals) for intervals in merged_region_intervals.values()))

    output_path = Path(output_path)
    cache = PositionalMotifHitCache(output_path, mode="w")
    metadata_group = cache.dataset.require_group("metadata", overwrite=True)
    motif_names_array = metadata_group.create_array(
        "motif_names",
        shape=(len(motif_names),),
        dtype=VariableLengthUTF8(),
        overwrite=True,
    )
    motif_names_array[:] = list(motif_names)
    metadata_group.create_array(
        "motif_center_offsets",
        data=center_offsets_np.astype(np.int32),
        overwrite=True,
    )
    metadata = {
        **motif_metadata,
        "schema_version": SCHEMA_VERSION,
        "genome_zarr_path": str(genome_zarr_path),
        "coverage_scope": "region_intervals" if use_region_intervals else "genome_tiles",
        "coverage_regions": coverage_regions if use_region_intervals else None,
        "coverage_bp": coverage_bp if use_region_intervals else None,
        "tile_size": int(tile_size),
        "tile_extension_bp": int(tile_extension_bp),
        "strand_specific": bool(strand_specific),
        "p_value_threshold": p_value_threshold,
        "score_threshold_vector_checksum": threshold_vector_checksum(thresholds),
        "score_threshold_vector_shape": threshold_shape,
        "columns": CACHE_COLUMNS,
        "row_dtype": str(np.dtype(CACHE_ROW_DTYPE)),
        "motif_count": len(motif_names),
        "cache_accumulation_semantics": CACHE_ACCUMULATION_SEMANTICS,
        "coordinate_frame": CACHE_COORDINATE_FRAME,
        "hit_storage_layout": HIT_STORAGE_LAYOUT,
        "hit_chunk_index_schema_version": HIT_CHUNK_INDEX_SCHEMA_VERSION,
    }
    metadata_group.attrs.update(_json_safe_attrs(metadata))

    hits_group = cache.dataset.require_group("hits/regions", overwrite=True)
    chunk_index_group = cache.dataset.require_group("hits/chunk_index", overwrite=True)
    temp_group = cache.dataset.require_group("_tmp_hits", overwrite=True)
    total_hits = 0
    total_bp = 0
    total_regions = 0
    motif_len = int(motif_kernels.shape[1])

    for chrom in chroms:
        chrom_size = chrom_sizes[chrom]
        temp_chrom_group = temp_group.require_group(chrom, overwrite=True)
        chrom_chunk_index = 0
        chrom_hit_count = 0
        tile_records: list[tuple[int, int, int, int]] = []
        if use_region_intervals:
            for tile_start, tile_end in merged_region_intervals.get(chrom, []):
                scan_start = max(tile_start - tile_extension_bp - motif_len, 0)
                scan_end = min(tile_end + tile_extension_bp + motif_len, chrom_size)
                if scan_end <= scan_start:
                    continue
                tile_records.append((tile_start, tile_end, scan_start, scan_end))
        else:
            for tile_start in range(0, chrom_size, tile_size):
                tile_end = min(tile_start + tile_size, chrom_size)
                scan_start = max(tile_start - tile_extension_bp - motif_len, 0)
                scan_end = min(tile_end + tile_extension_bp + motif_len, chrom_size)
                if scan_end <= scan_start:
                    continue
                tile_records.append((tile_start, tile_end, scan_start, scan_end))

        for batch_start in tqdm(
            range(0, len(tile_records), batch_size),
            desc=f"Building motif-hit cache {chrom}",
        ):
            batch_records = tile_records[batch_start : batch_start + batch_size]
            sequences = []
            sequence_starts = []
            sequence_lengths = []
            tile_bounds = []
            max_len = 0
            for tile_start, tile_end, scan_start, scan_end in batch_records:
                seq = sequence_database.get_track(
                    chrom, scan_start, scan_end, output_format="raw_array"
                ).astype(np.float32)
                if seq.ndim != 2 or seq.shape[1] != 4:
                    raise ValueError(
                        f"Expected one-hot sequence array with shape (bp, 4) for "
                        f"{chrom}:{tile_start}-{scan_end}, got {seq.shape}"
                    )
                if seq.shape[0] < motif_len:
                    continue
                sequences.append(seq)
                sequence_starts.append(scan_start)
                sequence_lengths.append(seq.shape[0])
                tile_bounds.append((tile_start, tile_end))
                max_len = max(max_len, seq.shape[0])
                total_bp += seq.shape[0]
                total_regions += 1
            if not sequences:
                continue
            batch_rows: list[np.ndarray] = []

            padded = [
                np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), mode="constant")
                if seq.shape[0] < max_len
                else seq
                for seq in sequences
            ]
            batch_tensor = torch.as_tensor(np.asarray(padded), dtype=dtype, device=device)
            sequence_starts_tensor = torch.as_tensor(
                sequence_starts, dtype=torch.int64, device=device
            )
            sequence_lengths_tensor = torch.as_tensor(
                sequence_lengths, dtype=torch.int64, device=device
            )
            strands = [(0, batch_tensor, False)]
            if not strand_specific:
                from .sequence_io import reverse_complement_batch

                strands.append((1, reverse_complement_batch(batch_tensor), True))

            for strand_id, strand_tensor, reverse_strand in strands:
                scores = _get_compiled_conv_fn()(
                    strand_tensor.permute(0, 2, 1),
                    motif_kernels.permute(0, 2, 1),
                )
                rows, row_batch_indices = collect_positional_hits(
                    scores,
                    thresholds,
                    sequence_starts_tensor,
                    center_offsets,
                    strand_id=strand_id,
                    motif_length=motif_len,
                    sequence_lengths=sequence_lengths_tensor,
                    padded_sequence_length=max_len,
                    reverse_strand=reverse_strand,
                    p_value_threshold=p_value_threshold,
                    return_batch_indices=True,
                )
                if rows.shape[0] == 0:
                    continue
                rows_np = rows.detach().cpu().numpy().astype(CACHE_ROW_DTYPE)
                row_batch_indices_np = row_batch_indices.detach().cpu().numpy()
                # Keep each hit once: its center must fall inside the original tile.
                keep = np.zeros(rows_np.shape[0], dtype=bool)
                for seq_i, (tile_start, tile_end) in enumerate(tile_bounds):
                    seq_mask = row_batch_indices_np == seq_i
                    center_mask = (rows_np[:, 1] >= tile_start) & (rows_np[:, 1] < tile_end)
                    keep |= seq_mask & center_mask
                if np.any(keep):
                    batch_rows.append(rows_np[keep])

            if batch_rows:
                chunk_data = np.concatenate(batch_rows, axis=0).astype(CACHE_ROW_DTYPE)
                order = np.lexsort((chunk_data[:, 2], chunk_data[:, 1]))
                chunk_data = chunk_data[order]
                chunk = temp_chrom_group.create_array(
                    f"chunk_{chrom_chunk_index:06d}",
                    data=chunk_data,
                    chunks=(
                        min(max(len(chunk_data), 1), 100_000),
                        len(CACHE_COLUMNS),
                    ),
                    compressors=COMPRESSOR,
                    overwrite=True,
                )
                chunk.attrs["columns"] = CACHE_COLUMNS
                chunk.attrs["n_hits"] = int(chunk_data.shape[0])
                chrom_chunk_index += 1
                chrom_hit_count += int(chunk_data.shape[0])

        if chrom_hit_count:
            chrom_array = hits_group.create_array(
                chrom,
                shape=(chrom_hit_count, len(CACHE_COLUMNS)),
                dtype=CACHE_ROW_DTYPE,
                chunks=(min(chrom_hit_count, 100_000), len(CACHE_COLUMNS)),
                compressors=COMPRESSOR,
                overwrite=True,
            )
            write_offset = 0
            previous_center = None
            for chunk_name in sorted(temp_chrom_group.keys()):
                chunk_data = temp_chrom_group[chunk_name][:]
                if previous_center is not None and chunk_data.shape[0]:
                    if chunk_data[0, 1] < previous_center:
                        raise RuntimeError(
                            f"Internal cache chunk ordering failed for {chrom}: "
                            f"{chunk_name} starts at {chunk_data[0, 1]} after "
                            f"{previous_center}"
                        )
                if chunk_data.shape[0]:
                    previous_center = float(chunk_data[-1, 1])
                chrom_array[write_offset : write_offset + chunk_data.shape[0]] = chunk_data
                write_offset += chunk_data.shape[0]
        else:
            chrom_array = hits_group.create_array(
                chrom,
                data=np.zeros((0, len(CACHE_COLUMNS)), dtype=CACHE_ROW_DTYPE),
                chunks=(1, len(CACHE_COLUMNS)),
                compressors=COMPRESSOR,
                overwrite=True,
            )
        chrom_array.attrs["columns"] = CACHE_COLUMNS
        chrom_array.attrs["n_hits"] = int(chrom_hit_count)
        chrom_array.attrs["storage_layout"] = HIT_STORAGE_LAYOUT
        chrom_array.attrs["n_build_chunks"] = int(chrom_chunk_index)
        _write_chrom_chunk_index(chunk_index_group, chrom, chrom_array)
        total_hits += int(chrom_hit_count)

    metadata_group.attrs["n_hits"] = int(total_hits)
    metadata_group.attrs["bp_scanned"] = int(total_bp)
    metadata_group.attrs["n_regions"] = int(total_regions)
    try:
        del cache.dataset["_tmp_hits"]
    except Exception as exc:
        logger.warning("Could not remove temporary cache shards: %s", exc)
    logger.info(
        "Built positional motif-hit cache at %s with %,d hits",
        output_path,
        total_hits,
    )
    return CacheBuildStats(
        path=str(output_path),
        n_hits=total_hits,
        bp_scanned=total_bp,
        n_regions=total_regions,
    )
