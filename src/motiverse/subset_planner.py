"""Multi-subset aggregation helpers for positional motif-hit caches."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec
from zarr.core.dtype import VariableLengthUTF8

from .positional_cache import PositionalMotifHitCache, accumulate_hit_rows

COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
SUBSET_ZARR_SCHEMA_VERSION = "subset_motif_aggregation_v1"
INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION = "interval_contribution_cache_v1"
INTERVAL_CONTRIBUTION_SEMANTICS = "sparse_hit_contribution_per_exact_interval"
CACHE_PROVENANCE_KEYS = (
    "schema_version",
    "cache_schema_version",
    "motif_source",
    "aligned_motif_path",
    "loaded_with_aligned",
    "_loaded_with_aligned",
    "use_aligned",
    "require_aligned",
    "motif_count",
    "motif_selection",
    "motif_names_checksum",
    "motif_kernel_checksum",
    "motif_kernel_shape",
    "threshold_mode",
    "p_value_threshold",
    "score_threshold",
    "score_threshold_vector_checksum",
    "score_threshold_vector_shape",
    "genome_assembly",
    "genome_zarr_path",
    "n_hits",
    "bp_scanned",
    "n_regions",
    "hit_storage_layout",
    "hit_chunk_index_schema_version",
    "row_dtype",
    "cache_accumulation_semantics",
    "coordinate_frame",
)


@dataclass(frozen=True)
class MultiSubsetAggregationPlan:
    subset_column: str
    n_input_rows: int
    n_dropped_invalid_regions: int
    n_subsets: int
    n_region_memberships: int
    n_unique_regions: int
    reused_region_memberships: int
    reuse_factor: float
    n_motifs: int
    window_size: int
    query_motif_index: int | None
    output_shape: tuple[int, ...]
    output_bytes: int
    dtype: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["output_shape"] = list(self.output_shape)
        return data


@dataclass(frozen=True)
class MultiSubsetAggregationResult:
    subset_ids: list[str]
    values: np.ndarray
    plan: MultiSubsetAggregationPlan
    timings: dict[str, float] | None = None


@dataclass(frozen=True)
class MultiSubsetZarrResult:
    path: str
    subset_ids: list[str]
    plan: MultiSubsetAggregationPlan
    checksum: str | None
    timings: dict[str, float] | None = None


@dataclass(frozen=True)
class IntervalContributionCacheResult:
    path: str
    n_unique_regions: int
    contribution_shape: tuple[int, ...]
    checksum: str | None
    timings: dict[str, float] | None = None


def cache_metadata_snapshot(cache: PositionalMotifHitCache) -> dict:
    return {key: cache.metadata.get(key) for key in CACHE_PROVENANCE_KEYS if key in cache.metadata}


def _contribution_shape(
    *,
    n_motifs: int,
    window_size: int,
    query_motif_index: int | None,
) -> tuple[int, ...]:
    width = 2 * int(window_size) + 1
    if query_motif_index is None:
        return (int(n_motifs), int(n_motifs), width)
    return (int(n_motifs), width)


def _unique_regions_frame(normalized: pd.DataFrame) -> pd.DataFrame:
    return (
        normalized[["chrom", "start", "end"]]
        .drop_duplicates()
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )


def _interval_key(chrom: str, start: int, end: int) -> tuple[str, int, int]:
    return str(chrom), int(start), int(end)


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Missing required column; tried {candidates}")


def _normalized_regions(df: pd.DataFrame, subset_column: str) -> pd.DataFrame:
    chrom_col = _resolve_column(df, ("chrom", "Chromosome"))
    start_col = _resolve_column(df, ("start", "Start"))
    end_col = _resolve_column(df, ("end", "End"))
    if subset_column not in df.columns:
        raise ValueError(f"Missing subset column: {subset_column}")
    starts = pd.to_numeric(df[start_col], errors="coerce")
    ends = pd.to_numeric(df[end_col], errors="coerce")
    valid = starts.notna() & ends.notna()
    valid &= np.isfinite(starts.to_numpy(dtype=float, na_value=np.nan))
    valid &= np.isfinite(ends.to_numpy(dtype=float, na_value=np.nan))
    valid &= starts >= 0
    valid &= ends > starts
    normalized = pd.DataFrame(
        {
            "subset_id": df.loc[valid, subset_column].astype(str),
            "chrom": df.loc[valid, chrom_col].astype(str),
            "start": starts.loc[valid].astype(np.int64),
            "end": ends.loc[valid].astype(np.int64),
        }
    )
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def plan_region_subset_aggregation(
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    n_motifs: int,
    window_size: int,
    query_motif_index: int | None = None,
    dtype: np.dtype | str = np.float32,
) -> MultiSubsetAggregationPlan:
    """Estimate query reuse and dense output size for a subset-family run."""
    n_input_rows = int(len(region_memberships_df))
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    unique_regions = normalized[["chrom", "start", "end"]].drop_duplicates()
    n_memberships = int(len(normalized))
    n_unique_regions = int(len(unique_regions))
    reused = n_memberships - n_unique_regions
    width = 2 * window_size + 1
    if query_motif_index is None:
        output_shape = (len(subset_ids), n_motifs, n_motifs, width)
    else:
        output_shape = (len(subset_ids), n_motifs, width)
    output_bytes = int(np.prod(output_shape, dtype=np.int64) * np.dtype(dtype).itemsize)
    return MultiSubsetAggregationPlan(
        subset_column=subset_column,
        n_input_rows=n_input_rows,
        n_dropped_invalid_regions=n_input_rows - n_memberships,
        n_subsets=len(subset_ids),
        n_region_memberships=n_memberships,
        n_unique_regions=n_unique_regions,
        reused_region_memberships=reused,
        reuse_factor=(n_memberships / n_unique_regions) if n_unique_regions else 0.0,
        n_motifs=int(n_motifs),
        window_size=int(window_size),
        query_motif_index=query_motif_index,
        output_shape=tuple(int(x) for x in output_shape),
        output_bytes=output_bytes,
        dtype=str(np.dtype(dtype)),
    )


def _membership_subset_counts(
    memberships: pd.DataFrame, subset_to_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    subset_indices = np.asarray(
        [subset_to_index[subset_id] for subset_id in memberships["subset_id"]],
        dtype=np.int64,
    )
    if subset_indices.size == 0:
        return subset_indices, np.zeros((0,), dtype=np.int64)
    return np.unique(subset_indices, return_counts=True)


def _contiguous_index_runs(
    subset_indices: np.ndarray, counts: np.ndarray
) -> list[tuple[int, int, np.ndarray]]:
    if subset_indices.size == 0:
        return []
    runs = []
    run_start = 0
    for idx in range(1, len(subset_indices)):
        if subset_indices[idx] != subset_indices[idx - 1] + 1:
            runs.append(
                (
                    int(subset_indices[run_start]),
                    int(subset_indices[idx - 1]) + 1,
                    counts[run_start:idx],
                )
            )
            run_start = idx
    runs.append(
        (
            int(subset_indices[run_start]),
            int(subset_indices[-1]) + 1,
            counts[run_start:],
        )
    )
    return runs


def _add_contribution_to_subset_values(
    values,
    subset_indices: np.ndarray,
    counts: np.ndarray,
    contribution: np.ndarray,
    *,
    dtype: np.dtype,
    chunk_subsets: int,
) -> None:
    if subset_indices.size == 0:
        return
    contribution = contribution.astype(dtype, copy=False)
    count_shape = (0,) + (1,) * contribution.ndim
    max_rows = max(1, int(chunk_subsets))
    values_is_ndarray = isinstance(values, np.ndarray)
    for run_start, run_end, run_counts in _contiguous_index_runs(subset_indices, counts):
        for start in range(run_start, run_end, max_rows):
            end = min(start + max_rows, run_end)
            offset = start - run_start
            local_counts = run_counts[offset : offset + (end - start)].astype(dtype)
            target = values[start:end]
            target_array = target if isinstance(target, np.ndarray) else np.asarray(target)
            if np.all(local_counts == 1):
                np.add(target_array, contribution, out=target_array)
            else:
                scaled = contribution * local_counts.reshape((len(local_counts),) + count_shape[1:])
                np.add(target_array, scaled, out=target_array)
            if not values_is_ndarray:
                values[start:end] = target_array


def accumulate_region_subsets(
    cache: PositionalMotifHitCache | str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    window_size: int,
    n_motifs: int | None = None,
    query_motif_index: int | None = None,
    dtype: np.dtype | str = np.float32,
    max_output_bytes: int | None = None,
) -> MultiSubsetAggregationResult:
    """Aggregate many region subsets while reusing duplicate interval work.

    Each unique ``chrom,start,end`` interval is queried and converted to a local
    co-occurrence contribution once. That contribution is then added to every
    subset containing the interval. This preserves repeated-subset semantics for
    duplicate memberships, while avoiding repeated cache queries and pair loops
    for shared intervals.
    """
    cache_obj = (
        cache if isinstance(cache, PositionalMotifHitCache) else PositionalMotifHitCache(cache)
    )
    if n_motifs is None:
        n_motifs = int(cache_obj.metadata.get("motif_count", len(cache_obj.motif_names)))
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    if max_output_bytes is not None and plan.output_bytes > max_output_bytes:
        raise MemoryError(
            f"Planned dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={max_output_bytes:,}. Use a narrower motif "
            "set, smaller window, one-to-all mode, or a streaming output strategy."
        )
    timings = {
        "query_wall_s": 0.0,
        "accumulate_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "nonzero_contribution_regions": 0.0,
    }
    values = np.zeros(plan.output_shape, dtype=dtype)
    if len(normalized) == 0:
        return MultiSubsetAggregationResult(subset_ids, values, plan, timings)

    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    width = 2 * window_size + 1
    if query_motif_index is None:
        contribution_shape = (n_motifs, n_motifs, width)
    else:
        contribution_shape = (n_motifs, width)

    for (chrom, start, end), memberships in grouped:
        timings["unique_regions_processed"] += 1
        contribution = np.zeros(contribution_shape, dtype=np.float32)
        query_start = time.perf_counter()
        region_hits = cache_obj.query(chrom, int(start), int(end))
        timings["query_wall_s"] += time.perf_counter() - query_start
        accumulate_start = time.perf_counter()
        accumulate_hit_rows(
            contribution,
            region_hits,
            window_size=window_size,
            query_motif_index=query_motif_index,
        )
        timings["accumulate_wall_s"] += time.perf_counter() - accumulate_start
        if not np.any(contribution):
            continue
        timings["nonzero_contribution_regions"] += 1
        subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
        update_start = time.perf_counter()
        _add_contribution_to_subset_values(
            values,
            subset_indices,
            counts,
            contribution,
            dtype=np.dtype(dtype),
            chunk_subsets=len(subset_indices),
        )
        timings["subset_update_wall_s"] += time.perf_counter() - update_start

    return MultiSubsetAggregationResult(subset_ids, values, plan, timings)


def _checksum_zarr_values(values_array) -> str:
    import hashlib

    digest = hashlib.sha256()
    for subset_i in range(values_array.shape[0]):
        digest.update(np.asarray(values_array[subset_i]).tobytes())
    return digest.hexdigest()


def _sample_subset_indices(n_subsets: int, sample_subsets: int) -> list[int]:
    if n_subsets <= 0 or sample_subsets <= 0:
        return []
    count = min(int(n_subsets), int(sample_subsets))
    return sorted(set(int(x) for x in np.linspace(0, n_subsets - 1, count)))


def _sample_checksum_zarr_values(
    values_array,
    *,
    sample_subsets: int,
) -> tuple[str, list[int]]:
    import hashlib

    digest = hashlib.sha256()
    indices = _sample_subset_indices(values_array.shape[0], sample_subsets)
    for subset_i in indices:
        digest.update(np.asarray([subset_i], dtype=np.int64).tobytes())
        digest.update(np.asarray(values_array[subset_i]).tobytes())
    return digest.hexdigest(), indices


def finalize_zarr_checksum(
    values_array,
    *,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
) -> tuple[str | None, dict]:
    """Return checksum plus metadata for full, sampled, or skipped verification."""
    mode = str(checksum_mode).lower()
    if mode not in {"full", "sample", "none"}:
        raise ValueError(
            f"checksum_mode must be one of 'full', 'sample', or 'none'; got {checksum_mode!r}"
        )
    if mode == "none":
        return None, {
            "values_checksum": None,
            "values_checksum_mode": mode,
            "values_checksum_status": "skipped",
            "values_checksum_sample_subsets": 0,
            "values_checksum_sample_indices": [],
        }
    if mode == "sample":
        checksum, indices = _sample_checksum_zarr_values(
            values_array,
            sample_subsets=checksum_sample_subsets,
        )
        return checksum, {
            "values_checksum": checksum,
            "values_checksum_mode": mode,
            "values_checksum_status": "sampled",
            "values_checksum_sample_subsets": len(indices),
            "values_checksum_sample_indices": indices,
        }
    checksum = _checksum_zarr_values(values_array)
    return checksum, {
        "values_checksum": checksum,
        "values_checksum_mode": mode,
        "values_checksum_status": "complete",
        "values_checksum_sample_subsets": 0,
        "values_checksum_sample_indices": [],
    }


def _interval_contribution_cache_metadata_snapshot(attrs: dict) -> dict:
    keys = (
        "schema_version",
        "complete",
        "semantics",
        "source_cache_path",
        "n_input_rows",
        "n_dropped_invalid_regions",
        "n_unique_regions",
        "n_region_memberships",
        "n_motifs",
        "window_size",
        "query_motif_index",
        "contribution_shape",
        "contribution_cache_bytes",
        "dtype",
        "values_checksum",
        "values_checksum_mode",
        "values_checksum_status",
    )
    return {key: attrs.get(key) for key in keys if key in attrs}


def _interval_contribution_key_to_index(root) -> dict[tuple[str, int, int], int]:
    chroms = root["intervals/chrom"][:].tolist()
    starts = root["intervals/start"][:].astype(np.int64)
    ends = root["intervals/end"][:].astype(np.int64)
    return {
        _interval_key(chrom, int(start), int(end)): idx
        for idx, (chrom, start, end) in enumerate(zip(chroms, starts, ends, strict=True))
    }


def build_interval_contribution_cache(
    cache: PositionalMotifHitCache | str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    window_size: int,
    n_motifs: int | None = None,
    query_motif_index: int | None = None,
    dtype: np.dtype | str = np.float32,
    max_cache_bytes: int | None = None,
    chunk_intervals: int = 1,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
) -> IntervalContributionCacheResult:
    """Build an exact per-interval contribution cache from positional hits.

    Each unique interval gets the same sparse-hit co-occurrence tensor produced
    by ``accumulate_hit_rows``. Later subset-family runs can apply those tensors
    to arbitrary subset memberships without re-querying motif-hit rows.
    """
    cache_obj = (
        cache if isinstance(cache, PositionalMotifHitCache) else PositionalMotifHitCache(cache)
    )
    if n_motifs is None:
        n_motifs = int(cache_obj.metadata.get("motif_count", len(cache_obj.motif_names)))
    dtype = np.dtype(dtype)
    normalized = _normalized_regions(region_memberships_df, subset_column)
    unique_regions = _unique_regions_frame(normalized)
    contribution_shape = _contribution_shape(
        n_motifs=int(n_motifs),
        window_size=int(window_size),
        query_motif_index=query_motif_index,
    )
    contribution_cache_bytes = int(
        len(unique_regions) * np.prod(contribution_shape, dtype=np.int64) * dtype.itemsize
    )
    if max_cache_bytes is not None and contribution_cache_bytes > max_cache_bytes:
        raise MemoryError(
            f"Planned interval contribution cache is {contribution_cache_bytes:,} bytes, "
            f"above max_cache_bytes={max_cache_bytes:,}."
        )

    output_path = Path(output_path)
    root = zarr.open_group(str(output_path), mode="w")
    metadata = root.require_group("metadata", overwrite=True)
    intervals = root.require_group("intervals", overwrite=True)
    intervals.create_array(
        "chrom",
        shape=(len(unique_regions),),
        dtype=VariableLengthUTF8(),
        overwrite=True,
    )[:] = unique_regions["chrom"].astype(str).tolist()
    intervals.create_array(
        "start",
        data=unique_regions["start"].to_numpy(dtype=np.int64),
        overwrite=True,
    )
    intervals.create_array(
        "end",
        data=unique_regions["end"].to_numpy(dtype=np.int64),
        overwrite=True,
    )
    values = root.create_array(
        "values",
        shape=(len(unique_regions),) + contribution_shape,
        chunks=(max(1, int(chunk_intervals)),) + contribution_shape,
        dtype=dtype,
        compressors=COMPRESSOR,
        overwrite=True,
    )
    metadata.attrs.update(
        {
            "schema_version": INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION,
            "complete": False,
            "semantics": INTERVAL_CONTRIBUTION_SEMANTICS,
            "source_cache_path": cache_obj.path,
            "subset_column": subset_column,
            "n_input_rows": int(len(region_memberships_df)),
            "n_dropped_invalid_regions": int(len(region_memberships_df) - len(normalized)),
            "n_region_memberships": int(len(normalized)),
            "n_unique_regions": int(len(unique_regions)),
            "n_motifs": int(n_motifs),
            "window_size": int(window_size),
            "query_motif_index": query_motif_index,
            "contribution_shape": list(contribution_shape),
            "contribution_cache_bytes": contribution_cache_bytes,
            "dtype": str(dtype),
            "cache_metadata": cache_metadata_snapshot(cache_obj),
        }
    )
    timings = {
        "query_wall_s": 0.0,
        "accumulate_wall_s": 0.0,
        "checksum_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "nonzero_contribution_regions": 0.0,
    }
    for interval_idx, row in enumerate(unique_regions.itertuples(index=False)):
        timings["unique_regions_processed"] += 1
        contribution = np.zeros(contribution_shape, dtype=np.float32)
        query_start = time.perf_counter()
        region_hits = cache_obj.query(str(row.chrom), int(row.start), int(row.end))
        timings["query_wall_s"] += time.perf_counter() - query_start
        accumulate_start = time.perf_counter()
        accumulate_hit_rows(
            contribution,
            region_hits,
            window_size=window_size,
            query_motif_index=query_motif_index,
        )
        timings["accumulate_wall_s"] += time.perf_counter() - accumulate_start
        if np.any(contribution):
            timings["nonzero_contribution_regions"] += 1
        values[interval_idx] = contribution.astype(dtype, copy=False)

    checksum_start = time.perf_counter()
    checksum, checksum_metadata = finalize_zarr_checksum(
        values,
        checksum_mode=checksum_mode,
        checksum_sample_subsets=checksum_sample_subsets,
    )
    timings["checksum_wall_s"] = time.perf_counter() - checksum_start
    metadata.attrs.update(timings)
    metadata.attrs.update(checksum_metadata)
    metadata.attrs["complete"] = True
    return IntervalContributionCacheResult(
        path=str(output_path),
        n_unique_regions=int(len(unique_regions)),
        contribution_shape=contribution_shape,
        checksum=checksum,
        timings=timings,
    )


def _accumulate_from_interval_contribution_cache_into_values(
    *,
    contribution_cache_root,
    normalized: pd.DataFrame,
    values,
    subset_to_index: dict[str, int],
    dtype: np.dtype,
    chunk_subsets: int,
    timings: dict[str, float],
) -> None:
    key_to_index = _interval_contribution_key_to_index(contribution_cache_root)
    contribution_values = contribution_cache_root["values"]
    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    for (chrom, start, end), memberships in grouped:
        key = _interval_key(chrom, int(start), int(end))
        if key not in key_to_index:
            raise ValueError(
                f"Interval contribution cache is missing required interval "
                f"{key[0]}:{key[1]}-{key[2]}"
            )
        lookup_start = time.perf_counter()
        contribution = np.asarray(contribution_values[key_to_index[key]])
        timings["contribution_lookup_wall_s"] += time.perf_counter() - lookup_start
        timings["contribution_cache_hits"] += 1
        if not np.any(contribution):
            continue
        timings["nonzero_contribution_regions"] += 1
        subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
        update_start = time.perf_counter()
        _add_contribution_to_subset_values(
            values,
            subset_indices,
            counts,
            contribution,
            dtype=dtype,
            chunk_subsets=chunk_subsets,
        )
        timings["subset_update_wall_s"] += time.perf_counter() - update_start


def accumulate_region_subsets_from_interval_contribution_cache(
    contribution_cache_path: str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    dtype: np.dtype | str | None = None,
    max_output_bytes: int | None = None,
) -> MultiSubsetAggregationResult:
    """Aggregate subset outputs using a precomputed interval contribution cache."""
    root = zarr.open_group(str(contribution_cache_path), mode="r")
    attrs = dict(root["metadata"].attrs)
    if attrs.get("schema_version") != INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Expected interval contribution cache schema "
            f"{INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION!r}; got "
            f"{attrs.get('schema_version')!r}."
        )
    if attrs.get("complete") is not True:
        raise ValueError("Interval contribution cache is incomplete.")
    n_motifs = int(attrs["n_motifs"])
    window_size = int(attrs["window_size"])
    query_motif_index = attrs.get("query_motif_index")
    if query_motif_index is not None:
        query_motif_index = int(query_motif_index)
    dtype = np.dtype(dtype or attrs.get("dtype", "float32"))
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    if max_output_bytes is not None and plan.output_bytes > max_output_bytes:
        raise MemoryError(
            f"Planned dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={max_output_bytes:,}."
        )
    values = np.zeros(plan.output_shape, dtype=dtype)
    timings = {
        "contribution_lookup_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "contribution_cache_hits": 0.0,
        "nonzero_contribution_regions": 0.0,
        "region_query_strategy": "interval_contribution_cache",
    }
    _accumulate_from_interval_contribution_cache_into_values(
        contribution_cache_root=root,
        normalized=normalized,
        values=values,
        subset_to_index=subset_to_index,
        dtype=dtype,
        chunk_subsets=max(1, len(subset_ids)),
        timings=timings,
    )
    return MultiSubsetAggregationResult(subset_ids, values, plan, timings)


def accumulate_region_subsets_from_interval_contribution_cache_to_zarr(
    contribution_cache_path: str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    dtype: np.dtype | str | None = None,
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
    buffered_subset_updates: bool = False,
    max_buffered_output_bytes: int | None = None,
) -> MultiSubsetZarrResult:
    """Write subset outputs from a precomputed interval contribution cache."""
    contribution_root = zarr.open_group(str(contribution_cache_path), mode="r")
    contribution_attrs = dict(contribution_root["metadata"].attrs)
    if contribution_attrs.get("schema_version") != INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Expected interval contribution cache schema "
            f"{INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION!r}; got "
            f"{contribution_attrs.get('schema_version')!r}."
        )
    if contribution_attrs.get("complete") is not True:
        raise ValueError("Interval contribution cache is incomplete.")
    n_motifs = int(contribution_attrs["n_motifs"])
    window_size = int(contribution_attrs["window_size"])
    query_motif_index = contribution_attrs.get("query_motif_index")
    if query_motif_index is not None:
        query_motif_index = int(query_motif_index)
    dtype = np.dtype(dtype or contribution_attrs.get("dtype", "float32"))
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    if max_output_bytes is not None and plan.output_bytes > max_output_bytes:
        raise MemoryError(
            f"Planned zarr subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={max_output_bytes:,}."
        )
    if buffered_subset_updates and (
        max_buffered_output_bytes is not None and plan.output_bytes > max_buffered_output_bytes
    ):
        raise MemoryError(
            f"Planned buffered subset output is {plan.output_bytes:,} bytes, "
            f"above max_buffered_output_bytes={max_buffered_output_bytes:,}."
        )

    output_path = Path(output_path)
    root = zarr.open_group(str(output_path), mode="w")
    metadata = root.require_group("metadata", overwrite=True)
    subset_array = metadata.create_array(
        "subset_ids",
        shape=(len(subset_ids),),
        dtype=VariableLengthUTF8(),
        overwrite=True,
    )
    subset_array[:] = subset_ids
    values = root.create_array(
        "values",
        shape=plan.output_shape,
        chunks=(max(1, int(chunk_subsets)),) + tuple(plan.output_shape[1:]),
        dtype=dtype,
        compressors=COMPRESSOR,
        overwrite=True,
    )
    contribution_metadata = _interval_contribution_cache_metadata_snapshot(contribution_attrs)
    metadata.attrs.update(
        {
            "schema_version": SUBSET_ZARR_SCHEMA_VERSION,
            "complete": False,
            "plan": plan.to_dict(),
            "subset_column": subset_column,
            "n_input_rows": plan.n_input_rows,
            "n_dropped_invalid_regions": plan.n_dropped_invalid_regions,
            "n_subsets": plan.n_subsets,
            "n_regions": plan.n_region_memberships,
            "n_unique_regions": plan.n_unique_regions,
            "reuse_factor": plan.reuse_factor,
            "motif_count": plan.n_motifs,
            "window_size": plan.window_size,
            "query_motif_index": query_motif_index,
            "output_bytes": plan.output_bytes,
            "dtype": str(dtype),
            "region_query_strategy": "interval_contribution_cache",
            "subset_update_mode": "buffered_full_output"
            if buffered_subset_updates
            else "zarr_interval_incremental",
            "buffered_output_bytes": plan.output_bytes if buffered_subset_updates else 0,
            "contribution_cache_path": str(contribution_cache_path),
            "contribution_cache_metadata": contribution_metadata,
            "cache_metadata": contribution_attrs.get("cache_metadata", {}),
        }
    )
    timings = {
        "contribution_lookup_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "checksum_wall_s": 0.0,
        "contribution_cache_hits": 0.0,
        "nonzero_contribution_regions": 0.0,
    }
    accumulation_values = values
    if buffered_subset_updates:
        accumulation_values = np.zeros(plan.output_shape, dtype=dtype)
    _accumulate_from_interval_contribution_cache_into_values(
        contribution_cache_root=contribution_root,
        normalized=normalized,
        values=accumulation_values,
        subset_to_index=subset_to_index,
        dtype=dtype,
        chunk_subsets=int(chunk_subsets),
        timings=timings,
    )
    if buffered_subset_updates:
        write_start = time.perf_counter()
        values[:] = accumulation_values
        timings["subset_update_wall_s"] += time.perf_counter() - write_start
    checksum_start = time.perf_counter()
    checksum, checksum_metadata = finalize_zarr_checksum(
        values,
        checksum_mode=checksum_mode,
        checksum_sample_subsets=checksum_sample_subsets,
    )
    timings["checksum_wall_s"] = time.perf_counter() - checksum_start
    metadata.attrs.update(timings)
    metadata.attrs.update(checksum_metadata)
    metadata.attrs["complete"] = True
    return MultiSubsetZarrResult(str(output_path), subset_ids, plan, checksum, timings)


def accumulate_region_subsets_to_zarr(
    cache: PositionalMotifHitCache | str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    window_size: int,
    n_motifs: int | None = None,
    query_motif_index: int | None = None,
    dtype: np.dtype | str = np.float32,
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
) -> MultiSubsetZarrResult:
    """Aggregate many region subsets into a zarr array without dense RAM output."""
    cache_obj = (
        cache if isinstance(cache, PositionalMotifHitCache) else PositionalMotifHitCache(cache)
    )
    if n_motifs is None:
        n_motifs = int(cache_obj.metadata.get("motif_count", len(cache_obj.motif_names)))
    dtype = np.dtype(dtype)
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    if max_output_bytes is not None and plan.output_bytes > max_output_bytes:
        raise MemoryError(
            f"Planned zarr subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={max_output_bytes:,}."
        )

    output_path = Path(output_path)
    root = zarr.open_group(str(output_path), mode="w")
    metadata = root.require_group("metadata", overwrite=True)
    subset_array = metadata.create_array(
        "subset_ids",
        shape=(len(subset_ids),),
        dtype=VariableLengthUTF8(),
        overwrite=True,
    )
    subset_array[:] = subset_ids
    metadata.attrs.update(
        {
            "schema_version": SUBSET_ZARR_SCHEMA_VERSION,
            "complete": False,
            "plan": plan.to_dict(),
            "subset_column": subset_column,
            "n_input_rows": plan.n_input_rows,
            "n_dropped_invalid_regions": plan.n_dropped_invalid_regions,
            "n_subsets": plan.n_subsets,
            "n_regions": plan.n_region_memberships,
            "n_unique_regions": plan.n_unique_regions,
            "reuse_factor": plan.reuse_factor,
            "motif_count": plan.n_motifs,
            "window_size": plan.window_size,
            "query_motif_index": query_motif_index,
            "output_bytes": plan.output_bytes,
            "dtype": str(dtype),
            "cache_path": cache_obj.path,
            "cache_metadata": cache_metadata_snapshot(cache_obj),
        }
    )
    timings = {
        "query_wall_s": 0.0,
        "accumulate_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "checksum_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "nonzero_contribution_regions": 0.0,
    }
    chunk_shape = (max(1, int(chunk_subsets)),) + tuple(plan.output_shape[1:])
    values = root.create_array(
        "values",
        shape=plan.output_shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressors=COMPRESSOR,
        overwrite=True,
    )
    if len(normalized) == 0:
        checksum_start = time.perf_counter()
        checksum, checksum_metadata = finalize_zarr_checksum(
            values,
            checksum_mode=checksum_mode,
            checksum_sample_subsets=checksum_sample_subsets,
        )
        timings["checksum_wall_s"] = time.perf_counter() - checksum_start
        metadata.attrs.update(timings)
        metadata.attrs.update(checksum_metadata)
        metadata.attrs["complete"] = True
        return MultiSubsetZarrResult(str(output_path), subset_ids, plan, checksum, timings)

    width = 2 * window_size + 1
    if query_motif_index is None:
        contribution_shape = (n_motifs, n_motifs, width)
    else:
        contribution_shape = (n_motifs, width)

    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    for (chrom, start, end), memberships in grouped:
        timings["unique_regions_processed"] += 1
        contribution = np.zeros(contribution_shape, dtype=np.float32)
        query_start = time.perf_counter()
        region_hits = cache_obj.query(chrom, int(start), int(end))
        timings["query_wall_s"] += time.perf_counter() - query_start
        accumulate_start = time.perf_counter()
        accumulate_hit_rows(
            contribution,
            region_hits,
            window_size=window_size,
            query_motif_index=query_motif_index,
        )
        timings["accumulate_wall_s"] += time.perf_counter() - accumulate_start
        if not np.any(contribution):
            continue
        timings["nonzero_contribution_regions"] += 1
        subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
        update_start = time.perf_counter()
        _add_contribution_to_subset_values(
            values,
            subset_indices,
            counts,
            contribution,
            dtype=dtype,
            chunk_subsets=chunk_subsets,
        )
        timings["subset_update_wall_s"] += time.perf_counter() - update_start

    checksum_start = time.perf_counter()
    checksum, checksum_metadata = finalize_zarr_checksum(
        values,
        checksum_mode=checksum_mode,
        checksum_sample_subsets=checksum_sample_subsets,
    )
    timings["checksum_wall_s"] = time.perf_counter() - checksum_start
    metadata.attrs.update(timings)
    metadata.attrs.update(checksum_metadata)
    metadata.attrs["complete"] = True
    return MultiSubsetZarrResult(str(output_path), subset_ids, plan, checksum, timings)
