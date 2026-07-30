"""Exact dense-score subset aggregation helpers.

These helpers preserve the existing dense accumulation semantics from
``accumulate_motif_cooccurrences`` and ``accumulate_around_query_motif``:
significant motif hits are anchors, but all partner motif scores inside the
window contribute. This is intentionally different from the positional
significant-hit cache, which stores only above-threshold hit pairs.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from numcodecs import get_codec
from numcodecs.vlen import VLenUTF8
from numcodecs.zstd import Zstd
from zarr.core.dtype import VariableLengthUTF8

from .accumulation import (
    accumulate_around_query_motif,
    accumulate_motif_cooccurrences,
)
from .processing import _get_compiled_conv_fn
from .sequence_io import SequenceDenseZarrIO, reverse_complement_batch
from .subset_planner import (
    COMPRESSOR,
    MultiSubsetAggregationPlan,
    _add_contribution_to_subset_values,
    _contiguous_index_runs,
    _contribution_shape,
    _interval_key,
    _membership_subset_counts,
    _normalized_regions,
    _unique_regions_frame,
    finalize_zarr_checksum,
    plan_region_subset_aggregation,
)

DenseScoreProvider = Callable[[str, int, int], torch.Tensor | Iterable[torch.Tensor]]
DENSE_SUBSET_SEMANTICS = "dense_window_scores_around_significant_anchors"
DENSE_SUBSET_COORDINATE_FRAME = "convolution_score_start"
DENSE_SUBSET_ZARR_SCHEMA_VERSION = "dense_subset_aggregation_v1"
DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION = "dense_interval_contribution_cache_v1"
DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS = "dense_window_contribution_per_exact_interval"
DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION = "dense_query_motif_hit_cache_v1"
DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS = "query_motif_hit_positions_per_exact_interval"


class DirectSequenceZarrV2Reader:
    """Read legacy directory-chunked one-hot sequence Zarr arrays directly."""

    def __init__(
        self,
        path: str | Path,
        *,
        dtype: np.dtype | str = np.float32,
        max_cached_chunks: int = 4,
    ):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.max_cached_chunks = max(0, int(max_cached_chunks))
        self._chrom_metadata: dict[str, dict[str, Any]] = {}
        self._chunk_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self.stats = {
            "direct_sequence_reader": True,
            "direct_sequence_chunk_cache_hits": 0,
            "direct_sequence_chunk_cache_misses": 0,
            "direct_sequence_chunk_cache_evictions": 0,
            "direct_sequence_chunks_decoded": 0,
        }
        if not (self.path / ".zgroup").exists() or not (self.path / "chrs").exists():
            raise FileNotFoundError(f"Expected zarr-v2 sequence store at {self.path}")

    def _metadata(self, chrom: str) -> dict[str, Any]:
        chrom = str(chrom)
        metadata = self._chrom_metadata.get(chrom)
        if metadata is not None:
            return metadata
        metadata_path = self.path / "chrs" / chrom / ".zarray"
        if not metadata_path.exists():
            raise ValueError(f"Chromosome {chrom!r} not present in {self.path}")
        payload = json.loads(metadata_path.read_text())
        if int(payload.get("zarr_format", 0)) != 2:
            raise ValueError(f"Expected zarr-v2 chromosome array at {metadata_path}")
        if len(payload.get("shape", [])) != 2 or len(payload.get("chunks", [])) != 2:
            raise ValueError(f"Expected 2D chromosome array metadata at {metadata_path}")
        metadata = {
            "shape": tuple(int(x) for x in payload["shape"]),
            "chunks": tuple(int(x) for x in payload["chunks"]),
            "dtype": np.dtype(payload["dtype"]),
            "order": payload.get("order", "C"),
            "codec": get_codec(payload["compressor"]),
        }
        self._chrom_metadata[chrom] = metadata
        return metadata

    def _chunk(self, chrom: str, chunk_index: int) -> np.ndarray:
        key = (str(chrom), int(chunk_index))
        cached = self._chunk_cache.get(key)
        if cached is not None:
            self.stats["direct_sequence_chunk_cache_hits"] += 1
            self._chunk_cache.move_to_end(key)
            return cached
        self.stats["direct_sequence_chunk_cache_misses"] += 1
        metadata = self._metadata(chrom)
        chunk_path = self.path / "chrs" / str(chrom) / f"{int(chunk_index)}.0"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing sequence zarr chunk: {chunk_path}")
        decoded = metadata["codec"].decode(chunk_path.read_bytes())
        values = np.frombuffer(decoded, dtype=metadata["dtype"])
        feature_size = int(metadata["shape"][1])
        chunk = values.reshape((-1, feature_size), order=metadata["order"])
        self.stats["direct_sequence_chunks_decoded"] += 1
        if self.max_cached_chunks:
            self._chunk_cache[key] = chunk
            self._chunk_cache.move_to_end(key)
            while len(self._chunk_cache) > self.max_cached_chunks:
                self._chunk_cache.popitem(last=False)
                self.stats["direct_sequence_chunk_cache_evictions"] += 1
        return chunk

    def get_track(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        output_format: str = "raw_array",
    ) -> np.ndarray:
        if output_format != "raw_array":
            raise ValueError("DirectSequenceZarrV2Reader only supports raw_array output.")
        metadata = self._metadata(chrom)
        chrom_size, feature_size = metadata["shape"]
        start = max(0, int(start))
        end = min(int(chrom_size), int(end))
        if end <= start:
            return np.zeros((0, int(feature_size)), dtype=self.dtype)
        chunk_bp = int(metadata["chunks"][0])
        first_chunk = start // chunk_bp
        last_chunk = (end - 1) // chunk_bp
        pieces: list[np.ndarray] = []
        for chunk_index in range(first_chunk, last_chunk + 1):
            chunk_start = chunk_index * chunk_bp
            chunk = self._chunk(chrom, chunk_index)
            piece_start = max(start, chunk_start) - chunk_start
            piece_end = min(end, chunk_start + chunk.shape[0]) - chunk_start
            if piece_end > piece_start:
                pieces.append(chunk[piece_start:piece_end])
        if not pieces:
            return np.zeros((0, int(feature_size)), dtype=self.dtype)
        if len(pieces) == 1:
            return pieces[0].astype(self.dtype, copy=False)
        return np.concatenate(pieces, axis=0).astype(self.dtype, copy=False)


def _add_contribution_to_subset_values_torch(
    values: torch.Tensor,
    subset_indices: np.ndarray,
    counts: np.ndarray,
    contribution: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    chunk_subsets: int,
) -> None:
    if subset_indices.size == 0:
        return
    contribution = contribution.to(device=device, dtype=dtype)
    count_shape = (0,) + (1,) * (contribution.ndim)
    max_rows = max(1, int(chunk_subsets))
    for run_start, run_end, run_counts in _contiguous_index_runs(
        subset_indices,
        counts,
    ):
        for start in range(run_start, run_end, max_rows):
            end = min(start + max_rows, run_end)
            offset = start - run_start
            local_counts = run_counts[offset : offset + (end - start)]
            if np.all(local_counts == 1):
                values[start:end].add_(contribution)
                continue
            count_tensor = torch.as_tensor(
                local_counts,
                dtype=dtype,
                device=device,
            ).reshape((len(local_counts),) + count_shape[1:])
            values[start:end].add_(contribution * count_tensor)


def _dtype_itemsize(dtype: np.dtype | str | torch.dtype) -> int:
    if isinstance(dtype, torch.dtype):
        return int(torch.empty((), dtype=dtype).element_size())
    return int(np.dtype(dtype).itemsize)


def _estimated_dense_block_score_bytes(
    span_bp: int,
    *,
    motif_length: int,
    n_motifs: int,
    dtype: np.dtype | str | torch.dtype,
    strands: int,
) -> int:
    score_positions = max(0, int(span_bp) - int(motif_length) + 1)
    return int(score_positions) * int(n_motifs) * int(strands) * _dtype_itemsize(dtype)


class DenseGenomeScoreProvider:
    """Load genome intervals and return dense motif-score tensors on demand."""

    def __init__(
        self,
        genome_zarr_path: str,
        motif_kernels: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        strand_specific: bool = False,
    ):
        self.sequence_database = self._open_sequence_database(genome_zarr_path)
        self.motif_kernels = motif_kernels.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.strand_specific = strand_specific
        self.motif_length = int(motif_kernels.shape[1])
        self.stats = {
            "provider_intervals": 0,
            "provider_bp_loaded": 0,
            "provider_score_batches": 0,
            "provider_scan_wall_s": 0.0,
            "provider_sequence_reader": type(self.sequence_database).__name__,
        }

    @staticmethod
    def _open_sequence_database(genome_zarr_path: str | Path):
        try:
            return DirectSequenceZarrV2Reader(genome_zarr_path, dtype=np.float32)
        except Exception:
            return SequenceDenseZarrIO(str(genome_zarr_path), mode="r")

    def __call__(self, chrom: str, start: int, end: int) -> list[torch.Tensor]:
        self.stats["provider_intervals"] += 1
        start = int(start)
        end = int(end)
        if end <= start:
            return []
        sequence = self.sequence_database.get_track(
            chrom,
            start,
            end,
            output_format="raw_array",
        ).astype(np.float32)
        self.stats["provider_bp_loaded"] += int(sequence.shape[0])
        if sequence.shape[0] < self.motif_length:
            empty = torch.zeros(
                (1, int(self.motif_kernels.shape[0]), 0),
                dtype=self.dtype,
                device=self.device,
            )
            return [empty]

        batch_tensor = torch.as_tensor(
            sequence[None, :, :],
            dtype=self.dtype,
            device=self.device,
        )
        kernels = self.motif_kernels.permute(0, 2, 1)
        conv_fn = _get_compiled_conv_fn()
        scan_start = time.perf_counter()
        scores = [
            conv_fn(
                batch_tensor.permute(0, 2, 1),
                kernels,
            )
        ]
        if not self.strand_specific:
            reverse_tensor = reverse_complement_batch(batch_tensor)
            scores.append(conv_fn(reverse_tensor.permute(0, 2, 1), kernels))
        self.stats["provider_scan_wall_s"] += time.perf_counter() - scan_start
        self.stats["provider_score_batches"] += len(scores)
        return scores


class DenseGenomeBlockScoreProvider:
    """Scan coalesced genome blocks once and slice dense interval scores."""

    def __init__(
        self,
        genome_zarr_path: str,
        motif_kernels: torch.Tensor,
        region_memberships_df: pd.DataFrame,
        *,
        subset_column: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        strand_specific: bool = False,
        max_gap_bp: int = 0,
        max_block_span_bp: int = 1_000_000,
        max_block_score_bytes: int | None = None,
        max_cached_blocks: int | None = None,
        max_sequence_cache_bytes: int | None = 512 * 1024 * 1024,
    ):
        self.sequence_database = DenseGenomeScoreProvider._open_sequence_database(genome_zarr_path)
        self.motif_kernels = motif_kernels.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.strand_specific = strand_specific
        self.motif_length = int(motif_kernels.shape[1])
        self.max_gap_bp = int(max_gap_bp)
        self.max_block_span_bp = int(max_block_span_bp)
        self.max_block_score_bytes = (
            None if max_block_score_bytes is None else int(max_block_score_bytes)
        )
        self.max_cached_blocks = (
            None if max_cached_blocks is None else max(0, int(max_cached_blocks))
        )
        self.max_sequence_cache_bytes = (
            None if max_sequence_cache_bytes is None else max(0, int(max_sequence_cache_bytes))
        )
        self._block_cache: OrderedDict[tuple[str, int, int], list[torch.Tensor]] = OrderedDict()
        self._sequence_chunk_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self._sequence_chunk_cache_bytes = 0
        self._interval_to_block, self._blocks = _plan_dense_genome_blocks(
            region_memberships_df,
            subset_column=subset_column,
            max_gap_bp=self.max_gap_bp,
            max_block_span_bp=self.max_block_span_bp,
            max_block_score_bytes=self.max_block_score_bytes,
            n_motifs=int(self.motif_kernels.shape[0]),
            motif_length=self.motif_length,
            dtype=self.dtype,
            strands=1 if self.strand_specific else 2,
        )
        block_count = len(self._blocks)
        block_spans = np.asarray(
            [int(end) - int(start) for _, start, end in self._blocks],
            dtype=np.int64,
        )
        block_score_bytes = np.asarray(
            [
                _estimated_dense_block_score_bytes(
                    int(end) - int(start),
                    motif_length=self.motif_length,
                    n_motifs=int(self.motif_kernels.shape[0]),
                    dtype=self.dtype,
                    strands=1 if self.strand_specific else 2,
                )
                for _, start, end in self._blocks
            ],
            dtype=np.int64,
        )
        unique_region_bp = int(
            sum(int(end) - int(start) for _, start, end in self._interval_to_block)
        )
        coalesced_block_bp = int(block_spans.sum()) if block_spans.size else 0
        block_score_bytes_max = int(block_score_bytes.max()) if block_score_bytes.size else 0
        if (
            self.max_block_score_bytes is not None
            and block_score_bytes_max > self.max_block_score_bytes
        ):
            raise MemoryError(
                "At least one required dense genome block has estimated score "
                f"size {block_score_bytes_max:,} bytes, above "
                f"max_block_score_bytes={self.max_block_score_bytes:,}."
            )
        block_plan_checksum = hashlib.sha256(
            json.dumps(self._blocks, separators=(",", ":")).encode()
        ).hexdigest()
        self.stats = {
            "provider_intervals": 0,
            "provider_unique_intervals": len(self._interval_to_block),
            "provider_blocks_planned": block_count,
            "provider_blocks_scanned": 0,
            "provider_bp_loaded": 0,
            "provider_unique_region_bp": unique_region_bp,
            "provider_coalesced_block_bp": coalesced_block_bp,
            "provider_block_bp_vs_unique_region_bp": (
                float(coalesced_block_bp) / float(unique_region_bp) if unique_region_bp else 0.0
            ),
            "provider_block_span_min": (int(block_spans.min()) if block_spans.size else 0),
            "provider_block_span_median": (
                float(np.median(block_spans)) if block_spans.size else 0.0
            ),
            "provider_block_span_max": (int(block_spans.max()) if block_spans.size else 0),
            "provider_block_score_bytes_sum": (
                int(block_score_bytes.sum()) if block_score_bytes.size else 0
            ),
            "provider_block_score_bytes_median": (
                float(np.median(block_score_bytes)) if block_score_bytes.size else 0.0
            ),
            "provider_block_score_bytes_max": block_score_bytes_max,
            "provider_block_plan_checksum": block_plan_checksum,
            "provider_score_batches": 0,
            "provider_scan_wall_s": 0.0,
            "provider_sequence_fetch_wall_s": 0.0,
            "provider_tensor_prepare_wall_s": 0.0,
            "provider_forward_scan_wall_s": 0.0,
            "provider_reverse_complement_wall_s": 0.0,
            "provider_reverse_scan_wall_s": 0.0,
            "provider_interval_slice_wall_s": 0.0,
            "provider_interval_slices": 0,
            "provider_sequence_fetch_mode": "direct_chrom_array",
            "provider_sequence_fetch_fallbacks": 0,
            "provider_max_gap_bp": self.max_gap_bp,
            "provider_max_block_span_bp": self.max_block_span_bp,
            "provider_max_block_score_bytes": self.max_block_score_bytes,
            "provider_max_cached_blocks": self.max_cached_blocks,
            "provider_block_cache_hits": 0,
            "provider_block_cache_misses": 0,
            "provider_block_cache_evictions": 0,
            "provider_block_cache_peak_blocks": 0,
            "provider_max_sequence_cache_bytes": self.max_sequence_cache_bytes,
            "provider_sequence_chunk_cache_enabled": (self.max_sequence_cache_bytes != 0),
            "provider_sequence_chunk_cache_hits": 0,
            "provider_sequence_chunk_cache_misses": 0,
            "provider_sequence_chunk_cache_evictions": 0,
            "provider_sequence_chunk_cache_bytes": 0,
            "provider_sequence_chunk_cache_peak_bytes": 0,
            "provider_sequence_chunk_bp_loaded": 0,
        }

    @staticmethod
    def _chrom_chunk_bp(chrom_ref: Any) -> int | None:
        chunks = getattr(chrom_ref, "chunks", None)
        if chunks is None:
            return None
        if isinstance(chunks, int):
            return int(chunks) if int(chunks) > 0 else None
        if len(chunks) == 0:
            return None
        chunk_bp = int(chunks[0])
        return chunk_bp if chunk_bp > 0 else None

    def _store_sequence_chunk(
        self,
        key: tuple[str, int, int],
        chunk: np.ndarray,
    ) -> None:
        if self.max_sequence_cache_bytes == 0:
            return
        chunk_bytes = int(chunk.nbytes)
        if (
            self.max_sequence_cache_bytes is not None
            and chunk_bytes > self.max_sequence_cache_bytes
        ):
            return
        previous = self._sequence_chunk_cache.pop(key, None)
        if previous is not None:
            self._sequence_chunk_cache_bytes -= int(previous.nbytes)
        self._sequence_chunk_cache[key] = chunk
        self._sequence_chunk_cache.move_to_end(key)
        self._sequence_chunk_cache_bytes += chunk_bytes
        while (
            self.max_sequence_cache_bytes is not None
            and self._sequence_chunk_cache_bytes > self.max_sequence_cache_bytes
            and self._sequence_chunk_cache
        ):
            _, evicted = self._sequence_chunk_cache.popitem(last=False)
            self._sequence_chunk_cache_bytes -= int(evicted.nbytes)
            self.stats["provider_sequence_chunk_cache_evictions"] += 1
        self.stats["provider_sequence_chunk_cache_bytes"] = int(self._sequence_chunk_cache_bytes)
        self.stats["provider_sequence_chunk_cache_peak_bytes"] = max(
            int(self.stats["provider_sequence_chunk_cache_peak_bytes"]),
            int(self._sequence_chunk_cache_bytes),
        )

    def _fetch_sequence_block_from_chunks(
        self,
        chrom: str,
        chrom_ref: Any,
        start: int,
        end: int,
        *,
        chunk_bp: int,
    ) -> np.ndarray:
        pieces: list[np.ndarray] = []
        first_chunk = int(start) // int(chunk_bp)
        last_chunk = (int(end) - 1) // int(chunk_bp)
        for chunk_index in range(first_chunk, last_chunk + 1):
            chunk_start = chunk_index * int(chunk_bp)
            chunk_end = min(int(chrom_ref.shape[0]), chunk_start + int(chunk_bp))
            key = (str(chrom), int(chunk_start), int(chunk_end))
            chunk = self._sequence_chunk_cache.get(key)
            if chunk is None:
                self.stats["provider_sequence_chunk_cache_misses"] += 1
                chunk = np.asarray(chrom_ref[chunk_start:chunk_end])
                self.stats["provider_sequence_chunk_bp_loaded"] += int(chunk.shape[0])
                self._store_sequence_chunk(key, chunk)
            else:
                self.stats["provider_sequence_chunk_cache_hits"] += 1
                self._sequence_chunk_cache.move_to_end(key)
            piece_start = max(int(start), chunk_start) - chunk_start
            piece_end = min(int(end), chunk_end) - chunk_start
            pieces.append(chunk[piece_start:piece_end])
        if not pieces:
            return np.zeros((0, int(chrom_ref.shape[1])), dtype=np.float32)
        if len(pieces) == 1:
            return pieces[0].astype(np.float32)
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def _fetch_sequence_block(self, chrom: str, start: int, end: int) -> np.ndarray:
        try:
            if (
                getattr(self.sequence_database, "dir_chunked", None) is False
                and getattr(self.sequence_database, "dataset", None) is not None
                and "chrs" in self.sequence_database.dataset
                and chrom in self.sequence_database.dataset["chrs"]
            ):
                chrom_ref = self.sequence_database.dataset["chrs"][chrom]
                chrom_size = int(chrom_ref.shape[0])
                bounded_start = max(0, int(start))
                bounded_end = min(chrom_size, int(end))
                if bounded_start >= bounded_end:
                    return np.zeros((0, int(chrom_ref.shape[1])), dtype=np.float32)
                chunk_bp = self._chrom_chunk_bp(chrom_ref)
                if self.max_sequence_cache_bytes != 0 and chunk_bp is not None:
                    self.stats["provider_sequence_fetch_mode"] = "direct_chrom_chunk_cache"
                    return self._fetch_sequence_block_from_chunks(
                        chrom,
                        chrom_ref,
                        bounded_start,
                        bounded_end,
                        chunk_bp=chunk_bp,
                    )
                return chrom_ref[bounded_start:bounded_end].astype(np.float32)
        except Exception:
            self.stats["provider_sequence_fetch_fallbacks"] += 1
        self.stats["provider_sequence_fetch_mode"] = "get_track_fallback"
        return self.sequence_database.get_track(
            chrom,
            start,
            end,
            output_format="raw_array",
        ).astype(np.float32)

    def _scan_block(self, block: tuple[str, int, int]) -> list[torch.Tensor]:
        if block in self._block_cache:
            self.stats["provider_block_cache_hits"] += 1
            self._block_cache.move_to_end(block)
            return self._block_cache[block]
        self.stats["provider_block_cache_misses"] += 1
        chrom, start, end = block
        sequence_start = time.perf_counter()
        sequence = self._fetch_sequence_block(chrom, start, end)
        self.stats["provider_sequence_fetch_wall_s"] += time.perf_counter() - sequence_start
        self.stats["provider_blocks_scanned"] += 1
        self.stats["provider_bp_loaded"] += int(sequence.shape[0])
        if sequence.shape[0] < self.motif_length:
            empty = torch.zeros(
                (1, int(self.motif_kernels.shape[0]), 0),
                dtype=self.dtype,
                device=self.device,
            )
            scores = [empty]
            self._store_block_scores(block, scores)
            return scores

        tensor_start = time.perf_counter()
        batch_tensor = torch.as_tensor(
            sequence[None, :, :],
            dtype=self.dtype,
            device=self.device,
        )
        kernels = self.motif_kernels.permute(0, 2, 1)
        self.stats["provider_tensor_prepare_wall_s"] += time.perf_counter() - tensor_start
        conv_fn = _get_compiled_conv_fn()
        scan_start = time.perf_counter()
        forward_start = time.perf_counter()
        scores = [conv_fn(batch_tensor.permute(0, 2, 1), kernels)]
        self.stats["provider_forward_scan_wall_s"] += time.perf_counter() - forward_start
        if not self.strand_specific:
            reverse_start = time.perf_counter()
            reverse_tensor = reverse_complement_batch(batch_tensor)
            self.stats["provider_reverse_complement_wall_s"] += time.perf_counter() - reverse_start
            reverse_scan_start = time.perf_counter()
            scores.append(conv_fn(reverse_tensor.permute(0, 2, 1), kernels))
            self.stats["provider_reverse_scan_wall_s"] += time.perf_counter() - reverse_scan_start
        self.stats["provider_scan_wall_s"] += time.perf_counter() - scan_start
        self.stats["provider_score_batches"] += len(scores)
        self._store_block_scores(block, scores)
        return scores

    def _store_block_scores(self, block: tuple[str, int, int], scores: list[torch.Tensor]) -> None:
        if self.max_cached_blocks == 0:
            return
        self._block_cache[block] = scores
        self._block_cache.move_to_end(block)
        while (
            self.max_cached_blocks is not None and len(self._block_cache) > self.max_cached_blocks
        ):
            self._block_cache.popitem(last=False)
            self.stats["provider_block_cache_evictions"] += 1
        self.stats["provider_block_cache_peak_blocks"] = max(
            self.stats["provider_block_cache_peak_blocks"],
            len(self._block_cache),
        )

    def __call__(self, chrom: str, start: int, end: int) -> list[torch.Tensor]:
        self.stats["provider_intervals"] += 1
        key = (str(chrom), int(start), int(end))
        block = self._interval_to_block.get(key, key)
        block_scores = self._scan_block(block)
        _, block_start, block_end = block
        interval_length = int(end) - int(start)
        score_length = max(0, interval_length - self.motif_length + 1)
        if score_length == 0:
            return [
                torch.zeros(
                    (1, int(self.motif_kernels.shape[0]), 0),
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in block_scores
            ]

        forward_start = int(start) - block_start
        forward_end = forward_start + score_length
        slice_start = time.perf_counter()
        sliced = [block_scores[0][:, :, forward_start:forward_end]]
        if not self.strand_specific and len(block_scores) > 1:
            block_length = block_end - block_start
            reverse_start = block_length - int(end) + block_start
            reverse_end = reverse_start + score_length
            sliced.append(block_scores[1][:, :, reverse_start:reverse_end])
        self.stats["provider_interval_slice_wall_s"] += time.perf_counter() - slice_start
        self.stats["provider_interval_slices"] += 1
        return sliced


@dataclass(frozen=True)
class DenseSubsetAggregationResult:
    subset_ids: list[str]
    values: np.ndarray
    plan: MultiSubsetAggregationPlan
    timings: dict[str, float]
    semantics: str = DENSE_SUBSET_SEMANTICS


@dataclass(frozen=True)
class DenseSubsetResourcePlan:
    plan: MultiSubsetAggregationPlan
    repeated_region_bp: int
    unique_region_bp: int
    coalesced_block_bp: int
    n_coalesced_blocks: int
    provider_max_gap_bp: int
    provider_max_block_span_bp: int
    provider_max_block_score_bytes: int | None
    coalesced_block_score_bytes_sum: int
    max_block_score_bytes_estimate: int
    median_block_score_bytes_estimate: float
    storage_threshold_bytes: int
    use_mnt_storage: bool
    recommended_output_path: str | None
    recommended_storage_base: str | None

    @property
    def bp_reuse_factor(self) -> float:
        if self.unique_region_bp == 0:
            return 0.0
        return self.repeated_region_bp / self.unique_region_bp

    @property
    def block_bp_vs_unique_region_bp(self) -> float:
        if self.unique_region_bp == 0:
            return 0.0
        return self.coalesced_block_bp / self.unique_region_bp

    def to_dict(self) -> dict:
        return {
            "schema_version": "dense_subset_resource_plan_v1",
            "plan": self.plan.to_dict(),
            "repeated_region_bp": self.repeated_region_bp,
            "unique_region_bp": self.unique_region_bp,
            "coalesced_block_bp": self.coalesced_block_bp,
            "n_coalesced_blocks": self.n_coalesced_blocks,
            "bp_reuse_factor": self.bp_reuse_factor,
            "block_bp_vs_unique_region_bp": self.block_bp_vs_unique_region_bp,
            "provider_max_gap_bp": self.provider_max_gap_bp,
            "provider_max_block_span_bp": self.provider_max_block_span_bp,
            "provider_max_block_score_bytes": self.provider_max_block_score_bytes,
            "coalesced_block_score_bytes_sum": self.coalesced_block_score_bytes_sum,
            "max_block_score_bytes_estimate": self.max_block_score_bytes_estimate,
            "median_block_score_bytes_estimate": (self.median_block_score_bytes_estimate),
            "storage_threshold_bytes": self.storage_threshold_bytes,
            "use_mnt_storage": self.use_mnt_storage,
            "recommended_output_path": self.recommended_output_path,
            "recommended_storage_base": self.recommended_storage_base,
        }


@dataclass(frozen=True)
class DenseIntervalContributionCachePlan:
    """Storage plan for exact full-curve per-interval contribution caching."""

    subset_plan: MultiSubsetAggregationPlan
    contribution_shape: tuple[int, ...]
    contribution_cache_bytes: int
    max_cache_bytes: int | None
    cache_feasible_under_max: bool
    cache_to_output_bytes_ratio: float
    cache_bytes_per_unique_region: int
    sparse_nonzero_scenarios: list[dict[str, Any]]
    recommended_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dense_interval_contribution_cache_plan_v1",
            "subset_plan": self.subset_plan.to_dict(),
            "contribution_shape": list(self.contribution_shape),
            "contribution_cache_bytes": int(self.contribution_cache_bytes),
            "max_cache_bytes": self.max_cache_bytes,
            "cache_feasible_under_max": bool(self.cache_feasible_under_max),
            "cache_to_output_bytes_ratio": float(self.cache_to_output_bytes_ratio),
            "cache_bytes_per_unique_region": int(self.cache_bytes_per_unique_region),
            "sparse_nonzero_scenarios": self.sparse_nonzero_scenarios,
            "recommended_strategy": self.recommended_strategy,
        }


@dataclass(frozen=True)
class DenseSubsetZarrResult:
    path: str
    subset_ids: list[str]
    plan: MultiSubsetAggregationPlan
    checksum: str | None
    timings: dict[str, float]
    semantics: str = DENSE_SUBSET_SEMANTICS


@dataclass(frozen=True)
class DenseIntervalContributionCacheResult:
    path: str
    n_unique_regions: int
    contribution_shape: tuple[int, ...]
    checksum: str | None
    timings: dict[str, float]
    semantics: str = DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS


@dataclass(frozen=True)
class DenseQueryMotifHitCacheResult:
    path: str
    n_unique_regions: int
    n_anchor_hits: int
    checksum: str
    timings: dict[str, float]
    semantics: str = DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS


@dataclass(frozen=True)
class DenseQueryMotifHitResult:
    """Target-anchor positions for one interval across score batches."""

    batch_indices: np.ndarray
    positions: np.ndarray
    n_hits: int


def _plan_dense_genome_blocks(
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    max_gap_bp: int = 0,
    max_block_span_bp: int = 1_000_000,
    max_block_score_bytes: int | None = None,
    n_motifs: int | None = None,
    motif_length: int | None = None,
    dtype: np.dtype | str | torch.dtype = np.float32,
    strands: int = 1,
) -> tuple[
    dict[tuple[str, int, int], tuple[str, int, int]],
    list[tuple[str, int, int]],
]:
    if max_block_score_bytes is not None and (n_motifs is None or motif_length is None):
        raise ValueError("n_motifs and motif_length are required with max_block_score_bytes.")
    normalized = _normalized_regions(region_memberships_df, subset_column)
    unique_regions = normalized[["chrom", "start", "end"]].drop_duplicates()
    interval_to_block: dict[tuple[str, int, int], tuple[str, int, int]] = {}
    blocks: list[tuple[str, int, int]] = []

    for chrom, chrom_regions in unique_regions.groupby("chrom", sort=True):
        records = sorted(
            (
                (str(chrom), int(row.start), int(row.end))
                for row in chrom_regions.itertuples(index=False)
            ),
            key=lambda item: (item[1], item[2]),
        )
        current_start: int | None = None
        current_end: int | None = None
        current_records: list[tuple[str, int, int]] = []
        for record in records:
            _, start, end = record
            if current_start is None:
                current_start = start
                current_end = end
                current_records = [record]
                continue

            proposed_end = max(int(current_end), end)
            proposed_span = proposed_end - int(current_start)
            close_enough = start <= int(current_end) + int(max_gap_bp)
            within_score_budget = True
            if max_block_score_bytes is not None:
                within_score_budget = _estimated_dense_block_score_bytes(
                    proposed_span,
                    motif_length=int(motif_length),
                    n_motifs=int(n_motifs),
                    dtype=dtype,
                    strands=int(strands),
                ) <= int(max_block_score_bytes)
            if close_enough and proposed_span <= int(max_block_span_bp) and within_score_budget:
                current_end = proposed_end
                current_records.append(record)
            else:
                block = (str(chrom), int(current_start), int(current_end))
                blocks.append(block)
                for item in current_records:
                    interval_to_block[item] = block
                current_start = start
                current_end = end
                current_records = [record]

        if current_start is not None:
            block = (str(chrom), int(current_start), int(current_end))
            blocks.append(block)
            for item in current_records:
                interval_to_block[item] = block
    return interval_to_block, blocks


def plan_dense_score_region_subset_resources(
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    n_motifs: int,
    window_size: int,
    query_motif_index: int | None = None,
    motif_length: int = 1,
    strands: int = 2,
    dtype: np.dtype | str = np.float32,
    provider_max_gap_bp: int = 0,
    provider_max_block_span_bp: int = 1_000_000,
    provider_max_block_score_bytes: int | None = None,
    output_path: str | Path | None = None,
    storage_threshold_bytes: int = 10 * 1024**3,
    mnt_storage_base: str | Path = str(Path.home() / ".cache" / "motiverse" / "benchmarks"),
) -> DenseSubsetResourcePlan:
    """Preflight exact dense subset output and block-scan resource shape."""
    dtype = np.dtype(dtype)
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    normalized = _normalized_regions(region_memberships_df, subset_column)
    repeated_region_bp = int((normalized["end"] - normalized["start"]).sum())
    unique_regions = normalized[["chrom", "start", "end"]].drop_duplicates()
    unique_region_bp = int((unique_regions["end"] - unique_regions["start"]).sum())
    _, blocks = _plan_dense_genome_blocks(
        region_memberships_df,
        subset_column=subset_column,
        max_gap_bp=provider_max_gap_bp,
        max_block_span_bp=provider_max_block_span_bp,
        max_block_score_bytes=provider_max_block_score_bytes,
        n_motifs=n_motifs,
        motif_length=motif_length,
        dtype=dtype,
        strands=strands,
    )
    coalesced_block_bp = int(sum(end - start for _, start, end in blocks))
    block_score_bytes = np.asarray(
        [
            _estimated_dense_block_score_bytes(
                end - start,
                motif_length=motif_length,
                n_motifs=n_motifs,
                dtype=dtype,
                strands=strands,
            )
            for _, start, end in blocks
        ],
        dtype=np.int64,
    )
    max_block_score_bytes_estimate = int(block_score_bytes.max()) if block_score_bytes.size else 0
    if provider_max_block_score_bytes is not None and max_block_score_bytes_estimate > int(
        provider_max_block_score_bytes
    ):
        raise MemoryError(
            "At least one required dense genome block has estimated score size "
            f"{max_block_score_bytes_estimate:,} bytes, above "
            f"provider_max_block_score_bytes={int(provider_max_block_score_bytes):,}."
        )
    use_mnt_storage = bool(plan.output_bytes >= int(storage_threshold_bytes))
    recommended_output_path = str(output_path) if output_path is not None else None
    recommended_storage_base = None
    if use_mnt_storage:
        recommended_storage_base = str(mnt_storage_base)
        if output_path is not None and not str(output_path).startswith(str(mnt_storage_base)):
            recommended_output_path = str(Path(mnt_storage_base) / Path(output_path).name)
    return DenseSubsetResourcePlan(
        plan=plan,
        repeated_region_bp=repeated_region_bp,
        unique_region_bp=unique_region_bp,
        coalesced_block_bp=coalesced_block_bp,
        n_coalesced_blocks=len(blocks),
        provider_max_gap_bp=int(provider_max_gap_bp),
        provider_max_block_span_bp=int(provider_max_block_span_bp),
        provider_max_block_score_bytes=(
            None if provider_max_block_score_bytes is None else int(provider_max_block_score_bytes)
        ),
        coalesced_block_score_bytes_sum=(
            int(block_score_bytes.sum()) if block_score_bytes.size else 0
        ),
        max_block_score_bytes_estimate=max_block_score_bytes_estimate,
        median_block_score_bytes_estimate=(
            float(np.median(block_score_bytes)) if block_score_bytes.size else 0.0
        ),
        storage_threshold_bytes=int(storage_threshold_bytes),
        use_mnt_storage=use_mnt_storage,
        recommended_output_path=recommended_output_path,
        recommended_storage_base=recommended_storage_base,
    )


def plan_dense_interval_contribution_cache_resources(
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    n_motifs: int,
    window_size: int,
    query_motif_index: int | None = None,
    dtype: np.dtype | str = np.float32,
    max_cache_bytes: int | None = None,
    sparse_nonzero_fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
) -> DenseIntervalContributionCachePlan:
    """Preflight exact per-interval full-curve contribution-cache storage.

    The exact interval contribution cache stores one dense contribution tensor
    for every unique ``chrom,start,end`` interval. That is signal-preserving, but
    for full-motif one-to-all it scales as ``unique_regions * motifs * window``.
    This planner makes that storage tradeoff explicit before a run starts.
    """
    dtype = np.dtype(dtype)
    subset_plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    contribution_shape = _contribution_shape(
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
    )
    bytes_per_region = int(np.prod(contribution_shape, dtype=np.int64) * dtype.itemsize)
    contribution_cache_bytes = int(subset_plan.n_unique_regions * bytes_per_region)
    cache_feasible_under_max = (
        True if max_cache_bytes is None else contribution_cache_bytes <= int(max_cache_bytes)
    )
    output_bytes = max(1, int(subset_plan.output_bytes))
    sparse_scenarios: list[dict[str, Any]] = []
    for fraction in sparse_nonzero_fractions:
        fraction = max(0.0, min(1.0, float(fraction)))
        nonzero_regions = int(np.ceil(subset_plan.n_unique_regions * fraction))
        sparse_bytes = int(nonzero_regions * bytes_per_region)
        sparse_scenarios.append(
            {
                "nonzero_fraction": fraction,
                "nonzero_region_count": nonzero_regions,
                "sparse_contribution_cache_bytes": sparse_bytes,
                "sparse_cache_to_dense_cache_ratio": (
                    float(sparse_bytes / contribution_cache_bytes)
                    if contribution_cache_bytes
                    else 0.0
                ),
                "sparse_cache_to_output_bytes_ratio": float(sparse_bytes / output_bytes),
                "sparse_cache_feasible_under_max": (
                    True if max_cache_bytes is None else sparse_bytes <= int(max_cache_bytes)
                ),
            }
        )
    if not cache_feasible_under_max:
        recommended_strategy = "skip_exact_interval_cache_use_union_direct_or_tiled_cache"
    elif subset_plan.reuse_factor <= 1.05:
        recommended_strategy = "build_only_if_reused_by_multiple_downstream_subset_families"
    else:
        recommended_strategy = "cache_can_amortize_exact_interval_reuse"
    return DenseIntervalContributionCachePlan(
        subset_plan=subset_plan,
        contribution_shape=contribution_shape,
        contribution_cache_bytes=contribution_cache_bytes,
        max_cache_bytes=max_cache_bytes,
        cache_feasible_under_max=cache_feasible_under_max,
        cache_to_output_bytes_ratio=float(contribution_cache_bytes / output_bytes),
        cache_bytes_per_unique_region=bytes_per_region,
        sparse_nonzero_scenarios=sparse_scenarios,
        recommended_strategy=recommended_strategy,
    )


def _as_score_batches(
    score_batches: torch.Tensor | Iterable[torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(score_batches, torch.Tensor):
        return [score_batches]
    return list(score_batches)


def dense_region_contribution_from_scores(
    score_batches: torch.Tensor | Iterable[torch.Tensor],
    *,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int | None = None,
    p_value_threshold: str = "p0.0001",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Return one exact dense accumulation contribution for one interval.

    ``score_batches`` may contain one tensor for forward-only analysis or
    multiple tensors for independently scanned strands. Each tensor must have
    shape ``(batch, motif, position)`` and is accumulated with the same dense
    algorithms used by direct region processing.
    """
    batches = _as_score_batches(score_batches)
    if device is None:
        device = batches[0].device if batches else "cpu"
    width = 2 * int(window_size) + 1
    if query_motif_index is None:
        contribution = torch.zeros((n_motifs, n_motifs, width), dtype=dtype, device=device)
    else:
        contribution = torch.zeros((n_motifs, width), dtype=dtype, device=device)

    for motif_scores in batches:
        motif_scores = motif_scores.to(device=device, dtype=dtype)
        thresholds = {
            key: value.to(device=device, dtype=dtype) for key, value in score_thresholds.items()
        }
        if query_motif_index is None:
            accumulate_motif_cooccurrences(
                motif_scores,
                contribution,
                thresholds,
                window_size=window_size,
                p_value_threshold=p_value_threshold,
            )
        else:
            accumulate_around_query_motif(
                motif_scores,
                contribution,
                thresholds,
                query_motif_index=query_motif_index,
                window_size=window_size,
                device=str(device),
                p_value_threshold=p_value_threshold,
            )
    return contribution.detach().cpu().numpy()


def dense_query_motif_hits_from_scores(
    score_batches: torch.Tensor | Iterable[torch.Tensor],
    *,
    score_thresholds: dict[str, torch.Tensor],
    query_motif_index: int,
    window_size: int,
    p_value_threshold: str = "p0.0001",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> DenseQueryMotifHitResult:
    """Find valid query-motif-hit positions without materializing full curves."""
    batches = _as_score_batches(score_batches)
    if device is None:
        device = batches[0].device if batches else "cpu"
    threshold = score_thresholds[p_value_threshold].to(device=device, dtype=dtype)[
        int(query_motif_index)
    ]
    batch_ids: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for batch_list_index, motif_scores in enumerate(batches):
        motif_scores = motif_scores.to(device=device, dtype=dtype)
        if motif_scores.numel() == 0:
            continue
        _, _, score_length = motif_scores.shape
        if score_length < 2 * int(window_size) + 1:
            continue
        query_scores = motif_scores[:, int(query_motif_index), :]
        valid_scores = query_scores[:, int(window_size) : score_length - int(window_size)]
        hit_batch_indices, valid_offsets = torch.where(valid_scores > threshold)
        if hit_batch_indices.numel() == 0:
            continue
        hit_positions = valid_offsets + int(window_size)
        encoded_batch = (
            torch.full_like(hit_batch_indices, int(batch_list_index)) * 1_000_000
            + hit_batch_indices
        )
        batch_ids.append(encoded_batch.detach().cpu().numpy().astype(np.int64))
        positions.append(hit_positions.detach().cpu().numpy().astype(np.int64))
    if not batch_ids:
        empty = np.zeros((0,), dtype=np.int64)
        return DenseQueryMotifHitResult(empty, empty, 0)
    batch_arr = np.concatenate(batch_ids).astype(np.int64)
    pos_arr = np.concatenate(positions).astype(np.int64)
    return DenseQueryMotifHitResult(batch_arr, pos_arr, int(pos_arr.size))


def dense_region_contribution_from_scores_and_query_motif_hits(
    score_batches: torch.Tensor | Iterable[torch.Tensor],
    anchors: DenseQueryMotifHitResult,
    *,
    window_size: int,
    n_motifs: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
    anchor_contribution_mode: str = "strict",
    return_tensor: bool = False,
) -> np.ndarray:
    """Accumulate one-to-all full curves from precomputed query motif hits."""
    batches = _as_score_batches(score_batches)
    if device is None:
        device = batches[0].device if batches else "cpu"
    if anchor_contribution_mode not in {"strict", "stack_all"}:
        raise ValueError(
            "anchor_contribution_mode must be one of 'strict' or 'stack_all'; "
            f"got {anchor_contribution_mode!r}."
        )
    width = 2 * int(window_size) + 1
    contribution = torch.zeros((n_motifs, width), dtype=dtype, device=device)
    if anchors.n_hits == 0:
        return contribution if return_tensor else contribution.detach().cpu().numpy()
    anchor_batch_codes = np.asarray(anchors.batch_indices, dtype=np.int64) // 1_000_000
    if anchors.n_hits > 2 and (
        anchor_contribution_mode == "stack_all"
        or bool(np.all(anchor_batch_codes == anchor_batch_codes[0]))
    ):
        slices = []
        for encoded_batch_index, anchor_position in zip(
            anchors.batch_indices,
            anchors.positions,
            strict=True,
        ):
            encoded_batch_index = int(encoded_batch_index)
            batch_list_index = encoded_batch_index // 1_000_000
            local_batch_index = encoded_batch_index - batch_list_index * 1_000_000
            motif_scores = batches[batch_list_index].to(device=device, dtype=dtype)
            start = int(anchor_position) - int(window_size)
            end = int(anchor_position) + int(window_size) + 1
            slices.append(motif_scores[int(local_batch_index), :, start:end])
        contribution = torch.stack(slices, dim=0).sum(dim=0)
        return contribution if return_tensor else contribution.detach().cpu().numpy()
    if anchors.n_hits <= 2:
        # For one or two query motif hits, direct slicing avoids constructing the
        # large broadcasted gather tensor. This keeps the same addition order as
        # the old gather/sum path for <=2 anchors, while leaving 3+ anchors on
        # the original reduction path to avoid float-order drift.
        for encoded_batch_index, anchor_position in zip(
            anchors.batch_indices,
            anchors.positions,
            strict=True,
        ):
            encoded_batch_index = int(encoded_batch_index)
            batch_list_index = encoded_batch_index // 1_000_000
            local_batch_index = encoded_batch_index - batch_list_index * 1_000_000
            motif_scores = batches[batch_list_index].to(device=device, dtype=dtype)
            start = int(anchor_position) - int(window_size)
            end = int(anchor_position) + int(window_size) + 1
            contribution += motif_scores[int(local_batch_index), :, start:end]
        return contribution if return_tensor else contribution.detach().cpu().numpy()
    anchor_batch_ids = torch.as_tensor(anchors.batch_indices, device=device)
    anchor_positions = torch.as_tensor(anchors.positions, device=device)
    relative_positions = torch.arange(-int(window_size), int(window_size) + 1, device=device)
    motif_indices = torch.arange(int(n_motifs), device=device).unsqueeze(0).unsqueeze(2)
    for batch_list_index, motif_scores in enumerate(batches):
        motif_scores = motif_scores.to(device=device, dtype=dtype)
        encoded_min = int(batch_list_index) * 1_000_000
        encoded_max = encoded_min + 1_000_000
        mask = (anchor_batch_ids >= encoded_min) & (anchor_batch_ids < encoded_max)
        if not torch.any(mask):
            continue
        local_batch_indices = (anchor_batch_ids[mask] - encoded_min).long()
        local_positions = anchor_positions[mask].long()
        absolute_positions = local_positions.unsqueeze(1) + relative_positions.unsqueeze(0)
        batch_indices_expanded = local_batch_indices.unsqueeze(1).unsqueeze(2)
        position_indices_expanded = absolute_positions.unsqueeze(1)
        extracted = motif_scores[
            batch_indices_expanded,
            motif_indices,
            position_indices_expanded,
        ]
        contribution += extracted.sum(dim=0)
    return contribution if return_tensor else contribution.detach().cpu().numpy()


def dense_region_one_to_all_sum_contribution_from_scores(
    score_batches: torch.Tensor | Iterable[torch.Tensor],
    *,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int,
    p_value_threshold: str = "p0.0001",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Return the window-summed one-to-all contribution for one interval.

    This is equivalent to
    ``dense_region_contribution_from_scores(..., query_motif_index=...).sum(-1)``
    but avoids materializing the full ``(motif, window)`` tensor. It preserves the
    dense semantics: target hits are thresholded, while all partner motif scores
    inside each query-centered window contribute.
    """
    batches = _as_score_batches(score_batches)
    if device is None:
        device = batches[0].device if batches else "cpu"
    reduced = torch.zeros((n_motifs,), dtype=dtype, device=device)
    if not batches:
        return reduced.detach().cpu().numpy()

    window_size = int(window_size)
    window_width = 2 * window_size + 1
    for motif_scores in batches:
        motif_scores = motif_scores.to(device=device, dtype=dtype)
        if motif_scores.numel() == 0:
            continue
        _, _, seq_length = motif_scores.shape
        if seq_length < window_width:
            continue
        query_threshold = score_thresholds[p_value_threshold].to(
            device=device,
            dtype=dtype,
        )[int(query_motif_index)]
        query_hits = motif_scores[:, int(query_motif_index), :] > query_threshold
        valid_hits = query_hits[:, window_size : seq_length - window_size]
        if not torch.any(valid_hits):
            continue
        hit_batch_indices, valid_offsets = torch.where(valid_hits)
        hit_positions = valid_offsets + window_size
        starts = hit_positions - window_size
        ends = hit_positions + window_size + 1
        zero_prefix = torch.zeros(
            (motif_scores.shape[0], motif_scores.shape[1], 1),
            dtype=dtype,
            device=device,
        )
        prefix_sums = torch.cat([zero_prefix, torch.cumsum(motif_scores, dim=2)], dim=2)
        prefix_by_position = prefix_sums.permute(0, 2, 1)
        window_sums = (
            prefix_by_position[hit_batch_indices, ends, :]
            - prefix_by_position[hit_batch_indices, starts, :]
        )
        reduced += window_sums.sum(dim=0)
    return reduced.detach().cpu().numpy()


def accumulate_dense_score_region_subsets(
    score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int | None = None,
    p_value_threshold: str = "p0.0001",
    dtype: np.dtype | str = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
) -> DenseSubsetAggregationResult:
    """Aggregate exact dense direct-scan contributions across many subsets.

    Each unique ``chrom,start,end`` interval is passed to ``score_provider``
    once. The resulting dense direct contribution is then added to every subset
    membership for that interval, preserving duplicate-membership semantics.
    """
    dtype = np.dtype(dtype)
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
            f"Planned exact dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={max_output_bytes:,}."
        )

    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    values = np.zeros(plan.output_shape, dtype=dtype)
    timings = {
        "score_provider_wall_s": 0.0,
        "dense_contribution_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "nonzero_contribution_regions": 0.0,
    }
    if len(normalized) == 0:
        return DenseSubsetAggregationResult(subset_ids, values, plan, timings)

    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    for (chrom, start, end), memberships in grouped:
        timings["unique_regions_processed"] += 1
        provider_start = time.perf_counter()
        score_batches = score_provider(str(chrom), int(start), int(end))
        timings["score_provider_wall_s"] += time.perf_counter() - provider_start

        contribution_start = time.perf_counter()
        contribution = dense_region_contribution_from_scores(
            score_batches,
            score_thresholds=score_thresholds,
            window_size=window_size,
            n_motifs=n_motifs,
            query_motif_index=query_motif_index,
            p_value_threshold=p_value_threshold,
            dtype=torch_dtype,
            device=device,
        )
        timings["dense_contribution_wall_s"] += time.perf_counter() - contribution_start
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

    return DenseSubsetAggregationResult(subset_ids, values, plan, timings)


def accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter(
    target_score_provider: DenseScoreProvider,
    full_score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int,
    p_value_threshold: str = "p0.0001",
    dtype: np.dtype | str = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
) -> DenseSubsetAggregationResult:
    """Aggregate one-to-all full curves after query-only anchor prefiltering.

    Intervals without valid query motif hits cannot contribute to one-to-all dense
    full curves, so this path scans full motif scores only for intervals where a
    query-only scan found at least one anchor.
    """
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    if max_output_bytes is not None and plan.output_bytes > int(max_output_bytes):
        raise MemoryError(
            f"Planned exact dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={int(max_output_bytes):,}."
        )
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    values = np.zeros(plan.output_shape, dtype=dtype)
    timings = {
        "query_motif_hit_provider_wall_s": 0.0,
        "query_motif_hit_wall_s": 0.0,
        "full_score_provider_wall_s": 0.0,
        "dense_contribution_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "query_motif_hit_hit_regions": 0.0,
        "query_motif_hit_hits": 0.0,
        "query_motif_hit_singleton_regions": 0.0,
        "query_motif_hit_pair_regions": 0.0,
        "query_motif_hit_multi_anchor_regions": 0.0,
        "full_score_regions_scanned": 0.0,
        "full_score_regions_skipped": 0.0,
        "region_query_strategy": "query_motif_hit_prefilter_full_curve",
    }
    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    for (chrom, start, end), memberships in grouped:
        timings["unique_regions_processed"] += 1
        query_start = time.perf_counter()
        query_scores = target_score_provider(str(chrom), int(start), int(end))
        timings["query_motif_hit_provider_wall_s"] += time.perf_counter() - query_start
        anchor_start = time.perf_counter()
        anchors = dense_query_motif_hits_from_scores(
            query_scores,
            score_thresholds=score_thresholds,
            query_motif_index=0,
            window_size=window_size,
            p_value_threshold=p_value_threshold,
            dtype=torch_dtype,
            device=device,
        )
        timings["query_motif_hit_wall_s"] += time.perf_counter() - anchor_start
        timings["query_motif_hit_hits"] += anchors.n_hits
        if anchors.n_hits == 0:
            timings["full_score_regions_skipped"] += 1
            continue
        timings["query_motif_hit_hit_regions"] += 1
        if anchors.n_hits == 1:
            timings["query_motif_hit_singleton_regions"] += 1
        elif anchors.n_hits == 2:
            timings["query_motif_hit_pair_regions"] += 1
        else:
            timings["query_motif_hit_multi_anchor_regions"] += 1
        provider_start = time.perf_counter()
        full_scores = full_score_provider(str(chrom), int(start), int(end))
        timings["full_score_provider_wall_s"] += time.perf_counter() - provider_start
        timings["full_score_regions_scanned"] += 1
        contribution_start = time.perf_counter()
        contribution = dense_region_contribution_from_scores_and_query_motif_hits(
            full_scores,
            anchors,
            window_size=window_size,
            n_motifs=n_motifs,
            dtype=torch_dtype,
            device=device,
        )
        timings["dense_contribution_wall_s"] += time.perf_counter() - contribution_start
        subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
        update_start = time.perf_counter()
        _add_contribution_to_subset_values(
            values,
            subset_indices,
            counts,
            contribution,
            dtype=np.dtype(dtype),
            chunk_subsets=chunk_subsets,
        )
        timings["subset_update_wall_s"] += time.perf_counter() - update_start
    return DenseSubsetAggregationResult(subset_ids, values, plan, timings)


def _dense_query_motif_hit_cache_checksum(
    interval_indices: np.ndarray,
    batch_indices: np.ndarray,
    positions: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    for values in (interval_indices, batch_indices, positions):
        array = np.asarray(values)
        hasher.update(str(array.dtype).encode())
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _dense_query_motif_hit_key_to_index(root) -> dict[tuple[str, int, int], int]:
    chroms = root["intervals/chrom"][:].tolist()
    starts = root["intervals/start"][:].astype(np.int64)
    ends = root["intervals/end"][:].astype(np.int64)
    return {
        _interval_key(chrom, int(start), int(end)): idx
        for idx, (chrom, start, end) in enumerate(zip(chroms, starts, ends, strict=True))
    }


def _zarr_json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _zarr_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_zarr_json_ready(item) for item in value]
    return value


def _write_zarr_v3_group(path: Path, *, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "attributes": _zarr_json_ready(attrs or {}),
        "zarr_format": 3,
        "node_type": "group",
    }
    (path / "zarr.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_zarr_v3_1d_array(path: Path, values: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    if values.dtype == object or values.dtype.kind in {"U", "S"}:
        data_type = "string"
        codecs = [
            {"name": "vlen-utf8", "configuration": {}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ]
        fill_value = ""
        chunk_bytes = VLenUTF8().encode(values.astype(object))
    else:
        data_type = "int64"
        codecs = [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ]
        fill_value = 0
        chunk_bytes = values.astype("<i8", copy=False).tobytes()
    chunk_shape = [max(1, int(values.shape[0]))]
    payload = {
        "shape": [int(values.shape[0])],
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    (path / "zarr.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
    if values.shape[0] == 0:
        return
    chunk_dir = path / "c"
    chunk_dir.mkdir(exist_ok=True)
    (chunk_dir / "0").write_bytes(Zstd(level=0, checksum=False).encode(chunk_bytes))


def _write_zarr_v3_nd_array(
    path: Path,
    values: np.ndarray,
    *,
    chunk_shape: tuple[int, ...],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    if values.dtype != np.dtype("float32"):
        values = values.astype(np.float32)
    chunk_shape = tuple(
        max(1, min(int(dim), int(chunk)))
        for dim, chunk in zip(values.shape, chunk_shape, strict=True)
    )
    payload = {
        "shape": [int(x) for x in values.shape],
        "data_type": "float32",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [int(x) for x in chunk_shape]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    (path / "zarr.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
    if values.size == 0:
        return
    n_chunks = [
        int(math.ceil(int(size) / int(chunk)))
        for size, chunk in zip(values.shape, chunk_shape, strict=True)
    ]
    for chunk_index in np.ndindex(*n_chunks):
        slices = tuple(
            slice(
                int(index) * int(chunk),
                min(int(size), (int(index) + 1) * int(chunk)),
            )
            for index, size, chunk in zip(chunk_index, values.shape, chunk_shape, strict=True)
        )
        chunk_values = np.ascontiguousarray(values[slices], dtype="<f4")
        # Zarr v3 stores edge chunks at the regular chunk shape; readers use
        # the array shape to trim the padded portion on selection.  Writing a
        # shorter final chunk makes the output unreadable through zarr-python.
        if chunk_values.shape != chunk_shape:
            chunk = np.zeros(chunk_shape, dtype="<f4")
            chunk[tuple(slice(0, size) for size in chunk_values.shape)] = chunk_values
        else:
            chunk = chunk_values
        chunk_path = path / "c" / Path(*[str(int(index)) for index in chunk_index])
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_bytes(Zstd(level=0, checksum=False).encode(chunk.tobytes()))


def _read_zarr_v3_array_metadata(array_path: Path) -> dict[str, Any]:
    metadata_path = array_path / "zarr.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing zarr v3 array metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise ValueError(f"Expected a zarr v3 array at {array_path}.")
    return metadata


def _decode_zarr_v3_chunk(raw: bytes, metadata: dict[str, Any]) -> bytes | np.ndarray:
    decoded: bytes | np.ndarray = raw
    for codec in reversed(metadata.get("codecs", [])):
        name = codec.get("name")
        if name == "zstd":
            config = codec.get("configuration", {})
            decoded = Zstd(
                level=int(config.get("level", 0)),
                checksum=bool(config.get("checksum", False)),
            ).decode(decoded)
        elif name == "vlen-utf8":
            decoded = VLenUTF8().decode(decoded)
        elif name == "bytes":
            continue
        else:
            raise ValueError(f"Unsupported zarr v3 codec {name!r}.")
    return decoded


def _read_zarr_v3_1d_array(array_path: Path) -> np.ndarray:
    metadata = _read_zarr_v3_array_metadata(array_path)
    shape = metadata.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"Expected a 1D zarr v3 array at {array_path}; got {shape!r}.")
    n_items = int(shape[0])
    data_type = metadata.get("data_type")
    if n_items == 0:
        return np.asarray([], dtype=object if data_type == "string" else np.int64)
    chunk_shape = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    chunk_items = int(chunk_shape[0])
    n_chunks = int(math.ceil(n_items / chunk_items)) if chunk_items else 0
    chunks: list[np.ndarray] = []
    for chunk_index in range(n_chunks):
        chunk_path = array_path / "c" / str(chunk_index)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing zarr v3 chunk: {chunk_path}")
        decoded = _decode_zarr_v3_chunk(chunk_path.read_bytes(), metadata)
        if data_type == "string":
            if not isinstance(decoded, np.ndarray):
                decoded = VLenUTF8().decode(decoded)
            chunk = np.asarray(decoded, dtype=object)
        elif data_type == "int64":
            if isinstance(decoded, np.ndarray):
                decoded = decoded.tobytes()
            chunk = np.frombuffer(decoded, dtype="<i8")
        else:
            raise ValueError(f"Unsupported zarr v3 data_type {data_type!r}.")
        chunks.append(chunk)
    return np.concatenate(chunks)[:n_items]


def _read_zarr_v3_nd_array(array_path: str | Path) -> np.ndarray:
    array_path = Path(array_path)
    metadata = _read_zarr_v3_array_metadata(array_path)
    shape = tuple(int(x) for x in metadata.get("shape", []))
    if not shape:
        return np.asarray([], dtype=np.float32)
    if metadata.get("data_type") != "float32":
        raise ValueError(f"Unsupported zarr v3 data_type {metadata.get('data_type')!r}.")
    chunk_shape = tuple(int(x) for x in metadata["chunk_grid"]["configuration"]["chunk_shape"])
    values = np.zeros(shape, dtype=np.float32)
    n_chunks = [
        int(math.ceil(int(size) / int(chunk)))
        for size, chunk in zip(shape, chunk_shape, strict=True)
    ]
    for chunk_index in np.ndindex(*n_chunks):
        chunk_path = array_path / "c" / Path(*[str(int(index)) for index in chunk_index])
        if not chunk_path.exists():
            continue
        decoded = _decode_zarr_v3_chunk(chunk_path.read_bytes(), metadata)
        if isinstance(decoded, np.ndarray):
            decoded = decoded.tobytes()
        slices = tuple(
            slice(
                int(index) * int(chunk),
                min(int(size), (int(index) + 1) * int(chunk)),
            )
            for index, size, chunk in zip(chunk_index, shape, chunk_shape, strict=True)
        )
        actual_shape = tuple(s.stop - s.start for s in slices)
        values[slices] = np.frombuffer(decoded, dtype="<f4").reshape(actual_shape)
    return values


def _read_dense_query_motif_hit_cache_directory(
    cache_path: str | Path,
) -> tuple[dict, dict[tuple[str, int, int], int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = Path(cache_path)
    metadata_json = cache_path / "metadata" / "zarr.json"
    metadata = json.loads(metadata_json.read_text())
    attrs = dict(metadata.get("attributes", {}))
    chroms = _read_zarr_v3_1d_array(cache_path / "intervals" / "chrom").tolist()
    starts = _read_zarr_v3_1d_array(cache_path / "intervals" / "start").astype(np.int64)
    ends = _read_zarr_v3_1d_array(cache_path / "intervals" / "end").astype(np.int64)
    key_to_index = {
        _interval_key(chrom, int(start), int(end)): idx
        for idx, (chrom, start, end) in enumerate(zip(chroms, starts, ends, strict=True))
    }
    interval_anchor_starts = _read_zarr_v3_1d_array(
        cache_path / "anchors" / "interval_anchor_start"
    ).astype(np.int64)
    interval_anchor_counts = _read_zarr_v3_1d_array(
        cache_path / "anchors" / "interval_anchor_count"
    ).astype(np.int64)
    anchor_batch_indices = _read_zarr_v3_1d_array(cache_path / "anchors" / "batch_index").astype(
        np.int64
    )
    anchor_positions = _read_zarr_v3_1d_array(cache_path / "anchors" / "position").astype(np.int64)
    return (
        attrs,
        key_to_index,
        interval_anchor_starts,
        interval_anchor_counts,
        anchor_batch_indices,
        anchor_positions,
    )


def _write_dense_query_motif_hit_cache_zarr_v3(
    output_path: Path,
    *,
    unique_regions: pd.DataFrame,
    flat_interval_indices: np.ndarray,
    flat_batch_indices: np.ndarray,
    flat_positions: np.ndarray,
    interval_anchor_starts: np.ndarray,
    interval_anchor_counts: np.ndarray,
    metadata_attrs: dict,
) -> None:
    output_path = Path(output_path)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    _write_zarr_v3_group(output_path)
    _write_zarr_v3_group(output_path / "metadata", attrs=metadata_attrs)
    _write_zarr_v3_group(output_path / "intervals")
    _write_zarr_v3_group(output_path / "anchors")
    _write_zarr_v3_1d_array(
        output_path / "intervals" / "chrom",
        unique_regions["chrom"].astype(str).to_numpy(dtype=object),
    )
    _write_zarr_v3_1d_array(
        output_path / "intervals" / "start",
        unique_regions["start"].to_numpy(dtype=np.int64),
    )
    _write_zarr_v3_1d_array(
        output_path / "intervals" / "end",
        unique_regions["end"].to_numpy(dtype=np.int64),
    )
    _write_zarr_v3_1d_array(
        output_path / "anchors" / "interval_index",
        np.asarray(flat_interval_indices, dtype=np.int64),
    )
    _write_zarr_v3_1d_array(
        output_path / "anchors" / "batch_index",
        np.asarray(flat_batch_indices, dtype=np.int64),
    )
    _write_zarr_v3_1d_array(
        output_path / "anchors" / "position",
        np.asarray(flat_positions, dtype=np.int64),
    )
    _write_zarr_v3_1d_array(
        output_path / "anchors" / "interval_anchor_start",
        np.asarray(interval_anchor_starts, dtype=np.int64),
    )
    _write_zarr_v3_1d_array(
        output_path / "anchors" / "interval_anchor_count",
        np.asarray(interval_anchor_counts, dtype=np.int64),
    )


def _metadata_value_equal(left, right) -> bool:
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    if isinstance(left, tuple):
        left = list(left)
    if isinstance(right, tuple):
        right = list(right)
    if isinstance(left, Path):
        left = str(left)
    if isinstance(right, Path):
        right = str(right)
    return left == right


def _validate_dense_query_motif_hit_cache_attrs(
    attrs: dict,
    *,
    expected_metadata: dict | None = None,
) -> None:
    if attrs.get("schema_version") != DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Expected dense query-motif-hit cache schema "
            f"{DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION!r}; got "
            f"{attrs.get('schema_version')!r}."
        )
    if attrs.get("complete") is not True:
        raise ValueError("Dense query-motif-hit cache is incomplete.")
    if attrs.get("semantics") != DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS:
        raise ValueError(
            f"Dense query-motif-hit cache has unexpected semantics {attrs.get('semantics')!r}."
        )
    if attrs.get("motif_source") != "aligned_pt" or attrs.get("loaded_with_aligned") is not True:
        raise ValueError(
            "Dense query-motif-hit cache must record motif_source='aligned_pt' "
            "and loaded_with_aligned=True."
        )
    if not expected_metadata:
        return
    compatibility_keys = (
        "genome_zarr_path",
        "motif_count",
        "motif_names_checksum",
        "motif_kernel_checksum",
        "motif_kernel_shape",
        "aligned_motif_path",
        "threshold_mode",
        "p_value_threshold",
        "score_threshold",
        "score_threshold_vector_checksum",
        "score_threshold_vector_shape",
        "window_size",
        "query_motif_index",
        "strand_specific",
        "strands",
        "dtype",
    )
    mismatches = []
    for key in compatibility_keys:
        if key not in expected_metadata:
            continue
        expected_value = expected_metadata[key]
        actual_value = attrs.get(key)
        if not _metadata_value_equal(actual_value, expected_value):
            mismatches.append((key, actual_value, expected_value))
    if mismatches:
        detail = "; ".join(
            f"{key}: cache={actual!r}, expected={expected!r}"
            for key, actual, expected in mismatches[:6]
        )
        raise ValueError(
            f"Dense query-motif-hit cache metadata is incompatible with this run: {detail}"
        )


def build_dense_query_motif_hit_cache(
    target_score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int,
    p_value_threshold: str = "p0.0001",
    dtype: np.dtype | str = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    metadata_extra: dict | None = None,
) -> DenseQueryMotifHitCacheResult:
    """Build a compact exact query-motif-hit cache per unique interval.

    The target score provider is expected to expose the selected query motif as
    motif index 0. The cache stores only significant query motif hit positions; it
    does not store partner motif scores or full curves.
    """
    dtype = np.dtype(dtype)
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    normalized = _normalized_regions(region_memberships_df, subset_column)
    unique_regions = _unique_regions_frame(normalized)
    interval_anchor_starts = np.zeros((len(unique_regions),), dtype=np.int64)
    interval_anchor_counts = np.zeros((len(unique_regions),), dtype=np.int64)
    anchor_interval_indices: list[np.ndarray] = []
    anchor_batch_indices: list[np.ndarray] = []
    anchor_positions: list[np.ndarray] = []
    timings = {
        "query_motif_hit_provider_wall_s": 0.0,
        "query_motif_hit_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "query_motif_hit_hit_regions": 0.0,
        "query_motif_hit_hits": 0.0,
    }

    anchor_offset = 0
    for interval_idx, row in enumerate(unique_regions.itertuples(index=False)):
        timings["unique_regions_processed"] += 1
        interval_anchor_starts[interval_idx] = anchor_offset
        provider_start = time.perf_counter()
        query_scores = target_score_provider(str(row.chrom), int(row.start), int(row.end))
        timings["query_motif_hit_provider_wall_s"] += time.perf_counter() - provider_start
        anchor_start = time.perf_counter()
        anchors = dense_query_motif_hits_from_scores(
            query_scores,
            score_thresholds=score_thresholds,
            query_motif_index=0,
            window_size=window_size,
            p_value_threshold=p_value_threshold,
            dtype=torch_dtype,
            device=device,
        )
        timings["query_motif_hit_wall_s"] += time.perf_counter() - anchor_start
        interval_anchor_counts[interval_idx] = anchors.n_hits
        timings["query_motif_hit_hits"] += anchors.n_hits
        if anchors.n_hits == 0:
            continue
        timings["query_motif_hit_hit_regions"] += 1
        anchor_interval_indices.append(np.full((anchors.n_hits,), interval_idx, dtype=np.int64))
        anchor_batch_indices.append(anchors.batch_indices.astype(np.int64, copy=False))
        anchor_positions.append(anchors.positions.astype(np.int64, copy=False))
        anchor_offset += anchors.n_hits

    if anchor_interval_indices:
        flat_interval_indices = np.concatenate(anchor_interval_indices).astype(np.int64)
        flat_batch_indices = np.concatenate(anchor_batch_indices).astype(np.int64)
        flat_positions = np.concatenate(anchor_positions).astype(np.int64)
    else:
        flat_interval_indices = np.zeros((0,), dtype=np.int64)
        flat_batch_indices = np.zeros((0,), dtype=np.int64)
        flat_positions = np.zeros((0,), dtype=np.int64)
    checksum = _dense_query_motif_hit_cache_checksum(
        flat_interval_indices,
        flat_batch_indices,
        flat_positions,
    )

    base_metadata = _dense_subset_base_metadata(
        plan=plan,
        subset_column=subset_column,
        query_motif_index=query_motif_index,
        dtype=dtype,
        metadata_extra=metadata_extra,
    )
    base_metadata.update(
        {
            "schema_version": DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION,
            "complete": True,
            "semantics": DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS,
            "target_score_motif_index": 0,
            "n_anchor_hits": int(flat_positions.size),
            "query_motif_hit_checksum": checksum,
        }
    )
    provider_stats = getattr(target_score_provider, "stats", None)
    if isinstance(provider_stats, dict):
        base_metadata.update(
            {f"query_motif_hit_provider_{key}": value for key, value in provider_stats.items()}
        )
    base_metadata.update(timings)
    output_path = Path(output_path)
    _write_dense_query_motif_hit_cache_zarr_v3(
        output_path,
        unique_regions=unique_regions,
        flat_interval_indices=flat_interval_indices,
        flat_batch_indices=flat_batch_indices,
        flat_positions=flat_positions,
        interval_anchor_starts=interval_anchor_starts,
        interval_anchor_counts=interval_anchor_counts,
        metadata_attrs=base_metadata,
    )
    return DenseQueryMotifHitCacheResult(
        path=str(output_path),
        n_unique_regions=int(len(unique_regions)),
        n_anchor_hits=int(flat_positions.size),
        checksum=checksum,
        timings=timings,
    )


def _interval_covering_tiles(
    chrom: str,
    start: int,
    end: int,
    *,
    tile_size: int,
    tile_extension_bp: int,
) -> list[tuple[str, int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if tile_extension_bp < 0:
        raise ValueError("tile_extension_bp must be non-negative.")
    first_tile = int(start) // int(tile_size)
    last_tile = (int(end) - 1) // int(tile_size)
    tiles = []
    for tile_index in range(first_tile, last_tile + 1):
        tile_start = tile_index * int(tile_size)
        tile_end = tile_start + int(tile_size)
        scan_start = max(0, tile_start - int(tile_extension_bp))
        scan_end = tile_end + int(tile_extension_bp)
        tiles.append((str(chrom), tile_start, tile_end, scan_start, scan_end))
    return tiles


def build_dense_query_motif_hit_cache_from_tiles(
    target_score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int,
    tile_size: int,
    tile_extension_bp: int | None = None,
    p_value_threshold: str = "p0.0001",
    dtype: np.dtype | str = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    metadata_extra: dict | None = None,
) -> DenseQueryMotifHitCacheResult:
    """Build an exact interval query-motif-hit cache by scanning reusable tiles.

    Tiles are a precompute/query layer only. The emitted cache keeps the same
    per-exact-interval schema as ``build_dense_query_motif_hit_cache`` by
    converting tile-local anchor positions back to interval-local positions and
    filtering by exact interval boundaries.
    """
    dtype = np.dtype(dtype)
    motif_length = int(getattr(target_score_provider, "motif_length", 1))
    requested_tile_extension_bp = int(
        window_size if tile_extension_bp is None else tile_extension_bp
    )
    minimum_tile_extension_bp = int(window_size) + max(0, int(motif_length) - 1)
    tile_extension_bp = max(requested_tile_extension_bp, minimum_tile_extension_bp)
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    normalized = _normalized_regions(region_memberships_df, subset_column)
    unique_regions = _unique_regions_frame(normalized)
    interval_anchor_starts = np.zeros((len(unique_regions),), dtype=np.int64)
    interval_anchor_counts = np.zeros((len(unique_regions),), dtype=np.int64)
    anchor_interval_indices: list[np.ndarray] = []
    anchor_batch_indices: list[np.ndarray] = []
    anchor_positions: list[np.ndarray] = []
    timings = {
        "query_motif_hit_provider_wall_s": 0.0,
        "query_motif_hit_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "query_motif_hit_hit_regions": 0.0,
        "query_motif_hit_hits": 0.0,
        "tile_query_motif_hit_provider_wall_s": 0.0,
        "tile_query_motif_hit_wall_s": 0.0,
        "unique_tiles_processed": 0.0,
        "tile_anchor_hits": 0.0,
        "tile_anchor_candidate_assignments": 0.0,
        "tile_anchor_duplicate_assignments": 0.0,
        "tile_anchor_boundary_filtered": 0.0,
        "tile_size_bp": int(tile_size),
        "tile_extension_bp": int(tile_extension_bp),
        "requested_tile_extension_bp": int(requested_tile_extension_bp),
        "minimum_tile_extension_bp": int(minimum_tile_extension_bp),
        "motif_length": int(motif_length),
        "region_query_strategy": "tile_assisted_dense_query_motif_hit_cache_build",
    }

    tile_keys: set[tuple[str, int, int, int, int]] = set()
    interval_tiles: list[list[tuple[str, int, int, int, int]]] = []
    for row in unique_regions.itertuples(index=False):
        tiles = _interval_covering_tiles(
            str(row.chrom),
            int(row.start),
            int(row.end),
            tile_size=int(tile_size),
            tile_extension_bp=int(tile_extension_bp),
        )
        interval_tiles.append(tiles)
        tile_keys.update(tiles)

    tile_anchor_lookup: dict[
        tuple[str, int, int, int, int],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    for tile in sorted(tile_keys):
        chrom, _tile_start, _tile_end, scan_start, scan_end = tile
        timings["unique_tiles_processed"] += 1
        provider_start = time.perf_counter()
        query_scores = target_score_provider(str(chrom), int(scan_start), int(scan_end))
        provider_wall = time.perf_counter() - provider_start
        timings["tile_query_motif_hit_provider_wall_s"] += provider_wall
        timings["query_motif_hit_provider_wall_s"] += provider_wall
        anchor_start = time.perf_counter()
        anchors = dense_query_motif_hits_from_scores(
            query_scores,
            score_thresholds=score_thresholds,
            query_motif_index=0,
            window_size=window_size,
            p_value_threshold=p_value_threshold,
            dtype=torch_dtype,
            device=device,
        )
        anchor_wall = time.perf_counter() - anchor_start
        timings["tile_query_motif_hit_wall_s"] += anchor_wall
        timings["query_motif_hit_wall_s"] += anchor_wall
        timings["tile_anchor_hits"] += anchors.n_hits
        if anchors.n_hits == 0:
            tile_anchor_lookup[tile] = (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
            continue
        tile_anchor_lookup[tile] = (
            anchors.batch_indices.astype(np.int64, copy=False),
            anchors.positions.astype(np.int64, copy=False),
        )

    anchor_offset = 0
    for interval_idx, row in enumerate(unique_regions.itertuples(index=False)):
        timings["unique_regions_processed"] += 1
        interval_anchor_starts[interval_idx] = anchor_offset
        interval_records: dict[tuple[int, int], None] = {}
        start = int(row.start)
        end = int(row.end)
        interval_score_length = max(0, end - start - int(motif_length) + 1)
        valid_start = int(window_size)
        valid_end = interval_score_length - int(window_size)
        if valid_end <= valid_start:
            timings["tile_anchor_boundary_filtered"] += sum(
                int(tile_anchor_lookup[tile][1].size) for tile in interval_tiles[interval_idx]
            )
            continue
        for tile in interval_tiles[interval_idx]:
            _chrom, _tile_start, _tile_end, scan_start, scan_end = tile
            batch_indices, tile_positions = tile_anchor_lookup[tile]
            if tile_positions.size == 0:
                continue
            for batch_index, genomic_position in zip(
                batch_indices,
                tile_positions,
                strict=True,
            ):
                batch_list_index = int(batch_index) // 1_000_000
                tile_position = int(genomic_position)
                if batch_list_index == 0:
                    local_position = int(scan_start) + tile_position - start
                else:
                    genomic_score_start = int(scan_end) - int(motif_length) - tile_position
                    local_position = end - int(motif_length) - genomic_score_start
                if not (valid_start <= local_position < valid_end):
                    timings["tile_anchor_boundary_filtered"] += 1
                    continue
                key = (int(batch_index), local_position)
                if key in interval_records:
                    timings["tile_anchor_duplicate_assignments"] += 1
                    continue
                interval_records[key] = None
                timings["tile_anchor_candidate_assignments"] += 1
        if not interval_records:
            continue
        records = sorted(interval_records)
        interval_anchor_counts[interval_idx] = len(records)
        timings["query_motif_hit_hit_regions"] += 1
        timings["query_motif_hit_hits"] += len(records)
        anchor_interval_indices.append(np.full((len(records),), interval_idx, dtype=np.int64))
        anchor_batch_indices.append(np.asarray([record[0] for record in records], dtype=np.int64))
        anchor_positions.append(np.asarray([record[1] for record in records], dtype=np.int64))
        anchor_offset += len(records)

    if anchor_interval_indices:
        flat_interval_indices = np.concatenate(anchor_interval_indices).astype(np.int64)
        flat_batch_indices = np.concatenate(anchor_batch_indices).astype(np.int64)
        flat_positions = np.concatenate(anchor_positions).astype(np.int64)
    else:
        flat_interval_indices = np.zeros((0,), dtype=np.int64)
        flat_batch_indices = np.zeros((0,), dtype=np.int64)
        flat_positions = np.zeros((0,), dtype=np.int64)
    checksum = _dense_query_motif_hit_cache_checksum(
        flat_interval_indices,
        flat_batch_indices,
        flat_positions,
    )

    base_metadata = _dense_subset_base_metadata(
        plan=plan,
        subset_column=subset_column,
        query_motif_index=query_motif_index,
        dtype=dtype,
        metadata_extra=metadata_extra,
    )
    base_metadata.update(
        {
            "schema_version": DENSE_QUERY_MOTIF_HIT_CACHE_SCHEMA_VERSION,
            "complete": True,
            "semantics": DENSE_QUERY_MOTIF_HIT_CACHE_SEMANTICS,
            "target_score_motif_index": 0,
            "n_anchor_hits": int(flat_positions.size),
            "query_motif_hit_checksum": checksum,
            "query_motif_hit_cache_build_mode": "tile_assisted_exact_interval_reconstruction",
            "tile_size_bp": int(tile_size),
            "tile_extension_bp": int(tile_extension_bp),
            "requested_tile_extension_bp": int(requested_tile_extension_bp),
            "minimum_tile_extension_bp": int(minimum_tile_extension_bp),
            "motif_length": int(motif_length),
            "tile_exactness_note": (
                "Tile-assisted query-motif-hit cache stores reconstructed exact "
                "interval anchors; full-curve replay still scans exact intervals."
            ),
        }
    )
    provider_stats = getattr(target_score_provider, "stats", None)
    if isinstance(provider_stats, dict):
        base_metadata.update(
            {f"query_motif_hit_provider_{key}": value for key, value in provider_stats.items()}
        )
    base_metadata.update(timings)
    output_path = Path(output_path)
    _write_dense_query_motif_hit_cache_zarr_v3(
        output_path,
        unique_regions=unique_regions,
        flat_interval_indices=flat_interval_indices,
        flat_batch_indices=flat_batch_indices,
        flat_positions=flat_positions,
        interval_anchor_starts=interval_anchor_starts,
        interval_anchor_counts=interval_anchor_counts,
        metadata_attrs=base_metadata,
    )
    return DenseQueryMotifHitCacheResult(
        path=str(output_path),
        n_unique_regions=int(len(unique_regions)),
        n_anchor_hits=int(flat_positions.size),
        checksum=checksum,
        timings=timings,
    )


def accumulate_dense_score_region_subsets_from_query_motif_hit_cache(
    query_motif_hit_cache_path: str | Path,
    full_score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    dtype: np.dtype | str | None = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
    coalesce_membership_patterns: bool = False,
    max_pattern_accumulator_bytes: int | None = None,
    expected_metadata: dict | None = None,
    anchor_contribution_mode: str = "strict",
    preload_query_motif_hit_arrays: bool = True,
    output_accumulator: str = "numpy",
) -> DenseSubsetAggregationResult:
    """Replay cached query motif hits and scan partner motif scores only on hits."""
    if anchor_contribution_mode not in {"strict", "stack_all"}:
        raise ValueError(
            "anchor_contribution_mode must be one of 'strict' or 'stack_all'; "
            f"got {anchor_contribution_mode!r}."
        )
    if output_accumulator not in {"numpy", "torch"}:
        raise ValueError(
            f"output_accumulator must be one of 'numpy' or 'torch'; got {output_accumulator!r}."
        )
    if coalesce_membership_patterns and output_accumulator == "torch":
        raise ValueError(
            "output_accumulator='torch' cannot be combined with "
            "coalesce_membership_patterns; benchmark these PASS_WARN speed modes "
            "separately."
        )
    direct_cache_path = Path(query_motif_hit_cache_path)
    use_direct_cache_reader = (direct_cache_path / "metadata" / "zarr.json").exists()
    if use_direct_cache_reader:
        (
            attrs,
            key_to_index,
            interval_anchor_starts,
            interval_anchor_counts,
            direct_anchor_batch_indices,
            direct_anchor_positions,
        ) = _read_dense_query_motif_hit_cache_directory(direct_cache_path)
        root = None
    else:
        root = zarr.open_group(str(query_motif_hit_cache_path), mode="r")
        attrs = dict(root["metadata"].attrs)
        key_to_index = _dense_query_motif_hit_key_to_index(root)
        interval_anchor_starts = root["anchors/interval_anchor_start"][:].astype(np.int64)
        interval_anchor_counts = root["anchors/interval_anchor_count"][:].astype(np.int64)
        direct_anchor_batch_indices = None
        direct_anchor_positions = None
    _validate_dense_query_motif_hit_cache_attrs(
        attrs,
        expected_metadata=expected_metadata,
    )
    n_motifs = int(attrs["motif_count"])
    window_size = int(attrs["window_size"])
    query_motif_index = int(attrs["query_motif_index"])
    dtype = np.dtype(dtype or attrs.get("dtype", "float32"))
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    if max_output_bytes is not None and plan.output_bytes > int(max_output_bytes):
        raise MemoryError(
            f"Planned exact dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={int(max_output_bytes):,}."
        )
    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}
    if output_accumulator == "torch":
        values_torch = torch.zeros(plan.output_shape, dtype=torch_dtype, device=device)
        values = None
    else:
        values_torch = None
        values = np.zeros(plan.output_shape, dtype=dtype)
    preload_start = time.perf_counter()
    if use_direct_cache_reader:
        anchor_batch_indices = direct_anchor_batch_indices
        anchor_positions = direct_anchor_positions
    elif preload_query_motif_hit_arrays:
        anchor_batch_indices = root["anchors/batch_index"][:].astype(np.int64)
        anchor_positions = root["anchors/position"][:].astype(np.int64)
    else:
        anchor_batch_indices = root["anchors/batch_index"]
        anchor_positions = root["anchors/position"]
    anchor_preload_wall_s = time.perf_counter() - preload_start
    anchor_array_bytes = int(
        getattr(anchor_batch_indices, "nbytes", 0) + getattr(anchor_positions, "nbytes", 0)
    )
    timings = {
        "query_motif_hit_lookup_wall_s": 0.0,
        "query_motif_hit_array_preload_wall_s": anchor_preload_wall_s,
        "query_motif_hit_arrays_preloaded": bool(preload_query_motif_hit_arrays),
        "query_motif_hit_array_bytes": anchor_array_bytes,
        "query_motif_hit_array_read_wall_s": 0.0,
        "full_score_provider_wall_s": 0.0,
        "dense_contribution_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "query_motif_hit_cache_total_intervals": float(len(key_to_index)),
        "query_motif_hit_cache_required_intervals": float(plan.n_unique_regions),
        "query_motif_hit_cache_missing_intervals": 0.0,
        "query_motif_hit_cache_coverage_fraction": 0.0,
        "query_motif_hit_cache_hits": 0.0,
        "query_motif_hit_hit_regions": 0.0,
        "query_motif_hit_hits": 0.0,
        "query_motif_hit_singleton_regions": 0.0,
        "query_motif_hit_pair_regions": 0.0,
        "query_motif_hit_multi_anchor_regions": 0.0,
        "full_score_regions_scanned": 0.0,
        "full_score_regions_skipped": 0.0,
        "region_query_strategy": "dense_query_motif_hit_cache_replay",
        "membership_pattern_coalescing": bool(coalesce_membership_patterns),
        "membership_pattern_count": 0.0,
        "membership_pattern_interval_assignments": 0.0,
        "membership_pattern_accumulator_bytes": 0.0,
        "membership_pattern_accumulate_wall_s": 0.0,
        "anchor_contribution_mode": anchor_contribution_mode,
        "anchor_stack_all_float_order_jitter": bool(anchor_contribution_mode == "stack_all"),
        "output_accumulator": output_accumulator,
        "output_accumulator_device": str(device) if output_accumulator == "torch" else "cpu",
        "output_accumulator_float_order_jitter": bool(output_accumulator == "torch"),
        "output_accumulator_to_numpy_wall_s": 0.0,
        "query_motif_hit_same_batch_multi_anchor_regions": 0.0,
    }
    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    pattern_contributions: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray] = {}
    pattern_accumulator_bytes = 0
    for (chrom, start, end), memberships in grouped:
        timings["unique_regions_processed"] += 1
        key = _interval_key(chrom, int(start), int(end))
        if key not in key_to_index:
            timings["query_motif_hit_cache_missing_intervals"] += 1
            raise ValueError(
                "Dense query-motif-hit cache is missing required interval "
                f"{key[0]}:{key[1]}-{key[2]}"
            )
        lookup_start = time.perf_counter()
        cache_index = int(key_to_index[key])
        anchor_start = int(interval_anchor_starts[cache_index])
        anchor_count = int(interval_anchor_counts[cache_index])
        timings["query_motif_hit_lookup_wall_s"] += time.perf_counter() - lookup_start
        timings["query_motif_hit_cache_hits"] += 1
        if anchor_count == 0:
            timings["full_score_regions_skipped"] += 1
            continue
        timings["query_motif_hit_hit_regions"] += 1
        timings["query_motif_hit_hits"] += anchor_count
        if anchor_count == 1:
            timings["query_motif_hit_singleton_regions"] = (
                timings.get("query_motif_hit_singleton_regions", 0.0) + 1
            )
        elif anchor_count == 2:
            timings["query_motif_hit_pair_regions"] = (
                timings.get("query_motif_hit_pair_regions", 0.0) + 1
            )
        else:
            timings["query_motif_hit_multi_anchor_regions"] = (
                timings.get("query_motif_hit_multi_anchor_regions", 0.0) + 1
            )
        provider_start = time.perf_counter()
        full_scores = full_score_provider(str(chrom), int(start), int(end))
        timings["full_score_provider_wall_s"] += time.perf_counter() - provider_start
        timings["full_score_regions_scanned"] += 1
        anchor_read_start = time.perf_counter()
        anchors = DenseQueryMotifHitResult(
            np.asarray(anchor_batch_indices[anchor_start : anchor_start + anchor_count]),
            np.asarray(anchor_positions[anchor_start : anchor_start + anchor_count]),
            anchor_count,
        )
        timings["query_motif_hit_array_read_wall_s"] += time.perf_counter() - anchor_read_start
        if anchor_count > 2:
            anchor_batch_codes = anchors.batch_indices // 1_000_000
            if bool(np.all(anchor_batch_codes == anchor_batch_codes[0])):
                timings["query_motif_hit_same_batch_multi_anchor_regions"] += 1
        contribution_start = time.perf_counter()
        contribution = dense_region_contribution_from_scores_and_query_motif_hits(
            full_scores,
            anchors,
            window_size=window_size,
            n_motifs=n_motifs,
            dtype=torch_dtype,
            device=device,
            anchor_contribution_mode=anchor_contribution_mode,
            return_tensor=bool(output_accumulator == "torch"),
        )
        timings["dense_contribution_wall_s"] += time.perf_counter() - contribution_start
        subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
        if coalesce_membership_patterns:
            if output_accumulator == "torch":
                contribution = contribution.detach().cpu().numpy()
            pattern_key = (
                tuple(int(x) for x in subset_indices),
                tuple(int(x) for x in counts),
            )
            if not pattern_key[0] or not np.any(contribution):
                continue
            accumulate_start = time.perf_counter()
            pattern_sum = pattern_contributions.get(pattern_key)
            if pattern_sum is None:
                pattern_sum = np.zeros_like(contribution, dtype=dtype)
                pattern_accumulator_bytes += int(pattern_sum.nbytes)
                if max_pattern_accumulator_bytes is not None and pattern_accumulator_bytes > int(
                    max_pattern_accumulator_bytes
                ):
                    raise MemoryError(
                        "Planned dense query-motif-hit membership-pattern "
                        "accumulator is "
                        f"{pattern_accumulator_bytes:,} bytes, above "
                        "max_pattern_accumulator_bytes="
                        f"{int(max_pattern_accumulator_bytes):,}."
                    )
                pattern_contributions[pattern_key] = pattern_sum
            pattern_sum += contribution.astype(dtype, copy=False)
            timings["membership_pattern_accumulate_wall_s"] += (
                time.perf_counter() - accumulate_start
            )
            timings["membership_pattern_interval_assignments"] += 1
            continue
        update_start = time.perf_counter()
        if output_accumulator == "torch":
            _add_contribution_to_subset_values_torch(
                values_torch,
                subset_indices,
                counts,
                contribution,
                dtype=torch_dtype,
                device=device,
                chunk_subsets=chunk_subsets,
            )
        else:
            _add_contribution_to_subset_values(
                values,
                subset_indices,
                counts,
                contribution,
                dtype=dtype,
                chunk_subsets=chunk_subsets,
            )
            timings["subset_update_wall_s"] += time.perf_counter() - update_start
    if timings["query_motif_hit_cache_required_intervals"]:
        timings["query_motif_hit_cache_coverage_fraction"] = float(
            timings["query_motif_hit_cache_hits"]
        ) / float(timings["query_motif_hit_cache_required_intervals"])
    if output_accumulator == "torch":
        copy_start = time.perf_counter()
        values = values_torch.detach().cpu().numpy().astype(dtype, copy=False)
        timings["output_accumulator_to_numpy_wall_s"] = time.perf_counter() - copy_start
    if coalesce_membership_patterns:
        timings["membership_pattern_count"] = float(len(pattern_contributions))
        timings["membership_pattern_accumulator_bytes"] = float(pattern_accumulator_bytes)
        for (
            subset_indices_tuple,
            counts_tuple,
        ), contribution in pattern_contributions.items():
            subset_indices = np.asarray(subset_indices_tuple, dtype=np.int64)
            counts = np.asarray(counts_tuple, dtype=np.int64)
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
    return DenseSubsetAggregationResult(subset_ids, values, plan, timings)


def _dense_interval_contribution_cache_metadata_snapshot(attrs: dict) -> dict:
    keys = (
        "schema_version",
        "complete",
        "semantics",
        "coordinate_frame",
        "subset_column",
        "n_input_rows",
        "n_dropped_invalid_regions",
        "n_regions",
        "n_unique_regions",
        "reuse_factor",
        "motif_count",
        "window_size",
        "query_motif_index",
        "contribution_shape",
        "contribution_cache_bytes",
        "dtype",
        "motif_source",
        "aligned_motif_path",
        "loaded_with_aligned",
        "motif_names_checksum",
        "motif_kernel_checksum",
        "motif_kernel_shape",
        "threshold_mode",
        "p_value_threshold",
        "score_threshold",
        "score_threshold_vector_checksum",
        "genome_zarr_path",
        "values_checksum",
        "values_checksum_mode",
        "values_checksum_status",
    )
    return {key: attrs.get(key) for key in keys if key in attrs}


def _dense_interval_contribution_key_to_index(root) -> dict[tuple[str, int, int], int]:
    chroms = root["intervals/chrom"][:].tolist()
    starts = root["intervals/start"][:].astype(np.int64)
    ends = root["intervals/end"][:].astype(np.int64)
    return {
        _interval_key(chrom, int(start), int(end)): idx
        for idx, (chrom, start, end) in enumerate(zip(chroms, starts, ends, strict=True))
    }


def build_dense_interval_contribution_cache(
    score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int | None = None,
    p_value_threshold: str = "p0.0001",
    dtype: np.dtype | str = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_cache_bytes: int | None = None,
    chunk_intervals: int = 1,
    metadata_extra: dict | None = None,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
) -> DenseIntervalContributionCacheResult:
    """Build an exact dense full-curve contribution cache per unique interval.

    Each unique ``chrom,start,end`` interval is scanned once by ``score_provider``.
    The stored value is the same dense contribution that would later be added to
    every subset containing that interval, preserving the original full-curve
    one-to-all/all-to-all semantics while making later subset replays independent
    of genome IO and motif convolution.
    """
    dtype = np.dtype(dtype)
    plan = plan_region_subset_aggregation(
        region_memberships_df,
        subset_column=subset_column,
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
        dtype=dtype,
    )
    normalized = _normalized_regions(region_memberships_df, subset_column)
    unique_regions = _unique_regions_frame(normalized)
    contribution_shape = _contribution_shape(
        n_motifs=n_motifs,
        window_size=window_size,
        query_motif_index=query_motif_index,
    )
    contribution_cache_bytes = int(
        len(unique_regions) * np.prod(contribution_shape, dtype=np.int64) * dtype.itemsize
    )
    if max_cache_bytes is not None and contribution_cache_bytes > int(max_cache_bytes):
        raise MemoryError(
            f"Planned dense interval contribution cache is "
            f"{contribution_cache_bytes:,} bytes, above "
            f"max_cache_bytes={int(max_cache_bytes):,}."
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
    base_metadata = _dense_subset_base_metadata(
        plan=plan,
        subset_column=subset_column,
        query_motif_index=query_motif_index,
        dtype=dtype,
        metadata_extra=metadata_extra,
    )
    base_metadata.update(
        {
            "schema_version": DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION,
            "complete": False,
            "semantics": DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS,
            "contribution_shape": list(contribution_shape),
            "contribution_cache_bytes": contribution_cache_bytes,
            "chunk_intervals": max(1, int(chunk_intervals)),
        }
    )
    metadata.attrs.update(base_metadata)
    timings = {
        "score_provider_wall_s": 0.0,
        "dense_contribution_wall_s": 0.0,
        "checksum_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "nonzero_contribution_regions": 0.0,
    }

    for interval_idx, row in enumerate(unique_regions.itertuples(index=False)):
        timings["unique_regions_processed"] += 1
        provider_start = time.perf_counter()
        score_batches = score_provider(str(row.chrom), int(row.start), int(row.end))
        timings["score_provider_wall_s"] += time.perf_counter() - provider_start

        contribution_start = time.perf_counter()
        contribution = dense_region_contribution_from_scores(
            score_batches,
            score_thresholds=score_thresholds,
            window_size=window_size,
            n_motifs=n_motifs,
            query_motif_index=query_motif_index,
            p_value_threshold=p_value_threshold,
            dtype=torch_dtype,
            device=device,
        )
        timings["dense_contribution_wall_s"] += time.perf_counter() - contribution_start
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
    provider_stats = getattr(score_provider, "stats", None)
    if isinstance(provider_stats, dict):
        metadata.attrs.update(provider_stats)
    metadata.attrs.update(checksum_metadata)
    metadata.attrs["complete"] = True
    return DenseIntervalContributionCacheResult(
        path=str(output_path),
        n_unique_regions=int(len(unique_regions)),
        contribution_shape=contribution_shape,
        checksum=checksum,
        timings=timings,
    )


def _accumulate_dense_interval_contribution_cache_into_values(
    *,
    contribution_cache_root,
    normalized: pd.DataFrame,
    values,
    subset_to_index: dict[str, int],
    dtype: np.dtype,
    chunk_subsets: int,
    timings: dict[str, float],
    missing_intervals_are_zero: bool = False,
) -> None:
    key_to_index = _dense_interval_contribution_key_to_index(contribution_cache_root)
    contribution_values = contribution_cache_root["values"]
    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    for (chrom, start, end), memberships in grouped:
        key = _interval_key(chrom, int(start), int(end))
        if key not in key_to_index:
            if missing_intervals_are_zero:
                timings["contribution_cache_missing_zero_intervals"] += 1
                continue
            raise ValueError(
                "Dense interval contribution cache is missing required interval "
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


def _membership_pattern_key(
    memberships: pd.DataFrame,
    subset_to_index: dict[str, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
    return tuple(int(x) for x in subset_indices), tuple(int(x) for x in counts)


def _accumulate_dense_interval_contribution_cache_into_values_by_pattern(
    *,
    contribution_cache_root,
    normalized: pd.DataFrame,
    values,
    subset_to_index: dict[str, int],
    dtype: np.dtype,
    chunk_subsets: int,
    timings: dict[str, float],
    max_pattern_accumulator_bytes: int | None = None,
    pattern_read_batch_intervals: int = 64,
    missing_intervals_are_zero: bool = False,
) -> None:
    key_to_index = _dense_interval_contribution_key_to_index(contribution_cache_root)
    contribution_values = contribution_cache_root["values"]
    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    interval_records: list[tuple[int, tuple[tuple[int, ...], tuple[int, ...]]]] = []
    for (chrom, start, end), memberships in grouped:
        key = _interval_key(chrom, int(start), int(end))
        if key not in key_to_index:
            if missing_intervals_are_zero:
                timings["contribution_cache_missing_zero_intervals"] += 1
                continue
            raise ValueError(
                "Dense interval contribution cache is missing required interval "
                f"{key[0]}:{key[1]}-{key[2]}"
            )
        pattern_key = _membership_pattern_key(memberships, subset_to_index)
        if not pattern_key[0]:
            continue
        interval_records.append((int(key_to_index[key]), pattern_key))
    interval_records.sort(key=lambda item: item[0])

    pattern_contributions: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray] = {}
    pattern_accumulator_bytes = 0
    batch_size = max(1, int(pattern_read_batch_intervals))
    timings["membership_pattern_read_batch_intervals"] = float(batch_size)
    for batch_start in range(0, len(interval_records), batch_size):
        batch_records = interval_records[batch_start : batch_start + batch_size]
        first_index = int(batch_records[0][0])
        last_index = int(batch_records[-1][0])
        lookup_start = time.perf_counter()
        contribution_batch = np.asarray(contribution_values[first_index : last_index + 1])
        timings["contribution_lookup_wall_s"] += time.perf_counter() - lookup_start
        timings["contribution_cache_read_batches"] += 1
        timings["contribution_cache_intervals_read"] += int(last_index - first_index + 1)
        timings["contribution_cache_interval_overread"] += int(
            last_index - first_index + 1 - len(batch_records)
        )
        for cache_index, pattern_key in batch_records:
            contribution = contribution_batch[int(cache_index) - first_index]
            timings["contribution_cache_hits"] += 1
            if not np.any(contribution):
                continue
            timings["nonzero_contribution_regions"] += 1
            accumulate_start = time.perf_counter()
            pattern_sum = pattern_contributions.get(pattern_key)
            if pattern_sum is None:
                pattern_sum = np.zeros_like(contribution, dtype=dtype)
                pattern_accumulator_bytes += int(pattern_sum.nbytes)
                if max_pattern_accumulator_bytes is not None and pattern_accumulator_bytes > int(
                    max_pattern_accumulator_bytes
                ):
                    raise MemoryError(
                        "Planned dense membership-pattern accumulator is "
                        f"{pattern_accumulator_bytes:,} bytes, above "
                        "max_pattern_accumulator_bytes="
                        f"{int(max_pattern_accumulator_bytes):,}."
                    )
                pattern_contributions[pattern_key] = pattern_sum
            pattern_sum += contribution.astype(dtype, copy=False)
            timings["membership_pattern_accumulate_wall_s"] += (
                time.perf_counter() - accumulate_start
            )
            timings["membership_pattern_interval_assignments"] += 1

    timings["membership_pattern_count"] = float(len(pattern_contributions))
    timings["membership_pattern_accumulator_bytes"] = float(pattern_accumulator_bytes)
    for (subset_indices_tuple, counts_tuple), contribution in pattern_contributions.items():
        subset_indices = np.asarray(subset_indices_tuple, dtype=np.int64)
        counts = np.asarray(counts_tuple, dtype=np.int64)
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


def _validate_dense_interval_contribution_cache_attrs(attrs: dict) -> None:
    if attrs.get("schema_version") != DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Expected dense interval contribution cache schema "
            f"{DENSE_INTERVAL_CONTRIBUTION_CACHE_SCHEMA_VERSION!r}; got "
            f"{attrs.get('schema_version')!r}."
        )
    if attrs.get("complete") is not True:
        raise ValueError("Dense interval contribution cache is incomplete.")
    if attrs.get("semantics") != DENSE_INTERVAL_CONTRIBUTION_CACHE_SEMANTICS:
        raise ValueError(
            "Dense interval contribution cache has unexpected semantics "
            f"{attrs.get('semantics')!r}."
        )


def accumulate_dense_score_region_subsets_from_contribution_cache(
    contribution_cache_path: str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    subset_column: str,
    dtype: np.dtype | str | None = None,
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
    coalesce_membership_patterns: bool = False,
    max_pattern_accumulator_bytes: int | None = None,
    membership_pattern_read_batch_intervals: int = 64,
    missing_intervals_are_zero: bool | None = None,
) -> DenseSubsetAggregationResult:
    """Replay exact dense interval contributions into many subset full curves."""
    root = zarr.open_group(str(contribution_cache_path), mode="r")
    attrs = dict(root["metadata"].attrs)
    _validate_dense_interval_contribution_cache_attrs(attrs)
    if missing_intervals_are_zero is None:
        missing_intervals_are_zero = bool(attrs.get("sparse_nonzero_contributions"))
    n_motifs = int(attrs["motif_count"])
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
    if max_output_bytes is not None and plan.output_bytes > int(max_output_bytes):
        raise MemoryError(
            f"Planned exact dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={int(max_output_bytes):,}."
        )
    values = np.zeros(plan.output_shape, dtype=dtype)
    timings = {
        "contribution_lookup_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "contribution_cache_hits": 0.0,
        "nonzero_contribution_regions": 0.0,
        "region_query_strategy": "dense_interval_contribution_cache",
        "membership_pattern_coalescing": bool(coalesce_membership_patterns),
        "membership_pattern_count": 0.0,
        "membership_pattern_interval_assignments": 0.0,
        "membership_pattern_accumulator_bytes": 0.0,
        "membership_pattern_accumulate_wall_s": 0.0,
        "membership_pattern_read_batch_intervals": float(
            max(1, int(membership_pattern_read_batch_intervals))
        ),
        "contribution_cache_read_batches": 0.0,
        "contribution_cache_intervals_read": 0.0,
        "contribution_cache_interval_overread": 0.0,
        "contribution_cache_missing_zero_intervals": 0.0,
        "missing_intervals_are_zero": bool(missing_intervals_are_zero),
    }
    if coalesce_membership_patterns:
        _accumulate_dense_interval_contribution_cache_into_values_by_pattern(
            contribution_cache_root=root,
            normalized=normalized,
            values=values,
            subset_to_index=subset_to_index,
            dtype=dtype,
            chunk_subsets=chunk_subsets,
            timings=timings,
            max_pattern_accumulator_bytes=max_pattern_accumulator_bytes,
            pattern_read_batch_intervals=membership_pattern_read_batch_intervals,
            missing_intervals_are_zero=bool(missing_intervals_are_zero),
        )
    else:
        _accumulate_dense_interval_contribution_cache_into_values(
            contribution_cache_root=root,
            normalized=normalized,
            values=values,
            subset_to_index=subset_to_index,
            dtype=dtype,
            chunk_subsets=chunk_subsets,
            timings=timings,
            missing_intervals_are_zero=bool(missing_intervals_are_zero),
        )
    return DenseSubsetAggregationResult(subset_ids, values, plan, timings)


def accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr(
    contribution_cache_path: str | Path,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    dtype: np.dtype | str | None = None,
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
    metadata_extra: dict | None = None,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
    buffered_subset_updates: bool = False,
    max_buffered_output_bytes: int | None = None,
    chunk_buffered_subset_updates: bool = False,
    max_chunk_output_bytes: int | None = None,
    max_active_subset_chunks: int = 4,
    coalesce_membership_patterns: bool = False,
    max_pattern_accumulator_bytes: int | None = None,
    membership_pattern_read_batch_intervals: int = 64,
    missing_intervals_are_zero: bool | None = None,
) -> DenseSubsetZarrResult:
    """Replay a dense interval contribution cache into a disk-backed zarr output."""
    if buffered_subset_updates and chunk_buffered_subset_updates:
        raise ValueError(
            "Use only one of buffered_subset_updates or chunk_buffered_subset_updates."
        )
    if coalesce_membership_patterns and chunk_buffered_subset_updates:
        raise ValueError(
            "coalesce_membership_patterns currently supports in-memory or "
            "direct zarr replay; disable chunk_buffered_subset_updates."
        )
    contribution_root = zarr.open_group(str(contribution_cache_path), mode="r")
    contribution_attrs = dict(contribution_root["metadata"].attrs)
    _validate_dense_interval_contribution_cache_attrs(contribution_attrs)
    if missing_intervals_are_zero is None:
        missing_intervals_are_zero = bool(contribution_attrs.get("sparse_nonzero_contributions"))
    n_motifs = int(contribution_attrs["motif_count"])
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
    if max_output_bytes is not None and plan.output_bytes > int(max_output_bytes):
        raise MemoryError(
            f"Planned exact dense subset zarr output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={int(max_output_bytes):,}."
        )
    if buffered_subset_updates and (
        max_buffered_output_bytes is not None and plan.output_bytes > int(max_buffered_output_bytes)
    ):
        raise MemoryError(
            f"Planned buffered exact dense subset output is {plan.output_bytes:,} "
            f"bytes, above max_buffered_output_bytes="
            f"{int(max_buffered_output_bytes):,}."
        )
    (
        requested_chunk_subsets,
        effective_chunk_subsets,
        per_subset_output_bytes,
    ) = _dense_subset_chunk_settings(
        plan,
        dtype=dtype,
        chunk_subsets=chunk_subsets,
        max_chunk_output_bytes=max_chunk_output_bytes if chunk_buffered_subset_updates else None,
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
    merged_metadata = {
        "region_query_strategy": "dense_interval_contribution_cache",
        "source_contribution_cache_path": str(contribution_cache_path),
        "contribution_cache_metadata": (
            _dense_interval_contribution_cache_metadata_snapshot(contribution_attrs)
        ),
    }
    if metadata_extra:
        merged_metadata.update(metadata_extra)
    base_metadata = _dense_subset_base_metadata(
        plan=plan,
        subset_column=subset_column,
        query_motif_index=query_motif_index,
        dtype=dtype,
        metadata_extra=merged_metadata,
    )
    base_metadata.update(
        {
            "subset_update_mode": "chunk_buffered_output"
            if chunk_buffered_subset_updates
            else "buffered_full_output"
            if buffered_subset_updates
            else "membership_pattern_coalesced_output"
            if coalesce_membership_patterns
            else "zarr_interval_incremental",
            "buffered_output_bytes": plan.output_bytes if buffered_subset_updates else 0,
            "requested_chunk_subsets": requested_chunk_subsets,
            "chunk_subsets": effective_chunk_subsets,
            "chunk_buffer_bytes": int(effective_chunk_subsets * per_subset_output_bytes),
            "max_active_subset_chunks": max(1, int(max_active_subset_chunks)),
        }
    )
    metadata.attrs.update(base_metadata)
    chunk_shape = (effective_chunk_subsets,) + tuple(plan.output_shape[1:])
    values = root.create_array(
        "values",
        shape=plan.output_shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressors=COMPRESSOR,
        overwrite=True,
    )
    timings = {
        "contribution_lookup_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "subset_buffer_update_wall_s": 0.0,
        "final_zarr_write_wall_s": 0.0,
        "subset_chunk_reads": 0.0,
        "subset_chunk_read_wall_s": 0.0,
        "subset_chunk_flushes": 0.0,
        "checksum_wall_s": 0.0,
        "contribution_cache_hits": 0.0,
        "nonzero_contribution_regions": 0.0,
        "region_query_strategy": "dense_interval_contribution_cache",
        "membership_pattern_coalescing": bool(coalesce_membership_patterns),
        "membership_pattern_count": 0.0,
        "membership_pattern_interval_assignments": 0.0,
        "membership_pattern_accumulator_bytes": 0.0,
        "membership_pattern_accumulate_wall_s": 0.0,
        "membership_pattern_read_batch_intervals": float(
            max(1, int(membership_pattern_read_batch_intervals))
        ),
        "contribution_cache_read_batches": 0.0,
        "contribution_cache_intervals_read": 0.0,
        "contribution_cache_interval_overread": 0.0,
        "contribution_cache_missing_zero_intervals": 0.0,
        "missing_intervals_are_zero": bool(missing_intervals_are_zero),
    }
    accumulation_values = values
    if buffered_subset_updates:
        accumulation_values = np.zeros(plan.output_shape, dtype=dtype)
    chunk_buffers: OrderedDict[int, np.ndarray] = OrderedDict()
    written_chunks: set[int] = set()
    grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
    key_to_index = _dense_interval_contribution_key_to_index(contribution_root)
    contribution_values = contribution_root["values"]
    if coalesce_membership_patterns:
        _accumulate_dense_interval_contribution_cache_into_values_by_pattern(
            contribution_cache_root=contribution_root,
            normalized=normalized,
            values=accumulation_values,
            subset_to_index=subset_to_index,
            dtype=dtype,
            chunk_subsets=effective_chunk_subsets,
            timings=timings,
            max_pattern_accumulator_bytes=max_pattern_accumulator_bytes,
            pattern_read_batch_intervals=membership_pattern_read_batch_intervals,
            missing_intervals_are_zero=bool(missing_intervals_are_zero),
        )
    else:
        for (chrom, start, end), memberships in grouped:
            key = _interval_key(chrom, int(start), int(end))
            if key not in key_to_index:
                if missing_intervals_are_zero:
                    timings["contribution_cache_missing_zero_intervals"] += 1
                    continue
                raise ValueError(
                    "Dense interval contribution cache is missing required interval "
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
            if chunk_buffered_subset_updates:
                _add_contribution_to_dense_subset_chunk_buffers(
                    values=values,
                    buffers=chunk_buffers,
                    written_chunks=written_chunks,
                    subset_indices=subset_indices,
                    counts=counts,
                    contribution=contribution,
                    dtype=dtype,
                    chunk_subsets=effective_chunk_subsets,
                    max_active_subset_chunks=max_active_subset_chunks,
                    output_shape=plan.output_shape,
                    timings=timings,
                )
            else:
                update_start = time.perf_counter()
                _add_contribution_to_subset_values(
                    accumulation_values,
                    subset_indices,
                    counts,
                    contribution,
                    dtype=dtype,
                    chunk_subsets=effective_chunk_subsets,
                )
                update_wall = time.perf_counter() - update_start
                timings["subset_buffer_update_wall_s"] += update_wall
                timings["subset_update_wall_s"] += update_wall

    if buffered_subset_updates:
        write_start = time.perf_counter()
        values[:] = accumulation_values
        timings["final_zarr_write_wall_s"] = time.perf_counter() - write_start
        timings["subset_update_wall_s"] += timings["final_zarr_write_wall_s"]
    if chunk_buffered_subset_updates:
        for chunk_index in list(chunk_buffers):
            _flush_dense_subset_chunk_buffer(
                values=values,
                buffers=chunk_buffers,
                written_chunks=written_chunks,
                chunk_index=chunk_index,
                chunk_subsets=effective_chunk_subsets,
                timings=timings,
            )

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
    return DenseSubsetZarrResult(str(output_path), subset_ids, plan, checksum, timings)


def _dense_subset_base_metadata(
    *,
    plan: MultiSubsetAggregationPlan,
    subset_column: str,
    query_motif_index: int | None,
    dtype: np.dtype,
    metadata_extra: dict | None,
) -> dict:
    metadata = {
        "schema_version": DENSE_SUBSET_ZARR_SCHEMA_VERSION,
        "complete": False,
        "semantics": DENSE_SUBSET_SEMANTICS,
        "coordinate_frame": DENSE_SUBSET_COORDINATE_FRAME,
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
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return metadata


def _dense_subset_chunk_settings(
    plan: MultiSubsetAggregationPlan,
    *,
    dtype: np.dtype,
    chunk_subsets: int,
    max_chunk_output_bytes: int | None,
) -> tuple[int, int, int]:
    per_subset_bytes = int(np.prod(plan.output_shape[1:], dtype=np.int64) * dtype.itemsize)
    requested_chunk_subsets = max(1, int(chunk_subsets))
    if max_chunk_output_bytes is None:
        return requested_chunk_subsets, requested_chunk_subsets, per_subset_bytes
    max_by_bytes = int(max_chunk_output_bytes) // max(1, per_subset_bytes)
    if max_by_bytes < 1:
        raise MemoryError(
            f"One subset output chunk is {per_subset_bytes:,} bytes, above "
            f"max_chunk_output_bytes={max_chunk_output_bytes:,}."
        )
    return (
        requested_chunk_subsets,
        min(requested_chunk_subsets, max_by_bytes),
        per_subset_bytes,
    )


def _flush_dense_subset_chunk_buffer(
    *,
    values,
    buffers: OrderedDict[int, np.ndarray],
    written_chunks: set[int],
    chunk_index: int,
    chunk_subsets: int,
    timings: dict,
) -> None:
    chunk_values = buffers.pop(chunk_index)
    chunk_start = chunk_index * chunk_subsets
    chunk_end = chunk_start + chunk_values.shape[0]
    write_start = time.perf_counter()
    values[chunk_start:chunk_end] = chunk_values
    write_wall = time.perf_counter() - write_start
    timings["final_zarr_write_wall_s"] += write_wall
    timings["subset_update_wall_s"] += write_wall
    timings["subset_chunk_flushes"] += 1
    written_chunks.add(chunk_index)


def _add_contribution_to_dense_subset_chunk_buffers(
    *,
    values,
    buffers: OrderedDict[int, np.ndarray],
    written_chunks: set[int],
    subset_indices: np.ndarray,
    counts: np.ndarray,
    contribution: np.ndarray,
    dtype: np.dtype,
    chunk_subsets: int,
    max_active_subset_chunks: int,
    output_shape: tuple[int, ...],
    timings: dict,
) -> None:
    if subset_indices.size == 0:
        return
    max_active_subset_chunks = max(1, int(max_active_subset_chunks))

    def get_chunk(chunk_index: int) -> np.ndarray:
        if chunk_index in buffers:
            buffers.move_to_end(chunk_index)
            return buffers[chunk_index]
        while len(buffers) >= max_active_subset_chunks:
            old_index = next(iter(buffers))
            _flush_dense_subset_chunk_buffer(
                values=values,
                buffers=buffers,
                written_chunks=written_chunks,
                chunk_index=old_index,
                chunk_subsets=chunk_subsets,
                timings=timings,
            )
        chunk_start = chunk_index * chunk_subsets
        chunk_end = min(chunk_start + chunk_subsets, output_shape[0])
        chunk_shape = (chunk_end - chunk_start,) + tuple(output_shape[1:])
        if chunk_index in written_chunks:
            read_start = time.perf_counter()
            chunk_values = np.asarray(values[chunk_start:chunk_end])
            timings["subset_chunk_read_wall_s"] += time.perf_counter() - read_start
            timings["subset_chunk_reads"] += 1
        else:
            chunk_values = np.zeros(chunk_shape, dtype=dtype)
        buffers[chunk_index] = chunk_values
        return chunk_values

    for chunk_index in np.unique(subset_indices // chunk_subsets):
        chunk_index = int(chunk_index)
        chunk_start = chunk_index * chunk_subsets
        chunk_end = min(chunk_start + chunk_subsets, output_shape[0])
        left = int(np.searchsorted(subset_indices, chunk_start, side="left"))
        right = int(np.searchsorted(subset_indices, chunk_end, side="left"))
        if left == right:
            continue
        chunk_values = get_chunk(chunk_index)
        update_start = time.perf_counter()
        _add_contribution_to_subset_values(
            chunk_values,
            subset_indices[left:right] - chunk_start,
            counts[left:right],
            contribution,
            dtype=dtype,
            chunk_subsets=chunk_values.shape[0],
        )
        update_wall = time.perf_counter() - update_start
        timings["subset_buffer_update_wall_s"] += update_wall
        timings["subset_update_wall_s"] += update_wall


def accumulate_dense_score_region_subsets_to_zarr(
    score_provider: DenseScoreProvider,
    region_memberships_df: pd.DataFrame,
    *,
    output_path: str | Path,
    subset_column: str,
    score_thresholds: dict[str, torch.Tensor],
    window_size: int,
    n_motifs: int,
    query_motif_index: int | None = None,
    p_value_threshold: str = "p0.0001",
    dtype: np.dtype | str = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_output_bytes: int | None = None,
    chunk_subsets: int = 1,
    metadata_extra: dict | None = None,
    checksum_mode: str = "full",
    checksum_sample_subsets: int = 16,
    buffered_subset_updates: bool = False,
    max_buffered_output_bytes: int | None = None,
    chunk_buffered_subset_updates: bool = False,
    max_chunk_output_bytes: int | None = None,
    max_active_subset_chunks: int = 4,
) -> DenseSubsetZarrResult:
    """Aggregate exact dense direct-scan contributions into a zarr array.

    This is the disk-backed counterpart to
    ``accumulate_dense_score_region_subsets``. It has the same exact dense
    accumulation semantics, but writes subset tensors incrementally so large
    outputs can spill to disk instead of requiring one dense RAM allocation.
    """
    if buffered_subset_updates and chunk_buffered_subset_updates:
        raise ValueError(
            "Use only one of buffered_subset_updates or chunk_buffered_subset_updates."
        )
    dtype = np.dtype(dtype)
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
            f"Planned exact dense subset zarr output is {plan.output_bytes:,} bytes, "
            f"above max_output_bytes={max_output_bytes:,}."
        )
    if buffered_subset_updates and (
        max_buffered_output_bytes is not None and plan.output_bytes > max_buffered_output_bytes
    ):
        raise MemoryError(
            f"Planned buffered exact dense subset output is {plan.output_bytes:,} bytes, "
            f"above max_buffered_output_bytes={max_buffered_output_bytes:,}."
        )
    (
        requested_chunk_subsets,
        effective_chunk_subsets,
        per_subset_output_bytes,
    ) = _dense_subset_chunk_settings(
        plan,
        dtype=dtype,
        chunk_subsets=chunk_subsets,
        max_chunk_output_bytes=max_chunk_output_bytes if chunk_buffered_subset_updates else None,
    )

    normalized = _normalized_regions(region_memberships_df, subset_column)
    subset_ids = sorted(normalized["subset_id"].unique().tolist())
    subset_to_index = {subset_id: idx for idx, subset_id in enumerate(subset_ids)}

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
    base_metadata = _dense_subset_base_metadata(
        plan=plan,
        subset_column=subset_column,
        query_motif_index=query_motif_index,
        dtype=dtype,
        metadata_extra=metadata_extra,
    )
    base_metadata.update(
        {
            "subset_update_mode": "chunk_buffered_output"
            if chunk_buffered_subset_updates
            else "buffered_full_output"
            if buffered_subset_updates
            else "zarr_interval_incremental",
            "buffered_output_bytes": plan.output_bytes if buffered_subset_updates else 0,
            "requested_chunk_subsets": requested_chunk_subsets,
            "chunk_subsets": effective_chunk_subsets,
            "chunk_buffer_bytes": int(effective_chunk_subsets * per_subset_output_bytes),
            "max_active_subset_chunks": max(1, int(max_active_subset_chunks)),
        }
    )
    metadata.attrs.update(base_metadata)
    timings = {
        "score_provider_wall_s": 0.0,
        "dense_contribution_wall_s": 0.0,
        "subset_update_wall_s": 0.0,
        "subset_buffer_update_wall_s": 0.0,
        "final_zarr_write_wall_s": 0.0,
        "subset_chunk_reads": 0.0,
        "subset_chunk_read_wall_s": 0.0,
        "subset_chunk_flushes": 0.0,
        "checksum_wall_s": 0.0,
        "unique_regions_processed": 0.0,
        "nonzero_contribution_regions": 0.0,
    }
    chunk_shape = (effective_chunk_subsets,) + tuple(plan.output_shape[1:])
    values = root.create_array(
        "values",
        shape=plan.output_shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressors=COMPRESSOR,
        overwrite=True,
    )
    accumulation_values = values
    if buffered_subset_updates:
        accumulation_values = np.zeros(plan.output_shape, dtype=dtype)
    chunk_buffers: OrderedDict[int, np.ndarray] = OrderedDict()
    written_chunks: set[int] = set()
    if len(normalized) > 0:
        grouped = normalized.groupby(["chrom", "start", "end"], sort=True)
        for (chrom, start, end), memberships in grouped:
            timings["unique_regions_processed"] += 1
            provider_start = time.perf_counter()
            score_batches = score_provider(str(chrom), int(start), int(end))
            timings["score_provider_wall_s"] += time.perf_counter() - provider_start

            contribution_start = time.perf_counter()
            contribution = dense_region_contribution_from_scores(
                score_batches,
                score_thresholds=score_thresholds,
                window_size=window_size,
                n_motifs=n_motifs,
                query_motif_index=query_motif_index,
                p_value_threshold=p_value_threshold,
                dtype=torch_dtype,
                device=device,
            )
            timings["dense_contribution_wall_s"] += time.perf_counter() - contribution_start
            if not np.any(contribution):
                continue
            timings["nonzero_contribution_regions"] += 1
            subset_indices, counts = _membership_subset_counts(memberships, subset_to_index)
            if chunk_buffered_subset_updates:
                _add_contribution_to_dense_subset_chunk_buffers(
                    values=values,
                    buffers=chunk_buffers,
                    written_chunks=written_chunks,
                    subset_indices=subset_indices,
                    counts=counts,
                    contribution=contribution,
                    dtype=dtype,
                    chunk_subsets=effective_chunk_subsets,
                    max_active_subset_chunks=max_active_subset_chunks,
                    output_shape=plan.output_shape,
                    timings=timings,
                )
            else:
                update_start = time.perf_counter()
                _add_contribution_to_subset_values(
                    accumulation_values,
                    subset_indices,
                    counts,
                    contribution,
                    dtype=dtype,
                    chunk_subsets=effective_chunk_subsets,
                )
                update_wall = time.perf_counter() - update_start
                timings["subset_buffer_update_wall_s"] += update_wall
                timings["subset_update_wall_s"] += update_wall

    if buffered_subset_updates:
        write_start = time.perf_counter()
        values[:] = accumulation_values
        timings["final_zarr_write_wall_s"] = time.perf_counter() - write_start
        timings["subset_update_wall_s"] += timings["final_zarr_write_wall_s"]
    if chunk_buffered_subset_updates:
        for chunk_index in list(chunk_buffers):
            _flush_dense_subset_chunk_buffer(
                values=values,
                buffers=chunk_buffers,
                written_chunks=written_chunks,
                chunk_index=chunk_index,
                chunk_subsets=effective_chunk_subsets,
                timings=timings,
            )

    checksum_start = time.perf_counter()
    checksum, checksum_metadata = finalize_zarr_checksum(
        values,
        checksum_mode=checksum_mode,
        checksum_sample_subsets=checksum_sample_subsets,
    )
    timings["checksum_wall_s"] = time.perf_counter() - checksum_start
    metadata.attrs.update(timings)
    provider_stats = getattr(score_provider, "stats", None)
    if isinstance(provider_stats, dict):
        metadata.attrs.update(provider_stats)
    metadata.attrs.update(checksum_metadata)
    metadata.attrs["complete"] = True
    return DenseSubsetZarrResult(str(output_path), subset_ids, plan, checksum, timings)
