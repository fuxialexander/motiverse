"""Small, dependency-light adapters for genomic sequence Zarr stores.

Motiverse understands the same ``chrs/<chrom>`` layout used by Caesarion but
does not require Caesarion to read it.  Both current array-backed stores and
the older ``chunk_N`` group layout are supported.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

logger = logging.getLogger(__name__)

NARROWPEAK_COLUMNS = [
    "chrom",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "signalValue",
    "pValue",
    "qValue",
    "peak",
]


class SequenceDenseZarrIO:
    """Read dense one-hot genomic sequences from a Zarr group."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        dtype: str = "int8",
        chroms: list[str] | None = None,
    ) -> None:
        self.path = str(path)
        self.mode = mode
        self.dtype = np.dtype(dtype)
        self.dataset = zarr.open_group(self.path, mode=mode)
        if "chrs" not in self.dataset:
            raise ValueError(f"{self.path!r} has no top-level 'chrs' group")
        available = list(self.dataset["chrs"].keys())
        self.chroms = (
            [chrom for chrom in chroms if chrom in available] if chroms is not None else available
        )
        if not self.chroms:
            raise ValueError(f"No usable chromosomes found in {self.path!r}")
        self.assembly = self.dataset.attrs.get("assembly")
        first = self.dataset["chrs"][self.chroms[0]]
        self.dir_chunked = isinstance(first, zarr.Group)
        self.chunk_size = self._chunk_size(first)
        self.chrom_sizes = {
            chrom: self._chrom_size(self.dataset["chrs"][chrom]) for chrom in self.chroms
        }
        self.feature_size = self._feature_size(first)
        self.chrom_n_chunks = {
            chrom: max(
                1,
                int(np.ceil(size / self.chunk_size)) if self.chunk_size else 1,
            )
            for chrom, size in self.chrom_sizes.items()
        }

    @staticmethod
    def _ordered_chunks(group: zarr.Group) -> list[str]:
        def key(name: str) -> tuple[int, str]:
            try:
                return int(name.rsplit("_", 1)[-1]), name
            except ValueError:
                return 0, name

        return sorted(group.keys(), key=key)

    def _chunk_size(self, chrom_ref: Any) -> int | None:
        if isinstance(chrom_ref, zarr.Array):
            return int(chrom_ref.chunks[0]) if chrom_ref.chunks else None
        names = self._ordered_chunks(chrom_ref)
        return int(chrom_ref[names[0]].shape[0]) if names else None

    def _chrom_size(self, chrom_ref: Any) -> int:
        if isinstance(chrom_ref, zarr.Array):
            return int(chrom_ref.shape[0])
        return int(sum(chrom_ref[name].shape[0] for name in self._ordered_chunks(chrom_ref)))

    def _feature_size(self, chrom_ref: Any) -> int:
        if isinstance(chrom_ref, zarr.Group):
            names = self._ordered_chunks(chrom_ref)
            if not names:
                return 1
            shape = chrom_ref[names[0]].shape
        else:
            shape = chrom_ref.shape
        return int(shape[1]) if len(shape) > 1 else 1

    def get_track(
        self,
        chr_name: str,
        start: int,
        end: int,
        *,
        output_format: str = "raw_array",
        **_: Any,
    ) -> np.ndarray:
        """Return ``[start, end)`` as a NumPy array."""
        if output_format != "raw_array":
            raise ValueError("Standalone Motiverse supports output_format='raw_array'")
        if chr_name not in self.chrom_sizes:
            raise ValueError(f"Chromosome {chr_name!r} is not present")
        start = max(0, int(start))
        end = min(int(end), self.chrom_sizes[chr_name])
        if end <= start:
            shape = (0, self.feature_size) if self.feature_size > 1 else (0,)
            return np.empty(shape, dtype=self.dtype)
        chrom_ref = self.dataset["chrs"][chr_name]
        if isinstance(chrom_ref, zarr.Array):
            return np.asarray(chrom_ref[start:end])

        pieces: list[np.ndarray] = []
        offset = 0
        for name in self._ordered_chunks(chrom_ref):
            chunk = chrom_ref[name]
            chunk_end = offset + int(chunk.shape[0])
            if chunk_end > start and offset < end:
                local_start = max(0, start - offset)
                local_end = min(int(chunk.shape[0]), end - offset)
                pieces.append(np.asarray(chunk[local_start:local_end]))
            offset = chunk_end
            if offset >= end:
                break
        if pieces:
            return np.concatenate(pieces, axis=0)
        shape = (0, self.feature_size) if self.feature_size > 1 else (0,)
        return np.empty(shape, dtype=self.dtype)

    def load_to_memory_dense(self) -> SequenceDenseZarrIO:
        """Compatibility no-op; reads remain lazy to keep memory use bounded."""
        return self


class SignalDenseZarrIO(SequenceDenseZarrIO):
    """Read a single dense signal track from the same Zarr layout."""

    def __init__(self, path: str | Path, mode: str = "r", **kwargs: Any) -> None:
        super().__init__(path, mode=mode, dtype="float32", **kwargs)


class CelltypeDenseZarrIO:
    """Minimal reader for ``<celltype>/chrs/<chrom>`` signal stores."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        celltype_ids: list[str] | None = None,
        **_: Any,
    ) -> None:
        self.path = str(path)
        self.dataset = zarr.open_group(self.path, mode=mode)
        candidates = [key for key in self.dataset.keys() if "chrs" in self.dataset[key]]
        self.ids = (
            [key for key in celltype_ids if key in candidates]
            if celltype_ids is not None
            else candidates
        )
        if not self.ids:
            raise ValueError("No cell-type groups with a 'chrs' subgroup were found")
        self.n_celltypes = len(self.ids)

    def load_to_memory_dense(self) -> CelltypeDenseZarrIO:
        return self

    def get_track(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        query: Any = None,
        **_: Any,
    ) -> np.ndarray:
        selected = [query.key] if isinstance(query, KeyQuery) else self.ids
        tracks = [
            np.asarray(self.dataset[celltype]["chrs"][chrom][start:end]) for celltype in selected
        ]
        return tracks[0] if len(tracks) == 1 else np.stack(tracks, axis=0)


class KeyQuery:
    def __init__(self, key: str) -> None:
        self.key = key


class ConcatAggregator:
    pass


class SumAggregator:
    pass


class NoNormalizer:
    pass


def _open_text(path: str | Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else Path(path).open()


def read_narrowpeak(path: str | Path) -> pd.DataFrame:
    """Read BED/narrowPeak input and normalize it to 2 kb centered windows."""
    skiprows: list[int] = []
    column_count = 0
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "track", "browser")):
                skiprows.append(line_number)
                continue
            column_count = len(stripped.split("\t"))
            break
    if column_count < 3:
        raise ValueError(f"Expected at least three BED columns in {path}")
    names = NARROWPEAK_COLUMNS[:column_count]
    if column_count > len(names):
        names += [f"extra_{i}" for i in range(len(names), column_count)]
    with _open_text(path) as handle:
        frame = pd.read_csv(
            handle,
            sep="\t",
            header=None,
            names=names,
            skiprows=skiprows,
        )
    frame["length"] = frame["end"] - frame["start"]
    frame["center"] = (frame["start"] + frame["end"]) / 2
    frame["start"] = (frame["center"] - 1000).astype(int)
    frame["end"] = (frame["center"] + 1000).astype(int)
    return frame.sample(n=1_000_000) if len(frame) > 1_000_000 else frame


def extract_sequences_from_regions(
    seq_io: SequenceDenseZarrIO,
    regions_df: pd.DataFrame,
    extend_bp: int = 0,
    extend_right_only: bool = False,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Extract region sequences while preserving input row order."""
    extracted: list[tuple[Any, np.ndarray, dict[str, Any]]] = []
    for original_index, row in regions_df.iterrows():
        chrom = str(row["chrom"])
        if chrom not in seq_io.chrom_sizes:
            continue
        original_start = int(row["start"])
        original_end = int(row["end"])
        start = original_start if extend_right_only else max(0, original_start - extend_bp)
        end = min(seq_io.chrom_sizes[chrom], original_end + extend_bp)
        if end <= start:
            continue
        track = seq_io.get_track(
            chr_name=chrom,
            start=start,
            end=end,
            output_format="raw_array",
        ).astype(np.float32)
        extracted.append(
            (
                original_index,
                track,
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "original_start": original_start,
                    "original_end": original_end,
                    "length": end - start,
                    "original_idx": original_index,
                },
            )
        )
    if not extracted:
        raise ValueError("No sequences could be extracted from the regions")
    extracted.sort(key=lambda item: item[0])
    return [item[1] for item in extracted], [item[2] for item in extracted]


def reverse_complement_batch(batch_tensor):
    """Reverse-complement a ``(batch, length, 4)`` DNA tensor."""
    import torch

    return torch.flip(torch.flip(batch_tensor, dims=[1]), dims=[2])
