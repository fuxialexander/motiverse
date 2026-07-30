"""Motif loading and provenance helpers for genome motif analysis."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from gcell.dna.hocomoco import HocomocoIO
except ImportError:
    HocomocoIO = None


@dataclass(frozen=True)
class MotifSourceMetadata:
    """Serializable metadata that identifies the motif basis used for a run."""

    motif_source: str
    aligned_motif_path: str | None
    loaded_with_aligned: bool
    use_aligned: bool
    require_aligned: bool
    motif_count: int
    motif_kernel_shape: tuple[int, ...]
    motif_kernel_checksum: str
    motif_names_checksum: str
    motif_selection: str | None
    cache_schema_version: str
    threshold_mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def motif_names_checksum(motif_names: list[str]) -> str:
    """Return a stable checksum for an ordered motif-name list."""
    payload = "\n".join(motif_names).encode()
    return hashlib.sha256(payload).hexdigest()


def kernel_checksum(motif_kernels: np.ndarray) -> str:
    """Return a stable checksum for motif kernels."""
    kernels = np.asarray(motif_kernels)
    return hashlib.sha256(kernels.tobytes()).hexdigest()


def select_motif_indices(motif_names: list[str], motif_selection: str | None) -> list[int]:
    """Resolve Motiverse motif-selection syntax against a loaded motif list.

    Supports comma-separated exact names, prefix matches, integer indices, and
    inclusive integer ranges such as ``0-63``. Duplicates are removed while
    preserving selection order.
    """
    if not motif_selection:
        return list(range(len(motif_names)))

    selected: list[int] = []
    seen: set[int] = set()

    def add_index(idx: int) -> None:
        if idx < 0 or idx >= len(motif_names):
            raise ValueError(f"Motif index {idx} out of range [0, {len(motif_names) - 1}]")
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)

    for raw_token in motif_selection.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if "-" in token and all(part.strip().isdigit() for part in token.split("-", 1)):
            start_s, end_s = token.split("-", 1)
            start_i, end_i = int(start_s), int(end_s)
            if start_i > end_i:
                raise ValueError(f"Invalid descending motif range: {token}")
            for idx in range(start_i, end_i + 1):
                add_index(idx)
            continue

        if token.isdigit():
            add_index(int(token))
            continue

        exact_matches = [i for i, name in enumerate(motif_names) if name == token]
        if exact_matches:
            for idx in exact_matches:
                add_index(idx)
            continue

        prefix_matches = [
            i
            for i, name in enumerate(motif_names)
            if name.startswith(token + ".") or name.startswith(token + "_")
        ]
        if prefix_matches:
            for idx in prefix_matches:
                add_index(idx)
            continue

        partial_matches = [i for i, name in enumerate(motif_names) if token.upper() in name.upper()]
        if partial_matches:
            for idx in partial_matches:
                add_index(idx)
            continue

        raise ValueError(f"Motif selection token '{token}' matched no motifs")

    if not selected:
        raise ValueError(f"Motif selection '{motif_selection}' matched no motifs")
    return selected


def _subset_threshold_dict(
    thresholds: dict[str, np.ndarray] | None, indices: list[int]
) -> dict[str, np.ndarray] | None:
    if not thresholds:
        return thresholds
    subset: dict[str, np.ndarray] = {}
    for key, value in thresholds.items():
        arr = np.asarray(value)
        if arr.shape[0] >= max(indices) + 1:
            subset[key] = arr[indices].astype(np.float32)
        else:
            subset[key] = arr.astype(np.float32)
    return subset


def load_hocomoco_motifs(
    *,
    motif_selection: str | None = None,
    aligned_motif_path: str | None = None,
    use_aligned_motifs: bool = True,
    require_aligned_motifs: bool = True,
    threshold_mode: str = "pvalue_mapping",
) -> tuple[Any, MotifSourceMetadata]:
    """Load HOCOMOCO motifs with explicit aligned-PT provenance.

    ``gcell.dna.hocomoco.HocomocoIO`` already knows how to load the aligned
    tensor, but filtering aligned motifs through its public ``filter_motifs``
    currently falls back to unsupported behavior. This helper forces motif data
    to load first, then subsets the already-loaded aligned tensors in place.
    """
    if HocomocoIO is None:
        raise ImportError(
            "HOCOMOCO p-value loading requires gcell. Install it with `pip install gcell`."
        )

    hocomoco_db = HocomocoIO(
        aligned_motif_path=aligned_motif_path,
        use_aligned=use_aligned_motifs,
    )

    # Force lazy load so provenance is known before filtering or GPU transfer.
    motif_names = list(hocomoco_db.motif_names)
    motif_kernels = np.asarray(hocomoco_db.motif_kernels, dtype=np.float32)
    loaded_with_aligned = bool(getattr(hocomoco_db, "_loaded_with_aligned", False))

    if require_aligned_motifs and not loaded_with_aligned:
        raise RuntimeError(
            "Aligned motifs were required but HocomocoIO did not load the aligned "
            f"PT tensor. Requested path: {hocomoco_db.aligned_motif_path}"
        )

    if motif_selection:
        indices = select_motif_indices(motif_names, motif_selection)
        motif_names = [motif_names[i] for i in indices]
        motif_kernels = motif_kernels[indices].astype(np.float32)
        hocomoco_db._motif_names = motif_names
        hocomoco_db._motif_kernels = motif_kernels
        hocomoco_db._similarity_matrix = None
        hocomoco_db._significance_thresholds = _subset_threshold_dict(
            getattr(hocomoco_db, "_significance_thresholds", None), indices
        )
        # P-value mappings are keyed by motif name and remain valid, but the
        # cached dict may have been built for the full set. Keep only selected
        # names if it was already materialized.
        if getattr(hocomoco_db, "_pvalue_mappings", None) is not None:
            hocomoco_db._pvalue_mappings = {
                name: value
                for name, value in hocomoco_db._pvalue_mappings.items()
                if name in set(motif_names)
            }
        logger.info(
            "Selected %d motifs from aligned motif tensor using '%s'",
            len(motif_names),
            motif_selection,
        )

    metadata = MotifSourceMetadata(
        motif_source="aligned_pt" if loaded_with_aligned else "pwm",
        aligned_motif_path=str(hocomoco_db.aligned_motif_path)
        if hocomoco_db.aligned_motif_path
        else None,
        loaded_with_aligned=loaded_with_aligned,
        use_aligned=use_aligned_motifs,
        require_aligned=require_aligned_motifs,
        motif_count=len(motif_names),
        motif_kernel_shape=tuple(np.asarray(hocomoco_db.motif_kernels).shape),
        motif_kernel_checksum=kernel_checksum(hocomoco_db.motif_kernels),
        motif_names_checksum=motif_names_checksum(motif_names),
        motif_selection=motif_selection,
        cache_schema_version="positional_hit_cache_v1",
        threshold_mode=threshold_mode,
    )
    return hocomoco_db, metadata


def write_metadata_json(path: str | Path, metadata: dict) -> None:
    """Write JSON metadata with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
