from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from gcell.dna.hocomoco import HocomocoIO

from motiverse.motif_source import (
    load_hocomoco_motifs,
    motif_names_checksum,
    select_motif_indices,
)


def test_select_motif_indices_mixed_syntax():
    names = [
        "A.H13CORE.0.P.A",
        "A.H13CORE.0.P.A_RC",
        "B.H13CORE.0.P.A",
        "CTCF.H13CORE.0.P.A",
    ]

    indices = select_motif_indices(names, "0-1,CTCF,B.H13CORE.0.P.A")

    assert indices == [0, 1, 3, 2]


def test_motif_names_checksum_is_order_sensitive():
    assert motif_names_checksum(["A", "B"]) != motif_names_checksum(["B", "A"])
    assert motif_names_checksum(["A", "B"]) == motif_names_checksum(["A", "B"])


def test_default_load_requests_aligned_pt_and_subsets_after_load(monkeypatch):
    calls = []

    class FakeHocomocoIO:
        def __init__(self, *, aligned_motif_path=None, use_aligned=True):
            calls.append(
                {
                    "aligned_motif_path": aligned_motif_path,
                    "use_aligned": use_aligned,
                }
            )
            self.aligned_motif_path = aligned_motif_path or "/tmp/motifs_with_rc_aligned.pt"
            self._loaded_with_aligned = True
            self._motif_names = ["A.H13CORE.0.P.A", "B.H13CORE.0.P.A", "C.H13CORE.0.P.A"]
            self._motif_kernels = np.arange(3 * 31 * 4, dtype=np.float32).reshape(3, 31, 4)
            self._similarity_matrix = "full"
            self._significance_thresholds = {"p0.0001": np.asarray([1.0, 2.0, 3.0])}
            self._pvalue_mappings = None

        @property
        def motif_names(self):
            return self._motif_names

        @property
        def motif_kernels(self):
            return self._motif_kernels

    monkeypatch.setattr(
        "motiverse.motif_source.HocomocoIO",
        FakeHocomocoIO,
    )

    db, metadata = load_hocomoco_motifs(motif_selection="0,2")

    assert calls == [{"aligned_motif_path": None, "use_aligned": True}]
    assert metadata.loaded_with_aligned is True
    assert metadata.motif_source == "aligned_pt"
    assert metadata.aligned_motif_path == "/tmp/motifs_with_rc_aligned.pt"
    assert metadata.motif_count == 2
    assert metadata.motif_kernel_shape == (2, 31, 4)
    assert db.motif_names == ["A.H13CORE.0.P.A", "C.H13CORE.0.P.A"]
    np.testing.assert_allclose(db._significance_thresholds["p0.0001"], [1.0, 3.0])


def test_default_load_requires_aligned_pt_and_can_subset():
    probe = HocomocoIO()
    aligned_path = Path(probe.aligned_motif_path)
    if not aligned_path.exists():
        pytest.skip(f"Aligned motif tensor not available: {aligned_path}")

    db, metadata = load_hocomoco_motifs(motif_selection="0-1")

    assert metadata.loaded_with_aligned is True
    assert metadata.motif_source == "aligned_pt"
    assert metadata.aligned_motif_path == str(aligned_path)
    assert metadata.motif_count == 2
    assert len(metadata.motif_kernel_checksum) == 64
    assert tuple(db.motif_kernels.shape[:2]) == (2, 31)
    assert getattr(db, "_loaded_with_aligned") is True
