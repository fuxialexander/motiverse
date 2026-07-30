from __future__ import annotations

import pandas as pd
import pytest
import torch

from motiverse import main as genome_motif_main
from motiverse.main import (
    _resolve_query_motif_index,
    analyze_genome_sequences,
    motif_info_payload,
)


class FakeHocomoco:
    motif_names = ["A.H13CORE.0.P.A", "P53.H13CORE.0.P.B"]
    motif_kernels = torch.zeros((2, 31, 4), dtype=torch.float32)

    def resolve_motif_index(self, name: str) -> int:
        if name == "P53.H13CORE.0.P.B":
            return 1
        raise ValueError(f"Motif '{name}' not found")

    def prepare_for_gpu(self, *, device, dtype, p_value):
        return self.motif_kernels.to(device=device, dtype=dtype), torch.ones(
            len(self.motif_names), device=device, dtype=dtype
        )

    def get_score_threshold_for_pvalue(self, p_value):
        return [0.1, 0.2]


class FakeMotifMetadata:
    motif_source = "aligned_pt"
    aligned_motif_path = "/home/xf2217/.gcell_data/annotations/hocomoco/motifs_with_rc_aligned.pt"

    def to_dict(self):
        return {
            "motif_source": self.motif_source,
            "aligned_motif_path": self.aligned_motif_path,
            "loaded_with_aligned": True,
            "use_aligned": True,
            "require_aligned": True,
            "motif_count": 2,
            "motif_kernel_shape": (2, 31, 4),
            "motif_kernel_checksum": "kernel123",
            "motif_names_checksum": "names123",
            "motif_selection": None,
            "cache_schema_version": "positional_hit_cache_v1",
            "threshold_mode": "pvalue_mapping",
        }


def test_resolve_query_motif_accepts_numeric_index():
    names = ["A.H13CORE.0.P.A", "P53.H13CORE.0.P.B"]

    index, name = _resolve_query_motif_index(FakeHocomoco(), names, "1")

    assert index == 1
    assert name == "P53.H13CORE.0.P.B"


def test_motif_info_payload_records_aligned_pt_and_resolved_target(monkeypatch):
    def fake_load_hocomoco_motifs(**kwargs):
        assert kwargs["use_aligned_motifs"] is True
        assert kwargs["require_aligned_motifs"] is True
        assert kwargs["threshold_mode"] == "pvalue_mapping"
        return FakeHocomoco(), FakeMotifMetadata()

    monkeypatch.setattr(
        genome_motif_main,
        "load_hocomoco_motifs",
        fake_load_hocomoco_motifs,
    )

    payload = motif_info_payload(motif_query="P53.H13CORE.0.P.B")

    assert payload["schema_version"] == "genome_motif_analysis_motif_info_v1"
    assert payload["motif_source"] == "aligned_pt"
    assert payload["loaded_with_aligned"] is True
    assert payload["aligned_motif_path"].endswith("motifs_with_rc_aligned.pt")
    assert payload["motif_count"] == 2
    assert payload["resolved_query_motif_index"] == 1
    assert payload["resolved_query_motif_name"] == "P53.H13CORE.0.P.B"


def test_build_hit_cache_with_narrowpeak_uses_module_reader(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeSequenceDenseZarrIO:
        chroms = ["chr1"]
        chrom_sizes = {"chr1": 1000}

        def __init__(self, path, mode="r"):
            captured["genome_zarr_path"] = path
            captured["genome_mode"] = mode

    def fake_load_hocomoco_motifs(**kwargs):
        assert kwargs["use_aligned_motifs"] is True
        assert kwargs["require_aligned_motifs"] is True
        return FakeHocomoco(), FakeMotifMetadata()

    def fake_read_narrowpeak(path):
        captured["narrowpeak_path"] = path
        return pd.DataFrame(
            {
                "chrom": ["chr1", "chrUn"],
                "start": [100, 10],
                "end": [120, 20],
            }
        )

    def fake_build_positional_motif_hit_cache(**kwargs):
        captured["cache_kwargs"] = kwargs

    monkeypatch.setattr(genome_motif_main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        genome_motif_main,
        "load_hocomoco_motifs",
        fake_load_hocomoco_motifs,
    )
    monkeypatch.setattr(
        genome_motif_main,
        "SequenceDenseZarrIO",
        FakeSequenceDenseZarrIO,
    )
    monkeypatch.setattr(genome_motif_main, "read_narrowpeak", fake_read_narrowpeak)
    monkeypatch.setattr(
        genome_motif_main,
        "build_positional_motif_hit_cache",
        fake_build_positional_motif_hit_cache,
    )

    analyze_genome_sequences(
        genome_zarr_path="toy_genome.zarr",
        target_chromosomes=["chr1"],
        output_directory=str(tmp_path / "out"),
        narrowpeak_file_path="toy_peaks.bed",
        region_extension_bp=10,
        use_mixed_precision=False,
        build_hit_cache=True,
        cache_output_path=str(tmp_path / "cache.zarr"),
    )

    assert captured["narrowpeak_path"] == "toy_peaks.bed"
    cache_kwargs = captured["cache_kwargs"]
    assert cache_kwargs["output_path"] == str(tmp_path / "cache.zarr")
    assert cache_kwargs["target_chromosomes"] == ["chr1"]
    assert cache_kwargs["motif_metadata"]["motif_source"] == "aligned_pt"
    assert cache_kwargs["motif_metadata"]["loaded_with_aligned"] is True
    intervals = cache_kwargs["region_intervals"]
    assert intervals[["chrom", "start", "end"]].to_dict("records") == [
        {"chrom": "chr1", "start": 90, "end": 130}
    ]


def test_resolve_query_motif_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        _resolve_query_motif_index(FakeHocomoco(), ["A.H13CORE.0.P.A"], "259")


def test_resolve_query_motif_falls_back_to_name_resolution():
    names = ["A.H13CORE.0.P.A", "P53.H13CORE.0.P.B"]

    index, name = _resolve_query_motif_index(
        FakeHocomoco(),
        names,
        "P53.H13CORE.0.P.B",
    )

    assert index == 1
    assert name == "P53.H13CORE.0.P.B"
