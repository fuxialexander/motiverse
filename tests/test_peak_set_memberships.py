from __future__ import annotations

from pathlib import Path

import pytest

from motiverse.peak_set_memberships import (
    build_peak_set_memberships,
    parse_peak_set_spec,
)


def _write_bed(path: Path, rows: list[tuple[str, int, int]]) -> None:
    path.write_text("".join(f"{chrom}\t{start}\t{end}\n" for chrom, start, end in rows))


def test_multiple_peak_sets_create_one_subset_per_input(tmp_path: Path) -> None:
    first = tmp_path / "first.bed"
    second = tmp_path / "second.bed"
    _write_bed(first, [("chr1", 100, 200), ("chr1", 300, 400)])
    _write_bed(second, [("chr1", 500, 600)])
    regions, inventory = build_peak_set_memberships(
        [f"first={first}", f"second={second}"],
        window_bp=200,
        pairwise_set_diff=False,
        min_peaks=0,
    )
    assert regions.groupby("subset_id").size().to_dict() == {"first": 2, "second": 1}
    assert inventory["retained"].tolist() == [True, True]


def test_pairwise_set_difference_uses_one_base_overlap_rule(tmp_path: Path) -> None:
    first = tmp_path / "first.bed"
    second = tmp_path / "second.bed"
    _write_bed(first, [("chr1", 100, 200), ("chr1", 500, 600)])
    _write_bed(second, [("chr1", 150, 250), ("chr1", 900, 1000)])
    regions, inventory = build_peak_set_memberships(
        [f"first={first}", f"second={second}"],
        window_bp=100,
        pairwise_set_diff=True,
        min_peaks=0,
    )
    assert inventory.set_index("subset_id")["n_windows"].to_dict() == {
        "first__minus__second": 1,
        "second__minus__first": 1,
    }
    assert set(regions["subset_id"]) == {"first__minus__second", "second__minus__first"}


def test_pairwise_set_difference_uses_shared_union_locus_memberships(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.bed"
    second = tmp_path / "second.bed"
    third = tmp_path / "third.bed"
    _write_bed(first, [("chr1", 100, 200)])
    _write_bed(second, [("chr1", 150, 250)])
    _write_bed(third, [("chr1", 200, 300)])

    regions, inventory, union_loci = build_peak_set_memberships(
        [f"first={first}", f"second={second}", f"third={third}"],
        window_bp=100,
        pairwise_set_diff=True,
        min_peaks=0,
        return_union_loci=True,
    )

    # The three normalized windows are one overlap-connected union locus.
    assert union_loci.to_dict("records") == [
        {
            "chrom": "chr1",
            "start": 100,
            "end": 300,
            "peak_set_memberships": "first,second,third",
        }
    ]
    assert regions.empty
    assert inventory["n_union_loci"].eq(0).all()


def test_peak_set_window_and_minimum_are_validated(tmp_path: Path) -> None:
    first = tmp_path / "first.bed"
    _write_bed(first, [("chr1", 100, 200)])
    with pytest.raises(ValueError, match="window-bp"):
        build_peak_set_memberships(
            [f"first={first}"],
            window_bp=0,
            pairwise_set_diff=False,
            min_peaks=0,
        )
    with pytest.raises(ValueError, match="min-peaks"):
        build_peak_set_memberships(
            [f"first={first}"],
            window_bp=100,
            pairwise_set_diff=False,
            min_peaks=-1,
        )


def test_peak_set_spec_requires_name_and_existing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_peak_set_spec("missing-separator")
    with pytest.raises(FileNotFoundError):
        parse_peak_set_spec(f"missing={tmp_path / 'missing.bed'}")
