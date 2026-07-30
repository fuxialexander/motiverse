"""Build union-locus memberships for multi-peak-set full-curve reuse."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_peak_set_spec(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = str(spec).partition("=")
    if not separator or not name or not raw_path:
        raise ValueError("--peak-set must be NAME=PATH")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Peak-set file does not exist: {path}")
    return name, path


def _centered_windows(path: Path, *, window_bp: int) -> list[tuple[str, int, int]]:
    raw = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
    )
    raw.columns = ["chrom", "start", "end"]
    raw["start"] = pd.to_numeric(raw["start"], errors="coerce")
    raw["end"] = pd.to_numeric(raw["end"], errors="coerce")
    raw = raw[raw["end"] > raw["start"]].copy()
    centers = ((raw["start"] + raw["end"]) // 2).astype("int64")
    starts = (centers - int(window_bp) // 2).clip(lower=0)
    ends = starts + int(window_bp)
    return sorted(set(zip(raw["chrom"].astype(str), starts.astype(int), ends.astype(int))))


def build_peak_set_union(specs: list[str], *, window_bp: int) -> tuple[list[str], pd.DataFrame]:
    """Construct a merged, overlap-connected locus universe with memberships.

    Each input peak is first normalized to a fixed centered window. Any windows
    sharing at least one base are merged into one union locus, and the locus is
    assigned the set of peak sets with at least one contributing window. This
    makes all downstream contrasts operate on one canonical locus universe.
    """
    if window_bp <= 0:
        raise ValueError("--peak-set-window-bp must be positive.")
    parsed = [parse_peak_set_spec(spec) for spec in specs]
    names = [name for name, _ in parsed]
    if len(set(names)) != len(names):
        raise ValueError("Each --peak-set name must be unique.")
    by_chrom: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for name, path in parsed:
        for chrom, start, end in _centered_windows(path, window_bp=window_bp):
            by_chrom[chrom].append((start, end, name))

    loci: list[dict[str, object]] = []
    for chrom in sorted(by_chrom):
        intervals = sorted(by_chrom[chrom])
        if not intervals:
            continue
        start, end, name = intervals[0]
        members = {name}
        for next_start, next_end, next_name in intervals[1:]:
            if next_start < end:  # One-base overlap joins the same union locus.
                end = max(end, next_end)
                members.add(next_name)
                continue
            loci.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "peak_set_memberships": ",".join(sorted(members)),
                }
            )
            start, end, members = next_start, next_end, {next_name}
        loci.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "peak_set_memberships": ",".join(sorted(members)),
            }
        )
    return names, pd.DataFrame(loci, columns=["chrom", "start", "end", "peak_set_memberships"])


def build_peak_set_memberships(
    specs: list[str],
    *,
    window_bp: int,
    pairwise_set_diff: bool,
    min_peaks: int,
    return_union_loci: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build output subset memberships from a shared union-locus universe."""
    if min_peaks < 0:
        raise ValueError("--min-peaks must be non-negative.")
    if len(specs) < (2 if pairwise_set_diff else 1):
        raise ValueError("Pairwise set difference requires at least two --peak-set inputs.")
    names, union_loci = build_peak_set_union(specs, window_bp=window_bp)
    membership_sets = union_loci["peak_set_memberships"].map(
        lambda value: frozenset(str(value).split(","))
    )
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    if pairwise_set_diff:
        for source in names:
            for reference in names:
                if source == reference:
                    continue
                selected = union_loci[
                    membership_sets.map(
                        lambda members: source in members and reference not in members
                    )
                ]
                subset_id = f"{source}__minus__{reference}"
                keep = len(selected) > min_peaks
                inventory.append(
                    {
                        "subset_id": subset_id,
                        "source_peak_set": source,
                        "reference_peak_set": reference,
                        "n_windows": len(selected),
                        "n_union_loci": len(selected),
                        "retained": keep,
                    }
                )
                if keep:
                    rows.extend(
                        {"subset_id": subset_id, **record}
                        for record in selected[["chrom", "start", "end"]].to_dict("records")
                    )
    else:
        for name in names:
            selected = union_loci[membership_sets.map(lambda members: name in members)]
            keep = len(selected) > min_peaks
            inventory.append(
                {
                    "subset_id": name,
                    "source_peak_set": name,
                    "reference_peak_set": None,
                    "n_windows": len(selected),
                    "n_union_loci": len(selected),
                    "retained": keep,
                }
            )
            if keep:
                rows.extend(
                    {"subset_id": name, **record}
                    for record in selected[["chrom", "start", "end"]].to_dict("records")
                )
    memberships = pd.DataFrame(rows, columns=["subset_id", "chrom", "start", "end"])
    inventory_frame = pd.DataFrame(inventory)
    if return_union_loci:
        return memberships, inventory_frame, union_loci
    return memberships, inventory_frame
