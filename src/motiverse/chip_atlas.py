"""Extract reproducible subset memberships from a ChIP-Atlas BED export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


def parse_attributes(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in unquote(str(value)).replace("<br>", ";").split(";"):
        if "=" in part:
            key, raw = part.split("=", 1)
            fields[key.strip().lower()] = raw.strip()
    return fields


def context_label(attrs: dict[str, str], grouping: str) -> str | None:
    if grouping == "srx":
        for value in [attrs.get("id", ""), *attrs.values()]:
            match = re.search(r"\bSRX\d+\b", value)
            if match:
                return match.group(0)
        return None
    if grouping == "celltype":
        match = re.search(r"@\s*([^)]+)\)", attrs.get("name", ""))
        if match:
            return match.group(1).strip()
        for key in ("cell line", "cell type", "source_name"):
            if attrs.get(key):
                return attrs[key]
        return None
    if grouping.startswith("attribute:"):
        return attrs.get(grouping.split(":", 1)[1].lower())
    raise ValueError("--group-by must be srx, celltype, or attribute:ATTRIBUTE_NAME")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-by", default="celltype")
    parser.add_argument(
        "--include-regex", help="Case-insensitive regex applied to decoded metadata."
    )
    parser.add_argument("--window-bp", type=int, default=2000)
    parser.add_argument("--min-peaks", type=int, default=0)
    parser.add_argument("--canonical-chromosomes", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.window_bp <= 0 or args.min_peaks < 0:
        raise ValueError("--window-bp must be positive and --min-peaks non-negative.")
    raw = pd.read_csv(
        args.bed,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "attrs"],
    )
    raw["start"] = pd.to_numeric(raw["start"], errors="coerce")
    raw["end"] = pd.to_numeric(raw["end"], errors="coerce")
    raw = raw[raw["end"] > raw["start"]].copy()
    raw["attrs"] = raw["attrs"].astype(str).map(unquote)
    if args.include_regex:
        raw = raw[
            raw["attrs"].str.contains(args.include_regex, case=False, regex=True, na=False)
        ].copy()
    if args.canonical_chromosomes:
        allowed = {*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY", "chrM"}
        raw = raw[raw["chrom"].astype(str).isin(allowed)].copy()
    raw["subset_id"] = (
        raw["attrs"].map(parse_attributes).map(lambda x: context_label(x, args.group_by))
    )
    raw = raw[raw["subset_id"].notna()].copy()
    counts = raw.groupby("subset_id").size()
    raw = raw[raw["subset_id"].isin(counts[counts > args.min_peaks].index)].copy()
    centers = ((raw["start"] + raw["end"]) // 2).astype("int64")
    raw["start"] = (centers - args.window_bp // 2).clip(lower=0)
    raw["end"] = raw["start"] + args.window_bp
    regions = (
        raw[["subset_id", "chrom", "start", "end"]]
        .astype({"start": "int64", "end": "int64"})
        .drop_duplicates()
        .sort_values(["subset_id", "chrom", "start", "end"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regions_path = args.output_dir / "chip_atlas_context_memberships.tsv"
    inventory_path = args.output_dir / "chip_atlas_context_inventory.tsv"
    regions.to_csv(regions_path, sep="\t", index=False)
    inventory = regions.groupby("subset_id").size().rename("n_windows").reset_index()
    inventory.to_csv(inventory_path, sep="\t", index=False)
    summary = {
        "schema_version": "chip_atlas_context_memberships_v1",
        "group_by": args.group_by,
        "include_regex": args.include_regex,
        "window_bp": args.window_bp,
        "n_subsets": int(len(inventory)),
        "n_memberships": int(len(regions)),
        "regions_tsv": str(regions_path),
        "inventory_tsv": str(inventory_path),
        "regions_sha256": hashlib.sha256(
            regions.to_csv(sep="\t", index=False).encode()
        ).hexdigest(),
    }
    (args.output_dir / "chip_atlas_context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
