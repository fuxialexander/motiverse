"""Rank motif profiles by cross-subset amplitude and shape variability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from .motif_source import load_hocomoco_motifs


def _subset_ids(root, regions_tsv: Path) -> list[str]:
    metadata = root.get("metadata")
    if metadata is not None and "subset_ids" in metadata:
        return [x.decode() if isinstance(x, bytes) else str(x) for x in metadata["subset_ids"][:]]
    return sorted(pd.read_csv(regions_tsv, sep="\t")["subset_id"].astype(str).unique())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values-zarr", required=True, type=Path)
    parser.add_argument("--regions-tsv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--motif-chunk-size", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--min-regions", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=50)
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer.")
    root = zarr.open_group(str(args.values_zarr), mode="r")
    values = root["values"]
    subset_ids = _subset_ids(root, args.regions_tsv)
    counts = pd.read_csv(args.regions_tsv, sep="\t").groupby("subset_id").size().reindex(subset_ids)
    if counts.isna().any() or len(subset_ids) != values.shape[0]:
        raise ValueError("Subset IDs in values Zarr and regions TSV do not agree.")
    motifs, _ = load_hocomoco_motifs(require_aligned_motifs=True)
    if len(motifs.motif_names) != values.shape[1]:
        raise ValueError("Aligned motif names do not match the values motif axis.")
    keep = counts.to_numpy() >= args.min_regions
    pad = args.smooth_window // 2
    rows = []
    for start in range(0, values.shape[1], args.motif_chunk_size):
        profiles = (
            np.asarray(values[:, start : start + args.motif_chunk_size, :], dtype=np.float64)
            / counts.to_numpy()[:, None, None]
        )
        baseline = np.median(profiles, axis=-1, keepdims=True)
        signal = profiles - baseline
        if pad:
            signal = np.apply_along_axis(
                lambda x: np.convolve(
                    np.pad(x, pad, mode="edge"),
                    np.ones(args.smooth_window) / args.smooth_window,
                    mode="valid",
                ),
                -1,
                signal,
            )
        amplitude = np.max(signal, axis=-1)
        norm = np.linalg.norm(signal, axis=-1, keepdims=True)
        shape = np.divide(signal, norm, out=np.zeros_like(signal), where=norm > 0)
        for local in range(signal.shape[1]):
            active = keep & (amplitude[:, local] > 0)
            shape_var = (
                float(np.mean(np.var(shape[active, local], axis=0)))
                if active.sum() >= 2
                else np.nan
            )
            amp_var = (
                float(np.std(np.log1p(amplitude[active, local]))) if active.sum() >= 2 else 0.0
            )
            rows.append(
                {
                    "motif_index": start + local,
                    "motif_name": motifs.motif_names[start + local],
                    "active_subset_count": int(active.sum()),
                    "shape_variance": shape_var,
                    "amplitude_variability": amp_var,
                }
            )
    scores = pd.DataFrame(rows)
    for column in ("shape_variance", "amplitude_variability"):
        scores[f"{column}_rank"] = scores[column].rank(
            ascending=False, method="min", na_option="bottom"
        )
    scores["cross_context_rank_score"] = (
        scores["shape_variance_rank"] + scores["amplitude_variability_rank"]
    )
    scores = scores.sort_values("cross_context_rank_score")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "motif_context_ranking.tsv", sep="\t", index=False)
    scores.head(args.top_n).to_csv(
        args.output_dir / "top_context_variable_motifs.tsv", sep="\t", index=False
    )
    summary = {
        "schema_version": "motif_context_ranking_v1",
        "values_zarr": str(args.values_zarr),
        "regions_tsv": str(args.regions_tsv),
        "n_subsets": len(subset_ids),
        "n_motifs": values.shape[1],
        "min_regions": args.min_regions,
        "ranking_tsv": str(args.output_dir / "motif_context_ranking.tsv"),
    }
    (args.output_dir / "motif_context_ranking_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
