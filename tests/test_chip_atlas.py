from __future__ import annotations

from argparse import Namespace

import pandas as pd

from motiverse.chip_atlas import run


def test_chip_atlas_contexts_groups_and_filters_metadata(tmp_path):
    bed = tmp_path / "atlas.bed"
    bed.write_text(
        "chr1\t100\t200\tname=x%40%20A2780);treatment=Nutlin\n"
        "chr1\t300\t400\tname=x%40%20SJSA-1);treatment=Nutlin\n"
        "chr1\t500\t600\tname=x%40%20A2780);treatment=DMSO\n"
    )
    summary = run(
        Namespace(
            bed=bed,
            output_dir=tmp_path / "out",
            group_by="celltype",
            include_regex="nutlin",
            window_bp=200,
            min_peaks=0,
            canonical_chromosomes=True,
        )
    )
    regions = pd.read_csv(summary["regions_tsv"], sep="\t")
    assert regions["subset_id"].tolist() == ["A2780", "SJSA-1"]
    assert regions[["start", "end"]].values.tolist() == [[50, 250], [250, 450]]
