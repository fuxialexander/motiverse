from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_package_cli_routes_full_curve_reuse(monkeypatch, capsys):
    main_module = importlib.import_module("motiverse.main")
    reuse_module = importlib.import_module("motiverse.full_curve_reuse")
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return {"validation_status": "PASS", "mode": "full_curve_reuse"}

    monkeypatch.setattr(reuse_module, "run_full_curve_one_to_all_reuse", fake_run)

    assert main_module._run_backend_subcommand_if_requested(
        [
            "full-curve-reuse",
            "--regions-tsv",
            "regions.tsv",
            "--output-dir",
            "out",
            "--genome-zarr",
            "genome.zarr",
            "--aligned-motif-path",
            "motifs.pt",
        ]
    )
    assert captured["args"].regions_tsv == "regions.tsv"
    assert Path(captured["args"].output_dir) == Path("out")
    assert json.loads(capsys.readouterr().out)["validation_status"] == "PASS"


def test_pairwise_peak_set_alias_enables_set_difference(monkeypatch, capsys):
    main_module = importlib.import_module("motiverse.main")
    reuse_module = importlib.import_module("motiverse.full_curve_reuse")
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return {"validation_status": "PASS", "mode": "full_curve_reuse"}

    monkeypatch.setattr(reuse_module, "run_full_curve_one_to_all_reuse", fake_run)

    assert main_module._run_backend_subcommand_if_requested(
        [
            "pairwise-peak-set-reuse",
            "--peak-set",
            "SJSA-1=SJSA-1.bed",
            "--peak-set",
            "A2780=A2780.bed",
            "--output-dir",
            "out",
            "--genome-zarr",
            "genome.zarr",
            "--aligned-motif-path",
            "motifs.pt",
        ]
    )
    assert captured["args"].pairwise_set_diff is True
    assert json.loads(capsys.readouterr().out)["validation_status"] == "PASS"


def test_human_facing_compare_alias_enables_set_difference(monkeypatch, capsys):
    main_module = importlib.import_module("motiverse.main")
    reuse_module = importlib.import_module("motiverse.full_curve_reuse")
    captured = {}

    def fake_run(args):
        captured["args"] = args
        return {"validation_status": "PASS"}

    monkeypatch.setattr(reuse_module, "run_full_curve_one_to_all_reuse", fake_run)
    assert main_module._run_backend_subcommand_if_requested(
        [
            "compare-peak-sets",
            "--peak-set",
            "A=a.bed",
            "--peak-set",
            "B=b.bed",
            "--output-dir",
            "out",
            "--genome-zarr",
            "genome.zarr",
            "--aligned-motif-path",
            "motifs.pt",
        ]
    )
    assert captured["args"].pairwise_set_diff is True
    capsys.readouterr()
