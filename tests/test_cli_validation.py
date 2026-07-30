from __future__ import annotations

import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "motiverse", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_hit_cache_rejects_sequence_accumulation_without_touching_genome():
    completed = _run_cli(
        "--genome-zarr",
        "missing.zarr",
        "--hit-cache",
        "cache.zarr",
        "--sequence-accumulation",
        "--narrowpeak",
        "peaks.bed",
    )

    assert completed.returncode == 2
    assert "--hit-cache does not support --sequence-accumulation" in completed.stderr


def test_hit_cache_requires_region_source_unless_building_cache():
    completed = _run_cli(
        "--genome-zarr",
        "missing.zarr",
        "--hit-cache",
        "cache.zarr",
    )

    assert completed.returncode == 2
    assert "--hit-cache requires --narrowpeak, --tiling, or --remap" in completed.stderr


def test_one_to_all_rejects_sequence_accumulation_without_touching_genome():
    completed = _run_cli(
        "--genome-zarr",
        "missing.zarr",
        "--one-to-all",
        "P53.H13CORE.0.P.B",
        "--sequence-accumulation",
    )

    assert completed.returncode == 2
    assert "--one-to-all cannot be used with --sequence-accumulation" in completed.stderr


def test_remap_requires_one_to_all_without_touching_genome():
    completed = _run_cli(
        "--genome-zarr",
        "missing.zarr",
        "--remap",
    )

    assert completed.returncode == 2
    assert "--remap requires --one-to-all" in completed.stderr


def test_build_hit_cache_rejects_remap_without_touching_genome():
    completed = _run_cli(
        "--genome-zarr",
        "missing.zarr",
        "--build-hit-cache",
        "--remap",
        "--one-to-all",
        "P53.H13CORE.0.P.B",
    )

    assert completed.returncode == 2
    assert "--build-hit-cache does not support --remap" in completed.stderr
