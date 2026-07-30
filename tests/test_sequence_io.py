from __future__ import annotations

import numpy as np
import zarr

from motiverse.sequence_io import SequenceDenseZarrIO


def _one_hot(length: int) -> np.ndarray:
    sequence = np.zeros((length, 4), dtype=np.int8)
    sequence[np.arange(length), np.arange(length) % 4] = 1
    return sequence


def test_reads_array_backed_sequence_zarr(tmp_path):
    path = tmp_path / "array-genome.zarr"
    root = zarr.open_group(path, mode="w")
    root.attrs["assembly"] = "toy"
    chromosomes = root.create_group("chrs")
    sequence = _one_hot(12)
    chromosomes.create_array("chrToy", data=sequence, chunks=(5, 4))

    genome = SequenceDenseZarrIO(path)

    assert genome.assembly == "toy"
    assert genome.chroms == ["chrToy"]
    assert genome.chrom_sizes == {"chrToy": 12}
    assert genome.chunk_size == 5
    np.testing.assert_array_equal(
        genome.get_track("chrToy", 3, 9, output_format="raw_array"),
        sequence[3:9],
    )


def test_reads_legacy_directory_chunked_sequence_zarr_across_boundary(tmp_path):
    path = tmp_path / "chunked-genome.zarr"
    root = zarr.open_group(path, mode="w")
    chromosome = root.create_group("chrs").create_group("chrToy")
    sequence = _one_hot(10)
    chromosome.create_array("chunk_0", data=sequence[:6])
    chromosome.create_array("chunk_1", data=sequence[6:])

    genome = SequenceDenseZarrIO(path)

    assert genome.dir_chunked is True
    assert genome.chrom_sizes == {"chrToy": 10}
    np.testing.assert_array_equal(
        genome.get_track("chrToy", 4, 9, output_format="raw_array"),
        sequence[4:9],
    )
