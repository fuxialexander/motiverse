# Motiverse

Motiverse is a standalone Python package for scanning genomic sequence with motif kernels and preserving the full positional curves needed to compare motif organization across peak groups.

It was extracted from Caesarion's genome motif analysis module. Motiverse now owns its core computation, dense sequence-Zarr reader, peak membership construction, sparse hit storage, and command-line interface. Caesarion is not required.

## What it answers

- Which motifs occur near a query motif, and at what offsets?
- How do full motif-position profiles differ across overlapping peak groups?
- Which loci are exclusive to one peak set versus another?
- Which motifs best distinguish biological contexts?
- Can repeated region families reuse sequence scans without reducing the result to scalar or top-k summaries?

The scalable workflow keeps the complete `(group, motif, position)` output. Use
an exact motif name (including its HOCOMOCO suffix) or `--query-motif-index`;
short family names can match both forward and reverse-complement entries.

## Interactive atlas

[Explore the 96-species comparative transposable-element atlas](https://fuxialexander.github.io/motiverse/atlas/), with 111,980 recomputed target points, repeat-divergence and Dfam-based evolutionary-age views. Scores reproduce the validated Figure 4 windows; all 96 species are available in divergence mode and CSV export.

## Install

Motiverse is currently installed from GitHub rather than PyPI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install \
  "motiverse[motifs] @ git+https://github.com/fuxialexander/motiverse.git"
motiverse --help
```

For development:

```bash
git clone https://github.com/fuxialexander/motiverse.git
cd motiverse
python -m pip install -e ".[motifs,plots,dev]"
pytest
```

Dependency sets:

- Base: NumPy, pandas, PyTorch, tqdm, Zarr, and numcodecs.
- `motifs`: HOCOMOCO loading and p-value mappings through `gcell`.
- `plots`: matplotlib and seaborn.
- `caesarion`: optional interoperability with the broader Caesarion toolkit; Motiverse itself does not require it.
- `dev`: tests, build tooling, and linting.

## Inputs

Genome sequences use a Zarr group with one-hot arrays in this layout:

```text
genome.zarr/
  attrs: assembly="hg38"
  chrs/
    chr1: shape (chromosome_length, 4)
    chr2: shape (chromosome_length, 4)
```

The older layout where each chromosome is a group of `chunk_0`, `chunk_1`, ... arrays is also supported.

Peak-group membership is a tab-separated table with:

```text
subset_id  chrom  start  end
group_a    chr1   1000   3000
group_b    chr1   1800   3800
```

Coordinates are 0-based, half-open.

## Quick workflow

Set paths once:

```bash
export MOTIVERSE_GENOME_ZARR=/path/to/hg38.zarr
export MOTIVERSE_ALIGNED_MOTIFS=/path/to/motifs_with_rc_aligned.pt
```

### 1. Prepare groups from ChIP-Atlas

```bash
motiverse prepare-chip-atlas-groups \
  --bed chip_atlas.bed \
  --output-dir prepared-groups
```

### 2. Profile every group

```bash
motiverse profile-peak-groups \
  --regions-tsv prepared-groups/chip_atlas_context_memberships.tsv \
  --query-motif CTCF.H13CORE.0.P.B \
  --output-dir results/ctcf-groups
```

This writes a complete full-curve profile for every retained group. Use `--genome-zarr` and `--aligned-motif-path` instead of environment variables when preferred.

### 3. Compare peak sets directly

```bash
motiverse compare-peak-sets \
  --peak-set treated=treated.narrowPeak \
  --peak-set control=control.narrowPeak \
  --query-motif TP53 \
  --output-dir results/treated-v-control
```

Motiverse constructs a shared union of overlap-connected loci and profiles the ordered set differences.

### 4. Rank context differences

```bash
motiverse rank-context-motifs \
  --values-zarr results/ctcf-groups/full_curve_one_to_all.zarr \
  --regions-tsv prepared-groups/chip_atlas_context_memberships.tsv \
  --output-dir results/ctcf-groups/context-ranking
```

Run `motiverse <command> --help` for the exact options supported by the installed version.

## Python API

```python
import torch
from motiverse.processing import _scan_motifs_conv1d

sequence = torch.tensor(
    [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]],
    dtype=torch.float32,
).permute(0, 2, 1)
motifs = torch.tensor(
    [[[1, 0, 0, 0], [0, 1, 0, 0]]],
    dtype=torch.float32,
).permute(0, 2, 1)

scores = _scan_motifs_conv1d(sequence, motifs)
```

Higher-level functions are exported from `motiverse`, while focused modules remain importable for reproducible workflows.

## Output and correctness

The primary biological output is a full positional tensor rather than a scalar screen. Workflow summaries record motif provenance, coordinate semantics, cache checksums, resource plans, and validation status. Core convolution, accumulation, positional-cache, dense-reuse, and subset-aggregation behavior is covered by synthetic regression tests.

## Relationship to Caesarion

Motiverse was extracted from `caesar.analysis.genome_motif_analysis` at Caesarion commit `af0212e`. Package imports and subprocess entry points now use `motiverse`; the only optional Caesarion relationship is the `caesarion` installation extra for users who also need Caesarion's broader multi-omics IO.
