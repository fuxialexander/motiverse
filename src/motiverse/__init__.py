"""Genome motif analysis module.

This module provides comprehensive motif analysis capabilities including
co-occurrence analysis, query-centered analysis, signal aggregation,
and sequence content analysis.

**Note:** This module requires PyTorch. Install with: ``pip install motiverse``.

For backward compatibility, all functions from the original genome_motif_analysis.py
file are imported and available at the module level.
"""

from __future__ import annotations

from importlib.util import find_spec


class TorchNotAvailableError(ImportError):
    """Raised when a torch-backed Motiverse operation is requested."""

    def __init__(self, feature: str):
        super().__init__(
            f"{feature} requires PyTorch. Install Motiverse with `pip install motiverse`."
        )


def is_torch_available() -> bool:
    return find_spec("torch") is not None


# Check if torch is available before importing torch-dependent modules
if not is_torch_available():
    # Create placeholder module that raises informative error on access
    def _raise_torch_error(*args, **kwargs):
        raise TorchNotAvailableError("Genome motif analysis")

    # Placeholder classes/functions that raise errors when called
    class _TorchRequiredPlaceholder:
        def __init__(self, *args, **kwargs):
            raise TorchNotAvailableError("Genome motif analysis")

        def __call__(self, *args, **kwargs):
            raise TorchNotAvailableError("Genome motif analysis")

    # Set all exports to placeholders
    accumulate_motif_cooccurrences = _raise_torch_error
    accumulate_around_query_motif = _raise_torch_error
    accumulate_signal_around_hits = _raise_torch_error
    accumulate_sequences_around_hits = _raise_torch_error
    initialize_analysis_tensor = _raise_torch_error
    apply_accumulation_strategy = _raise_torch_error
    process_chromosome_sequences = _raise_torch_error
    process_narrowpeak_regions = _raise_torch_error
    StreamingHitCollector = _TorchRequiredPlaceholder
    collect_max_hits_per_peak = _raise_torch_error
    collect_sum_hits_per_peak = _raise_torch_error
    load_hocomoco_motifs = _raise_torch_error
    PositionalMotifHitCache = _TorchRequiredPlaceholder
    build_positional_motif_hit_cache = _raise_torch_error
    threshold_vector_checksum = _raise_torch_error
    MultiSubsetAggregationPlan = _TorchRequiredPlaceholder
    MultiSubsetAggregationResult = _TorchRequiredPlaceholder
    MultiSubsetZarrResult = _TorchRequiredPlaceholder
    IntervalContributionCacheResult = _TorchRequiredPlaceholder
    accumulate_region_subsets = _raise_torch_error
    accumulate_region_subsets_from_interval_contribution_cache = _raise_torch_error
    accumulate_region_subsets_from_interval_contribution_cache_to_zarr = _raise_torch_error
    accumulate_region_subsets_to_zarr = _raise_torch_error
    build_interval_contribution_cache = _raise_torch_error
    plan_region_subset_aggregation = _raise_torch_error
    DenseSubsetAggregationResult = _TorchRequiredPlaceholder
    DenseIntervalContributionCacheResult = _TorchRequiredPlaceholder
    DenseIntervalContributionCachePlan = _TorchRequiredPlaceholder
    DenseQueryMotifHitCacheResult = _TorchRequiredPlaceholder
    DenseSubsetResourcePlan = _TorchRequiredPlaceholder
    DenseSubsetZarrResult = _TorchRequiredPlaceholder
    DenseGenomeScoreProvider = _TorchRequiredPlaceholder
    DenseGenomeBlockScoreProvider = _TorchRequiredPlaceholder
    accumulate_dense_score_region_subsets = _raise_torch_error
    accumulate_dense_score_region_subsets_from_contribution_cache = _raise_torch_error
    accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr = _raise_torch_error
    accumulate_dense_score_region_subsets_from_query_motif_hit_cache = _raise_torch_error
    accumulate_dense_score_region_subsets_to_zarr = _raise_torch_error
    accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter = _raise_torch_error
    build_dense_interval_contribution_cache = _raise_torch_error
    build_dense_query_motif_hit_cache = _raise_torch_error
    build_dense_query_motif_hit_cache_from_tiles = _raise_torch_error
    dense_region_contribution_from_scores = _raise_torch_error
    dense_subset_command = _raise_torch_error
    resolve_query_motif_index = _raise_torch_error
    run_full_curve_one_to_all_reuse = _raise_torch_error
    plan_dense_interval_contribution_cache_resources = _raise_torch_error
    plan_dense_score_region_subset_resources = _raise_torch_error
    generate_analysis_plots = _raise_torch_error
    create_query_motif_heatmap = _raise_torch_error
    get_signal_data = _raise_torch_error
    load_signal_database = _raise_torch_error
    download_remap_narrowpeak = _raise_torch_error
    save_analysis_results = _raise_torch_error
    parse_plot_pairs_string = _raise_torch_error
    analyze_genome_sequences = _raise_torch_error

else:
    # Import all functions when torch is available
    from .accumulation import (
        accumulate_around_query_motif,
        accumulate_motif_cooccurrences,
        accumulate_sequences_around_hits,
        accumulate_signal_around_hits,
        apply_accumulation_strategy,
        initialize_analysis_tensor,
    )
    from .collection import (
        StreamingHitCollector,
        collect_max_hits_per_peak,
        collect_sum_hits_per_peak,
    )
    from .config_utils import (
        parse_plot_pairs_string,
    )
    from .data_utils import (
        download_remap_narrowpeak,
        get_signal_data,
        load_signal_database,
        save_analysis_results,
    )
    from .dense_subset import (
        DenseGenomeBlockScoreProvider,
        DenseGenomeScoreProvider,
        DenseIntervalContributionCachePlan,
        DenseIntervalContributionCacheResult,
        DenseQueryMotifHitCacheResult,
        DenseSubsetAggregationResult,
        DenseSubsetResourcePlan,
        DenseSubsetZarrResult,
        accumulate_dense_score_region_subsets,
        accumulate_dense_score_region_subsets_from_contribution_cache,
        accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr,
        accumulate_dense_score_region_subsets_from_query_motif_hit_cache,
        accumulate_dense_score_region_subsets_to_zarr,
        accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter,
        build_dense_interval_contribution_cache,
        build_dense_query_motif_hit_cache,
        build_dense_query_motif_hit_cache_from_tiles,
        dense_region_contribution_from_scores,
        plan_dense_interval_contribution_cache_resources,
        plan_dense_score_region_subset_resources,
    )
    from .full_curve_reuse import (
        dense_subset_command,
        resolve_query_motif_index,
        run_full_curve_one_to_all_reuse,
    )
    from .main import (
        analyze_genome_sequences,
    )
    from .motif_source import (
        load_hocomoco_motifs,
    )
    from .positional_cache import (
        PositionalMotifHitCache,
        build_positional_motif_hit_cache,
        threshold_vector_checksum,
    )
    from .processing import (
        process_chromosome_sequences,
        process_narrowpeak_regions,
    )
    from .subset_planner import (
        IntervalContributionCacheResult,
        MultiSubsetAggregationPlan,
        MultiSubsetAggregationResult,
        MultiSubsetZarrResult,
        accumulate_region_subsets,
        accumulate_region_subsets_from_interval_contribution_cache,
        accumulate_region_subsets_from_interval_contribution_cache_to_zarr,
        accumulate_region_subsets_to_zarr,
        build_interval_contribution_cache,
        plan_region_subset_aggregation,
    )

    try:
        from .visualization import (
            create_query_motif_heatmap,
            generate_analysis_plots,
        )
    except ImportError:

        def _raise_plot_error(*args, **kwargs):
            raise ImportError(
                "Plotting requires the optional dependencies. Install with "
                "`pip install motiverse[plots]`."
            )

        create_query_motif_heatmap = _raise_plot_error
        generate_analysis_plots = _raise_plot_error

# For complete backward compatibility, expose all functions at module level
__all__ = [
    # Core accumulation functions
    "accumulate_motif_cooccurrences",
    "accumulate_around_query_motif",
    "accumulate_signal_around_hits",
    "accumulate_sequences_around_hits",
    "initialize_analysis_tensor",
    "apply_accumulation_strategy",
    # Processing functions
    "process_chromosome_sequences",
    "process_narrowpeak_regions",
    # Collection classes and functions
    "StreamingHitCollector",
    "collect_max_hits_per_peak",
    "collect_sum_hits_per_peak",
    "load_hocomoco_motifs",
    "PositionalMotifHitCache",
    "build_positional_motif_hit_cache",
    "threshold_vector_checksum",
    "MultiSubsetAggregationPlan",
    "MultiSubsetAggregationResult",
    "MultiSubsetZarrResult",
    "IntervalContributionCacheResult",
    "accumulate_region_subsets",
    "accumulate_region_subsets_from_interval_contribution_cache",
    "accumulate_region_subsets_from_interval_contribution_cache_to_zarr",
    "accumulate_region_subsets_to_zarr",
    "build_interval_contribution_cache",
    "plan_region_subset_aggregation",
    "DenseSubsetAggregationResult",
    "DenseIntervalContributionCacheResult",
    "DenseIntervalContributionCachePlan",
    "DenseSubsetResourcePlan",
    "DenseSubsetZarrResult",
    "DenseQueryMotifHitCacheResult",
    "DenseGenomeScoreProvider",
    "DenseGenomeBlockScoreProvider",
    "accumulate_dense_score_region_subsets",
    "accumulate_dense_score_region_subsets_from_contribution_cache",
    "accumulate_dense_score_region_subsets_from_contribution_cache_to_zarr",
    "accumulate_dense_score_region_subsets_from_query_motif_hit_cache",
    "accumulate_dense_score_region_subsets_to_zarr",
    "accumulate_dense_score_region_subsets_with_query_motif_hit_prefilter",
    "build_dense_interval_contribution_cache",
    "build_dense_query_motif_hit_cache",
    "build_dense_query_motif_hit_cache_from_tiles",
    "dense_region_contribution_from_scores",
    "dense_subset_command",
    "resolve_query_motif_index",
    "run_full_curve_one_to_all_reuse",
    "plan_dense_interval_contribution_cache_resources",
    "plan_dense_score_region_subset_resources",
    # Visualization functions
    "generate_analysis_plots",
    "create_query_motif_heatmap",
    # Data utilities
    "get_signal_data",
    "load_signal_database",
    "download_remap_narrowpeak",
    "save_analysis_results",
    # Configuration utilities
    "parse_plot_pairs_string",
    # Main functions
    "analyze_genome_sequences",
]
