"""Configuration utilities for parsing and validation.

This module contains utilities for parsing command-line arguments and configuration options.
"""

import logging

logger = logging.getLogger(__name__)


def parse_plot_pairs_string(pairs_string):
    """
    Simple parser for motif pairs string format.
    Returns raw pairs without name resolution.
    """
    if not pairs_string:
        return []

    pairs = []
    pair_specifications = pairs_string.split(";")

    for pair_spec in pair_specifications:
        try:
            motif_i, motif_j = pair_spec.strip().split(",")
            pairs.append((motif_i.strip(), motif_j.strip()))
        except ValueError:
            logger.warning(f"Invalid motif pair specification: '{pair_spec}'")
            continue

    return pairs
