"""
Plotting utilities for genome scanning analysis.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_cooccurrence_matrix(
    accumulation_tensor,
    motif_names_i=None,
    motif_names_j=None,
    smoothing_window=1,
    figsize_multiplier=(4, 3),
):
    """
    Plots a matrix of co-occurrence curves from an accumulation tensor.

    Args:
        accumulation_tensor (torch.Tensor): Shape (M, M, W).
        motif_names_i (list): Names for rows.
        motif_names_j (list): Names for columns.
        smoothing_window (int): Size of convolution window for smoothing.
        figsize_multiplier (tuple): Figure size per subplot.
    """
    num_rows = accumulation_tensor.shape[0]
    num_cols = accumulation_tensor.shape[1]
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(figsize_multiplier[0] * num_cols, figsize_multiplier[1] * num_rows),
        sharex=True,
        sharey=False,
    )

    for i in range(num_rows):
        for j in range(num_cols):
            curve = accumulation_tensor[i][j].float().numpy()
            if smoothing_window > 1:
                curve = np.convolve(
                    curve, np.ones(smoothing_window) / smoothing_window, mode="valid"
                )

            ax = (
                axes[i, j]
                if num_rows > 1 and num_cols > 1
                else axes[max(i, j)]
                if (num_rows > 1 or num_cols > 1)
                else axes
            )
            sns.lineplot(ax=ax, data=curve)
            title = f"[{i}][{j}]"
            if motif_names_i and motif_names_j:
                title = f"{motif_names_i[i]} vs {motif_names_j[j]}"
            ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_power_spectrum(y):
    """
    Calculates and plots the power spectrum of a 1D signal.

    Args:
        y (np.array): Input signal.
    """
    N = len(y)
    y_fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(N)
    magnitude = np.abs(y_fft)

    plt.figure()
    plt.plot(freqs[: N // 2], magnitude[: N // 2])  # Plot positive frequencies
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain")
    plt.show()

    # Find dominant frequencies
    sorted_indices = np.argsort(magnitude[1 : N // 2])[::-1]
    dominant_freqs = freqs[1 : N // 2][sorted_indices[:5]]
    periods = 1 / dominant_freqs
    print("Top 5 dominant frequencies and periods:")
    for f, p in zip(dominant_freqs, periods):
        print(f"  Freq: {f:.4f}, Period: {p:.2f}")


def plot_autocorrelation(curve, max_delay=500):
    """
    Calculates and plots the autocorrelation of a signal.

    Args:
        curve (np.array): Input signal.
        max_delay (int): Maximum delay to calculate autocorrelation for.
    """
    autocorrs = []
    for delay in range(1, max_delay):
        autocorr = np.corrcoef(curve[:-delay], curve[delay:])[0, 1]
        autocorrs.append(autocorr)

    plt.figure()
    sns.lineplot(x=range(1, max_delay), y=autocorrs)
    plt.xlabel("Delay")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Plot")
    plt.show()


def read_hocomoco_metrics(annotation_file):
    """
    Read HOCOMOCO annotation file and extract quality metrics.

    Args:
        annotation_file (str or Path): Path to the annotation file.

    Returns:
        pd.DataFrame: DataFrame with TF, chipseq, and selex quality scores.
    """
    records = []

    with Path(annotation_file).open() as f:
        for line in f:
            data = json.loads(line.strip())
            name = data["name"]
            tf_name = name.split(".")[0]
            chipseq_score = None
            selex_score = None

            if "ChIP-Seq" in data["metrics_summary"]:
                if "pwmeval" in data["metrics_summary"]["ChIP-Seq"]["overall"]:
                    chipseq_score = data["metrics_summary"]["ChIP-Seq"]["overall"]["pwmeval"][
                        "pr_auc"
                    ]["max"]
                else:
                    chipseq_score = data["metrics_summary"]["ChIP-Seq"]["overall"][
                        "pwmeval_chipseq"
                    ]["pr_auc"]["max"]
            if "HT-SELEX" in data["metrics_summary"]:
                selex_score = data["metrics_summary"]["HT-SELEX"]["overall"]["pwmeval_selex_10"][
                    "pr_auc"
                ]["max"]

            records.append(
                {"TF": tf_name, "name": name, "chipseq": chipseq_score, "selex": selex_score}
            )

    annotations = pd.DataFrame(records)
    return annotations
