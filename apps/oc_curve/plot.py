"""Plotting utilities for O-C curve approximation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from apps.oc_curve.logic import OCFitResult, oc_model


def plot_oc_fit(result: OCFitResult) -> Figure:
    """Create a publication-quality plot of the O-C data with the fitted model.

    Returns a matplotlib Figure so the caller can save or embed it.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Observed data
    ax.scatter(result.time, result.oc, s=8, color="black", zorder=3,
               label="Observed O-C data")

    # Smooth fitted curve
    time_smooth = np.linspace(result.time.min(), result.time.max(), 2000)
    oc_smooth = oc_model(time_smooth, result.c0, result.c1,
                         result.amplitude, result.period, result.phase)
    ax.plot(time_smooth, oc_smooth, color="red", linewidth=1.8, zorder=2,
            label="Fitted model (trend + sinusoid)")

    ax.set_xlabel("Time / Epoch (E)", fontsize=12)
    ax.set_ylabel("O-C", fontsize=12)
    ax.set_title("O-C Curve Approximation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
