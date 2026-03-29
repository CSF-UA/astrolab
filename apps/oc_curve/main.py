"""Standalone entry point for the O-C curve approximation tool.

Usage:
    uv run oc_curve                       # opens a file dialog
    uv run oc_curve path/to/data.txt      # loads the file directly
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "apps.oc_curve"

import matplotlib
matplotlib.use("qtagg")  # Use Qt-based backend to match PySide6

import numpy as np
import matplotlib.pyplot as plt

from apps.oc_curve.logic import load_oc_data, fit_oc_curve, OCDataError
from apps.oc_curve.plot import plot_oc_fit


def _choose_file() -> Path:
    """Open a file dialog so the user can pick any O-C data file."""
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open O-C Data File",
            str(Path.home()),
            "Text files (*.txt *.dat *.csv);;All files (*)",
        )
        if not path:
            print("No file selected. Exiting.")
            sys.exit(0)
        return Path(path)
    except ImportError:
        # Fallback: use tkinter
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Open O-C Data File",
            filetypes=[
                ("Text files", "*.txt *.dat *.csv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not path:
            print("No file selected. Exiting.")
            sys.exit(0)
        return Path(path)


def main() -> None:
    # Determine the data file path
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        filepath = _choose_file()

    print(f"Loading O-C data from: {filepath}")

    try:
        time, oc = load_oc_data(filepath)
    except OCDataError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(time)} data points (epochs {time[0]:.3f} – {time[-1]:.3f})")

    # Compute default period bounds so the user can see them
    delta_e = time[-1] - time[0]
    default_p_min = 100.0
    default_p_max = 3.0 * delta_e

    # Prompt for optional period bounds
    raw_min = input(
        f"  Period lower bound [default: {default_p_min:.1f}]: "
    ).strip()
    raw_max = input(
        f"  Period upper bound [default: {default_p_max:.1f}]: "
    ).strip()

    period_min = float(raw_min) if raw_min else None
    period_max = float(raw_max) if raw_max else None

    try:
        result = fit_oc_curve(time, oc,
                              period_min=period_min,
                              period_max=period_max)
    except OCDataError as e:
        print(f"Error during fitting: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print()
    print("=" * 52)
    print("  Fitted Parameters")
    print("=" * 52)
    print(f"  Period (P₂):    {result.period:.5f}")
    print(f"  Amplitude (R):  {result.amplitude:.8f}")
    print(f"  Phase (φ):      {result.phase:.5f} rad")
    print(f"  Linear C₀:     {result.c0:.10f}")
    print(f"  Linear C₁:     {result.c1:.10e}")
    print(f"  RMS residual:   {result.rms:.8f}")
    print("-" * 52)

    # Parameter uncertainties
    if result.param_errors is not None and not np.all(np.isinf(result.param_errors)):
        errs = result.param_errors
        print("  Parameter uncertainties (σ):")
        print(f"    σ(C₀):       {errs[0]:.10e}")
        print(f"    σ(C₁):       {errs[1]:.10e}")
        print(f"    σ(R):         {errs[2]:.10e}")
        print(f"    σ(P₂):       {errs[3]:.5f}")
        print(f"    σ(φ):         {errs[4]:.5f}")
    else:
        print("  ⚠  Parameter uncertainties unavailable (degenerate pcov).")

    print("-" * 52)

    # BIC model comparison
    print("  Model comparison (BIC):")
    print(f"    BIC (linear):     {result.bic_linear:.4f}")
    print(f"    BIC (sinusoidal): {result.bic_sinusoidal:.4f}")
    if result.sinusoid_preferred:
        print("    → Sinusoidal model preferred.")
    else:
        print("    → Linear model sufficient (sinusoid not justified).")
    print("=" * 52)
    print()

    # Plot
    fig = plot_oc_fit(result)
    plt.show()


if __name__ == "__main__":
    main()
