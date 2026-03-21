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

    try:
        result = fit_oc_curve(time, oc)
    except OCDataError as e:
        print(f"Error during fitting: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print()
    print("=" * 48)
    print("  Determined Sinusoidal Parameters")
    print("=" * 48)
    print(f"  Period (P₂):    {result.period:.5f}")
    print(f"  Amplitude (R):  {result.amplitude:.8f}")
    print(f"  Phase:          {result.phase:.5f} rad")
    print(f"  Linear C₀:     {result.c0:.10f}")
    print(f"  Linear C₁:     {result.c1:.10e}")
    print("=" * 48)

    # Residuals
    residuals = result.oc - result.oc_fitted
    rms = np.sqrt(np.mean(residuals**2))
    print(f"  RMS residual:   {rms:.8f}")
    print()

    # Plot
    fig = plot_oc_fit(result)
    plt.show()


if __name__ == "__main__":
    main()
