"""Standalone entry point for the Maxima Parameters tool.

Computes re-radiation parameters and O'Connell effect from eclipsing
binary light curves.

Usage:
    uv run maxima_params                              # file dialogs
    uv run maxima_params curve.tess intervals.txt     # CLI args
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "apps.maxima_params"

from apps.maxima_params.logic import (
    AnalysisResult,
    DataLoadError,
    analyze_maxima,
    load_light_curve,
)


def _choose_file(title: str = "Open file", filter_str: str = "All files (*)") -> Path:
    """Open a Qt file dialog."""
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog

        app = QApplication.instance() or QApplication(sys.argv)
        path, _ = QFileDialog.getOpenFileName(
            None,
            title,
            str(Path.home()),
            filter_str,
        )
        if not path:
            print("No file selected. Exiting.")
            sys.exit(0)
        return Path(path)
    except ImportError:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title)
        root.destroy()
        if not path:
            print("No file selected. Exiting.")
            sys.exit(0)
        return Path(path)


def _load_intervals_file(path: Path) -> tuple[list[int], list[int]]:
    """Load start/end index pairs from a whitespace-separated text file."""
    starts, ends = [], []
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                s, e = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if 0 <= s < e:
                starts.append(s)
                ends.append(e)
    if not starts:
        raise DataLoadError(f"No usable intervals in {path}")
    return starts, ends


def _print_results(result: AnalysisResult) -> None:
    """Pretty-print analysis results to stdout."""
    print()
    print("=" * 64)
    print("  Reflection effect  (Δm_refl = m_start − m_end)")
    print("=" * 64)
    if not result.maxima:
        print("  No maxima found.")
    else:
        print(f"  {'#':>3}  {'T0':>16}  {'m_start':>10}  {'m_end':>10}  {'Δm_refl':>10}")
        print(f"  {'---':>3}  {'---':>16}  {'---':>10}  {'---':>10}  {'---':>10}")
        for m in result.maxima:
            print(
                f"  {m.index + 1:3d}  {m.t0:16.6f}  "
                f"{m.m_start:10.6f}  {m.m_end:10.6f}  {m.delta_m_refl:+10.6f}"
            )
    print()
    print("=" * 64)
    print("  O'CONNELL EFFECT  (Δm = m_max2 − m_max1)")
    print("=" * 64)
    if not result.oconnell:
        print("  Not enough consecutive maxima to compute O'Connell effect.")
    else:
        print(f"  {'Pair':>6}  {'JD_max1':>16}  {'JD_max2':>16}  {'m_max1':>10}  {'m_max2':>10}  {'Δm':>10}")
        print(f"  {'----':>6}  {'---':>16}  {'---':>16}  {'---':>10}  {'---':>10}  {'---':>10}")
        for oc in result.oconnell:
            pair_str = f"{oc.max1_index + 1}-{oc.max2_index + 1}"
            print(
                f"  {pair_str:>6}  {oc.t0_max1:16.6f}  {oc.t0_max2:16.6f}  "
                f"{oc.y_max1:10.6f}  {oc.y_max2:10.6f}  {oc.delta_m:+10.6f}"
            )
    print()


def main() -> None:
    # Parse arguments: both light curve and intervals are required
    lc_path: Path | None = None
    iv_path: Path | None = None

    args = sys.argv[1:]
    if len(args) >= 2:
        lc_path = Path(args[0])
        iv_path = Path(args[1])
    elif len(args) == 1:
        lc_path = Path(args[0])

    # Prompt for missing files via dialog
    if lc_path is None:
        lc_path = _choose_file(
            "Select light curve (.tess)",
            "TESS files (*.tess);;All files (*)",
        )
    if iv_path is None:
        iv_path = _choose_file(
            "Select intervals file (.txt)",
            "Text files (*.txt);;All files (*)",
        )

    # Load light curve
    print(f"Loading light curve from: {lc_path}")
    try:
        times, mags = load_light_curve(lc_path)
    except DataLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(times):,} points  (time: {times[0]:.4f} – {times[-1]:.4f})")

    # Load intervals (mandatory)
    print(f"Loading intervals from: {iv_path}")
    try:
        interval_starts, interval_ends = _load_intervals_file(iv_path)
    except DataLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(interval_starts)} intervals")

    # Ask for period T0
    try:
        T0_input = input("Enter the orbital period P (days): ").strip()
        T0 = float(T0_input)
    except (ValueError, EOFError):
        print("Invalid period. Exiting.", file=sys.stderr)
        sys.exit(1)

    if T0 <= 0:
        print("Period must be positive. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning analysis with P = {T0:.8f} days ...")
    result = analyze_maxima(
        times, mags, T0,
        interval_starts=interval_starts,
        interval_ends=interval_ends,
    )

    _print_results(result)
    print(f"Found {len(result.maxima)} maxima, {len(result.oconnell)} O'Connell pairs.")


if __name__ == "__main__":
    main()
