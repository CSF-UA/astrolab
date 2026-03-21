"""O-C curve approximation logic.

Fits O-C data to a combined polynomial trend + sinusoidal model:
    O-C = (Σ Cj·(E - E0)^j) + R·cos(2π(E - E_max) / P2)

The implementation:
    1. Loads tab-separated (epoch, O-C) data from a text file.
    2. Detrends with a linear fit to obtain initial guesses.
    3. Uses a Lomb-Scargle periodogram (scipy) for the initial period guess.
    4. Performs non-linear curve fitting with scipy.optimize.curve_fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lombscargle


class OCDataError(Exception):
    """Raised when O-C data cannot be loaded or parsed."""


@dataclass
class OCFitResult:
    """Stores the results of the O-C curve fit."""
    c0: float           # constant offset
    c1: float           # linear trend coefficient
    amplitude: float    # semi-amplitude R of sinusoidal component
    period: float       # long (secondary) period P2
    phase: float        # phase in radians
    time: np.ndarray    # time/epoch array used
    oc: np.ndarray      # observed O-C values
    oc_fitted: np.ndarray  # fitted O-C values at the observed epochs


def load_oc_data(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load O-C data from a tab/space-separated text file.

    Expects two columns: Time/Epoch and O-C value.
    No header row. Values may use scientific notation (e.g. 3.84E-05).

    Returns sorted (time, oc) arrays with duplicates removed.
    """
    p = Path(path)
    if not p.exists():
        raise OCDataError(f"O-C data file not found: {p}")

    data_rows: list[Tuple[float, float]] = []
    with p.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                oc = float(parts[1])
                data_rows.append((t, oc))
            except ValueError:
                continue

    if not data_rows:
        raise OCDataError(f"No usable data in {p}")

    data = np.array(data_rows, dtype=float)
    time = data[:, 0]
    oc = data[:, 1]

    # Sort chronologically
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    oc = oc[sort_idx]

    # Remove exact duplicates (same epoch)
    _, unique_idx = np.unique(time, return_index=True)
    time = time[unique_idx]
    oc = oc[unique_idx]

    if time.size < 4:
        raise OCDataError("Need at least 4 data points for the fit.")

    return time, oc


def oc_model(t: np.ndarray, c0: float, c1: float, amplitude: float,
             period: float, phase: float) -> np.ndarray:
    """Combined linear trend + sinusoidal model for O-C data."""
    return c0 + c1 * t + amplitude * np.sin(2 * np.pi * t / period + phase)


def fit_oc_curve(time: np.ndarray, oc: np.ndarray) -> OCFitResult:
    """Fit the O-C data to a linear trend + sinusoid.

    Steps:
        1. Linear detrending via polyfit.
        2. Lomb-Scargle periodogram on the residuals for initial period guess.
        3. Levenberg-Marquardt curve_fit for all 5 parameters.
    """
    # 1. Linear trend for initial guess
    trend_coeffs = np.polyfit(time, oc, 1)  # [slope, intercept]
    oc_detrended = oc - np.polyval(trend_coeffs, time)

    # 2. Lomb-Scargle via scipy.signal.lombscargle
    #    - Need angular frequencies: omega = 2*pi*f
    #    - Build a frequency grid from the data span
    dt = time[-1] - time[0]
    n_pts = len(time)
    freq_min = 1.0 / dt
    freq_max = n_pts / (2.0 * dt)  # Nyquist-like upper bound
    n_freqs = max(1000, 5 * n_pts)
    freqs = np.linspace(freq_min, freq_max, n_freqs)
    angular_freqs = 2.0 * np.pi * freqs

    # Normalize the detrended data (zero-mean) for lombscargle
    oc_zero_mean = oc_detrended - np.mean(oc_detrended)
    power = lombscargle(time, oc_zero_mean, angular_freqs, normalize=True)

    best_freq = freqs[np.argmax(power)]
    guess_period = 1.0 / best_freq

    # Guess amplitude from the detrended data
    guess_amplitude = (np.max(oc_detrended) - np.min(oc_detrended)) / 2.0

    # Initial guesses: [c0 (intercept), c1 (slope), amplitude, period, phase]
    initial_guesses = [trend_coeffs[1], trend_coeffs[0],
                       guess_amplitude, guess_period, 0.0]

    # 3. Curve fitting
    try:
        popt, _ = curve_fit(
            oc_model, time, oc, p0=initial_guesses,
            maxfev=50000,
        )
    except RuntimeError as e:
        raise OCDataError(f"Curve fitting failed to converge: {e}")

    c0_opt, c1_opt, amplitude_opt, period_opt, phase_opt = popt

    # Normalize: ensure amplitude is positive
    if amplitude_opt < 0:
        amplitude_opt = -amplitude_opt
        phase_opt += np.pi

    # Normalize phase to [0, 2π)
    phase_opt = phase_opt % (2 * np.pi)

    # Compute fitted curve at observed epochs
    oc_fitted = oc_model(time, c0_opt, c1_opt, amplitude_opt,
                         period_opt, phase_opt)

    return OCFitResult(
        c0=c0_opt,
        c1=c1_opt,
        amplitude=amplitude_opt,
        period=period_opt,
        phase=phase_opt,
        time=time,
        oc=oc,
        oc_fitted=oc_fitted,
    )
